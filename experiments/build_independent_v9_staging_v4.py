#!/usr/bin/env python3
"""Unscored ten-family Typesense/Solr task-validity staging generator."""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import sha256_file, write_checksums, write_json, write_jsonl  # noqa: E402
from experiments.build_independent_v9_staging import _candidate  # noqa: E402
from experiments.build_independent_v9_staging_v2 import (  # noqa: E402
    SOURCES as BASE_SOURCES,
    VERSIONS,
    _case as _base_case,
    _contract,
    _core_create,
)

SOURCES = {
    **BASE_SOURCES,
    "typesense_curation_set_assignment": "https://typesense.org/docs/30.2/api/curation.html",
    "typesense_preset_assignment": "https://typesense.org/docs/30.2/api/search.html",
    "solr_request_params_assignment": "https://solr.apache.org/guide/solr/9_10/configuration-guide/request-parameters-api.html",
    "solr_dynamic_field_assignment": "https://solr.apache.org/guide/solr/9_10/indexing-guide/schema-api.html",
}
START = datetime(2026, 10, 18, 16, 0, tzinfo=UTC)


def _case(family: str, number: int, positive: bool) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
    if family in BASE_SOURCES:
        return _base_case(family, number, positive)
    suffix = f"{number:03d}"
    if family == "typesense_curation_set_assignment":
        target, decoy = f"v9_cur_target_{suffix}", f"v9_cur_decoy_{suffix}"
        amber, violet = f"amber{suffix}", f"violet{suffix}"
        left, right = (amber, violet) if positive else (violet, amber)
        query = f"Create curation set {target} with exact rule query {amber}, and set {decoy} with exact rule query {violet}; inspect both."
        plans = [
            {"method": "PUT", "path": f"/curation_sets/{name}", "json": {"items": [{
                "id": "rule", "rule": {"query": word, "match": "exact"},
                "excludes": [{"id": "1"}],
            }]}}
            for name, word in ((target, left), (decoy, right))
        ] + [{"method": "GET", "path": f"/curation_sets/{target}"}]
        checks = [
            {"path": f"/curation_sets/{name}", "status": 200, "assertions": [
                {"path": ["name"], "equals": name},
                {"path": ["items", 0, "rule", "query"], "equals": word},
            ]}
            for name, word in ((target, amber), (decoy, violet))
        ]
        return "typesense", query, plans, _contract([200] * 3, checks)
    if family == "typesense_preset_assignment":
        target, decoy = f"v9_preset_target_{suffix}", f"v9_preset_decoy_{suffix}"
        amber, violet = f"amber{suffix}", f"violet{suffix}"
        left, right = (amber, violet) if positive else (violet, amber)
        query = f"Create search preset {target} filtering stage to {amber} and preset {decoy} filtering stage to {violet}; inspect both."
        plans = [
            {"method": "PUT", "path": f"/presets/{name}", "json": {"value": {
                "q": "*", "query_by": "title", "filter_by": f"stage:={value}"}}}
            for name, value in ((target, left), (decoy, right))
        ] + [{"method": "GET", "path": f"/presets/{target}"}]
        checks = [
            {"path": f"/presets/{name}", "status": 200, "assertions": [
                {"path": ["name"], "equals": name},
                {"path": ["value", "filter_by"], "equals": f"stage:={value}"},
            ]}
            for name, value in ((target, amber), (decoy, violet))
        ]
        return "typesense", query, plans, _contract([200] * 3, checks)
    if family == "solr_request_params_assignment":
        target, decoy = f"v9_params_target_{suffix}", f"v9_params_decoy_{suffix}"
        small, large = 3 + number, 20 + number
        left, right = (small, large) if positive else (large, small)
        query = f"Create Solr cores {target} and {decoy}; set parameter set corem rows to {small} on {target} and {large} on {decoy}; inspect both."
        plans = [_core_create(name) for name in (target, decoy)] + [
            {"method": "POST", "path": f"/solr/{name}/config/params", "json": {
                "set": {"corem": {"rows": str(value)}}}}
            for name, value in ((target, left), (decoy, right))
        ] + [{"method": "GET", "path": f"/solr/{target}/config/params/corem?wt=json"}]
        checks = [
            {"path": f"/solr/{name}/config/params/corem?wt=json", "status": 200, "assertions": [
                {"path": ["response", "params", "corem", "rows"], "equals": str(value)},
            ]}
            for name, value in ((target, small), (decoy, large))
        ]
        return "solr", query, plans, _contract([200] * 5, checks)
    if family == "solr_dynamic_field_assignment":
        target, decoy = f"v9_dyn_target_{suffix}", f"v9_dyn_decoy_{suffix}"
        amber, violet = f"amber{suffix}_*", f"violet{suffix}_*"
        left, right = (amber, violet) if positive else (violet, amber)
        query = f"Create Solr cores {target} and {decoy}; add dynamic string field pattern {amber} to {target} and {violet} to {decoy}; inspect target."
        plans = [_core_create(name) for name in (target, decoy)] + [
            {"method": "POST", "path": f"/solr/{name}/schema", "json": {
                "add-dynamic-field": {"name": pattern, "type": "string", "stored": True}}}
            for name, pattern in ((target, left), (decoy, right))
        ] + [{"method": "GET", "path": f"/solr/{target}/schema/dynamicfields/{amber.replace('*', '%2A')}?wt=json"}]
        checks = [
            {"path": f"/solr/{name}/schema/dynamicfields/{pattern.replace('*', '%2A')}?wt=json",
             "status": 200, "assertions": [
                 {"path": ["dynamicField", "name"], "equals": pattern},
                 {"path": ["dynamicField", "type"], "equals": "string"},
             ]}
            for name, pattern in ((target, amber), (decoy, violet))
        ]
        return "solr", query, plans, _contract([200] * 5, checks)
    raise ValueError(f"unsupported v9 staging family: {family}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-label-output", type=Path, required=True)
    parser.add_argument("--salt-file", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.private_label_output.exists():
        parser.error("refusing to overwrite v9 staging")
    salt = args.salt_file.read_bytes()
    if len(salt) != 32:
        raise ValueError("private assignment salt must be 32 bytes")
    events, queries, memories, edges, candidates, oracle, negative = [], [], [], [], [], [], []
    for family_index, family in enumerate(SOURCES):
        sides = [0] * 4 + [1] * 4
        seed = hashlib.sha256(salt + f"v9-v4-assignment|{family}".encode()).digest()
        random.Random(int.from_bytes(seed[:8], "big")).shuffle(sides)
        for offset in range(8):
            number = offset + 1
            event_id = f"v9-staging-v4:{family}:event-{number:03d}"
            service, query, plans, contract = _case(family, number, True)
            event = {
                "event_id": event_id, "incident_id": event_id,
                "service_id": service, "provider_id": service,
                "version_id": VERSIONS[service],
                "event_time": (START + timedelta(minutes=family_index * 8 + offset)).isoformat().replace("+00:00", "Z"),
                "query": query, "task_key": family, "operation_count": len(plans),
                "workflow_contract": contract, "checker_contract_id": "independent-v9-assertions-v1",
                "required_evidence_atoms": [f"role:step{index}" for index in range(1, len(plans) + 1)] + ["role:verify"],
                "required_binding_atoms": [], "observation_tier": "prospective_unscored_staging",
            }
            events.append(event)
            queries.append({key: event[key] for key in ("event_id", "service_id", "provider_id", "version_id", "event_time", "query")})
            for side in (0, 1):
                correct = side == sides[offset]
                _service, _query, candidate_plans, _contract_unused = _case(family, number, correct)
                rows, edge, candidate = _candidate(
                    salt, event_id, family, service, side, candidate_plans, SOURCES[family]
                )
                memories.extend(rows)
                edges.append(edge)
                candidates.append(candidate)
                selection = {
                    "run_id": f"v9-staging-v4:{event_id}:{side}",
                    "event_id": event_id, "method": "oracle" if correct else "complete_negative",
                    "candidate_edge_id": edge["edge_id"],
                    "selected_memory_ids": candidate["selected_memory_ids"],
                    "token_cost": 24 * len(candidate["selected_memory_ids"]),
                }
                (oracle if correct else negative).append(selection)
    if Counter(row["service_id"] for row in events) != {"typesense": 40, "solr": 40}:
        raise ValueError("provider balance failed")
    args.output.mkdir(parents=True)
    files = {"events.jsonl": events, "query_only_events.jsonl": queries,
             "memories.jsonl": memories, "dependency_hyperedges.jsonl": edges,
             "candidate_selections.jsonl": candidates}
    for name, rows in files.items():
        write_jsonl(args.output / name, rows)
    write_json(args.output / "manifest.json", {
        "status": "prospective_unscored_staging_only",
        "claim_boundary": "Ten of sixteen planned families; no model score or independent method claim.",
        "families": list(SOURCES), "events": len(events), "candidate_rows": len(candidates),
        "source_uris": SOURCES, "assignment_salt_commitment_sha256": hashlib.sha256(salt).hexdigest(),
        "generator_sha256": sha256_file(Path(__file__)),
    })
    write_checksums(args.output, (*files, "manifest.json"))
    args.private_label_output.mkdir(parents=True, mode=0o700)
    write_jsonl(args.private_label_output / "oracle_selections.jsonl", oracle)
    write_jsonl(args.private_label_output / "complete_negative_selections.jsonl", negative)
    write_json(args.private_label_output / "manifest.json", {
        "status": "private_v9_staging_labels", "rows_per_label": len(oracle),
        "public_manifest_sha256": sha256_file(args.output / "manifest.json"),
    })
    write_checksums(args.private_label_output, ("oracle_selections.jsonl", "complete_negative_selections.jsonl", "manifest.json"))
    print(f"wrote {len(events)} prospective unscored v9 staging events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
