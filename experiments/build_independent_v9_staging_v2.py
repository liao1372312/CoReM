#!/usr/bin/env python3
"""Expand unscored Typesense/Solr staging to six distinct API families.

This remains task-validity staging, not a complete or sealed confirmation
benchmark. Do not expose private assignments to a selector.
"""

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
from experiments.build_independent_v9_staging import _candidate, _case as _v1_case  # noqa: E402

SOURCES = {
    "typesense_alias_assignment": "https://typesense.org/docs/30.2/api/collection-alias.html",
    "typesense_stopwords_assignment": "https://typesense.org/docs/30.2/api/stopwords.html",
    "typesense_synonym_set_assignment": "https://typesense.org/docs/30.2/api/synonyms.html",
    "solr_user_property_assignment": "https://solr.apache.org/guide/solr/9_10/configuration-guide/config-api.html",
    "solr_common_property_assignment": "https://solr.apache.org/guide/solr/9_10/configuration-guide/config-api.html",
    "solr_schema_field_assignment": "https://solr.apache.org/guide/solr/9_10/indexing-guide/schema-api.html",
}
VERSIONS = {"typesense": "30.2", "solr": "9.10.1"}
START = datetime(2026, 10, 18, 16, 0, tzinfo=UTC)
VALID_FROM = "2026-10-18T00:00:00Z"


def _contract(statuses: list[int], checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {"kind": "assertions_v1", "trace_statuses": statuses,
            "trace_assertions": [], "state_checks": checks}


def _core_create(name: str) -> dict[str, Any]:
    return {"method": "GET", "path": f"/solr/admin/cores?action=CREATE&name={name}&instanceDir={name}&configSet=_default&wt=json"}


def _case(family: str, number: int, positive: bool) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
    if family in {"typesense_alias_assignment", "solr_user_property_assignment"}:
        return _v1_case(family, number, positive)
    suffix = f"{number:03d}"
    if family == "typesense_stopwords_assignment":
        target, decoy = f"v9_stop_target_{suffix}", f"v9_stop_decoy_{suffix}"
        amber, violet = f"amber{suffix}", f"violet{suffix}"
        left, right = (amber, violet) if positive else (violet, amber)
        query = f"Configure stopword set {target} with {amber} and {decoy} with {violet}; inspect both sets."
        plans = [
            {"method": "PUT", "path": f"/stopwords/{name}", "json": {"stopwords": [word]}}
            for name, word in ((target, left), (decoy, right))
        ] + [{"method": "GET", "path": f"/stopwords/{target}"}]
        checks = [
            {"path": f"/stopwords/{name}", "status": 200, "assertions": [
                {"path": ["stopwords", "id"], "equals": name},
                {"path": ["stopwords", "stopwords"], "equals": [word]},
            ]}
            for name, word in ((target, amber), (decoy, violet))
        ]
        return "typesense", query, plans, _contract([200, 200, 200], checks)
    if family == "typesense_synonym_set_assignment":
        target, decoy = f"v9_syn_target_{suffix}", f"v9_syn_decoy_{suffix}"
        amber, violet = f"amber{suffix}", f"violet{suffix}"
        left, right = (amber, violet) if positive else (violet, amber)
        query = f"Create synonym set {target} pairing common with {amber}, and set {decoy} pairing common with {violet}; inspect both."
        plans = [
            {"method": "PUT", "path": f"/synonym_sets/{name}", "json": {
                "items": [{"id": "pair", "synonyms": ["common", word]}]}}
            for name, word in ((target, left), (decoy, right))
        ] + [{"method": "GET", "path": f"/synonym_sets/{target}"}]
        checks = [
            {"path": f"/synonym_sets/{name}", "status": 200, "assertions": [
                {"path": ["name"], "equals": name},
                {"path": ["items", 0, "synonyms"], "equals": ["common", word]},
            ]}
            for name, word in ((target, amber), (decoy, violet))
        ]
        return "typesense", query, plans, _contract([200, 200, 200], checks)
    if family == "solr_common_property_assignment":
        target, decoy = f"v9_prop_target_{suffix}", f"v9_prop_decoy_{suffix}"
        fast, slow = 5000 + number, 9000 + number
        left, right = (fast, slow) if positive else (slow, fast)
        query = f"Create Solr cores {target} and {decoy}; set updateHandler.autoCommit.maxTime to {fast} on {target} and {slow} on {decoy}; inspect both overlays."
        plans = [_core_create(name) for name in (target, decoy)] + [
            {"method": "POST", "path": f"/solr/{name}/config", "json": {
                "set-property": {"updateHandler.autoCommit.maxTime": value}}}
            for name, value in ((target, left), (decoy, right))
        ] + [{"method": "GET", "path": f"/solr/{target}/config/overlay?wt=json"}]
        checks = [
            {"path": f"/solr/{name}/config/overlay?wt=json", "status": 200, "assertions": [
                {"path": ["overlay", "props", "updateHandler", "autoCommit", "maxTime"], "equals": value},
            ]}
            for name, value in ((target, fast), (decoy, slow))
        ]
        return "solr", query, plans, _contract([200] * 5, checks)
    if family == "solr_schema_field_assignment":
        target, decoy = f"v9_schema_target_{suffix}", f"v9_schema_decoy_{suffix}"
        amber, violet = f"amber_{suffix}", f"violet_{suffix}"
        left, right = (amber, violet) if positive else (violet, amber)
        query = f"Create Solr cores {target} and {decoy}; add string field {amber} to {target} and {violet} to {decoy}; inspect the target field."
        plans = [_core_create(name) for name in (target, decoy)] + [
            {"method": "POST", "path": f"/solr/{name}/schema", "json": {
                "add-field": {"name": field, "type": "string", "stored": True}}}
            for name, field in ((target, left), (decoy, right))
        ] + [{"method": "GET", "path": f"/solr/{target}/schema/fields/{amber}?wt=json"}]
        checks = [
            {"path": f"/solr/{name}/schema/fields/{field}?wt=json", "status": 200, "assertions": [
                {"path": ["field", "name"], "equals": field},
                {"path": ["field", "type"], "equals": "string"},
            ]}
            for name, field in ((target, amber), (decoy, violet))
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
        parser.error("refusing to overwrite prior v9 staging")
    salt = args.salt_file.read_bytes()
    if len(salt) != 32:
        raise ValueError("private assignment salt must be 32 bytes")
    events, queries, memories, edges, candidates, oracle, negative = [], [], [], [], [], [], []
    for family_index, family in enumerate(SOURCES):
        sides = [0] * 4 + [1] * 4
        seed = hashlib.sha256(salt + f"v9-v2-assignment|{family}".encode()).digest()
        random.Random(int.from_bytes(seed[:8], "big")).shuffle(sides)
        for offset in range(8):
            number = offset + 1
            event_id = f"v9-staging-v2:{family}:event-{number:03d}"
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
                _service, _query, candidate_plans, _contract = _case(family, number, correct)
                rows, edge, candidate = _candidate(
                    salt, event_id, family, service, side, candidate_plans, SOURCES[family]
                )
                memories.extend(rows)
                edges.append(edge)
                candidates.append(candidate)
                selection = {
                    "run_id": f"v9-staging-v2:{event_id}:{side}",
                    "event_id": event_id, "method": "oracle" if correct else "complete_negative",
                    "candidate_edge_id": edge["edge_id"],
                    "selected_memory_ids": candidate["selected_memory_ids"],
                    "token_cost": 24 * len(candidate["selected_memory_ids"]),
                }
                (oracle if correct else negative).append(selection)
    if Counter(row["service_id"] for row in events) != {"typesense": 24, "solr": 24}:
        raise ValueError("provider balance failed")
    args.output.mkdir(parents=True)
    files = {"events.jsonl": events, "query_only_events.jsonl": queries,
             "memories.jsonl": memories, "dependency_hyperedges.jsonl": edges,
             "candidate_selections.jsonl": candidates}
    for name, rows in files.items():
        write_jsonl(args.output / name, rows)
    write_json(args.output / "manifest.json", {
        "status": "prospective_unscored_staging_only",
        "claim_boundary": "Six of sixteen planned families; no model scoring, independent method claim, or submission claim.",
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
