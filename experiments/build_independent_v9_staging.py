#!/usr/bin/env python3
"""Create unscored v9 task-validity staging pairs on new real services.

This initial two-family staging set is not the planned sixteen-family sealed
benchmark. It must not be scored by a model or reported as confirmation.
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

from corem.real_eval import write_checksums, write_json, write_jsonl  # noqa: E402

SOURCES = {
    "typesense_alias_assignment": "https://typesense.org/docs/30.2/api/collection-alias.html",
    "solr_user_property_assignment": "https://solr.apache.org/guide/solr/9_10/configuration-guide/config-api.html",
}
VERSIONS = {"typesense": "30.2", "solr": "9.10.1"}
START = datetime(2026, 10, 19, 0, 0, tzinfo=UTC)
VALID_FROM = "2026-10-18T00:00:00Z"


def _case(family: str, number: int, positive: bool) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
    suffix = f"{number:03d}"
    if family == "typesense_alias_assignment":
        target, decoy = f"v9_alias_target_{suffix}", f"v9_alias_decoy_{suffix}"
        blue, amber = f"v9_collection_blue_{suffix}", f"v9_collection_amber_{suffix}"
        target_collection, decoy_collection = (blue, amber) if positive else (amber, blue)
        query = f"Create collections {blue} and {amber}; route alias {target} to {blue} and alias {decoy} to {amber}; inspect both alias assignments."
        plans = [
            {"method": "POST", "path": "/collections", "json": {"name": name, "fields": [{"name": "title", "type": "string"}]}}
            for name in (blue, amber)
        ] + [
            {"method": "PUT", "path": f"/aliases/{name}", "json": {"collection_name": collection}}
            for name, collection in ((target, target_collection), (decoy, decoy_collection))
        ] + [{"method": "GET", "path": f"/aliases/{target}"}]
        checks = [
            {"path": f"/aliases/{name}", "status": 200, "assertions": [
                {"path": ["name"], "equals": name},
                {"path": ["collection_name"], "equals": collection},
            ]}
            for name, collection in ((target, blue), (decoy, amber))
        ]
        return "typesense", query, plans, {"kind": "assertions_v1", "trace_statuses": [201, 201, 200, 200, 200], "trace_assertions": [], "state_checks": checks}
    if family == "solr_user_property_assignment":
        target, decoy = f"v9_core_target_{suffix}", f"v9_core_decoy_{suffix}"
        alpha, beta = f"alpha_{suffix}", f"beta_{suffix}"
        target_value, decoy_value = (alpha, beta) if positive else (beta, alpha)
        query = f"Create Solr cores {target} and {decoy}; set user property owner on {target} to {alpha} and on {decoy} to {beta}; inspect both overlays."
        plans = [
            {"method": "GET", "path": f"/solr/admin/cores?action=CREATE&name={name}&instanceDir={name}&configSet=_default&wt=json"}
            for name in (target, decoy)
        ] + [
            {"method": "POST", "path": f"/solr/{name}/config", "json": {"set-user-property": {"owner": value}}}
            for name, value in ((target, target_value), (decoy, decoy_value))
        ] + [{"method": "GET", "path": f"/solr/{target}/config/overlay?wt=json"}]
        checks = [
            {"path": f"/solr/{name}/config/overlay?wt=json", "status": 200, "assertions": [
                {"path": ["overlay", "userProps", "owner"], "equals": value},
            ]}
            for name, value in ((target, alpha), (decoy, beta))
        ]
        return "solr", query, plans, {"kind": "assertions_v1", "trace_statuses": [200] * 5, "trace_assertions": [], "state_checks": checks}
    raise ValueError(f"unknown v9 staging family: {family}")


def _candidate(
    salt: bytes, event_id: str, family: str, service: str, side: int, plans: list[dict[str, Any]],
    provenance_uri: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    digest = hashlib.sha256(salt + f"v9|{event_id}|{side}".encode()).hexdigest()
    edge_id = f"{event_id}:candidate-{digest[:24]}"
    records = []
    members = []
    for index, plan in enumerate(plans, 1):
        memory_id = f"{edge_id}:record-{index}"
        members.append(memory_id)
        records.append({
            "memory_id": memory_id, "service_id": service, "provider_id": service,
            "version_scope": [VERSIONS[service]], "valid_from": VALID_FROM,
            "valid_until": None, "role": "workflow_step", "step_index": index,
            "text": f"Execute {service} operation {index}: {plan['method']} {plan['path']}.",
            "plan": plan, "token_cost": 24, "storage_cost": 48,
            "provenance_uri": provenance_uri or SOURCES[family], "provenance_verified": True,
            "evidence_atoms": [f"role:step{index}"],
        })
    verification_id = f"{edge_id}:record-{len(plans) + 1}"
    members.append(verification_id)
    records.append({
        "memory_id": verification_id, "service_id": service, "provider_id": service,
        "version_scope": [VERSIONS[service]], "valid_from": VALID_FROM,
        "valid_until": None, "role": "verification", "step_index": None,
        "text": f"Independently inspect final {service} state.",
        "token_cost": 24, "storage_cost": 48,
        "provenance_uri": provenance_uri or SOURCES[family], "provenance_verified": True,
        "evidence_atoms": ["role:verify"],
    })
    edge = {
        "edge_id": edge_id, "members": members, "service_id": service,
        "version_scope": [VERSIONS[service]], "valid_from": VALID_FROM,
        "valid_until": None, "verified": True,
        "relation_type": "role-balanced-v9-staging-workflow",
        "evidence_atoms": [f"role:step{index}" for index in range(1, len(plans) + 1)] + ["role:verify"],
    }
    candidate = {"event_id": event_id, "candidate_edge_id": edge_id, "selected_memory_ids": members}
    return records, edge, candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-label-output", type=Path, required=True)
    parser.add_argument("--salt-file", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.private_label_output.exists():
        parser.error("refusing to overwrite v9 staging files")
    salt = args.salt_file.read_bytes()
    if len(salt) != 32:
        raise ValueError("private assignment salt must be 32 bytes")
    events, queries, memories, edges, candidates, oracle, negative = [], [], [], [], [], [], []
    for family_index, family in enumerate(SOURCES):
        sides = [0] * 4 + [1] * 4
        seed = hashlib.sha256(salt + f"v9-assignment|{family}".encode()).digest()
        random.Random(int.from_bytes(seed[:8], "big")).shuffle(sides)
        for offset in range(8):
            number = offset + 1
            event_id = f"v9-staging:{family}:event-{number:03d}"
            service, query, plans, contract = _case(family, number, True)
            event = {
                "event_id": event_id, "incident_id": event_id, "service_id": service,
                "provider_id": service, "version_id": VERSIONS[service],
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
                rows, edge, candidate = _candidate(salt, event_id, family, service, side, candidate_plans)
                memories.extend(rows)
                edges.append(edge)
                candidates.append(candidate)
                selection = {"run_id": f"v9-staging:{event_id}:{side}", "event_id": event_id,
                             "method": "oracle" if correct else "complete_negative",
                             "candidate_edge_id": edge["edge_id"],
                             "selected_memory_ids": candidate["selected_memory_ids"],
                             "token_cost": 24 * len(candidate["selected_memory_ids"])}
                (oracle if correct else negative).append(selection)
    if Counter(row["service_id"] for row in events) != {"typesense": 8, "solr": 8}:
        raise ValueError("v9 staging balance failed")
    args.output.mkdir(parents=True)
    files = {"events.jsonl": events, "query_only_events.jsonl": queries,
             "memories.jsonl": memories, "dependency_hyperedges.jsonl": edges,
             "candidate_selections.jsonl": candidates}
    for name, rows in files.items():
        write_jsonl(args.output / name, rows)
    write_json(args.output / "manifest.json", {
        "status": "prospective_unscored_staging_only",
        "claim_boundary": "Only two of sixteen planned families; no model scoring, no independent method claim, and no submission claim.",
        "families": list(SOURCES), "events": len(events), "candidate_rows": len(candidates),
        "source_uris": SOURCES, "assignment_salt_commitment_sha256": hashlib.sha256(salt).hexdigest(),
    })
    write_checksums(args.output, (*files, "manifest.json"))
    args.private_label_output.mkdir(parents=True, mode=0o700)
    write_jsonl(args.private_label_output / "oracle_selections.jsonl", oracle)
    write_jsonl(args.private_label_output / "complete_negative_selections.jsonl", negative)
    write_json(args.private_label_output / "manifest.json", {
        "status": "private_v9_staging_labels", "rows_per_label": len(oracle),
        "public_manifest_sha256": hashlib.sha256((args.output / "manifest.json").read_bytes()).hexdigest(),
    })
    write_checksums(args.private_label_output, ("oracle_selections.jsonl", "complete_negative_selections.jsonl", "manifest.json"))
    print(f"wrote {len(events)} prospective unscored v9 staging events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
