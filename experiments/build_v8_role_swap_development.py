#!/usr/bin/env python3
"""Build post-v8 role-swap development tasks with matched literal multisets.

These tasks were designed after v8 outcomes were inspected. They must never
be reclassified as an independent confirmation set.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import write_checksums, write_json, write_jsonl  # noqa: E402

SOURCES = {
    "qdrant_config_role_swap": "https://api.qdrant.tech/v-1-15-x/api-reference/collections/update-collection",
    "qdrant_payload_role_swap": "https://api.qdrant.tech/v-1-15-x/api-reference/points/set-payload",
    "openbao_metadata_role_swap": "https://openbao.org/docs/secrets/kv/kv-v2/",
}
VERSIONS = {"qdrant": "1.15.5", "openbao": "2.7.0"}
BASE_TIME = datetime(2026, 10, 18, 15, 0, tzinfo=UTC)
VALID_FROM = "2026-10-17T00:00:00Z"


def _case(family: str, number: int, positive: bool) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
    suffix = f"{number:02d}"
    if family == "qdrant_config_role_swap":
        target = f"v8role_cfg_target_{suffix}"
        decoy = f"v8role_cfg_decoy_{suffix}"
        desired = 22000 + number * 100
        other = desired + 5000
        first, second = (desired, other) if positive else (other, desired)
        query = f"Create collections {target} and {decoy}; set optimizer indexing threshold of {target} to {desired} and {decoy} to {other}, then inspect {target}."
        plans = [
            {"method": "PUT", "path": f"/collections/{target}", "json": {"vectors": {"size": 2, "distance": "Cosine"}}},
            {"method": "PUT", "path": f"/collections/{decoy}", "json": {"vectors": {"size": 2, "distance": "Cosine"}}},
            {"method": "PATCH", "path": f"/collections/{target}", "json": {"optimizers_config": {"indexing_threshold": first}}},
            {"method": "PATCH", "path": f"/collections/{decoy}", "json": {"optimizers_config": {"indexing_threshold": second}}},
            {"method": "GET", "path": f"/collections/{target}"},
        ]
        checks = [
            {"path": f"/collections/{name}", "status": 200, "assertions": [
                {"path": ["result", "config", "optimizer_config", "indexing_threshold"], "equals": value}
            ]}
            for name, value in ((target, desired), (decoy, other))
        ]
        return "qdrant", query, plans, {"kind": "assertions_v1", "trace_statuses": [200] * 5, "trace_assertions": [], "state_checks": checks}
    if family == "qdrant_payload_role_swap":
        collection = f"v8role_payload_{suffix}"
        first, second = ("published", "draft") if positive else ("draft", "published")
        query = f"Create collection {collection} with points 1 and 2; set point 1 stage to published and point 2 stage to draft, then inspect point 1."
        plans = [
            {"method": "PUT", "path": f"/collections/{collection}", "json": {"vectors": {"size": 2, "distance": "Cosine"}}},
            {"method": "PUT", "path": f"/collections/{collection}/points?wait=true", "json": {"points": [
                {"id": 1, "vector": [1.0, 0.0], "payload": {"owner": "alpha"}},
                {"id": 2, "vector": [0.0, 1.0], "payload": {"owner": "beta"}},
            ]}},
            {"method": "POST", "path": f"/collections/{collection}/points/payload?wait=true", "json": {"points": [1], "payload": {"stage": first}}},
            {"method": "POST", "path": f"/collections/{collection}/points/payload?wait=true", "json": {"points": [2], "payload": {"stage": second}}},
            {"method": "GET", "path": f"/collections/{collection}/points/1?with_payload=true"},
        ]
        checks = [
            {"path": f"/collections/{collection}/points/{point}?with_payload=true", "status": 200, "assertions": [
                {"path": ["result", "payload", "owner"], "equals": owner},
                {"path": ["result", "payload", "stage"], "equals": stage},
            ]}
            for point, owner, stage in ((1, "alpha", "published"), (2, "beta", "draft"))
        ]
        return "qdrant", query, plans, {"kind": "assertions_v1", "trace_statuses": [200] * 5, "trace_assertions": [], "state_checks": checks}
    if family == "openbao_metadata_role_swap":
        target = f"v8role_meta_target_{suffix}"
        decoy = f"v8role_meta_decoy_{suffix}"
        first, second = ("alpha", "beta") if positive else ("beta", "alpha")
        query = f"Create OpenBao keys {target} and {decoy}; set custom metadata owner of {target} to alpha and {decoy} to beta, then inspect {target}."
        plans = [
            {"method": "POST", "path": f"/v1/secret/data/{target}", "json": {"data": {"marker": "original-target"}}},
            {"method": "POST", "path": f"/v1/secret/data/{decoy}", "json": {"data": {"marker": "original-decoy"}}},
            {"method": "POST", "path": f"/v1/secret/metadata/{target}", "json": {"custom_metadata": {"owner": first}}},
            {"method": "POST", "path": f"/v1/secret/metadata/{decoy}", "json": {"custom_metadata": {"owner": second}}},
            {"method": "GET", "path": f"/v1/secret/metadata/{target}"},
        ]
        checks = [
            {"path": f"/v1/secret/metadata/{name}", "status": 200, "assertions": [
                {"path": ["data", "custom_metadata", "owner"], "equals": owner}
            ]}
            for name, owner in ((target, "alpha"), (decoy, "beta"))
        ]
        checks.extend([
            {"path": f"/v1/secret/data/{name}", "status": 200, "assertions": [
                {"path": ["data", "data", "marker"], "equals": marker},
                {"path": ["data", "metadata", "version"], "equals": 1},
            ]}
            for name, marker in ((target, "original-target"), (decoy, "original-decoy"))
        ])
        return "openbao", query, plans, {"kind": "assertions_v1", "trace_statuses": [200, 200, 204, 204, 200], "trace_assertions": [], "state_checks": checks}
    raise ValueError(f"unsupported role-swap development family: {family}")


def _candidate(
    salt: bytes, event_id: str, family: str, service: str, side: int, plans: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    digest = hashlib.sha256(salt + f"role-swap|{event_id}|{side}".encode()).hexdigest()
    edge_id = f"{event_id}:candidate-{digest[:24]}"
    rows = []
    members = []
    for index, plan in enumerate(plans, 1):
        memory_id = f"{edge_id}:record-{index}"
        members.append(memory_id)
        rows.append({
            "memory_id": memory_id,
            "service_id": service,
            "provider_id": service,
            "version_scope": [VERSIONS[service]],
            "valid_from": VALID_FROM,
            "valid_until": None,
            "role": "workflow_step",
            "step_index": index,
            "text": f"Execute {service} role-binding operation {index} using {plan['method']} {plan['path']}.",
            "plan": plan,
            "token_cost": 24,
            "storage_cost": 48,
            "provenance_uri": SOURCES[family],
            "provenance_verified": True,
            "evidence_atoms": [f"role:step{index}"],
        })
    verification_id = f"{edge_id}:record-{len(plans) + 1}"
    members.append(verification_id)
    rows.append({
        "memory_id": verification_id,
        "service_id": service,
        "provider_id": service,
        "version_scope": [VERSIONS[service]],
        "valid_from": VALID_FROM,
        "valid_until": None,
        "role": "verification",
        "step_index": None,
        "text": f"Independently inspect final {service} state.",
        "token_cost": 24,
        "storage_cost": 48,
        "provenance_uri": SOURCES[family],
        "provenance_verified": True,
        "evidence_atoms": ["role:verify"],
    })
    edge = {
        "edge_id": edge_id,
        "members": members,
        "service_id": service,
        "version_scope": [VERSIONS[service]],
        "valid_from": VALID_FROM,
        "valid_until": None,
        "verified": True,
        "relation_type": "role-swap-development-workflow",
        "evidence_atoms": [f"role:step{index}" for index in range(1, len(plans) + 1)] + ["role:verify"],
    }
    return rows, edge, {"event_id": event_id, "candidate_edge_id": edge_id, "selected_memory_ids": members}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-label-output", type=Path, required=True)
    parser.add_argument("--salt-file", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.private_label_output.exists():
        parser.error("refusing to overwrite prior role-swap development data")
    salt = args.salt_file.read_bytes()
    if len(salt) != 32:
        raise ValueError("existing private assignment salt must be 32 bytes")
    events = []
    queries = []
    memories = []
    edges = []
    candidates = []
    oracle = []
    negative = []
    for family_index, family in enumerate(SOURCES):
        sides = [0] * 4 + [1] * 4
        family_seed = hashlib.sha256(salt + f"role-swap-assignment|{family}".encode()).digest()
        random.Random(int.from_bytes(family_seed[:8], "big")).shuffle(sides)
        for offset in range(8):
            number = offset + 1
            event_id = f"v8-dev-role:{family}:event-{number:03d}"
            service, query, _plans, contract = _case(family, number, True)
            event = {
                "event_id": event_id,
                "incident_id": event_id,
                "service_id": service,
                "provider_id": service,
                "version_id": VERSIONS[service],
                "event_time": (BASE_TIME + timedelta(minutes=family_index * 8 + offset)).isoformat().replace("+00:00", "Z"),
                "query": query,
                "task_key": family,
                "operation_count": len(_plans),
                "workflow_contract": contract,
                "checker_contract_id": "independent-v8-assertions-v1",
                "required_evidence_atoms": [f"role:step{index}" for index in range(1, len(_plans) + 1)] + ["role:verify"],
                "required_binding_atoms": [],
                "observation_tier": "post_inspection_development_only",
            }
            events.append(event)
            queries.append({key: event[key] for key in ("event_id", "service_id", "provider_id", "version_id", "event_time", "query")})
            for side in (0, 1):
                positive = side == sides[offset]
                _service, _query, plans, _contract = _case(family, number, positive)
                rows, edge, candidate = _candidate(salt, event_id, family, service, side, plans)
                memories.extend(rows)
                edges.append(edge)
                candidates.append(candidate)
                selection = {
                    "run_id": f"v8-role-development:{event_id}:{side}",
                    "event_id": event_id,
                    "method": "oracle" if positive else "complete_negative",
                    "selected_memory_ids": candidate["selected_memory_ids"],
                    "token_cost": 24 * len(candidate["selected_memory_ids"]),
                }
                (oracle if positive else negative).append(selection)
    args.output.mkdir(parents=True)
    files = {
        "events.jsonl": events,
        "query_only_events.jsonl": queries,
        "memories.jsonl": memories,
        "dependency_hyperedges.jsonl": edges,
        "candidate_selections.jsonl": candidates,
    }
    for name, rows in files.items():
        write_jsonl(args.output / name, rows)
    write_json(args.output / "manifest.json", {
        "status": "post_v8_inspection_development_only",
        "claim_boundary": "Role-swap tasks were designed after v8 scores were inspected; they cannot serve as independent confirmation.",
        "families": list(SOURCES),
        "events": len(events),
        "candidate_rows": len(candidates),
        "private_assignment_commitment_sha256": hashlib.sha256(salt).hexdigest(),
        "source_uris": SOURCES,
    })
    write_checksums(args.output, (*files, "manifest.json"))
    args.private_label_output.mkdir(parents=True, mode=0o700)
    write_jsonl(args.private_label_output / "oracle_selections.jsonl", oracle)
    write_jsonl(args.private_label_output / "complete_negative_selections.jsonl", negative)
    write_json(args.private_label_output / "manifest.json", {
        "status": "private_post_v8_development_labels",
        "public_manifest_sha256": hashlib.sha256((args.output / "manifest.json").read_bytes()).hexdigest(),
        "rows_per_label": len(oracle),
    })
    write_checksums(args.private_label_output, ("oracle_selections.jsonl", "complete_negative_selections.jsonl", "manifest.json"))
    print(f"wrote {len(events)} post-inspection role-swap development events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
