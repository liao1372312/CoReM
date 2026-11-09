#!/usr/bin/env python3
"""Build two visibly exploratory Qdrant/OpenBao workflow pilot incidents."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import write_checksums, write_json, write_jsonl  # noqa: E402

TIME = "2026-10-18T14:00:00Z"
VALID_FROM = "2026-10-17T00:00:00Z"


def candidate_id(event_id: str, side: int) -> str:
    digest = hashlib.sha256(f"v8-development|2027|{event_id}|{side}".encode()).hexdigest()
    return f"{event_id}:candidate-{digest[:20]}"


def pilot_spec(service: str, marker: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    if service == "qdrant":
        plans = [
            {
                "method": "PUT",
                "path": "/collections/v8pilot",
                "json": {"vectors": {"size": 2, "distance": "Cosine"}},
            },
            {
                "method": "PUT",
                "path": "/collections/v8pilot/points?wait=true",
                "json": {
                    "points": [
                        {"id": 1, "vector": [0.9, 0.1], "payload": {"tag": marker}},
                        {"id": 2, "vector": [0.1, 0.9], "payload": {"tag": "beta"}},
                    ]
                },
            },
            {
                "method": "POST",
                "path": "/collections/v8pilot/points/query",
                "json": {"query": [1.0, 0.0], "limit": 1, "with_payload": True},
            },
        ]
        contract = {
            "kind": "assertions_v1",
            "trace_statuses": [200, 200, 200],
            "trace_assertions": [
                {
                    "path": ["steps", 2, "response", "body", "result", "points", 0, "id"],
                    "equals": 1,
                }
            ],
            "state_checks": [
                {
                    "path": "/collections/v8pilot/points/1?with_payload=true",
                    "status": 200,
                    "assertions": [
                        {"path": ["result", "id"], "equals": 1},
                        {"path": ["result", "payload", "tag"], "equals": "alpha"},
                    ],
                }
            ],
        }
        text = f"Create a cosine collection, insert point 1 tagged {marker}, and query its nearest point."
        return text, plans, contract
    if service == "openbao":
        plans = [
            {
                "method": "POST",
                "path": "/v1/secret/data/v8pilot",
                "json": {"data": {"marker": marker}},
                "capture": {
                    "name": "secret_version",
                    "source": "body_path",
                    "path": ["data", "version"],
                },
            },
            {"method": "GET", "path": "/v1/secret/data/v8pilot?version={secret_version}"},
            {"method": "GET", "path": "/v1/secret/metadata/v8pilot"},
        ]
        contract = {
            "kind": "assertions_v1",
            "trace_statuses": [200, 200, 200],
            "trace_assertions": [
                {
                    "path": ["steps", 1, "response", "body", "data", "data", "marker"],
                    "equals": "good",
                },
                {"path": ["bindings", "secret_version"], "equals": 1},
            ],
            "state_checks": [
                {
                    "path": "/v1/secret/data/v8pilot",
                    "status": 200,
                    "assertions": [
                        {"path": ["data", "data", "marker"], "equals": "good"},
                        {"path": ["data", "metadata", "version"], "equals": 1},
                    ],
                }
            ],
        }
        text = f"Write a versioned secret with marker {marker}, read its returned version, and verify its metadata."
        return text, plans, contract
    raise ValueError(f"unknown pilot service: {service}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    events = []
    query_events = []
    memories = []
    edges = []
    candidates = []
    positive_selections = []
    negative_selections = []
    for event_index, (service, version, query) in enumerate(
        (
            (
                "qdrant",
                "1.15.5",
                "In a fresh cosine vector collection, store point 1 with tag alpha and retrieve it by nearest-neighbor query.",
            ),
            (
                "openbao",
                "2.7.0",
                "Write a versioned secret with marker good, read that version, and confirm the stored value.",
            ),
        )
    ):
        event_id = f"v8-dev:{service}:event-001"
        positive_side = event_index % 2
        _, _, contract = pilot_spec(service, "alpha" if service == "qdrant" else "good")
        event = {
            "event_id": event_id,
            "incident_id": event_id,
            "service_id": service,
            "provider_id": service,
            "version_id": version,
            "event_time": TIME,
            "query": query,
            "task_key": f"{service}_pilot",
            "operation_count": 3,
            "workflow_contract": contract,
            "checker_contract_id": "independent-v8-assertions-v1",
            "required_evidence_atoms": [
                "role:step1",
                "role:step2",
                "role:step3",
                "role:verify",
            ],
            "required_binding_atoms": ["binding:secret_version"] if service == "openbao" else [],
            "observation_tier": "development_pilot_only",
        }
        events.append(event)
        query_events.append(
            {
                key: event[key]
                for key in (
                    "event_id",
                    "event_time",
                    "service_id",
                    "provider_id",
                    "version_id",
                    "query",
                )
            }
        )
        for side in (0, 1):
            positive = side == positive_side
            marker = (
                "alpha" if service == "qdrant" else "good"
            ) if positive else "wrong"
            description, plans, _ = pilot_spec(service, marker)
            edge_id = candidate_id(event_id, side)
            selected = []
            for step_index, plan in enumerate(plans, 1):
                memory_id = f"{edge_id}:record-{step_index}"
                selected.append(memory_id)
                memories.append(
                    {
                        "memory_id": memory_id,
                        "service_id": service,
                        "provider_id": service,
                        "version_scope": [version],
                        "valid_from": VALID_FROM,
                        "valid_until": None,
                        "role": "workflow_step",
                        "step_index": step_index,
                        "text": f"{description} Step {step_index}: {plan['method']} {plan['path']}.",
                        "plan": plan,
                        "token_cost": 32,
                        "storage_cost": 48,
                        "provenance_uri": (
                            "https://qdrant.tech/documentation/manage-data/points/"
                            if service == "qdrant"
                            else "https://openbao.org/api-docs/"
                        ),
                        "provenance_verified": True,
                        "evidence_atoms": [f"role:step{step_index}"],
                    }
                )
            verify_id = f"{edge_id}:record-4"
            selected.append(verify_id)
            memories.append(
                {
                    "memory_id": verify_id,
                    "service_id": service,
                    "provider_id": service,
                    "version_scope": [version],
                    "valid_from": VALID_FROM,
                    "valid_until": None,
                    "role": "verification",
                    "step_index": None,
                    "text": f"Independently read final state for this {service} workflow.",
                    "token_cost": 32,
                    "storage_cost": 48,
                    "provenance_uri": (
                        "https://qdrant.tech/documentation/manage-data/points/"
                        if service == "qdrant"
                        else "https://openbao.org/api-docs/"
                    ),
                    "provenance_verified": True,
                    "evidence_atoms": ["role:verify"],
                }
            )
            edges.append(
                {
                    "edge_id": edge_id,
                    "members": selected,
                    "service_id": service,
                    "version_scope": [version],
                    "valid_from": VALID_FROM,
                    "valid_until": None,
                    "verified": True,
                    "relation_type": "development-complete-workflow",
                    "evidence_atoms": [
                        "role:step1",
                        "role:step2",
                        "role:step3",
                        "role:verify",
                    ],
                }
            )
            candidate = {
                "event_id": event_id,
                "candidate_edge_id": edge_id,
                "selected_memory_ids": selected,
            }
            candidates.append(candidate)
            selection = {
                "event_id": event_id,
                "method": "development_positive" if positive else "development_negative",
                "selected_memory_ids": selected,
                "token_cost": 128,
            }
            (positive_selections if positive else negative_selections).append(selection)
    args.output.mkdir(parents=True)
    filenames = (
        "events.jsonl",
        "query_only_events.jsonl",
        "memories.jsonl",
        "dependency_hyperedges.jsonl",
        "candidate_selections.jsonl",
        "pilot_positive_selections.jsonl",
        "pilot_negative_selections.jsonl",
    )
    for filename, rows in zip(
        filenames,
        (
            events,
            query_events,
            memories,
            edges,
            candidates,
            positive_selections,
            negative_selections,
        ),
    ):
        write_jsonl(args.output / filename, rows)
    write_json(
        args.output / "manifest.json",
        {
            "status": "development_pilot_only",
            "claim_boundary": "Both operation patterns were inspected before any model score.",
            "services": ["qdrant:1.15.5", "openbao:2.7.0"],
            "events": 2,
            "candidates_per_event": 2,
            "seed": 2027,
        },
    )
    write_checksums(args.output, (*filenames, "manifest.json"))
    print(f"wrote two new-service development pilot incidents to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
