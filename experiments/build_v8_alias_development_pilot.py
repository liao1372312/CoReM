#!/usr/bin/env python3
"""Build one inspected Qdrant alias-routing development probe.

This probe tests whether candidate serialization preserves operation bodies.
It is never an untouched confirmation family.
"""

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

EVENT_ID = "v8-dev:qdrant:alias-routing-001"
VERSION = "1.15.5"
NOW = "2026-10-18T15:00:00Z"
SOURCE = "https://api.qdrant.tech/v-1-15-x/api-reference/aliases/update-aliases"


def build_candidate(
    side: int, *, target: str, event_id: str = EVENT_ID, balanced_actions: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    digest_input = (
        f"v8-alias-development|2027|{event_id}|{side}"
        if balanced_actions
        else f"v8-alias-development|2027|{side}"
    )
    opaque = hashlib.sha256(digest_input.encode()).hexdigest()[:20]
    edge_id = f"{event_id}:candidate-{opaque}"
    actions = [{"create_alias": {"collection_name": target, "alias_name": "v8_active"}}]
    if balanced_actions:
        other = "v8_alias_decoy" if target == "v8_alias_target" else "v8_alias_target"
        actions.append({"create_alias": {"collection_name": other, "alias_name": "v8_spare"}})
    plans = [
        {"method": "PUT", "path": "/collections/v8_alias_target", "json": {"vectors": {"size": 2, "distance": "Cosine"}}},
        {"method": "PUT", "path": "/collections/v8_alias_decoy", "json": {"vectors": {"size": 2, "distance": "Cosine"}}},
        {"method": "POST", "path": "/collections/aliases", "json": {"actions": actions}},
        {"method": "GET", "path": "/collections/v8_alias_target/aliases"},
    ]
    memories: list[dict[str, Any]] = []
    selected: list[str] = []
    for index, plan in enumerate(plans, 1):
        memory_id = f"{edge_id}:record-{index}"
        selected.append(memory_id)
        memories.append(
            {
                "memory_id": memory_id,
                "service_id": "qdrant",
                "provider_id": "qdrant",
                "version_scope": [VERSION],
                "valid_from": "2026-10-17T00:00:00Z",
                "valid_until": None,
                "role": "workflow_step",
                "step_index": index,
                # Deliberately identical text across candidates: the API body
                # contains the alias destination and must not be discarded.
                "text": f"Perform Qdrant alias-routing workflow step {index}: {plan['method']} {plan['path']}.",
                "plan": plan,
                "token_cost": 28,
                "storage_cost": 48,
                "provenance_uri": SOURCE,
                "provenance_verified": True,
                "evidence_atoms": [f"role:step{index}"],
            }
        )
    verify_id = f"{edge_id}:record-5"
    selected.append(verify_id)
    memories.append(
        {
            "memory_id": verify_id,
            "service_id": "qdrant",
            "provider_id": "qdrant",
            "version_scope": [VERSION],
            "valid_from": "2026-10-17T00:00:00Z",
            "valid_until": None,
            "role": "verification",
            "step_index": None,
            "text": "Independently read the alias mapping from both collections.",
            "token_cost": 28,
            "storage_cost": 48,
            "provenance_uri": SOURCE,
            "provenance_verified": True,
            "evidence_atoms": ["role:verify"],
        }
    )
    edge = {
        "edge_id": edge_id,
        "members": selected,
        "service_id": "qdrant",
        "version_scope": [VERSION],
        "valid_from": "2026-10-17T00:00:00Z",
        "valid_until": None,
        "verified": True,
        "relation_type": "development-alias-routing",
        "evidence_atoms": [f"role:step{index}" for index in range(1, 5)] + ["role:verify"],
    }
    candidate = {
        "event_id": event_id,
        "candidate_edge_id": edge_id,
        "selected_memory_ids": selected,
    }
    return memories, edge, candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--balanced-actions", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    event_id = (
        "v8-dev:qdrant:alias-role-swap-001" if args.balanced_actions else EVENT_ID
    )
    query = (
        "Create two cosine collections; route alias v8_active to v8_alias_target "
        "and alias v8_spare to v8_alias_decoy, then inspect the active mapping."
        if args.balanced_actions
        else "Create two cosine collections and route alias v8_active to v8_alias_target, then inspect its mapping."
    )
    event = {
        "event_id": event_id,
        "incident_id": event_id,
        "service_id": "qdrant",
        "provider_id": "qdrant",
        "version_id": VERSION,
        "event_time": NOW,
        "query": query,
        "task_key": "alias_role_swap_development_only" if args.balanced_actions else "alias_routing_development_only",
        "operation_count": 4,
        "workflow_contract": {
            "kind": "assertions_v1",
            "trace_statuses": [200, 200, 200, 200],
            "trace_assertions": [],
            "state_checks": [
                {
                    "path": "/collections/v8_alias_target/aliases",
                    "status": 200,
                    "assertions": [
                        {"path": ["result", "aliases", 0, "alias_name"], "equals": "v8_active"},
                        {"path": ["result", "aliases", 0, "collection_name"], "equals": "v8_alias_target"},
                    ],
                },
                {
                    "path": "/collections/v8_alias_decoy/aliases",
                    "status": 200,
                    "assertions": (
                        [{"path": ["result", "aliases", 0, "alias_name"], "equals": "v8_spare"}]
                        if args.balanced_actions
                        else [{"path": ["result", "aliases"], "equals": []}]
                    ),
                },
            ],
        },
        "checker_contract_id": "independent-v8-assertions-v1",
        "required_evidence_atoms": [f"role:step{index}" for index in range(1, 5)] + ["role:verify"],
        "required_binding_atoms": [],
        "observation_tier": "development_pilot_only",
    }
    query_view = {
        key: event[key]
        for key in ("event_id", "service_id", "provider_id", "version_id", "event_time", "query")
    }
    all_memories: list[dict[str, Any]] = []
    all_edges: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    positive: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    for side, target in enumerate(("v8_alias_decoy", "v8_alias_target")):
        memories, edge, candidate = build_candidate(
            side, target=target, event_id=event_id, balanced_actions=args.balanced_actions
        )
        all_memories.extend(memories)
        all_edges.append(edge)
        candidates.append(candidate)
        selection = {
            "event_id": event_id,
            "method": "development_positive" if side == 1 else "development_negative",
            "selected_memory_ids": candidate["selected_memory_ids"],
            "token_cost": 140,
        }
        (positive if side == 1 else negative).append(selection)
    args.output.mkdir(parents=True)
    files = {
        "events.jsonl": [event],
        "query_only_events.jsonl": [query_view],
        "memories.jsonl": all_memories,
        "dependency_hyperedges.jsonl": all_edges,
        "candidate_selections.jsonl": candidates,
        "pilot_positive_selections.jsonl": positive,
        "pilot_negative_selections.jsonl": negative,
    }
    for name, rows in files.items():
        write_jsonl(args.output / name, rows)
    manifest = {
            "status": "development_pilot_only",
            "claim_boundary": "The alias-routing family is inspected and excluded from confirmation.",
            "service": "qdrant:1.15.5",
            "events": 1,
            "candidate_rows": 2,
            "positive_side": 1,
            "source": SOURCE,
    }
    if args.balanced_actions:
        manifest["balanced_actions"] = True
    write_json(args.output / "manifest.json", manifest)
    write_checksums(args.output, (*files, "manifest.json"))
    print(f"wrote alias-routing development pilot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
