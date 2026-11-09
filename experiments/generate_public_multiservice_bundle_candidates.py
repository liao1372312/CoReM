#!/usr/bin/env python3
"""Enumerate every admissible atomic dependency bundle for v5 supervision."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import (
    load_jsonl,
    parse_utc,
    write_checksums,
    write_json,
    write_jsonl,
)
from experiments.generate_public_challenge_selections import DEFAULT_SEEDS
from experiments.generate_public_multiservice_challenge_ablations import request_for
from experiments.generate_public_workflow_selections import (
    bm25_scores,
    make_graph,
    ranked_rows,
)

ROLE_ATOMS = ("role:step1", "role:step2", "role:step3", "role:verify")


def candidate_bundles(
    event: dict[str, Any],
    memories: list[dict[str, Any]],
    hyperedges: list[dict[str, Any]],
    token_budget: int,
) -> list[tuple[str, list[str], int]]:
    """Return all event-admissible complete hyperedges without outcome ranking."""
    graph = make_graph(memories, hyperedges)
    request = request_for(event)
    timestamp = parse_utc(event["event_time"], "event_time").timestamp()
    available = {
        str(row["memory_id"])
        for row in memories
        if row.get("service_id") == event.get("service_id")
        and event.get("version_id") in row.get("version_scope", [])
        and parse_utc(row["valid_from"], "valid_from").timestamp() <= timestamp
    }
    candidates = []
    for edge_id, edge in sorted(graph.hyperedges.items()):
        if not graph.is_hyperedge_admissible(edge, request):
            continue
        if not edge.members.issubset(available):
            continue
        if not graph.is_evidence_complete(request, edge.members):
            continue
        if any(
            not graph.is_admissible(memory_id, request) for memory_id in edge.members
        ):
            continue
        cost = sum(graph.records[memory_id].token_cost for memory_id in edge.members)
        if cost <= token_budget:
            candidates.append((edge_id, sorted(edge.members), cost))
    return candidates


def lexical_rolewise_candidates(
    event: dict[str, Any],
    memories: list[dict[str, Any]],
    token_budget: int,
) -> list[tuple[str, list[str], int]]:
    """Build label-blind, role-complete BM25 sets as non-hyperedge hard negatives."""
    timestamp = parse_utc(event["event_time"], "event_time").timestamp()
    active = [
        row
        for row in memories
        if row.get("service_id") == event.get("service_id")
        and event.get("version_id") in row.get("version_scope", [])
        and parse_utc(row["valid_from"], "valid_from").timestamp() <= timestamp
    ]
    scores = bm25_scores(str(event["query"]), active)
    output = []
    observed: set[tuple[str, ...]] = set()
    for seed in DEFAULT_SEEDS:
        selected = []
        for role in ROLE_ATOMS:
            eligible = [row for row in active if role in row.get("evidence_atoms", [])]
            ranked = ranked_rows(
                eligible,
                {
                    str(row["memory_id"]): scores[str(row["memory_id"])]
                    for row in eligible
                },
                seed=seed,
                event_id=str(event["event_id"]),
            )
            if not ranked:
                selected = []
                break
            selected.append(str(ranked[0]["memory_id"]))
        key = tuple(sorted(selected))
        if len(key) != 4 or key in observed:
            continue
        observed.add(key)
        cost = sum(
            int(next(row["token_cost"] for row in active if row["memory_id"] == key_id))
            for key_id in key
        )
        if cost <= token_budget:
            output.append((f"lexical-rolewise:{seed}", list(key), cost))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=160)
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2

    events = load_jsonl(args.dataset / "events.jsonl")
    memories = load_jsonl(args.dataset / "memories.jsonl")
    hyperedges = load_jsonl(args.dataset / "dependency_hyperedges.jsonl")
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    hyperedge_counts: dict[str, int] = {}
    ordered_events = sorted(
        events,
        key=lambda row: (
            parse_utc(row["event_time"], "event_time"),
            str(row["event_id"]),
        ),
    )
    for event in ordered_events:
        event_id = str(event["event_id"])
        timestamp = parse_utc(event["event_time"], "event_time").timestamp()
        available = [
            row
            for row in memories
            if row.get("service_id") == event.get("service_id")
            and parse_utc(row["valid_from"], "valid_from").timestamp() <= timestamp
        ]
        verified = candidate_bundles(
            event,
            memories,
            hyperedges,
            args.token_budget,
        )
        lexical = lexical_rolewise_candidates(event, memories, args.token_budget)
        verified_sets = {tuple(selected) for _, selected, _ in verified}
        candidates = verified + [
            row for row in lexical if tuple(row[1]) not in verified_sets
        ]
        counts[event_id] = len(candidates)
        hyperedge_counts[event_id] = len(verified)
        for candidate_index, (edge_id, selected, cost) in enumerate(candidates):
            method = f"bundle_candidate_{candidate_index:02d}"
            rows.append(
                {
                    "run_id": f"{method}:0:{event_id}",
                    "event_id": event_id,
                    "seed": 0,
                    "method": method,
                    "candidate_edge_id": edge_id,
                    "selected_memory_ids": selected,
                    "token_cost": cost,
                    "latency_ms": 0.0,
                    "llm_calls": 0,
                    "active_bank_cost": sum(
                        int(row["storage_cost"]) for row in available
                    ),
                    "retrieval_status": "enumerated",
                    "evaluations": len(candidates),
                }
            )

    if set(hyperedge_counts.values()) != {4} or min(counts.values()) < 4:
        distribution: dict[int, int] = {}
        for count in counts.values():
            distribution[count] = distribution.get(count, 0) + 1
        print(
            "expected four verified bundles plus optional hard negatives per event, "
            f"observed verified={set(hyperedge_counts.values())}, total={distribution}"
        )
        return 2

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "selections.jsonl", rows)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": "corem-public-multiservice-bundle-candidates-v1",
            "dataset": str(args.dataset.resolve()),
            "token_budget": args.token_budget,
            "events": len(events),
            "verified_bundles_per_event": 4,
            "candidate_count_distribution": {
                str(count): sum(value == count for value in counts.values())
                for count in sorted(set(counts.values()))
            },
            "selection_rows": len(rows),
            "label_blind": True,
            "enumeration_contract": (
                "Enumerate every current-service/current-version verified dependency "
                "hyperedge that is generic-role complete and budget feasible, plus "
                "deduplicated BM25 role-wise complete hard-negative sets. No event task "
                "key, exact required evidence ID, or outcome is used."
            ),
            "claim_boundary": (
                "Candidate executions supervise bundle outcome prediction on this "
                "controlled benchmark; they are not natural production observations."
            ),
        },
    )
    write_checksums(args.output, ("manifest.json", "selections.jsonl"))
    print(f"wrote {len(rows)} exhaustive bundle candidates to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
