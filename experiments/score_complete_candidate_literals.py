#!/usr/bin/env python3
"""Label-blind literal-value control over the same complete workflow bundles."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_jsonl, write_checksums, write_json, write_jsonl  # noqa: E402
from corem.v8_candidates import validate_candidate  # noqa: E402
from experiments.summarize_independent_v7_confirmation import sha256  # noqa: E402

QUERY_FIELDS = {"event_id", "service_id", "provider_id", "version_id", "event_time", "query"}


def literals(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set().union(*(literals(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(literals(item) for item in value))
    if isinstance(value, str):
        return {value.casefold()} if len(value) >= 3 else set()
    return set()


def literal_score(query: str, candidate: dict[str, Any], memories: dict[str, dict[str, Any]]) -> int:
    query = query.casefold()
    values = set().union(
        *(
            literals(memories[memory_id].get("plan", {}).get("json", {}))
            | literals(memories[memory_id].get("plan", {}).get("path", ""))
            for memory_id in candidate["selected_memory_ids"]
        )
    )
    return sum(len(value) for value in values if value in query)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--hyperedges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.token_budget < 1:
        parser.error("token budget must be positive")
    event_rows = load_jsonl(args.events)
    if not event_rows or any(set(row) != QUERY_FIELDS for row in event_rows):
        raise ValueError("query view contains missing fields or outcomes")
    events = {str(row["event_id"]): row for row in event_rows}
    memory_rows = load_jsonl(args.memories)
    edge_rows = load_jsonl(args.hyperedges)
    memories = {str(row["memory_id"]): row for row in memory_rows}
    edges = {str(row["edge_id"]): row for row in edge_rows}
    if (
        len(events) != len(event_rows)
        or len(memories) != len(memory_rows)
        or len(edges) != len(edge_rows)
    ):
        raise ValueError("duplicate event, memory, or edge")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for candidate in load_jsonl(args.candidates):
        event_id = str(candidate["event_id"])
        if event_id not in events:
            raise ValueError(f"candidate for unknown event: {event_id}")
        candidate_key = (event_id, str(candidate["candidate_edge_id"]))
        if candidate_key in seen:
            raise ValueError(f"duplicate candidate: {candidate_key}")
        seen.add(candidate_key)
        validate_candidate(candidate, events[event_id], memories, edges)
        cost = sum(int(memories[memory_id]["token_cost"]) for memory_id in candidate["selected_memory_ids"])
        if cost > args.token_budget:
            raise ValueError(f"candidate exceeds budget: {event_id}")
        grouped[event_id].append(candidate)
    if set(grouped) != set(events) or any(len(rows) < 2 for rows in grouped.values()):
        raise ValueError("each event needs at least two complete candidates")
    selections = []
    score_rows = []
    for event_id in sorted(events):
        ranked = []
        for candidate in grouped[event_id]:
            ids = sorted(map(str, candidate["selected_memory_ids"]))
            score = literal_score(str(events[event_id]["query"]), candidate, memories)
            digest = hashlib.sha256(f"{args.seed}|{event_id}|{'|'.join(ids)}".encode()).hexdigest()
            ranked.append((score, digest, ids))
            score_rows.append(
                {
                    "event_id": event_id,
                    "candidate_edge_id": candidate["candidate_edge_id"],
                    "selected_memory_ids": ids,
                    "literal_score": score,
                }
            )
        chosen = max(ranked)
        selections.append(
            {
                "run_id": f"candidate_literals:{args.seed}:{event_id}",
                "event_id": event_id,
                "method": "candidate_literals",
                "seed": args.seed,
                "selected_memory_ids": chosen[2],
                "token_cost": sum(int(memories[memory_id]["token_cost"]) for memory_id in chosen[2]),
                "retrieval_status": "ok",
                "evaluations": len(ranked),
            }
        )
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "candidate_scores.jsonl", score_rows)
    write_jsonl(args.output / "selections.jsonl", selections)
    write_json(
        args.output / "manifest.json",
        {
            "status": "label_blind_literal_control",
            "outcomes_loaded": False,
            "rule": "sum lengths of distinct casefolded plan JSON/path string values of length >= 3 that occur verbatim in the query",
            "seed": args.seed,
            "token_budget": args.token_budget,
            "events": len(events),
            "input_sha256": {
                "events": sha256(args.events),
                "candidates": sha256(args.candidates),
                "memories": sha256(args.memories),
                "hyperedges": sha256(args.hyperedges),
            },
        },
    )
    write_checksums(args.output, ("candidate_scores.jsonl", "manifest.json", "selections.jsonl"))
    print(f"wrote {len(selections)} literal-control selections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
