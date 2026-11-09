#!/usr/bin/env python3
"""Label-blind BM25 baseline over exactly the same complete candidate bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import (  # noqa: E402
    load_jsonl,
    parse_utc,
    write_checksums,
    write_json,
    write_jsonl,
)
from corem.v8_candidates import validate_candidate  # noqa: E402
from experiments.generate_public_workflow_selections import bm25_scores  # noqa: E402
from experiments.summarize_independent_v7_confirmation import sha256  # noqa: E402

QUERY_FIELDS = {
    "event_id",
    "service_id",
    "provider_id",
    "version_id",
    "event_time",
    "query",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--hyperedges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregation", choices=("sum", "max"), default="sum")
    parser.add_argument(
        "--document-view",
        choices=("text", "text_and_plan"),
        default="text",
        help="Include executable API paths and JSON bodies in the lexical control.",
    )
    parser.add_argument("--token-budget", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.token_budget < 1:
        parser.error("token budget must be positive")
    event_rows = load_jsonl(args.events)
    if any(set(row) != QUERY_FIELDS for row in event_rows):
        raise ValueError("query view includes labels or checker targets")
    events = {str(row["event_id"]): row for row in event_rows}
    if len(events) != len(event_rows) or not events:
        raise ValueError("empty or duplicate query events")
    memory_rows = load_jsonl(args.memories)
    memories = {str(row["memory_id"]): row for row in memory_rows}
    edge_rows = load_jsonl(args.hyperedges)
    edges = {str(row["edge_id"]): row for row in edge_rows}
    if len(memories) != len(memory_rows) or len(edges) != len(edge_rows):
        raise ValueError("duplicate memory or hyperedge identifiers")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_keys: set[tuple[str, str]] = set()
    for row in load_jsonl(args.candidates):
        event_id = str(row["event_id"])
        if event_id not in events:
            raise ValueError(f"candidate for unknown event: {event_id}")
        candidate_key = (event_id, str(row["candidate_edge_id"]))
        if candidate_key in candidate_keys:
            raise ValueError(f"duplicate candidate: {candidate_key}")
        candidate_keys.add(candidate_key)
        validate_candidate(row, events[event_id], memories, edges)
        grouped[event_id].append(row)
    if set(grouped) != set(events) or any(len(rows) < 2 for rows in grouped.values()):
        raise ValueError("each query requires at least two complete candidates")

    selections = []
    score_rows = []
    for event_id in sorted(events):
        started = time.perf_counter()
        event = events[event_id]
        timestamp = parse_utc(event["event_time"], "event_time")
        active = [
            row
            for row in memory_rows
            if row["service_id"] == event["service_id"]
            and row.get("provider_id", event.get("provider_id", event["service_id"]))
            == event.get("provider_id", event["service_id"])
            and event["version_id"] in row["version_scope"]
            and row["provenance_verified"] is True
            and parse_utc(row["valid_from"], "valid_from") <= timestamp
            and (
                row.get("valid_until") is None
                or parse_utc(row["valid_until"], "valid_until") > timestamp
            )
        ]
        if args.document_view == "text_and_plan":
            documents = [
                {
                    **row,
                    "text": (
                        str(row["text"])
                        + " "
                        + json.dumps(row.get("plan", {}), sort_keys=True)
                    ),
                }
                for row in active
            ]
        else:
            documents = active
        lexical = bm25_scores(str(event["query"]), documents)
        ranked = []
        for candidate in grouped[event_id]:
            ids = sorted(map(str, candidate["selected_memory_ids"]))
            cost = sum(int(memories[memory_id]["token_cost"]) for memory_id in ids)
            if cost > args.token_budget:
                raise ValueError(f"candidate exceeds token budget: {event_id}")
            values = [float(lexical[memory_id]) for memory_id in ids]
            score = sum(values) if args.aggregation == "sum" else max(values)
            digest = hashlib.sha256(
                f"{args.seed}|{event_id}|{'|'.join(ids)}".encode()
            ).hexdigest()
            ranked.append((score, digest, ids, cost))
            score_rows.append(
                {
                    "event_id": event_id,
                    "candidate_edge_id": candidate["candidate_edge_id"],
                    "selected_memory_ids": ids,
                    "bm25_score": score,
                }
            )
        chosen = max(ranked)
        selections.append(
            {
                "run_id": f"candidate_bm25_{args.aggregation}:{args.seed}:{event_id}",
                "event_id": event_id,
                "method": f"candidate_bm25_{args.aggregation}",
                "seed": args.seed,
                "selected_memory_ids": chosen[2],
                "token_cost": chosen[3],
                "latency_ms": (time.perf_counter() - started) * 1000,
                "retrieval_status": "ok",
                "evaluations": len(ranked),
            }
        )
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "selections.jsonl", selections)
    write_jsonl(args.output / "candidate_scores.jsonl", score_rows)
    write_json(
        args.output / "manifest.json",
        {
            "status": "label_blind_same_candidate_selection",
            "outcomes_loaded": False,
            "aggregation": args.aggregation,
            "document_view": args.document_view,
            "token_budget": args.token_budget,
            "tie_break_policy": "seeded_hash",
            "seed": args.seed,
            "events": len(events),
            "candidate_rows": len(score_rows),
            "input_sha256": {
                "events": sha256(args.events),
                "candidates": sha256(args.candidates),
                "memories": sha256(args.memories),
                "hyperedges": sha256(args.hyperedges),
            },
        },
    )
    write_checksums(
        args.output,
        ("candidate_scores.jsonl", "manifest.json", "selections.jsonl"),
    )
    print(f"wrote {len(selections)} same-candidate BM25 selections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
