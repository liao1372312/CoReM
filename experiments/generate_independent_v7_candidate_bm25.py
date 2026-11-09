#!/usr/bin/env python3
"""Select from the exact v7 candidate pool with label-blind BM25 only."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import (  # noqa: E402
    load_jsonl,
    parse_utc,
    sha256_file,
    write_checksums,
    write_json,
    write_jsonl,
)
from experiments.generate_public_multiservice_challenge_ablations import (  # noqa: E402
    request_for,
)
from experiments.generate_public_workflow_selections import (  # noqa: E402
    bm25_scores,
    make_graph,
)

CANDIDATE_FIELDS = {"event_id", "candidate_edge_id", "selected_memory_ids"}
QUERY_FIELDS = {
    "event_id",
    "service_id",
    "provider_id",
    "version_id",
    "event_time",
    "query",
}


def _score(values: list[float], aggregation: str) -> float:
    if aggregation == "max":
        return max(values)
    if aggregation == "sum":
        return sum(values)
    raise ValueError(f"unsupported aggregation: {aggregation}")


def _tie_hash(seed: int, event_id: str, ids: list[str]) -> str:
    return hashlib.sha256(f"{seed}|{event_id}|{'|'.join(ids)}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--query-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregation", choices=("max", "sum"), default="max")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--token-budget", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite baseline selections")
    if args.token_budget <= 0:
        parser.error("token budget must be positive")
    events = load_jsonl(args.query_view)
    if any(set(row) != QUERY_FIELDS for row in events):
        parser.error("query view includes labels or checker targets")
    event_by_id = {str(row["event_id"]): row for row in events}
    target_ids = {
        str(row["event_id"])
        for row in load_jsonl(args.splits)
        if row.get("split") == "test"
    }
    if not target_ids or not target_ids.issubset(event_by_id):
        parser.error("test split has missing query-only events")
    memories = load_jsonl(args.dataset / "memories.jsonl")
    graph = make_graph(
        memories, load_jsonl(args.dataset / "dependency_hyperedges.jsonl")
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in load_jsonl(args.candidates):
        if set(candidate) != CANDIDATE_FIELDS:
            parser.error("candidate input includes labels or outcomes")
        event_id = str(candidate["event_id"])
        if event_id in target_ids:
            grouped[event_id].append(candidate)
    if set(grouped) != target_ids or any(len(rows) < 2 for rows in grouped.values()):
        parser.error("each test event requires at least two complete candidates")
    output = []
    scored = []
    for event_id in sorted(target_ids):
        start = time.perf_counter()
        event = event_by_id[event_id]
        request = request_for(event)
        timestamp = parse_utc(event["event_time"], "event_time").timestamp()
        active = [
            row
            for row in memories
            if row.get("service_id") == event["service_id"]
            and event["version_id"] in row.get("version_scope", [])
            and parse_utc(row["valid_from"], "valid_from").timestamp() <= timestamp
        ]
        lexical = bm25_scores(request.text, active)
        ranked = []
        for candidate in grouped[event_id]:
            ids = sorted(map(str, candidate["selected_memory_ids"]))
            selected = frozenset(ids)
            cost = sum(graph.records[memory_id].token_cost for memory_id in ids)
            if (
                cost > args.token_budget
                or not graph.is_evidence_complete(request, selected)
                or not all(graph.is_admissible(memory_id, request) for memory_id in ids)
                or any(memory_id not in lexical for memory_id in ids)
            ):
                raise ValueError(f"infeasible candidate for {event_id}")
            score = _score(
                [float(lexical[memory_id]) for memory_id in ids], args.aggregation
            )
            ranked.append((score, _tie_hash(args.seed, event_id, ids), ids, cost))
            scored.append(
                {
                    "event_id": event_id,
                    "candidate_edge_id": candidate["candidate_edge_id"],
                    "selected_memory_ids": ids,
                    "bm25_score": score,
                }
            )
        winner = max(ranked)
        output.append(
            {
                "run_id": f"candidate_bm25_{args.aggregation}:{args.seed}:{event_id}",
                "event_id": event_id,
                "method": f"candidate_bm25_{args.aggregation}",
                "seed": args.seed,
                "selected_memory_ids": winner[2],
                "token_cost": winner[3],
                "latency_ms": (time.perf_counter() - start) * 1000.0,
                "llm_calls": 0,
                "retrieval_status": "ok",
                "evaluations": len(ranked),
            }
        )
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "selections.jsonl", output)
    write_jsonl(args.output / "candidate_scores.jsonl", scored)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": "corem-v7-candidate-bm25-v1",
            "status": "label_blind_selections",
            "dataset": str(args.dataset.resolve()),
            "query_view_sha256": sha256_file(args.query_view),
            "candidate_input_sha256": sha256_file(args.candidates),
            "aggregation": args.aggregation,
            "tie_break_policy": "seeded_hash",
            "seed": args.seed,
            "token_budget": args.token_budget,
            "events": len(output),
            "outcomes_loaded": False,
        },
    )
    write_checksums(
        args.output, ("selections.jsonl", "candidate_scores.jsonl", "manifest.json")
    )
    print(f"wrote {len(output)} label-blind BM25 selections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
