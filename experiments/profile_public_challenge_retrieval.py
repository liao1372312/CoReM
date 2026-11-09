#!/usr/bin/env python3
"""Profile complete query-to-selection latency for the frozen v4 challenge."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.model import SetUtilityModel
from corem.real_eval import (
    load_jsonl,
    parse_utc,
    write_checksums,
    write_json,
    write_jsonl,
)
from corem.retrieval import BundleBeamRetriever, RetrievalConfig
from corem.types import Request
from experiments.generate_public_challenge_selections import (
    GENERIC_REQUIRED_ATOMS,
    METHODS,
)
from experiments.generate_public_workflow_selections import (
    bm25_scores,
    budgeted,
    make_graph,
    ranked_rows,
)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def select_once(
    method: str,
    event: dict[str, Any],
    seed: int,
    memories: list[dict[str, Any]],
    graph: Any,
    retriever: BundleBeamRetriever,
    budget: int,
) -> tuple[list[str], int, int]:
    event_id = str(event["event_id"])
    event_timestamp = parse_utc(event["event_time"], "event_time").timestamp()
    available = [
        row
        for row in memories
        if row.get("service_id") == event.get("service_id")
        and parse_utc(row["valid_from"], "valid_from").timestamp() <= event_timestamp
    ]
    active_version = [
        row
        for row in available
        if event.get("version_id") in row.get("version_scope", [])
    ]
    if method == "no_memory":
        return [], 0, 0
    if method == "service_bm25":
        scores = bm25_scores(str(event["query"]), available)
        selected, cost = budgeted(
            ranked_rows(available, scores, seed=seed, event_id=event_id), budget
        )
        return selected, cost, len(available)
    if method == "recency_bm25":
        scores = bm25_scores(str(event["query"]), available)
        selected, cost = budgeted(
            ranked_rows(
                available,
                scores,
                seed=seed,
                event_id=event_id,
                recency_weight=0.20,
            ),
            budget,
        )
        return selected, cost, len(available)
    if method == "version_bm25":
        scores = bm25_scores(str(event["query"]), active_version)
        selected, cost = budgeted(
            ranked_rows(active_version, scores, seed=seed, event_id=event_id), budget
        )
        return selected, cost, len(active_version)
    request = Request(
        request_id=event_id,
        text=str(event["query"]),
        required_atoms=GENERIC_REQUIRED_ATOMS,
        operation_ids=frozenset(),
        timestamp=event_timestamp,
        allowed_scopes=frozenset((str(event["service_id"]),)),
        active_versions=frozenset((str(event["version_id"]),)),
    )
    result = retriever.retrieve(
        request,
        [str(row["memory_id"]) for row in available],
        graph,
    )
    return sorted(result.selected_ids), result.token_cost, result.evaluations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--reference-selections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=160)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    if args.budget <= 0 or args.repeats <= 0:
        print("budget and repeats must be positive")
        return 2

    events = load_jsonl(args.dataset / "events.jsonl")
    memories = load_jsonl(args.dataset / "memories.jsonl")
    hyperedges = load_jsonl(args.dataset / "dependency_hyperedges.jsonl")
    assignments = load_jsonl(args.splits)
    test_ids = {
        str(row["event_id"]) for row in assignments if row.get("split") == "test"
    }
    test_events = sorted(
        (row for row in events if str(row["event_id"]) in test_ids),
        key=lambda row: (row["event_time"], row["event_id"]),
    )
    reference_rows = [
        row
        for row in load_jsonl(args.reference_selections)
        if str(row["event_id"]) in test_ids
    ]
    reference = {
        (str(row["method"]), int(row["seed"]), str(row["event_id"])): row
        for row in reference_rows
    }
    seeds = sorted({int(row["seed"]) for row in reference_rows})
    expected_keys = len(METHODS) * len(seeds) * len(test_events)
    if len(reference) != expected_keys:
        raise ValueError(
            f"reference has {len(reference)} unique test keys; expected {expected_keys}"
        )

    graph = make_graph(memories, hyperedges)
    retriever = BundleBeamRetriever(
        SetUtilityModel(),
        RetrievalConfig(
            retrieval_budget=args.budget,
            beam_width=16,
            max_depth=4,
            candidate_pool_size=100,
            dependency_hops=2,
            max_closure_size=8,
            enforce_dependency_closure=True,
        ),
    )
    rows: list[dict[str, Any]] = []
    parity_errors: list[str] = []
    for method in METHODS:
        for seed in seeds:
            for event in test_events:
                event_id = str(event["event_id"])
                expected = reference[(method, seed, event_id)]
                warm_selected, warm_cost, _ = select_once(
                    method,
                    event,
                    seed,
                    memories,
                    graph,
                    retriever,
                    args.budget,
                )
                if warm_selected != expected["selected_memory_ids"] or warm_cost != int(
                    expected["token_cost"]
                ):
                    parity_errors.append(f"{method}:{seed}:{event_id}")
                    continue
                for repeat in range(args.repeats):
                    started = time.perf_counter_ns()
                    selected, cost, evaluations = select_once(
                        method,
                        event,
                        seed,
                        memories,
                        graph,
                        retriever,
                        args.budget,
                    )
                    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
                    rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "event_id": event_id,
                            "repeat": repeat,
                            "latency_ms": elapsed_ms,
                            "token_cost": cost,
                            "evaluations": evaluations,
                            "selection_matches_reference": selected
                            == expected["selected_memory_ids"],
                        }
                    )
    if parity_errors or not all(row["selection_matches_reference"] for row in rows):
        raise RuntimeError(
            f"profiling selector differs from frozen selections: {parity_errors[:10]}"
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    summary: list[dict[str, Any]] = []
    for method in METHODS:
        members = grouped[method]
        latencies = [float(row["latency_ms"]) for row in members]
        summary.append(
            {
                "method": method,
                "measurements": len(members),
                "incidents": len({row["event_id"] for row in members}),
                "seeds": len({row["seed"] for row in members}),
                "repeats": args.repeats,
                "median_latency_ms": statistics.median(latencies),
                "p95_latency_ms": percentile(latencies, 0.95),
                "mean_token_cost": statistics.mean(
                    float(row["token_cost"]) for row in members
                ),
                "median_evaluations": statistics.median(
                    int(row["evaluations"]) for row in members
                ),
            }
        )

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "latency_measurements.jsonl", rows)
    with (args.output / "method_latency.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_json(
        args.output / "result.json",
        {
            "status": "passed",
            "scope": "complete query-to-selection CPU latency; excludes service execution and offline training",
            "budget": args.budget,
            "test_events": len(test_events),
            "seeds": seeds,
            "repeats": args.repeats,
            "measurements": len(rows),
            "reference_parity_errors": len(parity_errors),
            "methods": summary,
        },
    )
    write_checksums(
        args.output,
        ("latency_measurements.jsonl", "method_latency.csv", "result.json"),
    )
    print(f"wrote {len(rows)} frozen-selector latency measurements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
