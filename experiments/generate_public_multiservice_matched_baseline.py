#!/usr/bin/env python3
"""Generate a version-BM25 baseline with the same verified bundle closure as CoReM."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import (  # noqa: E402
    load_jsonl,
    parse_utc,
    sha256_file,
    write_checksums,
    write_json,
    write_jsonl,
)
from corem.types import EvidenceGraph, Request  # noqa: E402
from experiments.generate_public_challenge_selections import (  # noqa: E402
    DEFAULT_SEEDS,
)
from experiments.generate_public_multiservice_challenge_ablations import request_for  # noqa: E402
from experiments.generate_public_workflow_selections import (  # noqa: E402
    bm25_scores,
    make_graph,
    ranked_rows,
)

METHOD = "version_bm25_bundle"
QUERY_FIELDS = {
    "event_id",
    "service_id",
    "provider_id",
    "version_id",
    "event_time",
    "query",
}


def select_bundle(
    request: Request,
    active_rows: list[dict[str, Any]],
    graph: EvidenceGraph,
    *,
    seed: int,
    event_id: str,
    token_budget: int,
) -> tuple[list[str], int, int]:
    """Return the first BM25-ranked seed whose verified closure is feasible."""
    lexical = bm25_scores(request.text, active_rows)
    ranked = ranked_rows(active_rows, lexical, seed=seed, event_id=event_id)
    bank = frozenset(str(row["memory_id"]) for row in active_rows)
    evaluations = 0
    for row in ranked:
        evaluations += 1
        seed_id = str(row["memory_id"])
        closure = graph.dependency_closure(
            request,
            frozenset((seed_id,)),
            bank,
            max_hops=2,
            max_size=8,
        )
        cost = sum(graph.records[memory_id].token_cost for memory_id in closure)
        if (
            cost <= token_budget
            and graph.is_evidence_complete(request, closure)
            and all(graph.is_admissible(memory_id, request) for memory_id in closure)
        ):
            return sorted(closure), cost, evaluations
    return [], 0, evaluations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--event-view",
        type=Path,
        help="Label-free query-only event view for sealed evaluation",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=160)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    if len(set(args.seeds)) != len(args.seeds):
        print("seeds must be unique")
        return 2

    events = load_jsonl(args.event_view or args.dataset / "events.jsonl")
    if args.event_view is not None and any(set(row) != QUERY_FIELDS for row in events):
        parser.error("event view contains checker targets or other fields")
    memories = load_jsonl(args.dataset / "memories.jsonl")
    graph = make_graph(
        memories,
        load_jsonl(args.dataset / "dependency_hyperedges.jsonl"),
    )
    ordered_events = sorted(
        events,
        key=lambda row: (
            parse_utc(row["event_time"], "event_time"),
            str(row["event_id"]),
        ),
    )
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for event in ordered_events:
            event_id = str(event["event_id"])
            event_timestamp = parse_utc(event["event_time"], "event_time").timestamp()
            active_rows = [
                row
                for row in memories
                if row.get("service_id") == event.get("service_id")
                and event.get("version_id") in row.get("version_scope", [])
                and parse_utc(row["valid_from"], "valid_from").timestamp()
                <= event_timestamp
            ]
            started = time.perf_counter()
            selected, cost, evaluations = select_bundle(
                request_for(event),
                active_rows,
                graph,
                seed=seed,
                event_id=event_id,
                token_budget=args.token_budget,
            )
            rows.append(
                {
                    "run_id": f"{METHOD}:{seed}:{event_id}",
                    "event_id": event_id,
                    "seed": seed,
                    "method": METHOD,
                    "selected_memory_ids": selected,
                    "token_cost": cost,
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                    "llm_calls": 0,
                    "active_bank_cost": sum(
                        int(row["storage_cost"]) for row in active_rows
                    ),
                    "retrieval_status": "ok" if selected else "fallback",
                    "evaluations": evaluations,
                }
            )

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "selections.jsonl", rows)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": "corem-public-multiservice-matched-baseline-v1",
            "dataset": str(args.dataset.resolve()),
            "event_view": (
                {
                    "path": str(args.event_view.resolve()),
                    "sha256": sha256_file(args.event_view),
                }
                if args.event_view is not None
                else None
            ),
            "method": METHOD,
            "seeds": args.seeds,
            "token_budget": args.token_budget,
            "selection_rows": len(rows),
            "label_blind": True,
            "outcomes_loaded": False,
            "control_contract": (
                "Rank active-version memories with the same BM25 and tie policy as "
                "version_bm25, then atomically select the first budget-feasible, "
                "evidence-complete closure from the same verified dependency hypergraph "
                "available to CoReM. No learned or CoReM utility score is used."
            ),
        },
    )
    write_checksums(args.output, ("manifest.json", "selections.jsonl"))
    print(f"wrote {len(rows)} matched-baseline selections to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
