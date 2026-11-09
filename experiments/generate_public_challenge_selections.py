#!/usr/bin/env python3
"""Generate label-blind selections for the three-step public challenge."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.model import SetUtilityModel  # noqa: E402
from corem.real_eval import (  # noqa: E402
    load_jsonl,
    parse_utc,
    write_checksums,
    write_json,
    write_jsonl,
)
from corem.retrieval import BundleBeamRetriever, RetrievalConfig  # noqa: E402
from corem.types import Request  # noqa: E402
from experiments.generate_public_workflow_selections import (  # noqa: E402
    DEFAULT_SEEDS,
    bm25_scores,
    budgeted,
    make_graph,
    ranked_rows,
)

METHODS = (
    "no_memory",
    "service_bm25",
    "recency_bm25",
    "version_bm25",
    "corem_query_only",
)
GENERIC_REQUIRED_ATOMS = frozenset(
    ("role:step1", "role:step2", "role:step3", "role:verify")
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=160)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    if len(set(args.seeds)) != len(args.seeds):
        print("seeds must be unique")
        return 2
    if len(set(args.methods)) != len(args.methods):
        print("methods must be unique")
        return 2

    events = load_jsonl(args.dataset / "events.jsonl")
    memories = load_jsonl(args.dataset / "memories.jsonl")
    hyperedges = load_jsonl(args.dataset / "dependency_hyperedges.jsonl")
    graph = make_graph(memories, hyperedges)
    retriever = BundleBeamRetriever(
        SetUtilityModel(),
        RetrievalConfig(
            retrieval_budget=args.token_budget,
            beam_width=16,
            max_depth=4,
            candidate_pool_size=100,
            dependency_hops=2,
            max_closure_size=8,
            enforce_dependency_closure=True,
        ),
    )
    rows: list[dict[str, Any]] = []
    ordered_events = sorted(
        events,
        key=lambda row: (
            parse_utc(row["event_time"], "event_time"),
            str(row["event_id"]),
        ),
    )
    for seed in args.seeds:
        for event in ordered_events:
            event_id = str(event["event_id"])
            event_timestamp = parse_utc(event["event_time"], "event_time").timestamp()
            available = [
                row
                for row in memories
                if row.get("service_id") == event.get("service_id")
                and parse_utc(row["valid_from"], "valid_from").timestamp()
                <= event_timestamp
            ]
            active_version = [
                row
                for row in available
                if event.get("version_id") in row.get("version_scope", [])
            ]
            lexical = bm25_scores(str(event["query"]), available)
            version_lexical = bm25_scores(str(event["query"]), active_version)
            for method in args.methods:
                started = time.perf_counter()
                status = "ok"
                evaluations = 0
                if method == "no_memory":
                    selected, cost = [], 0
                    status = "fallback"
                elif method == "service_bm25":
                    selected, cost = budgeted(
                        ranked_rows(
                            available,
                            lexical,
                            seed=seed,
                            event_id=event_id,
                        ),
                        args.token_budget,
                    )
                    evaluations = len(available)
                elif method == "recency_bm25":
                    selected, cost = budgeted(
                        ranked_rows(
                            available,
                            lexical,
                            seed=seed,
                            event_id=event_id,
                            recency_weight=0.20,
                        ),
                        args.token_budget,
                    )
                    evaluations = len(available)
                elif method == "version_bm25":
                    selected, cost = budgeted(
                        ranked_rows(
                            active_version,
                            version_lexical,
                            seed=seed,
                            event_id=event_id,
                        ),
                        args.token_budget,
                    )
                    evaluations = len(active_version)
                else:
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
                    selected = sorted(result.selected_ids)
                    cost = result.token_cost
                    status = result.status
                    evaluations = result.evaluations
                elapsed = (time.perf_counter() - started) * 1000.0
                rows.append(
                    {
                        "run_id": f"{method}:{seed}:{event_id}",
                        "event_id": event_id,
                        "seed": seed,
                        "method": method,
                        "selected_memory_ids": selected,
                        "token_cost": cost,
                        "latency_ms": elapsed,
                        "llm_calls": 0,
                        "active_bank_cost": sum(
                            int(row["storage_cost"]) for row in available
                        ),
                        "retrieval_status": status,
                        "evaluations": evaluations,
                    }
                )

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "selections.jsonl", rows)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": "corem-public-challenge-selections-v1",
            "dataset": str(args.dataset.resolve()),
            "methods": args.methods,
            "seeds": args.seeds,
            "token_budget": args.token_budget,
            "selection_rows": len(rows),
            "label_blind": True,
            "retrieval_inputs": {
                "query": True,
                "service_and_active_version": True,
                "generic_workflow_roles": True,
                "event_task_key": False,
                "exact_required_evidence_atoms": False,
                "checker_outcomes": False,
            },
            "seed_policy": "Seeds affect deterministic SHA-256 tie-breaking only; no seed is selected by outcome.",
            "method_note": "corem_query_only is the untrained repository retriever with generic role closure, no event task key, and no exact evidence identifiers.",
        },
    )
    write_checksums(args.output, ("manifest.json", "selections.jsonl"))
    print(f"wrote {len(rows)} challenge selections to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
