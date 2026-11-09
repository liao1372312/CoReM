#!/usr/bin/env python3
"""Generate frozen selections from a trained public-challenge utility head."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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
from experiments.generate_public_challenge_selections import (
    DEFAULT_SEEDS,
    GENERIC_REQUIRED_ATOMS,
)
from experiments.generate_public_workflow_selections import make_graph
from experiments.train_public_challenge_model import request_for, sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=160)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    payload = json.loads(args.model.read_text(encoding="utf-8"))
    weights = payload["weights"]
    model = SetUtilityModel(
        base_weights=list(map(float, weights["base"])),
        unary_weights=list(map(float, weights["unary"])),
        complement_weights=list(map(float, weights["complement"])),
        redundancy_weights=list(map(float, weights["redundancy"])),
        relation_heads={
            str(key): list(map(float, values))
            for key, values in weights["relation_heads"].items()
        },
    )
    events = load_jsonl(args.dataset / "events.jsonl")
    memories = load_jsonl(args.dataset / "memories.jsonl")
    graph = make_graph(
        memories, load_jsonl(args.dataset / "dependency_hyperedges.jsonl")
    )
    retriever = BundleBeamRetriever(
        model,
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
    rows = []
    ordered_events = sorted(
        events,
        key=lambda row: (
            parse_utc(row["event_time"], "event_time"),
            str(row["event_id"]),
        ),
    )
    for seed in args.seeds:
        for event in ordered_events:
            event_timestamp = parse_utc(event["event_time"], "event_time").timestamp()
            available = [
                row
                for row in memories
                if row.get("service_id") == event.get("service_id")
                and parse_utc(row["valid_from"], "valid_from").timestamp()
                <= event_timestamp
            ]
            started = time.perf_counter()
            result = retriever.retrieve(
                request_for(event),
                [str(row["memory_id"]) for row in available],
                graph,
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            event_id = str(event["event_id"])
            rows.append(
                {
                    "run_id": f"corem_trained:{seed}:{event_id}",
                    "event_id": event_id,
                    "seed": seed,
                    "method": "corem_trained",
                    "selected_memory_ids": sorted(result.selected_ids),
                    "token_cost": result.token_cost,
                    "latency_ms": elapsed,
                    "llm_calls": 0,
                    "active_bank_cost": sum(
                        int(row["storage_cost"]) for row in available
                    ),
                    "retrieval_status": result.status,
                    "evaluations": result.evaluations,
                }
            )
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "selections.jsonl", rows)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": "corem-public-challenge-trained-selections-v1",
            "dataset": str(args.dataset.resolve()),
            "method": "corem_trained",
            "seeds": args.seeds,
            "token_budget": args.token_budget,
            "selection_rows": len(rows),
            "model_path": str(args.model.resolve()),
            "model_sha256": sha256(args.model),
            "label_blind_at_inference": True,
            "generic_required_atoms": sorted(GENERIC_REQUIRED_ATOMS),
        },
    )
    write_checksums(args.output, ("manifest.json", "selections.jsonl"))
    print(f"wrote {len(rows)} frozen trained selections to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
