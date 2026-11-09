#!/usr/bin/env python3
"""Frozen MiniLM query-to-record bundle baseline on label-free candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import (  # noqa: E402
    load_jsonl,
    write_checksums,
    write_json,
    write_jsonl,
)
from experiments.generate_public_multiservice_challenge_ablations import (  # noqa: E402
    request_for,
)
from experiments.generate_public_multiservice_hgt_ensemble_selections import (  # noqa: E402
    rank_candidates,
    validate_label_free_candidate,
    validate_query_only_event,
)
from experiments.generate_public_workflow_selections import make_graph  # noqa: E402

MODEL_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
    / "snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--event-view", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--candidate-selections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--aggregation", choices=("max", "sum", "mean"), default="sum")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-snapshot", type=Path, default=MODEL_SNAPSHOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite baseline output: {args.output}")
    weights = args.model_snapshot / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"missing pinned local MiniLM weights: {weights}")
    events = {}
    for row in load_jsonl(args.event_view):
        validate_query_only_event(row)
        events[str(row["event_id"])] = row
    target_ids = {
        str(row["event_id"])
        for row in load_jsonl(args.splits)
        if row["split"] == args.split
    }
    if not target_ids or not target_ids.issubset(events):
        raise ValueError("missing target query-only events")
    memories = load_jsonl(args.dataset / "memories.jsonl")
    graph = make_graph(
        memories, load_jsonl(args.dataset / "dependency_hyperedges.jsonl")
    )
    candidates: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(args.candidate_selections):
        validate_label_free_candidate(row)
        event_id = str(row["event_id"])
        if event_id in target_ids:
            candidates.setdefault(event_id, []).append(row)
    if set(candidates) != target_ids:
        raise ValueError("candidate enumeration misses target events")
    texts = [str(events[event_id]["query"]) for event_id in sorted(target_ids)]
    text_keys = [("query", event_id) for event_id in sorted(target_ids)]
    for row in memories:
        texts.append(str(row["text"]))
        text_keys.append(("memory", str(row["memory_id"])))
    model = SentenceTransformer(
        str(args.model_snapshot), device=args.device, local_files_only=True
    )
    vectors = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    encoded = {key: vector for key, vector in zip(text_keys, vectors)}
    selections, scores = [], []
    for event_id in sorted(target_ids):
        started = time.perf_counter()
        event = events[event_id]
        request = request_for(event)
        rows = []
        for candidate in candidates[event_id]:
            selected = frozenset(map(str, candidate["selected_memory_ids"]))
            if not graph.is_evidence_complete(request, selected):
                raise ValueError(f"incomplete candidate for {event_id}")
            ordered_ids = sorted(selected)
            similarities = [
                float(
                    np.dot(encoded[("query", event_id)], encoded[("memory", memory_id)])
                )
                for memory_id in ordered_ids
            ]
            score = (
                max(similarities)
                if args.aggregation == "max"
                else sum(similarities)
                if args.aggregation == "sum"
                else float(np.mean(similarities))
            )
            rows.append(
                {
                    "event_id": event_id,
                    "candidate_edge_id": candidate["candidate_edge_id"],
                    "selected_memory_ids": ordered_ids,
                    "score": score,
                    "record_similarities": similarities,
                }
            )
        ranked = rank_candidates(rows, event_id, args.seed, "seeded_hash")
        chosen = ranked[0]
        selected_ids = list(chosen["selected_memory_ids"])
        selections.append(
            {
                "run_id": f"minilm_{args.aggregation}:{args.seed}:{event_id}",
                "event_id": event_id,
                "method": f"minilm_{args.aggregation}",
                "seed": args.seed,
                "selected_memory_ids": selected_ids,
                "token_cost": sum(
                    graph.records[memory_id].token_cost for memory_id in selected_ids
                ),
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "selection_score": chosen["score"],
                "top_score_margin": (
                    float(ranked[0]["score"]) - float(ranked[1]["score"])
                    if len(ranked) > 1
                    else None
                ),
            }
        )
        scores.extend(rows)
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "selections.jsonl", selections)
    write_jsonl(args.output / "candidate_scores.jsonl", scores)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": "corem-semantic-bundle-baseline-v1",
            "method": f"minilm_{args.aggregation}",
            "status": "development/baseline; no CoReM learning",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "model_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            "model_license": "Apache-2.0",
            "model_weights_sha256": _digest(weights),
            "dataset": str(args.dataset.resolve()),
            "event_view_sha256": _digest(args.event_view),
            "candidate_input_sha256": _digest(args.candidate_selections),
            "split": args.split,
            "aggregation": args.aggregation,
            "seed": args.seed,
            "device": args.device,
            "outcomes_loaded": False,
            "selection_rows": len(selections),
        },
    )
    write_checksums(
        args.output, ("selections.jsonl", "candidate_scores.jsonl", "manifest.json")
    )
    print(json.dumps({"selections": len(selections), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
