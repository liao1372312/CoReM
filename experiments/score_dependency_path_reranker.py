#!/usr/bin/env python3
"""Score label-free complete workflow candidates with a frozen path checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import (  # noqa: E402
    load_jsonl,
    write_checksums,
    write_json,
    write_jsonl,
)
from corem.v8_candidates import validate_candidate  # noqa: E402
from experiments.generate_public_multiservice_hgt_ensemble_selections import (  # noqa: E402
    validate_query_only_event,
)
from experiments.score_v7_path_nli_development import serialize_path  # noqa: E402
from experiments.path_serialization import serialize_path_body_v1  # noqa: E402
from experiments.summarize_independent_v7_confirmation import sha256  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--hyperedges", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--token-budget", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--serialization",
        choices=("legacy", "body_v1"),
        default="legacy",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.batch_size < 1 or args.max_length < 32 or args.token_budget < 1:
        parser.error("batch size, max length, and token budget must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    training_manifest = args.model_dir.parent / "result.json"
    if not training_manifest.is_file():
        raise ValueError("path checkpoint lacks a training manifest")
    training = json.loads(training_manifest.read_text(encoding="utf-8"))
    if (
        training.get("serialization", "legacy") != args.serialization
        or training.get("max_length") != args.max_length
        or training.get("full_development_checkpoint_sha256")
        != sha256(args.model_dir / "model.safetensors")
    ):
        raise ValueError("checkpoint serialization, sequence length, or weights mismatch")

    events = {}
    for row in load_jsonl(args.events):
        validate_query_only_event(row)
        event_id = str(row["event_id"])
        if event_id in events:
            raise ValueError(f"duplicate event: {event_id}")
        events[event_id] = row
    if not events:
        raise ValueError("no query events")
    memory_rows = load_jsonl(args.memories)
    edge_rows = load_jsonl(args.hyperedges)
    memories = {str(row["memory_id"]): row for row in memory_rows}
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
        cost = sum(int(memories[memory_id]["token_cost"]) for memory_id in row["selected_memory_ids"])
        if cost > args.token_budget:
            raise ValueError(f"candidate exceeds token budget: {event_id}")
        grouped[event_id].append(row)
    if set(grouped) != set(events) or any(len(rows) < 2 for rows in grouped.values()):
        raise ValueError("each event needs at least two valid candidates")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, local_files_only=True
    ).to(args.device)
    model.eval()
    labels = {value.lower(): int(key) for key, value in model.config.id2label.items()}
    if "entailment" not in labels or "contradiction" not in labels:
        raise ValueError(f"path checkpoint lacks NLI score labels: {labels}")
    candidate_rows = [row for event_id in sorted(grouped) for row in grouped[event_id]]
    serializer = serialize_path if args.serialization == "legacy" else serialize_path_body_v1
    amortized_latency_ms: dict[str, float] = defaultdict(float)
    candidate_scores = []
    for start in range(0, len(candidate_rows), args.batch_size):
        batch_started = time.perf_counter()
        batch = candidate_rows[start : start + args.batch_size]
        premises = [serializer(row, memories) for row in batch]
        hypotheses = [
            "This workflow fulfills the request: " + events[str(row["event_id"])]["query"]
            for row in batch
        ]
        tokens = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(args.device)
        with torch.inference_mode():
            logits = model(**tokens).logits.float().cpu()
        batch_elapsed_ms = (time.perf_counter() - batch_started) * 1000
        for row in batch:
            amortized_latency_ms[str(row["event_id"])] += batch_elapsed_ms / len(batch)
        for row, values in zip(batch, logits):
            candidate_scores.append(
                {
                    "event_id": row["event_id"],
                    "candidate_edge_id": row["candidate_edge_id"],
                    "selected_memory_ids": sorted(map(str, row["selected_memory_ids"])),
                    "score": (
                        values[labels["entailment"]]
                        - values[labels["contradiction"]]
                    ).item(),
                }
            )

    scored: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_scores:
        scored[str(row["event_id"])].append(row)
    selections = []
    for event_id in sorted(scored):
        def rank_key(row: dict[str, Any]) -> tuple[float, str]:
            digest = hashlib.sha256(
                f"{args.seed}|{event_id}|{'|'.join(row['selected_memory_ids'])}".encode()
            ).hexdigest()
            return float(row["score"]), digest

        ranked = sorted(scored[event_id], key=rank_key, reverse=True)
        chosen = ranked[0]
        selected = chosen["selected_memory_ids"]
        selections.append(
            {
                "run_id": f"path_reranker:{args.seed}:{event_id}",
                "event_id": event_id,
                "method": "pairwise_path_reranker",
                "seed": args.seed,
                "selected_memory_ids": selected,
                "token_cost": sum(int(memories[memory_id]["token_cost"]) for memory_id in selected),
                "score_margin": float(ranked[0]["score"]) - float(ranked[1]["score"]),
                "amortized_selection_latency_ms": amortized_latency_ms[event_id],
                "retrieval_status": "ok",
                "evaluations": len(ranked),
            }
        )
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "candidate_scores.jsonl", candidate_scores)
    write_jsonl(args.output / "selections.jsonl", selections)
    write_json(
        args.output / "manifest.json",
        {
            "status": "label_blind_selection_only",
            "outcomes_loaded": False,
            "model_weights_sha256": sha256(args.model_dir / "model.safetensors"),
            "model_config_sha256": sha256(args.model_dir / "config.json"),
            "training_manifest_sha256": sha256(training_manifest),
            "events_sha256": sha256(args.events),
            "candidates_sha256": sha256(args.candidates),
            "memories_sha256": sha256(args.memories),
            "hyperedges_sha256": sha256(args.hyperedges),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "serialization": args.serialization,
            "token_budget": args.token_budget,
            "seed": args.seed,
            "events": len(events),
            "candidate_rows": len(candidate_rows),
            "latency_definition": (
                "Shared-batch tokenization plus forward-pass time divided across "
                "candidate rows; not serial per-query wall-clock latency."
            ),
        },
    )
    write_checksums(
        args.output,
        ("candidate_scores.jsonl", "manifest.json", "selections.jsonl"),
    )
    print(f"wrote {len(selections)} label-blind path-reranker selections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
