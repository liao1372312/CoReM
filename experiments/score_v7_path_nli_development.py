#!/usr/bin/env python3
"""Exploratory zero-shot NLI scoring of complete v7 workflow paths.

Never use this already-inspected v7 split as independent confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_jsonl, write_checksums, write_json, write_jsonl  # noqa: E402
from experiments.generate_public_multiservice_hgt_ensemble_selections import (  # noqa: E402
    validate_label_free_candidate,
    validate_query_only_event,
)
from experiments.summarize_independent_v7_confirmation import sha256  # noqa: E402


def serialize_path(candidate: dict[str, Any], memories: dict[str, dict[str, Any]]) -> str:
    records = [memories[str(memory_id)] for memory_id in candidate["selected_memory_ids"]]
    records.sort(key=lambda row: (int(row.get("step_index") or 10_000), str(row["memory_id"])))
    steps = []
    for row in records:
        plan = row.get("plan") or {}
        api = " ".join(str(plan.get(key, "")) for key in ("method", "path"))
        steps.append(
            f"Step {row.get('step_index', 'verify')}: {row['text']} "
            f"API {api}. Bindings {json.dumps(plan.get('capture', {}), sort_keys=True)}."
        )
    return " ".join(steps)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    events = {}
    for row in load_jsonl(args.events):
        validate_query_only_event(row)
        event_id = str(row["event_id"])
        if event_id in events:
            raise ValueError(f"duplicate event: {event_id}")
        events[event_id] = row
    memories = {str(row["memory_id"]): row for row in load_jsonl(args.memories)}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(args.candidates):
        validate_label_free_candidate(row)
        grouped[str(row["event_id"])].append(row)
    if set(grouped) != set(events) or len(events) != 128:
        raise ValueError("expected complete 128-event query-only v7 view")
    if any(len(rows) != 2 for rows in grouped.values()):
        raise ValueError("expected exactly two complete candidates per event")

    snapshot = Path(snapshot_download(args.model, local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        snapshot, local_files_only=True
    ).to(args.device)
    model.eval()
    labels = {value.lower(): int(key) for key, value in model.config.id2label.items()}
    if "entailment" not in labels or "contradiction" not in labels:
        raise ValueError(f"NLI labels unavailable: {labels}")

    candidates = [row for event_id in sorted(grouped) for row in grouped[event_id]]
    scores = []
    for start in range(0, len(candidates), args.batch_size):
        batch = candidates[start : start + args.batch_size]
        premises = [serialize_path(row, memories) for row in batch]
        hypotheses = [
            "This workflow fulfills the request: " + events[str(row["event_id"])]["query"]
            for row in batch
        ]
        tokenized = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(args.device)
        with torch.inference_mode():
            logits = model(**tokenized).logits.float().cpu()
        for row, values in zip(batch, logits):
            score = (
                values[labels["entailment"]] - values[labels["contradiction"]]
            ).item()
            scores.append(
                {
                    "event_id": row["event_id"],
                    "candidate_edge_id": row["candidate_edge_id"],
                    "selected_memory_ids": sorted(row["selected_memory_ids"]),
                    "entailment_minus_contradiction": score,
                }
            )

    scored: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        scored[str(row["event_id"])].append(row)
    selections = []
    for event_id in sorted(scored):
        ranked = sorted(
            scored[event_id],
            key=lambda row: (
                row["entailment_minus_contradiction"],
                hashlib.sha256(
                    f"2027|{event_id}|{'|'.join(row['selected_memory_ids'])}".encode()
                ).hexdigest(),
            ),
            reverse=True,
        )
        selections.append(
            {
                "event_id": event_id,
                "selected_memory_ids": ranked[0]["selected_memory_ids"],
                "score_margin": ranked[0]["entailment_minus_contradiction"]
                - ranked[1]["entailment_minus_contradiction"],
            }
        )
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "candidate_scores.jsonl", scores)
    write_jsonl(args.output / "selections.jsonl", selections)
    model_files = {
        path.name: sha256(path)
        for path in snapshot.iterdir()
        if path.name in {"model.safetensors", "pytorch_model.bin", "tokenizer.json", "config.json"}
    }
    write_json(
        args.output / "manifest.json",
        {
            "status": "development_only_label_blind_selection",
            "claim_boundary": "v7 outcomes were already inspected before this model was tried",
            "model": args.model,
            "model_files_sha256": model_files,
            "device": args.device,
            "events_sha256": sha256(args.events),
            "candidates_sha256": sha256(args.candidates),
            "memories_sha256": sha256(args.memories),
            "events": len(events),
            "candidate_rows": len(scores),
            "outcomes_loaded": False,
        },
    )
    write_checksums(
        args.output,
        ("candidate_scores.jsonl", "manifest.json", "selections.jsonl"),
    )
    print(f"wrote {len(selections)} NLI path selections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
