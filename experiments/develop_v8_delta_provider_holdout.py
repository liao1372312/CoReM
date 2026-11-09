#!/usr/bin/env python3
"""Inspected-v8 development check that holds out an entire service.

This uses already-seen v8 labels and is not independent validation.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import load_jsonl, sha256_file, write_checksums, write_json, write_jsonl  # noqa: E402
from corem.v8_candidates import validate_candidate  # noqa: E402
from experiments.develop_v8_contrastive_path_input import difference_focused_paths  # noqa: E402
from experiments.develop_v8_delta_pairwise_model import _selected, _validity_gate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--oracle-gate", type=Path, required=True)
    parser.add_argument("--negative-gate", type=Path, required=True)
    parser.add_argument("--bm25-selections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an earlier development result")
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 32 or args.learning_rate <= 0:
        parser.error("invalid training hyperparameters")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    events = {str(row["event_id"]): row for row in load_jsonl(args.dataset / "query_only_events.jsonl")}
    memories = {str(row["memory_id"]): row for row in load_jsonl(args.dataset / "memories.jsonl")}
    edges = {str(row["edge_id"]): row for row in load_jsonl(args.dataset / "dependency_hyperedges.jsonl")}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(args.dataset / "candidate_selections.jsonl"):
        event_id = str(row["event_id"])
        validate_candidate(row, events[event_id], memories, edges)
        grouped[event_id].append(row)
    if len(events) != 128 or set(grouped) != set(events) or any(len(rows) != 2 for rows in grouped.values()):
        raise ValueError("expected 128 valid two-candidate events")
    providers = {str(row["provider_id"]) for row in events.values()}
    if providers != {"qdrant", "openbao"}:
        raise ValueError("unexpected provider split")
    oracle = _validity_gate(args.oracle_gate, True, set(events))
    negative = _validity_gate(args.negative_gate, False, set(events))
    bm25_rows = load_jsonl(args.bm25_selections)
    bm25 = {str(row["event_id"]): _selected(row) for row in bm25_rows}
    if len(bm25_rows) != 128 or len(bm25) != 128 or set(bm25) != set(events):
        raise ValueError("incomplete BM25 reference")
    event_ids = sorted(events)
    candidate_rows = []
    positive_index: dict[str, int] = {}
    premises = []
    hypotheses = []
    for event_id in event_ids:
        rows = grouped[event_id]
        if oracle[event_id] == negative[event_id]:
            raise ValueError(f"gate candidates are equal: {event_id}")
        paths = difference_focused_paths(rows[0], rows[1], memories)
        for row, path in zip(rows, paths[:2]):
            if _selected(row) == oracle[event_id]:
                positive_index[event_id] = len(candidate_rows)
            candidate_rows.append(row)
            premises.append(path)
            hypotheses.append("This workflow fulfills the request: " + str(events[event_id]["query"]))
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, local_files_only=True)
    tokenized = tokenizer(
        premises, hypotheses, padding="max_length", truncation=True,
        max_length=args.max_length, return_tensors="pt",
    )
    predictions = []
    for fold, held_provider in enumerate(sorted(providers)):
        torch.manual_seed(args.seed + fold)
        random.seed(args.seed + fold)
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed + fold)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model_dir, local_files_only=True
        ).to(args.device)
        labels = {value.lower(): int(index) for index, value in model.config.id2label.items()}
        if "entailment" not in labels or "contradiction" not in labels:
            raise ValueError("base model lacks entailment/contradiction labels")
        train_pairs = []
        held_events = []
        for event_index, event_id in enumerate(event_ids):
            pair = (event_index * 2, event_index * 2 + 1)
            if events[event_id]["provider_id"] == held_provider:
                held_events.append((event_id, pair))
            else:
                positive = positive_index[event_id]
                other = pair[1] if positive == pair[0] else pair[0]
                train_pairs.append((positive, other))
        if len(train_pairs) != 64 or len(held_events) != 64:
            raise ValueError("provider holdout is unbalanced")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        model.train()
        for epoch in range(args.epochs):
            random.Random(args.seed + fold * 100 + epoch).shuffle(train_pairs)
            for start in range(0, len(train_pairs), args.batch_size):
                batch_pairs = train_pairs[start : start + args.batch_size]
                indices = [index for pair in batch_pairs for index in pair]
                inputs = {name: values[indices].to(args.device) for name, values in tokenized.items()}
                logits = model(**inputs).logits.float()
                score = logits[:, labels["entailment"]] - logits[:, labels["contradiction"]]
                margins = score.reshape(-1, 2)
                loss = F.softplus(margins[:, 1] - margins[:, 0]).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.inference_mode():
            for event_id, pair in held_events:
                inputs = {name: values[list(pair)].to(args.device) for name, values in tokenized.items()}
                logits = model(**inputs).logits.float()
                scores = (logits[:, labels["entailment"]] - logits[:, labels["contradiction"]]).cpu().tolist()
                winner = max(range(2), key=lambda index: (scores[index], _selected(candidate_rows[pair[index]])))
                choice = _selected(candidate_rows[pair[winner]])
                predictions.append({
                    "event_id": event_id,
                    "held_provider": held_provider,
                    "family": event_id.split(":")[1],
                    "selected_memory_ids": list(choice),
                    "verified_success": int(choice == oracle[event_id]),
                    "bm25_verified_success": int(bm25[event_id] == oracle[event_id]),
                    "score_margin": abs(scores[0] - scores[1]),
                })
        print(f"completed held-out provider {held_provider}", flush=True)
        del model, optimizer
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    by_provider = Counter()
    for row in predictions:
        by_provider[row["held_provider"]] += row["verified_success"]
    if len(predictions) != 128 or {row["event_id"] for row in predictions} != set(events):
        raise ValueError("incomplete provider-held-out predictions")
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "heldout_predictions.jsonl", predictions)
    write_json(args.output / "result.json", {
        "status": "inspected_v8_provider_holdout_development_only",
        "claim_boundary": "The v8 labels were inspected before model design. Holding out one of only two providers is a stress test, not independent confirmation.",
        "successes": sum(row["verified_success"] for row in predictions),
        "bm25_successes": sum(row["bm25_verified_success"] for row in predictions),
        "by_held_provider_successes": dict(sorted(by_provider.items())),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "base_model_sha256": sha256_file(args.base_model_dir / "model.safetensors"),
        "oracle_gate_sha256": sha256_file(args.oracle_gate / "execution_results.jsonl"),
        "negative_gate_sha256": sha256_file(args.negative_gate / "execution_results.jsonl"),
        "bm25_selection_sha256": sha256_file(args.bm25_selections),
        "script_sha256": sha256_file(Path(__file__)),
    })
    write_checksums(args.output, ("heldout_predictions.jsonl", "result.json"))
    print(f"inspected-v8 provider-held-out development score: {sum(row['verified_success'] for row in predictions)}/128")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
