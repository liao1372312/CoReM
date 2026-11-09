#!/usr/bin/env python3
"""Development-only provider holdout with inspected v7+v8 training data.

The v7 checkpoint already saw v7 labels. This experiment probes whether
additional diverse training examples help transfer to a held-out v8 service;
it does not supply a fresh test of the method.
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
from experiments.develop_v8_contrastive_path_input import difference_focused_paths  # noqa: E402
from experiments.develop_v8_delta_pairwise_model import _selected, _validity_gate  # noqa: E402


def _training_pairs_from_v7(
    query_path: Path, candidate_path: Path, memory_path: Path, execution_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], dict[str, tuple[str, ...]]]:
    events = {str(row["event_id"]): row for row in load_jsonl(query_path)}
    memories = {str(row["memory_id"]): row for row in load_jsonl(memory_path)}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(candidate_path):
        grouped[str(row["event_id"])].append(row)
    if len(events) != 128 or set(grouped) != set(events) or any(len(rows) != 2 for rows in grouped.values()):
        raise ValueError("v7 development candidate set is incomplete")
    executions = load_jsonl(execution_path)
    positives: dict[str, tuple[str, ...]] = {}
    observed: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for row in executions:
        if (
            row.get("reset_verified") is not True
            or row.get("execution_error") is not None
            or row.get("checker_error") is not None
        ):
            raise ValueError("v7 development execution has an error")
        event_id = str(row["event_id"])
        selected = _selected(row)
        if selected in observed[event_id]:
            raise ValueError("v7 candidate was checked twice")
        observed[event_id].add(selected)
        if row["checker"]["goal_satisfied"]:
            if event_id in positives:
                raise ValueError("v7 event has two oracle candidates")
            positives[event_id] = selected
    if len(executions) != 256 or set(observed) != set(events) or set(positives) != set(events):
        raise ValueError("v7 development physical outcomes are incomplete")
    for event_id, rows in grouped.items():
        if observed[event_id] != {_selected(row) for row in rows}:
            raise ValueError("v7 physical outcomes do not match candidate pool")
    return events, grouped, memories, positives


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-queries", type=Path, required=True)
    parser.add_argument("--v7-candidates", type=Path, required=True)
    parser.add_argument("--v7-memories", type=Path, required=True)
    parser.add_argument("--v7-executions", type=Path, required=True)
    parser.add_argument("--v8-dataset", type=Path, required=True)
    parser.add_argument("--v8-oracle-gate", type=Path, required=True)
    parser.add_argument("--v8-negative-gate", type=Path, required=True)
    parser.add_argument("--v8-bm25-selections", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
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
    v7_events, v7_grouped, v7_memories, v7_oracle = _training_pairs_from_v7(
        args.v7_queries, args.v7_candidates, args.v7_memories, args.v7_executions
    )
    v8_events = {str(row["event_id"]): row for row in load_jsonl(args.v8_dataset / "query_only_events.jsonl")}
    v8_memories = {str(row["memory_id"]): row for row in load_jsonl(args.v8_dataset / "memories.jsonl")}
    v8_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(args.v8_dataset / "candidate_selections.jsonl"):
        v8_grouped[str(row["event_id"])].append(row)
    if len(v8_events) != 128 or set(v8_grouped) != set(v8_events) or any(len(rows) != 2 for rows in v8_grouped.values()):
        raise ValueError("v8 development candidate set is incomplete")
    v8_oracle = _validity_gate(args.v8_oracle_gate, True, set(v8_events))
    v8_negative = _validity_gate(args.v8_negative_gate, False, set(v8_events))
    v8_bm25_rows = load_jsonl(args.v8_bm25_selections)
    v8_bm25 = {str(row["event_id"]): _selected(row) for row in v8_bm25_rows}
    if len(v8_bm25_rows) != 128 or len(v8_bm25) != 128 or set(v8_bm25) != set(v8_events):
        raise ValueError("incomplete v8 BM25 reference")
    all_events = {**v7_events, **v8_events}
    all_groups = {**v7_grouped, **v8_grouped}
    all_memories = {**v7_memories, **v8_memories}
    positives = {**v7_oracle, **v8_oracle}
    if len(all_events) != 256 or len(all_groups) != 256 or len(all_memories) != len(v7_memories) + len(v8_memories):
        raise ValueError("v7/v8 development sets overlap")
    if any(v8_oracle[event_id] == v8_negative[event_id] for event_id in v8_events):
        raise ValueError("v8 positive and negative candidates overlap")
    event_ids = sorted(all_events)
    candidate_rows = []
    positive_index = {}
    premises = []
    hypotheses = []
    for event_id in event_ids:
        rows = all_groups[event_id]
        paths = difference_focused_paths(rows[0], rows[1], all_memories)
        for row, path in zip(rows, paths[:2]):
            if _selected(row) == positives[event_id]:
                positive_index[event_id] = len(candidate_rows)
            candidate_rows.append(row)
            premises.append(path)
            hypotheses.append("This workflow fulfills the request: " + str(all_events[event_id]["query"]))
    if len(positive_index) != 256:
        raise ValueError("not every development event has one positive candidate")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, local_files_only=True)
    tokenized = tokenizer(
        premises, hypotheses, padding="max_length", truncation=True,
        max_length=args.max_length, return_tensors="pt",
    )
    predictions = []
    for fold, held_provider in enumerate(("openbao", "qdrant")):
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
            if event_id in v8_events and v8_events[event_id]["provider_id"] == held_provider:
                held_events.append((event_id, pair))
            else:
                positive = positive_index[event_id]
                other = pair[1] if positive == pair[0] else pair[0]
                train_pairs.append((positive, other))
        if len(train_pairs) != 192 or len(held_events) != 64:
            raise ValueError("v7+v8 provider holdout is unbalanced")
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
                    "verified_success": int(choice == v8_oracle[event_id]),
                    "bm25_verified_success": int(v8_bm25[event_id] == v8_oracle[event_id]),
                    "score_margin": abs(scores[0] - scores[1]),
                })
        print(f"completed v7+v8 training, held-out provider {held_provider}", flush=True)
        del model, optimizer
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    by_provider = Counter()
    for row in predictions:
        by_provider[row["held_provider"]] += row["verified_success"]
    if len(predictions) != 128 or {row["event_id"] for row in predictions} != set(v8_events):
        raise ValueError("incomplete held-out provider predictions")
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "heldout_predictions.jsonl", predictions)
    write_json(args.output / "result.json", {
        "status": "inspected_v7_v8_provider_holdout_development_only",
        "claim_boundary": "The v7/v8 labels and task templates were inspected, and the base checkpoint already trained on v7. This is only a transfer diagnostic.",
        "successes": sum(row["verified_success"] for row in predictions),
        "bm25_successes": sum(row["bm25_verified_success"] for row in predictions),
        "by_held_provider_successes": dict(sorted(by_provider.items())),
        "train_pairs_per_fold": 192,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "base_model_sha256": sha256_file(args.base_model_dir / "model.safetensors"),
        "v7_executions_sha256": sha256_file(args.v7_executions),
        "v8_oracle_gate_sha256": sha256_file(args.v8_oracle_gate / "execution_results.jsonl"),
        "v8_negative_gate_sha256": sha256_file(args.v8_negative_gate / "execution_results.jsonl"),
        "v8_bm25_selection_sha256": sha256_file(args.v8_bm25_selections),
        "script_sha256": sha256_file(Path(__file__)),
    })
    write_checksums(args.output, ("heldout_predictions.jsonl", "result.json"))
    print(f"inspected-v7+v8 provider-held-out development score: {sum(row['verified_success'] for row in predictions)}/128")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
