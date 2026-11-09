#!/usr/bin/env python3
"""Family-held-out development training on inspected v8 candidate pairs.

All labels in this script have been inspected. No score here is an
independent confirmation or a replacement for the frozen v8 comparison.
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


def _selected(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(map(str, row["selected_memory_ids"])))


def _validity_gate(path: Path, expected: bool, event_ids: set[str]) -> dict[str, tuple[str, ...]]:
    manifest = __import__("json").loads((path / "run_manifest.json").read_text())
    rows = load_jsonl(path / "execution_results.jsonl")
    if (
        manifest.get("rows") != 128
        or manifest.get("verified_successes") != (128 if expected else 0)
        or manifest.get("runner_errors") != 0
        or manifest.get("checker_errors") != 0
        or len(rows) != 128
        or {str(row["event_id"]) for row in rows} != event_ids
    ):
        raise ValueError(f"incomplete physical gate: {path}")
    selected = {}
    for row in rows:
        if (
            row.get("reset_verified") is not True
            or row.get("execution_error") is not None
            or row.get("checker_error") is not None
            or row.get("checker", {}).get("goal_satisfied") is not expected
        ):
            raise ValueError(f"unverified candidate: {row['event_id']}")
        selected[str(row["event_id"])] = _selected(row)
    return selected


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
        parser.error("refusing to overwrite an existing development run")
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
    oracle = _validity_gate(args.oracle_gate, True, set(events))
    negative = _validity_gate(args.negative_gate, False, set(events))
    bm25_rows = load_jsonl(args.bm25_selections)
    bm25 = {str(row["event_id"]): _selected(row) for row in bm25_rows}
    if len(bm25_rows) != 128 or len(bm25) != 128 or set(bm25) != set(events):
        raise ValueError("incomplete BM25 reference")
    candidate_rows = []
    positive_index: dict[str, int] = {}
    hypothesis = []
    premises = []
    for event_id in sorted(events):
        rows = grouped[event_id]
        if oracle[event_id] == negative[event_id]:
            raise ValueError(f"same candidate on both gates: {event_id}")
        paths = difference_focused_paths(rows[0], rows[1], memories)
        for row, path in zip(rows, paths[:2]):
            if _selected(row) == oracle[event_id]:
                positive_index[event_id] = len(candidate_rows)
            candidate_rows.append(row)
            premises.append(path)
            hypothesis.append("This workflow fulfills the request: " + str(events[event_id]["query"]))
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, local_files_only=True)
    tokenized = tokenizer(
        premises, hypothesis, padding="max_length", truncation=True,
        max_length=args.max_length, return_tensors="pt",
    )
    families = sorted({event_id.split(":")[1] for event_id in events})
    if len(families) != 16:
        raise ValueError("expected sixteen task families")
    fold_by_family = {family: index % 4 for index, family in enumerate(families)}
    predictions = []
    for fold in range(4):
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
        for event_id in sorted(events):
            first = 2 * sorted(events).index(event_id)
            pair = (first, first + 1)
            if fold_by_family[event_id.split(":")[1]] == fold:
                held_events.append((event_id, pair))
            else:
                positive = positive_index[event_id]
                negative_index = pair[1] if positive == pair[0] else pair[0]
                train_pairs.append((positive, negative_index))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        model.train()
        for epoch in range(args.epochs):
            random.Random(args.seed + fold * 100 + epoch).shuffle(train_pairs)
            for start in range(0, len(train_pairs), args.batch_size):
                batch_pairs = train_pairs[start : start + args.batch_size]
                indices = [index for pair in batch_pairs for index in pair]
                inputs = {name: tensor[indices].to(args.device) for name, tensor in tokenized.items()}
                logits = model(**inputs).logits.float()
                scores = logits[:, labels["entailment"]] - logits[:, labels["contradiction"]]
                margins = scores.reshape(-1, 2)
                loss = F.softplus(margins[:, 1] - margins[:, 0]).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.inference_mode():
            for event_id, pair in held_events:
                inputs = {name: tensor[list(pair)].to(args.device) for name, tensor in tokenized.items()}
                logits = model(**inputs).logits.float()
                scores = (logits[:, labels["entailment"]] - logits[:, labels["contradiction"]]).cpu().tolist()
                ranked = max(range(2), key=lambda index: (scores[index], _selected(candidate_rows[pair[index]])))
                choice = _selected(candidate_rows[pair[ranked]])
                predictions.append({
                    "event_id": event_id,
                    "family": event_id.split(":")[1],
                    "fold": fold,
                    "selected_memory_ids": list(choice),
                    "verified_success": int(choice == oracle[event_id]),
                    "bm25_verified_success": int(bm25[event_id] == oracle[event_id]),
                    "score_margin": abs(scores[0] - scores[1]),
                })
        print(f"completed held-out fold {fold}", flush=True)
        del model, optimizer
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if len(predictions) != 128 or {row["event_id"] for row in predictions} != set(events):
        raise ValueError("incomplete family-held-out predictions")
    by_family = Counter()
    for row in predictions:
        by_family[row["family"]] += row["verified_success"]
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "heldout_predictions.jsonl", predictions)
    write_json(args.output / "result.json", {
        "status": "inspected_v8_family_heldout_development_only",
        "claim_boundary": "The v8 dataset and outcomes were inspected before this architecture was designed; cross-validation is a development diagnostic, not a fresh test.",
        "successes": sum(row["verified_success"] for row in predictions),
        "bm25_successes": sum(row["bm25_verified_success"] for row in predictions),
        "by_family_successes": dict(sorted(by_family.items())),
        "folds": 4,
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
    print(f"inspected-v8 held-out development score: {sum(row['verified_success'] for row in predictions)}/128")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
