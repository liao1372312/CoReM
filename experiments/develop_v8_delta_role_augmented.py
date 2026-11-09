#!/usr/bin/env python3
"""Development-only provider holdout with physically checked role-swap examples.

All v7/v8 and role-swap labels were inspected before this experiment. A good
score here would not be independent validation on new providers.
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
from experiments.develop_v8_delta_cross_service import _training_pairs_from_v7  # noqa: E402
from experiments.develop_v8_delta_pairwise_model import _selected, _validity_gate  # noqa: E402


def _dataset(path: Path) -> tuple[dict, dict, dict]:
    events = {str(row["event_id"]): row for row in load_jsonl(path / "query_only_events.jsonl")}
    memories = {str(row["memory_id"]): row for row in load_jsonl(path / "memories.jsonl")}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(path / "candidate_selections.jsonl"):
        grouped[str(row["event_id"])].append(row)
    if set(grouped) != set(events) or any(len(rows) != 2 for rows in grouped.values()):
        raise ValueError(f"incomplete paired development dataset: {path}")
    return events, memories, grouped


def _role_oracle(gate: Path, events: set[str]) -> dict[str, tuple[str, ...]]:
    import json

    manifest = json.loads((gate / "run_manifest.json").read_text())
    rows = load_jsonl(gate / "execution_results.jsonl")
    if (manifest.get("rows"), manifest.get("verified_successes"), manifest.get("runner_errors"), manifest.get("checker_errors")) != (24, 24, 0, 0):
        raise ValueError("incomplete role-swap physical oracle gate")
    if len(rows) != 24 or {str(row["event_id"]) for row in rows} != events:
        raise ValueError("role-swap gate event IDs do not match")
    if any(row.get("checker", {}).get("goal_satisfied") is not True for row in rows):
        raise ValueError("role-swap gate contains failed oracle")
    return {str(row["event_id"]): _selected(row) for row in rows}


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
    parser.add_argument("--role-dataset", type=Path, required=True)
    parser.add_argument("--role-oracle-gate", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--role-repeat", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists() or args.epochs < 1 or args.role_repeat < 1 or args.batch_size < 1:
        parser.error("output must be new and training parameters positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    v7_events, v7_groups, v7_memories, v7_oracle = _training_pairs_from_v7(
        args.v7_queries, args.v7_candidates, args.v7_memories, args.v7_executions
    )
    v8_events, v8_memories, v8_groups = _dataset(args.v8_dataset)
    role_events, role_memories, role_groups = _dataset(args.role_dataset)
    if len(v8_events) != 128 or len(role_events) != 24:
        raise ValueError("expected 128 v8 and 24 role-swap development pairs")
    v8_oracle = _validity_gate(args.v8_oracle_gate, True, set(v8_events))
    v8_negative = _validity_gate(args.v8_negative_gate, False, set(v8_events))
    if any(v8_oracle[key] == v8_negative[key] for key in v8_events):
        raise ValueError("v8 physical gates agree on a candidate")
    role_oracle = _role_oracle(args.role_oracle_gate, set(role_events))
    bm25_rows = load_jsonl(args.v8_bm25_selections)
    bm25 = {str(row["event_id"]): _selected(row) for row in bm25_rows}
    if len(bm25_rows) != 128 or set(bm25) != set(v8_events):
        raise ValueError("incomplete information-matched BM25 reference")
    all_events = {**v7_events, **v8_events, **role_events}
    all_groups = {**v7_groups, **v8_groups, **role_groups}
    all_memories = {**v7_memories, **v8_memories, **role_memories}
    positives = {**v7_oracle, **v8_oracle, **role_oracle}
    if len(all_events) != 280 or len(all_groups) != 280 or len(positives) != 280:
        raise ValueError("development event IDs overlap")
    event_ids = sorted(all_events)
    candidate_rows = []
    premises, hypotheses = [], []
    positive_index = {}
    for event_id in event_ids:
        rows = all_groups[event_id]
        paths = difference_focused_paths(rows[0], rows[1], all_memories)
        for row, path in zip(rows, paths[:2]):
            if _selected(row) == positives[event_id]:
                positive_index[event_id] = len(candidate_rows)
            candidate_rows.append(row)
            premises.append(path)
            hypotheses.append("This workflow fulfills the request: " + str(all_events[event_id]["query"]))
    if len(positive_index) != 280:
        raise ValueError("development label does not match candidate pair")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, local_files_only=True)
    tokenized = tokenizer(premises, hypotheses, padding="max_length", truncation=True,
                          max_length=args.max_length, return_tensors="pt")
    predictions = []
    for fold, held_provider in enumerate(("openbao", "qdrant")):
        torch.manual_seed(args.seed + fold)
        random.seed(args.seed + fold)
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed + fold)
        model = AutoModelForSequenceClassification.from_pretrained(args.base_model_dir, local_files_only=True).to(args.device)
        labels = {value.lower(): int(index) for index, value in model.config.id2label.items()}
        if "entailment" not in labels or "contradiction" not in labels:
            raise ValueError("base checkpoint lacks NLI labels")
        train_pairs, held_events = [], []
        role_training_pairs = 0
        for event_index, event_id in enumerate(event_ids):
            pair = (2 * event_index, 2 * event_index + 1)
            provider = str(all_events[event_id]["provider_id"])
            if event_id in v8_events and provider == held_provider:
                held_events.append((event_id, pair))
                continue
            if event_id in role_events and provider == held_provider:
                continue
            positive = positive_index[event_id]
            negative = pair[1] if positive == pair[0] else pair[0]
            repeats = args.role_repeat if event_id in role_events else 1
            train_pairs.extend([(positive, negative)] * repeats)
            role_training_pairs += int(event_id in role_events)
        if len(held_events) != 64 or role_training_pairs not in {8, 16}:
            raise ValueError("provider-holdout split is incomplete")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        model.train()
        for epoch in range(args.epochs):
            random.Random(args.seed + fold * 100 + epoch).shuffle(train_pairs)
            for start in range(0, len(train_pairs), args.batch_size):
                pairs = train_pairs[start:start + args.batch_size]
                indices = [index for pair in pairs for index in pair]
                inputs = {key: values[indices].to(args.device) for key, values in tokenized.items()}
                logits = model(**inputs).logits.float()
                scores = logits[:, labels["entailment"]] - logits[:, labels["contradiction"]]
                loss = F.softplus(scores.reshape(-1, 2)[:, 1] - scores.reshape(-1, 2)[:, 0]).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.inference_mode():
            for event_id, pair in held_events:
                inputs = {key: values[list(pair)].to(args.device) for key, values in tokenized.items()}
                logits = model(**inputs).logits.float()
                scores = (logits[:, labels["entailment"]] - logits[:, labels["contradiction"]]).cpu().tolist()
                selected = max(range(2), key=lambda index: (scores[index], _selected(candidate_rows[pair[index]])))
                choice = _selected(candidate_rows[pair[selected]])
                predictions.append({"event_id": event_id, "held_provider": held_provider,
                                    "selected_memory_ids": list(choice),
                                    "verified_success": int(choice == v8_oracle[event_id]),
                                    "bm25_verified_success": int(bm25[event_id] == v8_oracle[event_id]),
                                    "score_margin": abs(scores[0] - scores[1])})
        print(f"completed development holdout {held_provider}: {role_training_pairs} role-swap pairs", flush=True)
        del model, optimizer
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if len(predictions) != 128 or {row["event_id"] for row in predictions} != set(v8_events):
        raise ValueError("missing provider-held-out development predictions")
    by_provider = Counter()
    for row in predictions:
        by_provider[row["held_provider"]] += row["verified_success"]
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "heldout_predictions.jsonl", predictions)
    write_json(args.output / "result.json", {
        "status": "inspected_v7_v8_role_augmented_provider_holdout_development_only",
        "claim_boundary": "All labels and task templates were inspected; these are development diagnostics, not untouched validation.",
        "successes": sum(row["verified_success"] for row in predictions),
        "bm25_successes": sum(row["bm25_verified_success"] for row in predictions),
        "by_held_provider_successes": dict(sorted(by_provider.items())),
        "epochs": args.epochs, "role_repeat": args.role_repeat,
        "batch_size": args.batch_size, "max_length": args.max_length,
        "learning_rate": args.learning_rate, "seed": args.seed,
        "base_model_sha256": sha256_file(args.base_model_dir / "model.safetensors"),
        "v7_executions_sha256": sha256_file(args.v7_executions),
        "v8_oracle_gate_sha256": sha256_file(args.v8_oracle_gate / "execution_results.jsonl"),
        "v8_negative_gate_sha256": sha256_file(args.v8_negative_gate / "execution_results.jsonl"),
        "role_oracle_gate_sha256": sha256_file(args.role_oracle_gate / "execution_results.jsonl"),
        "bm25_selection_sha256": sha256_file(args.v8_bm25_selections),
        "script_sha256": sha256_file(Path(__file__)),
    })
    write_checksums(args.output, ("heldout_predictions.jsonl", "result.json"))
    print(f"development-only provider-held-out score: {sum(row['verified_success'] for row in predictions)}/128")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
