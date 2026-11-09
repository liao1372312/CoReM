#!/usr/bin/env python3
"""Train a prospective v9 neural tie-breaker on inspected development data.

No v9 outcomes are read. This checkpoint is a method candidate, not evidence
of independent success. Its policy must be frozen before any v9 scoring.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, __version__ as transformers_version

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import load_jsonl, sha256_file, write_checksums, write_json  # noqa: E402
from experiments.develop_v8_contrastive_path_input import difference_focused_paths  # noqa: E402
from experiments.develop_v8_delta_cross_service import _training_pairs_from_v7  # noqa: E402
from experiments.develop_v8_delta_pairwise_model import _selected, _validity_gate  # noqa: E402
from experiments.develop_v8_delta_role_augmented import _dataset, _role_oracle  # noqa: E402


def _checked_negative_gate(path: Path, event_ids: set[str]) -> dict[str, tuple[str, ...]]:
    manifest = json.loads((path / "run_manifest.json").read_text())
    rows = load_jsonl(path / "execution_results.jsonl")
    if (
        len(rows) != 24 or manifest.get("rows") != 24
        or manifest.get("verified_successes") != 0
        or manifest.get("runner_errors") != 0
        or manifest.get("checker_errors") != 0
        or {str(row["event_id"]) for row in rows} != event_ids
        or any(row.get("checker", {}).get("goal_satisfied") is not False for row in rows)
    ):
        raise ValueError("incomplete role-swap negative physical gate")
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
    parser.add_argument("--role-dataset", type=Path, required=True)
    parser.add_argument("--role-oracle-gate", type=Path, required=True)
    parser.add_argument("--role-negative-gate", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--role-repeat", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (
        args.output.exists() or args.epochs < 1 or args.role_repeat < 1
        or args.batch_size < 1 or args.max_length < 32 or args.learning_rate <= 0
    ):
        parser.error("output must be new and training parameters valid")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    v7_events, v7_groups, v7_memories, v7_oracle = _training_pairs_from_v7(
        args.v7_queries, args.v7_candidates, args.v7_memories, args.v7_executions
    )
    v8_events, v8_memories, v8_groups = _dataset(args.v8_dataset)
    role_events, role_memories, role_groups = _dataset(args.role_dataset)
    if len(v7_events) != 128 or len(v8_events) != 128 or len(role_events) != 24:
        raise ValueError("expected 128 v7, 128 v8, and 24 role-swap development events")
    v8_oracle = _validity_gate(args.v8_oracle_gate, True, set(v8_events))
    v8_negative = _validity_gate(args.v8_negative_gate, False, set(v8_events))
    role_oracle = _role_oracle(args.role_oracle_gate, set(role_events))
    role_negative = _checked_negative_gate(args.role_negative_gate, set(role_events))
    if any(v8_oracle[key] == v8_negative[key] for key in v8_events) or any(
        role_oracle[key] == role_negative[key] for key in role_events
    ):
        raise ValueError("positive and negative development candidates overlap")
    events = {**v7_events, **v8_events, **role_events}
    groups = {**v7_groups, **v8_groups, **role_groups}
    memories = {**v7_memories, **v8_memories, **role_memories}
    oracle = {**v7_oracle, **v8_oracle, **role_oracle}
    if len(events) != 280 or len(groups) != 280 or len(oracle) != 280:
        raise ValueError("development event IDs overlap")
    premises, hypotheses = [], []
    pairs = []
    by_source: dict[str, int] = defaultdict(int)
    for event_id in sorted(events):
        rows = groups[event_id]
        left, right, _changed = difference_focused_paths(rows[0], rows[1], memories)
        first_index = len(premises)
        premises.extend((left, right))
        hypothesis = "This workflow fulfills the request: " + str(events[event_id]["query"])
        hypotheses.extend((hypothesis, hypothesis))
        positive = first_index if _selected(rows[0]) == oracle[event_id] else first_index + 1
        negative = first_index + 1 if positive == first_index else first_index
        if _selected(rows[positive - first_index]) != oracle[event_id]:
            raise ValueError(f"missing development positive: {event_id}")
        source = "v7" if event_id in v7_events else "v8" if event_id in v8_events else "role_swap"
        repeats = args.role_repeat if source == "role_swap" else 1
        pairs.extend([(positive, negative)] * repeats)
        by_source[source] += 1
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, local_files_only=True)
    tokenized = tokenizer(premises, hypotheses, padding="max_length", truncation=True,
                          max_length=args.max_length, return_tensors="pt")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model_dir, local_files_only=True).to(args.device)
    labels = {value.lower(): int(index) for index, value in model.config.id2label.items()}
    if "entailment" not in labels or "contradiction" not in labels:
        raise ValueError("base checkpoint lacks entailment/contradiction labels")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    epoch_losses = []
    model.train()
    for epoch in range(args.epochs):
        shuffled = list(pairs)
        random.Random(args.seed + epoch).shuffle(shuffled)
        total_loss = 0.0
        batches = 0
        for start in range(0, len(shuffled), args.batch_size):
            batch_pairs = shuffled[start:start + args.batch_size]
            indices = [index for pair in batch_pairs for index in pair]
            inputs = {key: tensor[indices].to(args.device) for key, tensor in tokenized.items()}
            logits = model(**inputs).logits.float()
            scores = logits[:, labels["entailment"]] - logits[:, labels["contradiction"]]
            margins = scores.reshape(-1, 2)
            loss = F.softplus(margins[:, 1] - margins[:, 0]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        epoch_losses.append(total_loss / batches)
        print(f"completed inspected-data training epoch {epoch + 1}/{args.epochs}", flush=True)
    args.output.mkdir(parents=True)
    model_dir = args.output / "full_model"
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    model_files = sorted(path for path in model_dir.iterdir() if path.is_file())
    write_json(args.output / "result.json", {
        "status": "trained_on_inspected_development_data_for_prospective_v9_only",
        "claim_boundary": "No v9 labels or outcomes were used; this checkpoint alone does not establish a fresh method advantage.",
        "source_event_counts": dict(sorted(by_source.items())),
        "effective_pair_count_per_epoch": len(pairs),
        "epochs": args.epochs, "role_repeat": args.role_repeat,
        "batch_size": args.batch_size, "max_length": args.max_length,
        "learning_rate": args.learning_rate, "seed": args.seed,
        "torch_version": torch.__version__, "transformers_version": transformers_version,
        "device": str(args.device),
        "epoch_mean_losses": epoch_losses,
        "base_model_sha256": sha256_file(args.base_model_dir / "model.safetensors"),
        "development_input_sha256": {
            "v7_queries": sha256_file(args.v7_queries),
            "v7_candidates": sha256_file(args.v7_candidates),
            "v7_memories": sha256_file(args.v7_memories),
            "v8_queries": sha256_file(args.v8_dataset / "query_only_events.jsonl"),
            "v8_candidates": sha256_file(args.v8_dataset / "candidate_selections.jsonl"),
            "v8_memories": sha256_file(args.v8_dataset / "memories.jsonl"),
            "role_queries": sha256_file(args.role_dataset / "query_only_events.jsonl"),
            "role_candidates": sha256_file(args.role_dataset / "candidate_selections.jsonl"),
            "role_memories": sha256_file(args.role_dataset / "memories.jsonl"),
        },
        "v7_executions_sha256": sha256_file(args.v7_executions),
        "v8_oracle_gate_sha256": sha256_file(args.v8_oracle_gate / "execution_results.jsonl"),
        "v8_negative_gate_sha256": sha256_file(args.v8_negative_gate / "execution_results.jsonl"),
        "role_oracle_gate_sha256": sha256_file(args.role_oracle_gate / "execution_results.jsonl"),
        "role_negative_gate_sha256": sha256_file(args.role_negative_gate / "execution_results.jsonl"),
        "model_files_sha256": {path.name: sha256_file(path) for path in model_files},
        "script_sha256": sha256_file(Path(__file__)),
    })
    write_checksums(args.output, ("result.json", *(f"full_model/{path.name}" for path in model_files)))
    print(f"saved prospective neural tie-breaker: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
