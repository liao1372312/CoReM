#!/usr/bin/env python3
"""Exploratory family-held-out pairwise path reranking on inspected v7 data."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_jsonl, write_checksums, write_json, write_jsonl  # noqa: E402
from experiments.analyze_online_group_controls import verify_checksums  # noqa: E402
from experiments.generate_public_multiservice_hgt_ensemble_selections import (  # noqa: E402
    validate_label_free_candidate,
    validate_query_only_event,
)
from experiments.score_v7_path_nli_development import serialize_path  # noqa: E402
from experiments.path_serialization import serialize_path_body_v1  # noqa: E402
from experiments.summarize_independent_v7_confirmation import sha256  # noqa: E402


def key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(map(str, row["selected_memory_ids"])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--executions", type=Path, required=True)
    parser.add_argument("--bm25-selections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--serialization",
        choices=("legacy", "body_v1"),
        default="legacy",
    )
    parser.add_argument(
        "--train-full",
        action="store_true",
        help="Also fit all 128 inspected development pairs and save a checkpoint.",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.learning_rate <= 0
        or args.max_length < 32
    ):
        parser.error("epochs, batch size, learning rate, and max length must be valid")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    event_rows = load_jsonl(args.events)
    for row in event_rows:
        validate_query_only_event(row)
    events = {str(row["event_id"]): row for row in event_rows}
    memories = {str(row["memory_id"]): row for row in load_jsonl(args.memories)}
    candidate_rows = load_jsonl(args.candidates)
    for row in candidate_rows:
        validate_label_free_candidate(row)
    candidate_by_event: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(candidate_rows):
        candidate_by_event[str(row["event_id"])].append(index)
    if len(events) != 128 or set(candidate_by_event) != set(events):
        raise ValueError("expected complete 128-event v7 candidate set")
    if any(len(indices) != 2 for indices in candidate_by_event.values()):
        raise ValueError("expected two candidates for each event")

    verify_checksums(args.executions)
    outcomes: dict[tuple[str, tuple[str, ...]], int] = {}
    for row in load_jsonl(args.executions / "execution_results.jsonl"):
        if (
            row["reset_verified"] is not True
            or row["checker_independent"] is not True
            or row["execution_error"] is not None
            or row["checker_error"] is not None
        ):
            raise ValueError(f"unverified execution: {row['event_id']}")
        lookup = (str(row["event_id"]), key(row))
        if lookup in outcomes:
            raise ValueError(f"duplicate checked candidate: {lookup}")
        outcomes[lookup] = int(row["checker"]["goal_satisfied"])
    if len(outcomes) != 256:
        raise ValueError("expected 256 physically checked candidates")

    for event_id, indices in candidate_by_event.items():
        if sorted(outcomes[event_id, key(candidate_rows[i])] for i in indices) != [0, 1]:
            raise ValueError(f"invalid oracle/negative pair: {event_id}")
    bm25_rows = load_jsonl(args.bm25_selections)
    bm25 = {str(row["event_id"]): key(row) for row in bm25_rows}
    if len(bm25) != 128 or set(bm25) != set(events):
        raise ValueError("incomplete BM25 baseline selections")

    snapshot = Path(snapshot_download(args.model, local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    serializer = serialize_path if args.serialization == "legacy" else serialize_path_body_v1
    premises = [serializer(row, memories) for row in candidate_rows]
    hypotheses = [
        "This workflow fulfills the request: " + events[str(row["event_id"])]["query"]
        for row in candidate_rows
    ]
    tokenized = tokenizer(
        premises,
        hypotheses,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    families = sorted({event_id.split(":")[1] for event_id in events})
    if len(families) != 16:
        raise ValueError("expected 16 operation families")
    folds = {
        family: index % 4 for index, family in enumerate(families)
    }
    result_rows = []
    for fold in range(4):
        torch.manual_seed(args.seed + fold)
        random.seed(args.seed + fold)
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed + fold)
        model = AutoModelForSequenceClassification.from_pretrained(
            snapshot, local_files_only=True
        ).to(args.device)
        labels = {value.lower(): int(index) for index, value in model.config.id2label.items()}
        if "entailment" not in labels or "contradiction" not in labels:
            raise ValueError(f"NLI labels unavailable: {labels}")
        train_pairs = []
        held_indices = []
        for event_id, indices in candidate_by_event.items():
            if folds[event_id.split(":")[1]] == fold:
                held_indices.extend(indices)
            else:
                ordered = sorted(
                    indices,
                    key=lambda index: outcomes[event_id, key(candidate_rows[index])],
                    reverse=True,
                )
                train_pairs.append(tuple(ordered))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        model.train()
        for epoch in range(args.epochs):
            rng = random.Random(args.seed + fold * 100 + epoch)
            rng.shuffle(train_pairs)
            for start in range(0, len(train_pairs), args.batch_size):
                batch_pairs = train_pairs[start : start + args.batch_size]
                indices = [index for pair in batch_pairs for index in pair]
                inputs = {
                    name: values[indices].to(args.device)
                    for name, values in tokenized.items()
                }
                logits = model(**inputs).logits.float()
                score = logits[:, labels["entailment"]] - logits[:, labels["contradiction"]]
                margins = score.reshape(-1, 2)
                loss = F.softplus(margins[:, 1] - margins[:, 0]).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        model.eval()
        scored = {}
        with torch.inference_mode():
            for start in range(0, len(held_indices), args.batch_size * 2):
                indices = held_indices[start : start + args.batch_size * 2]
                inputs = {
                    name: values[indices].to(args.device)
                    for name, values in tokenized.items()
                }
                logits = model(**inputs).logits.float()
                values = logits[:, labels["entailment"]] - logits[:, labels["contradiction"]]
                scored.update(zip(indices, values.cpu().tolist()))
        for event_id, indices in sorted(candidate_by_event.items()):
            if folds[event_id.split(":")[1]] != fold:
                continue
            ranked = sorted(
                indices,
                key=lambda index: (scored[index], key(candidate_rows[index])),
                reverse=True,
            )
            selected = key(candidate_rows[ranked[0]])
            result_rows.append(
                {
                    "event_id": event_id,
                    "family": event_id.split(":")[1],
                    "fold": fold,
                    "selected_memory_ids": list(selected),
                    "score_margin": scored[ranked[0]] - scored[ranked[1]],
                    "verified_success": outcomes[event_id, selected],
                    "bm25_verified_success": outcomes[event_id, bm25[event_id]],
                }
            )
        print(f"finished family-held-out fold {fold}", flush=True)
        del model, optimizer
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if len(result_rows) != 128:
        raise ValueError("incomplete cross-family predictions")
    successes = sum(row["verified_success"] for row in result_rows)
    bm25_successes = sum(row["bm25_verified_success"] for row in result_rows)
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "heldout_predictions.jsonl", result_rows)
    full_model_sha256 = None
    if args.train_full:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed)
        full_model = AutoModelForSequenceClassification.from_pretrained(
            snapshot, local_files_only=True
        ).to(args.device)
        full_labels = {
            value.lower(): int(index)
            for index, value in full_model.config.id2label.items()
        }
        all_pairs = []
        for event_id, indices in candidate_by_event.items():
            ordered = sorted(
                indices,
                key=lambda index: outcomes[event_id, key(candidate_rows[index])],
                reverse=True,
            )
            all_pairs.append(tuple(ordered))
        optimizer = torch.optim.AdamW(full_model.parameters(), lr=args.learning_rate)
        full_model.train()
        for epoch in range(args.epochs):
            rng = random.Random(args.seed + epoch)
            rng.shuffle(all_pairs)
            for start in range(0, len(all_pairs), args.batch_size):
                batch_pairs = all_pairs[start : start + args.batch_size]
                indices = [index for pair in batch_pairs for index in pair]
                inputs = {
                    name: values[indices].to(args.device)
                    for name, values in tokenized.items()
                }
                logits = full_model(**inputs).logits.float()
                score = (
                    logits[:, full_labels["entailment"]]
                    - logits[:, full_labels["contradiction"]]
                )
                margins = score.reshape(-1, 2)
                loss = F.softplus(margins[:, 1] - margins[:, 0]).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        checkpoint_dir = args.output / "full_model"
        checkpoint_dir.mkdir()
        full_model.save_pretrained(checkpoint_dir, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint_dir)
        full_model_sha256 = sha256(checkpoint_dir / "model.safetensors")
        print(f"saved development checkpoint SHA-256 {full_model_sha256}", flush=True)
    write_json(
        args.output / "result.json",
        {
            "status": "exploratory_old_v7_development_only",
            "claim_boundary": (
                "All v7 outcomes were inspected before this model was designed; "
                "family-held-out CV is a development diagnostic, not independent confirmation."
            ),
            "events": 128,
            "families": 16,
            "folds": 4,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "serialization": args.serialization,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "model": args.model,
            "model_config_sha256": sha256(snapshot / "config.json"),
            "full_development_checkpoint_sha256": full_model_sha256,
            "source_sha256": {
                "events": sha256(args.events),
                "candidates": sha256(args.candidates),
                "memories": sha256(args.memories),
                "executions": sha256(args.executions / "execution_results.jsonl"),
                "bm25_selections": sha256(args.bm25_selections),
            },
            "successes": {
                "family_held_out_path_reranker": successes,
                "candidate_bm25_sum": bm25_successes,
            },
            "discordant_wins": sum(
                row["verified_success"] > row["bm25_verified_success"]
                for row in result_rows
            ),
            "discordant_losses": sum(
                row["verified_success"] < row["bm25_verified_success"]
                for row in result_rows
            ),
        },
    )
    checked_files = ["heldout_predictions.jsonl", "result.json"]
    if args.train_full:
        checked_files.extend(
            (
                "full_model/model.safetensors",
                "full_model/config.json",
                "full_model/tokenizer.json",
            )
        )
    write_checksums(args.output, tuple(checked_files))
    print(f"cross-family reranker {successes}/128; BM25-sum {bm25_successes}/128")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
