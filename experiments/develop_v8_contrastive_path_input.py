#!/usr/bin/env python3
"""Diagnose a difference-focused path input on already inspected v8 tasks.

This script is development-only. It cannot establish an independent method
advantage because the v8 outcome and family breakdown have been inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import load_jsonl, sha256_file, write_checksums, write_json, write_jsonl  # noqa: E402
from corem.v8_candidates import validate_candidate  # noqa: E402


def _ordered_plans(candidate: dict[str, Any], memories: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records = [memories[str(memory_id)] for memory_id in candidate["selected_memory_ids"]]
    steps = [row for row in records if row.get("role") == "workflow_step"]
    return sorted(steps, key=lambda row: int(row["step_index"]))


def difference_focused_paths(
    first: dict[str, Any], second: dict[str, Any], memories: dict[str, dict[str, Any]]
) -> tuple[str, str, int]:
    """Serialize only changed operations, preserving their original step indices."""
    left = _ordered_plans(first, memories)
    right = _ordered_plans(second, memories)
    if len(left) != len(right) or not left:
        raise ValueError("candidate pair has incompatible workflow lengths")
    changed = [index for index, (a, b) in enumerate(zip(left, right)) if a["plan"] != b["plan"]]
    if not changed:
        raise ValueError("candidate pair has no executable difference")

    def render(rows: list[dict[str, Any]]) -> str:
        operations = []
        for index in changed:
            row = rows[index]
            plan = row["plan"]
            body = json.dumps(plan.get("json", {}), sort_keys=True, separators=(",", ":"))
            capture = json.dumps(plan.get("capture", {}), sort_keys=True, separators=(",", ":"))
            operations.append(
                f"Call {row['step_index']}: {plan['method']} {plan['path']} body {body} capture {capture}."
            )
        return "API sequence: " + " ".join(operations) + " Evidence: independently inspect final service state."

    return render(left), render(right), len(changed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--oracle-selections", type=Path, required=True)
    parser.add_argument("--reference-selections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a prior development result")
    if args.batch_size < 1 or args.max_length < 32:
        parser.error("batch size and sequence length must be positive")
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
        raise ValueError("expected 128 two-candidate events")
    oracle_rows = load_jsonl(args.oracle_selections)
    oracle = {
        str(row["event_id"]): tuple(sorted(map(str, row["selected_memory_ids"])))
        for row in oracle_rows
    }
    reference_rows = load_jsonl(args.reference_selections)
    reference = {
        str(row["event_id"]): tuple(sorted(map(str, row["selected_memory_ids"])))
        for row in reference_rows
    }
    if (
        len(oracle) != 128 or len(oracle_rows) != 128
        or len(reference) != 128 or len(reference_rows) != 128
        or set(oracle) != set(events) or set(reference) != set(events)
    ):
        raise ValueError("incomplete oracle or reference selection map")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True).to(args.device)
    model.eval()
    labels = {value.lower(): int(key) for key, value in model.config.id2label.items()}
    if "entailment" not in labels or "contradiction" not in labels:
        raise ValueError("checkpoint has no NLI scoring labels")
    inputs = []
    for event_id in sorted(events):
        rows = grouped[event_id]
        paths = difference_focused_paths(rows[0], rows[1], memories)
        hypothesis = "This workflow fulfills the request: " + str(events[event_id]["query"])
        for row, path in zip(rows, paths[:2]):
            inputs.append((event_id, row, path, hypothesis, paths[2]))
    scores = []
    for start in range(0, len(inputs), args.batch_size):
        batch = inputs[start : start + args.batch_size]
        encoded = tokenizer(
            [item[2] for item in batch],
            [item[3] for item in batch],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(args.device)
        with torch.inference_mode():
            logits = model(**encoded).logits.float().cpu()
        for (event_id, row, path, _hypothesis, changed), values in zip(batch, logits):
            scores.append({
                "event_id": event_id,
                "candidate_edge_id": row["candidate_edge_id"],
                "selected_memory_ids": sorted(map(str, row["selected_memory_ids"])),
                "score": float(values[labels["entailment"]] - values[labels["contradiction"]]),
                "changed_operations": changed,
                "serialized_path": path,
            })
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        by_event[str(row["event_id"])].append(row)
    selected_rows = []
    success_by_family = Counter()
    reference_by_family = Counter()
    for event_id in sorted(events):
        def rank_key(row: dict[str, Any]) -> tuple[float, str]:
            digest = hashlib.sha256(
                f"{args.seed}|{event_id}|{'|'.join(row['selected_memory_ids'])}".encode()
            ).hexdigest()
            return row["score"], digest

        ranked = sorted(by_event[event_id], key=rank_key, reverse=True)
        chosen = tuple(ranked[0]["selected_memory_ids"])
        family = str(events[event_id]["task_key"]) if "task_key" in events[event_id] else event_id.split(":")[1]
        success_by_family[family] += chosen == oracle[event_id]
        reference_by_family[family] += reference[event_id] == oracle[event_id]
        selected_rows.append({
            "event_id": event_id,
            "selected_memory_ids": list(chosen),
            "score_margin": ranked[0]["score"] - ranked[1]["score"],
            "changed_operations": ranked[0]["changed_operations"],
        })
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "candidate_scores.jsonl", scores)
    write_jsonl(args.output / "selections.jsonl", selected_rows)
    write_json(args.output / "result.json", {
        "status": "inspected_v8_development_only",
        "claim_boundary": "Outcome labels and family effects were inspected before this input was designed; no independent transfer claim is permitted.",
        "events": 128,
        "successes": sum(success_by_family.values()),
        "reference_successes": sum(reference_by_family.values()),
        "by_family_successes": dict(sorted(success_by_family.items())),
        "reference_by_family_successes": dict(sorted(reference_by_family.items())),
        "model_weights_sha256": sha256_file(args.model_dir / "model.safetensors"),
        "dataset_query_sha256": sha256_file(args.dataset / "query_only_events.jsonl"),
        "oracle_selections_sha256": sha256_file(args.oracle_selections),
        "reference_selections_sha256": sha256_file(args.reference_selections),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "seed": args.seed,
    })
    write_checksums(args.output, ("candidate_scores.jsonl", "selections.jsonl", "result.json"))
    print(f"development-only difference-focused path score: {sum(success_by_family.values())}/128")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
