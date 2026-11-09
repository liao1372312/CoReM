#!/usr/bin/env python3
"""Aggregate-only first-turn token parity for TRAIN dependency/disjoint hints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "data" / "private_corem"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_appworld_v23_prompt_tokens import (  # noqa: E402
    _load_train_task_metadata,
    summarize_counts,
)
from experiments.run_appworld_local_agent_pilot import _initial_messages  # noqa: E402

TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--appworld-python", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--dependency-view", required=True, type=Path)
    parser.add_argument("--disjoint-view", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    views = {
        "dependency": args.dependency_view.resolve(),
        "disjoint": args.disjoint_view.resolve(),
    }
    if any(not path.is_relative_to(PRIVATE_ROOT.resolve()) for path in (root, output, *views.values())):
        parser.error("benchmark, views, and output must stay private")
    if output.exists() or output in views.values():
        parser.error("refusing to overwrite or alias a private artifact")
    payloads = {name: json.loads(path.read_text()) for name, path in views.items()}
    if any(
        payload.get("split") != "train" or payload.get("seed") != 2027
        or payload.get("num_tasks") != 9
        for payload in payloads.values()
    ):
        raise ValueError("views do not match the frozen nine-task train pilot")
    if payloads["dependency"]["selection_audit"] != payloads["disjoint"]["selection_audit"]:
        raise ValueError("views have different selected sources")
    if payloads["disjoint"].get("control_edge_overlap_with_original") != 0:
        raise ValueError("control is not disjoint")

    task_rows = _load_train_task_metadata(
        args.appworld_python.resolve(), root, 2027, 9
    )
    target_hashes = {str(row["task_hash"]) for row in task_rows}
    if len(target_hashes) != 9 or any(set(payload["hints"]) != target_hashes for payload in payloads.values()):
        raise ValueError("view target set differs from private train pilot")
    tokenizer_dir = args.tokenizer_dir.resolve()
    tokenizer_sha = hashlib.sha256((tokenizer_dir / "tokenizer.json").read_bytes()).hexdigest()
    if tokenizer_sha != TOKENIZER_SHA256:
        raise ValueError("tokenizer differs from frozen model snapshot")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    counts: dict[str, list[int]] = {name: [] for name in views}
    for row in task_rows:
        target_hash = str(row["task_hash"])
        supervisor = SimpleNamespace(**row["supervisor"])
        for name, payload in payloads.items():
            messages = _initial_messages(
                str(row["instruction"]), supervisor, "checklist", payload["hints"][target_hash]
            )
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            counts[name].append(len(tokenizer(prompt)["input_ids"]))
    delta = [
        actual - control
        for actual, control in zip(counts["dependency"], counts["disjoint"])
    ]
    result = {
        "status": "train_only_first_turn_disjoint_cost_not_agent_outcome",
        "split": "train",
        "seed": 2027,
        "num_tasks": 9,
        "per_arm": {name: summarize_counts(values) for name, values in counts.items()},
        "dependency_minus_disjoint": summarize_counts(delta),
        "max_absolute_per_task_delta": max(abs(value) for value in delta),
        "views_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in views.items()},
        "tokenizer_json_sha256": tokenizer_sha,
        "train_split_sha256": hashlib.sha256((root / "data" / "datasets" / "train.txt").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "target_ground_truth_loaded": False,
        "dev_or_test_task_loaded": False,
        "task_text_persisted_or_printed_by_parent": False,
        "task_ids_exported": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("per_arm", "dependency_minus_disjoint", "max_absolute_per_task_delta")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
