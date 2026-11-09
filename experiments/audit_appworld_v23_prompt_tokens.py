#!/usr/bin/env python3
"""Measure private v23 TRAIN first-turn prompt costs with the frozen tokenizer.

Only aggregate token counts and hashes are exported. No model inference or
task ground truth is used; later turns may be longer and are not measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "data" / "private_corem"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_appworld_local_agent_pilot import (  # noqa: E402
    _initial_messages,
    _sample_task_ids,
)


def summarize_counts(counts: list[int]) -> dict[str, int | float]:
    if not counts:
        raise ValueError("empty prompt cost list")
    return {
        "count": len(counts),
        "min": min(counts),
        "max": max(counts),
        "total": sum(counts),
        "mean": round(sum(counts) / len(counts), 3),
    }


def _emit_train_task_metadata() -> int:
    """Private IPC worker: stdout is captured by the tokenizer process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-task-metadata", action="store_true")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error("benchmark root must stay private")
    from appworld import load_task_ids, update_root
    from appworld.task import Task

    update_root(str(root))
    task_ids = _sample_task_ids(load_task_ids("train"), args.seed, args.num_tasks)
    rows = []
    for task_id in task_ids:
        task = Task.load(task_id, load_ground_truth=False)
        rows.append(
            {
                "task_hash": hashlib.sha256(task_id.encode()).hexdigest(),
                "instruction": task.instruction,
                "supervisor": {
                    name: getattr(task.supervisor, name)
                    for name in ("first_name", "last_name", "email", "phone_number")
                },
            }
        )
    print(json.dumps(rows))
    return 0


def _load_train_task_metadata(
    appworld_python: Path, root: Path, seed: int, num_tasks: int
) -> list[dict[str, object]]:
    command = [
        str(appworld_python),
        str(Path(__file__).resolve()),
        "--worker-task-metadata",
        "--root",
        str(root),
        "--seed",
        str(seed),
        "--num-tasks",
        str(num_tasks),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("private AppWorld metadata worker failed")
    rows = json.loads(result.stdout)
    if not isinstance(rows, list) or len(rows) != num_tasks:
        raise ValueError("private AppWorld worker returned unexpected task count")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--appworld-python", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--order-view", type=Path, required=True)
    parser.add_argument("--dependency-view", type=Path, required=True)
    parser.add_argument("--mirrored-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--num-tasks", type=int, default=9)
    parser.add_argument("--prompt-variant", choices=("checklist", "worked_example"), default="checklist")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    views = {
        "order": args.order_view.resolve(),
        "dependency": args.dependency_view.resolve(),
        "mirrored": args.mirrored_view.resolve(),
    }
    if any(
        not path.is_relative_to(PRIVATE_ROOT.resolve())
        for path in (root, output, *views.values())
    ):
        parser.error("benchmark, views, and output must stay private")
    if output.exists() or output in views.values():
        parser.error("refusing to overwrite a private artifact")
    if not 1 <= args.num_tasks <= 30:
        parser.error("invalid task count")

    from transformers import AutoTokenizer

    task_rows = _load_train_task_metadata(
        args.appworld_python.resolve(), root, args.seed, args.num_tasks
    )
    target_hashes = {str(row["task_hash"]) for row in task_rows}
    if len(target_hashes) != args.num_tasks:
        raise ValueError("private AppWorld worker returned duplicate tasks")
    payloads = {name: json.loads(path.read_text()) for name, path in views.items()}
    for payload in payloads.values():
        if payload.get("split") != "train" or payload.get("seed") != args.seed:
            raise ValueError("wrong split or seed in memory view")
        if set(payload.get("hints", {})) != target_hashes:
            raise ValueError("memory view does not match target tasks")
    if not (
        payloads["order"]["selection_audit"]
        == payloads["dependency"]["selection_audit"]
        == payloads["mirrored"]["selection_audit"]
    ):
        raise ValueError("memory views do not use identical source selections")
    tokenizer_dir = args.tokenizer_dir.resolve()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    counts: dict[str, list[int]] = {name: [] for name in views}
    for task in task_rows:
        task_hash = str(task["task_hash"])
        supervisor = SimpleNamespace(**task["supervisor"])
        for name, payload in payloads.items():
            messages = _initial_messages(
                str(task["instruction"]),
                supervisor,
                args.prompt_variant,
                payload["hints"][task_hash],
            )
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            counts[name].append(len(tokenizer(prompt)["input_ids"]))
    reference = counts["order"]
    result = {
        "status": "train_only_first_turn_prompt_cost_not_agent_outcome",
        "split": "train",
        "seed": args.seed,
        "num_tasks": len(task_rows),
        "prompt_variant": args.prompt_variant,
        "disable_thinking": True,
        "per_arm": {name: summarize_counts(values) for name, values in counts.items()},
        "delta_vs_order": {
            name: summarize_counts([value - base for value, base in zip(values, reference)])
            for name, values in counts.items()
            if name != "order"
        },
        "dependency_minus_mirrored": summarize_counts(
            [
                actual - control
                for actual, control in zip(counts["dependency"], counts["mirrored"])
            ]
        ),
        "views_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in views.items()
        },
        "tokenizer_json_sha256": hashlib.sha256(
            (tokenizer_dir / "tokenizer.json").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "train_split_sha256": hashlib.sha256(
            (root / "data" / "datasets" / "train.txt").read_bytes()
        ).hexdigest(),
        "target_ground_truth_loaded": False,
        "dev_or_test_task_loaded": False,
        "task_text_persisted_or_printed_by_parent": False,
        "private_task_metadata_transferred_via_captured_child_pipe": True,
        "task_ids_exported": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                "per_arm": result["per_arm"],
                "delta_vs_order": result["delta_vs_order"],
                "dependency_minus_mirrored": result["dependency_minus_mirrored"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker-task-metadata":
        raise SystemExit(_emit_train_task_metadata())
    raise SystemExit(main())
