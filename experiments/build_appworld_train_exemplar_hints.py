#!/usr/bin/env python3
"""Freeze generator-disjoint AppWorld TRAIN exemplars for a private agent pilot.

Only query instructions enter retrieval. A selected source's ground truth is
loaded afterward; the target's ground truth is never loaded by this script.
The resulting protected hints must remain in data/private_corem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = REPOSITORY_ROOT / "data" / "private_corem"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.generate_public_workflow_selections import bm25_scores  # noqa: E402
from experiments.run_appworld_local_agent_pilot import _sample_task_ids  # noqa: E402


def select_source(
    target: dict[str, str], candidates: list[dict[str, str]]
) -> dict[str, str]:
    """Rank only other-generator instructions, with a stable tie break."""
    eligible = [
        row for row in candidates if row["generator_id"] != target["generator_id"]
    ]
    if not eligible:
        raise ValueError("no generator-disjoint source")
    scores = bm25_scores(
        target["instruction"],
        [{"memory_id": row["task_id"], "text": row["instruction"]} for row in eligible],
    )
    return min(eligible, key=lambda row: (-scores[row["task_id"]], row["task_id"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--num-tasks", type=int, default=9)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if not root.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error("AppWorld root must be private")
    if not output.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error("protected output must stay under data/private_corem")
    if output.exists():
        parser.error("refusing to overwrite protected output")
    if not 1 <= args.num_tasks <= 30:
        parser.error("invalid task count")

    from appworld import load_task_ids, update_root
    from appworld.task import Task

    update_root(str(root))
    train_ids = load_task_ids("train")
    selected_ids = _sample_task_ids(train_ids, args.seed, args.num_tasks)
    rows = [
        {
            "task_id": task.id,
            "generator_id": task.generator_id,
            "instruction": task.instruction,
        }
        for task in (
            Task.load(
                task_id, load_ground_truth=False, include_api_response_schemas=False
            )
            for task_id in train_ids
        )
    ]
    by_id = {row["task_id"]: row for row in rows}
    hints: dict[str, str] = {}
    selections: list[dict[str, str]] = []
    for task_id in selected_ids:
        target = by_id[task_id]
        source = select_source(target, rows)
        source_task = Task.load(
            source["task_id"],
            ground_truth_mode="full",
            include_api_response_schemas=False,
        )
        if source_task.ground_truth is None:
            raise ValueError("selected train source has no ground truth")
        solution = source_task.ground_truth.compiled_solution_code
        if not solution.strip():
            raise ValueError("selected train source has no compiled solution")
        target_hash = hashlib.sha256(task_id.encode()).hexdigest()
        source_hash = hashlib.sha256(source["task_id"].encode()).hexdigest()
        hints[target_hash] = (
            "Reference from a DIFFERENT training task. It is an example, not the "
            "answer to the current task. Adapt only after checking the current "
            "API docs and returned values. Never reuse example IDs or credentials.\n"
            "Example task:\n"
            + source["instruction"]
            + "\nExample reference code (not executable as-is for this task):\n"
            + solution
        )
        selections.append(
            {
                "target_sha256": target_hash,
                "source_sha256": source_hash,
                "target_generator_sha256": hashlib.sha256(
                    target["generator_id"].encode()
                ).hexdigest(),
                "source_generator_sha256": hashlib.sha256(
                    source["generator_id"].encode()
                ).hexdigest(),
                "source_code_sha256": hashlib.sha256(solution.encode()).hexdigest(),
            }
        )
    if any(
        row["target_generator_sha256"] == row["source_generator_sha256"]
        for row in selections
    ):
        raise AssertionError("generator overlap")
    payload = {
        "status": "protected_local_only_train_exploratory",
        "split": "train",
        "method": "bm25_top1_other_generator_compiled_train_exemplar",
        "seed": args.seed,
        "num_tasks": args.num_tasks,
        "train_split_sha256": hashlib.sha256(
            (root / "data/datasets/train.txt").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection_audit": selections,
        "hints": hints,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        f"Private train exemplar view frozen for {len(hints)} tasks; "
        "all sources are generator-disjoint, and no target ground truth was loaded."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
