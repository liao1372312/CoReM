#!/usr/bin/env python3
"""Audit static API-name coverage against verified TRAIN runtime traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "data" / "private_corem"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.appworld_static_dependency import extract_static_dependency_sketch  # noqa: E402
from experiments.build_appworld_train_ordered_trace_hints import (  # noqa: E402
    ordered_api_names,
    route_pattern,
)
from experiments.run_appworld_local_agent_pilot import _sample_task_ids  # noqa: E402


def name_set_difference(
    static_names: tuple[str, ...], runtime_names: list[str]
) -> tuple[int, int]:
    static_set = set(static_names)
    runtime_set = set(runtime_names)
    return len(static_set - runtime_set), len(runtime_set - static_set)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--num-target-generators", type=int, default=9)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if not root.is_relative_to(PRIVATE_ROOT.resolve()) or not output.is_relative_to(
        PRIVATE_ROOT.resolve()
    ):
        parser.error("benchmark and output must stay private")
    if output.exists():
        parser.error("refusing to overwrite audit output")
    if not 1 <= args.num_target_generators <= 30:
        parser.error("invalid target-generator count")

    from appworld import load_task_ids, update_root
    from appworld.task import Task, task_id_to_generator_id

    update_root(str(root))
    train_ids = load_task_ids("train")
    target_generators = {
        task_id_to_generator_id(task_id)
        for task_id in _sample_task_ids(train_ids, args.seed, args.num_target_generators)
    }
    source_ids = [
        task_id
        for task_id in train_ids
        if task_id_to_generator_id(task_id) not in target_generators
    ]
    first = Task.load(train_ids[0], load_ground_truth=False)
    docs = [
        (str(doc["method"]).upper(), route_pattern(str(doc["path"])), f"{app}.{api}")
        for app, app_docs in first.api_docs.items()
        for api, doc in app_docs.items()
    ]
    differences: list[tuple[int, int]] = []
    for task_id in source_ids:
        source = Task.load(
            task_id, ground_truth_mode="full", include_api_response_schemas=False
        )
        sketch = extract_static_dependency_sketch(
            source.ground_truth.compiled_solution_code
        )
        source_root = root / "experiments" / "outputs" / "verification" / "tasks" / task_id
        report = (source_root / "evaluation" / "report.md").read_text()
        failed = re.search(r"Failed Tests\s*:\s*(\d+)", report)
        passed = re.search(r"Passed Tests\s*:\s*(\d+)", report)
        if failed is None or passed is None or int(failed[1]) != 0 or int(passed[1]) < 1:
            raise ValueError("source reference verification is not clean")
        runtime_names = ordered_api_names(source_root / "logs" / "api_calls.jsonl", docs)
        differences.append(name_set_difference(sketch.api_calls, runtime_names))
    result = {
        "status": "train_only_static_api_name_coverage_not_dependency_verification",
        "source_tasks": len(source_ids),
        "target_generators_excluded": len(target_generators),
        "all_static_names_seen_at_runtime": sum(left == 0 for left, _ in differences),
        "all_runtime_names_seen_statically": sum(right == 0 for _, right in differences),
        "static_only_name_count": sum(left for left, _ in differences),
        "runtime_only_name_count": sum(right for _, right in differences),
        "seed": args.seed,
        "train_split_sha256": hashlib.sha256(
            (root / "data" / "datasets" / "train.txt").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task_text_exported": False,
        "task_ids_exported": False,
        "runtime_edges_verified": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
