#!/usr/bin/env python3
"""Aggregate-only TRAIN audit of static API sketches against reference traces.

Order agreement is a necessary sanity check, not dynamic dataflow evidence.
No task identifiers, instructions, solutions, or trace contents are exported.
"""

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


def lcs_length(left: tuple[str, ...], right: list[str]) -> int:
    """Longest common subsequence length, retaining API occurrence order."""
    previous = [0] * (len(right) + 1)
    for left_name in left:
        current = [0] * (len(right) + 1)
        for index, right_name in enumerate(right, start=1):
            current[index] = (
                previous[index - 1] + 1
                if left_name == right_name
                else max(previous[index], current[index - 1])
            )
        previous = current
    return previous[-1]


def edge_order_compatible(
    names: tuple[str, ...], edges: tuple[tuple[int, int], ...], runtime: list[str]
) -> tuple[int, int]:
    """Count candidate edges whose API names can occur in the right order.

    Matching names does not prove the runtime consumer used the producer value.
    """
    compatible = 0
    for producer, consumer in edges:
        if not 1 <= producer < consumer <= len(names):
            raise ValueError("invalid static edge occurrence")
        producer_name, consumer_name = names[producer - 1], names[consumer - 1]
        seen_producer = False
        for name in runtime:
            if name == consumer_name and seen_producer:
                compatible += 1
                break
            if name == producer_name:
                seen_producer = True
    return compatible, len(edges)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--num-target-generators", type=int, default=9)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
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
    static_calls = runtime_calls = lcs_total = compatible_edges = total_edges = 0
    full_subsequence_tasks = identical_sequence_tasks = 0
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
        runtime = ordered_api_names(source_root / "logs" / "api_calls.jsonl", docs)
        matched = lcs_length(sketch.api_calls, runtime)
        static_calls += len(sketch.api_calls)
        runtime_calls += len(runtime)
        lcs_total += matched
        full_subsequence_tasks += matched == len(sketch.api_calls)
        identical_sequence_tasks += sketch.api_calls == tuple(runtime)
        compatible, count = edge_order_compatible(sketch.api_calls, sketch.edges, runtime)
        compatible_edges += compatible
        total_edges += count

    result = {
        "status": "train_only_order_sanity_check_not_dynamic_binding_or_target_success",
        "source_tasks": len(source_ids),
        "target_generators_excluded": len(target_generators),
        "static_api_occurrences": static_calls,
        "runtime_api_occurrences": runtime_calls,
        "lcs_matched_static_occurrences": lcs_total,
        "full_static_subsequence_tasks": full_subsequence_tasks,
        "identical_sequence_tasks": identical_sequence_tasks,
        "potential_static_edges": total_edges,
        "name_order_compatible_edges": compatible_edges,
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
