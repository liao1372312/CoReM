#!/usr/bin/env python3
"""Build paired, strict-source TRAIN hints from static API dataflow sketches."""

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

from experiments.appworld_static_dependency import (  # noqa: E402
    StaticDependencySketch,
    extract_static_dependency_sketch,
)
from experiments.build_appworld_train_exemplar_hints import select_source  # noqa: E402
from experiments.run_appworld_local_agent_pilot import _sample_task_ids  # noqa: E402


def strict_source_rows(
    rows: list[dict[str, str]], target_generators: set[str]
) -> list[dict[str, str]]:
    eligible = [row for row in rows if row["generator_id"] not in target_generators]
    if not eligible or any(row["generator_id"] in target_generators for row in eligible):
        raise ValueError("strict source bank is empty or overlaps targets")
    return eligible


def render_order(sketch: StaticDependencySketch) -> str:
    if not sketch.api_calls:
        raise ValueError("source reference has no static AppWorld API calls")
    return "\n".join(
        f"{index}. {name}" for index, name in enumerate(sketch.api_calls, start=1)
    )


def render_edges(sketch: StaticDependencySketch) -> str:
    if not sketch.edges:
        return "None detected in this source."
    return ", ".join(f"{producer}->{consumer}" for producer, consumer in sketch.edges)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--order-output", type=Path, required=True)
    parser.add_argument("--dependency-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--num-tasks", type=int, default=9)
    args = parser.parse_args()
    root = args.root.resolve()
    order_output = args.order_output.resolve()
    dependency_output = args.dependency_output.resolve()
    if any(
        not path.is_relative_to(PRIVATE_ROOT.resolve())
        for path in (root, order_output, dependency_output)
    ):
        parser.error("benchmark and hint files must stay private")
    if order_output == dependency_output or order_output.exists() or dependency_output.exists():
        parser.error("refusing to overwrite or alias paired outputs")
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
            Task.load(task_id, load_ground_truth=False, include_api_response_schemas=False)
            for task_id in train_ids
        )
    ]
    by_id = {row["task_id"]: row for row in rows}
    target_generators = {by_id[task_id]["generator_id"] for task_id in selected_ids}
    sources = strict_source_rows(rows, target_generators)
    if len(sources) != 3 * (30 - args.num_tasks):
        raise ValueError("unexpected strict source-bank size")

    order_hints: dict[str, str] = {}
    dependency_hints: dict[str, str] = {}
    audits: list[dict[str, object]] = []
    for task_id in selected_ids:
        target = by_id[task_id]
        source = select_source(target, sources)
        report_path = (
            root
            / "experiments"
            / "outputs"
            / "verification"
            / "tasks"
            / source["task_id"]
            / "evaluation"
            / "report.md"
        )
        report = report_path.read_text()
        failed = re.search(r"Failed Tests\s*:\s*(\d+)", report)
        passed = re.search(r"Passed Tests\s*:\s*(\d+)", report)
        if failed is None or passed is None or int(failed[1]) != 0 or int(passed[1]) < 1:
            raise ValueError("selected source lacks clean reference verification")
        source_task = Task.load(
            source["task_id"],
            ground_truth_mode="full",
            include_api_response_schemas=False,
        )
        code = source_task.ground_truth.compiled_solution_code
        sketch = extract_static_dependency_sketch(code)
        target_hash = hashlib.sha256(task_id.encode()).hexdigest()
        common = (
            "Example from a DIFFERENT train generator; do not copy entity IDs, "
            "credentials, or argument values. Inspect current API docs and "
            "actual responses before acting. Static source-code call order "
            "is not guaranteed runtime order.\n"
            "Example task:\n"
            + source["instruction"]
            + "\nStatic AppWorld API-call occurrences:\n"
            + render_order(sketch)
        )
        order_hints[target_hash] = common
        dependency_hints[target_hash] = (
            common
            + "\nPotential source-code def-use edges between occurrence numbers "
            "(not verified runtime bindings):\n"
            + render_edges(sketch)
        )
        audits.append(
            {
                "target_sha256": target_hash,
                "source_sha256": hashlib.sha256(source["task_id"].encode()).hexdigest(),
                "target_generator_sha256": hashlib.sha256(
                    target["generator_id"].encode()
                ).hexdigest(),
                "source_generator_sha256": hashlib.sha256(
                    source["generator_id"].encode()
                ).hexdigest(),
                "source_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "verification_report_sha256": hashlib.sha256(
                    report_path.read_bytes()
                ).hexdigest(),
                "static_api_occurrences": len(sketch.api_calls),
                "static_candidate_edges": len(sketch.edges),
            }
        )
    if any(
        row["source_generator_sha256"]
        in {item["target_generator_sha256"] for item in audits}
        for row in audits
    ):
        raise AssertionError("source bank overlaps a target generator")

    common_payload = {
        "status": "protected_local_only_train_exploratory",
        "split": "train",
        "seed": args.seed,
        "num_tasks": args.num_tasks,
        "strict_source_generators": len({row["generator_id"] for row in sources}),
        "strict_source_tasks": len(sources),
        "train_split_sha256": hashlib.sha256(
            (root / "data" / "datasets" / "train.txt").read_bytes()
        ).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "extractor_sha256": hashlib.sha256(
            (ROOT / "experiments" / "appworld_static_dependency.py").read_bytes()
        ).hexdigest(),
        "selection_audit": audits,
        "target_ground_truth_loaded": False,
        "dev_or_test_task_loaded": False,
    }
    for path, method, hints in (
        (
            order_output,
            "bm25_strict_source_static_api_order_only",
            order_hints,
        ),
        (
            dependency_output,
            "bm25_strict_source_static_api_order_plus_candidate_defuse",
            dependency_hints,
        ),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({**common_payload, "method": method, "hints": hints}, sort_keys=True)
            + "\n"
        )
    print(
        "Paired private train hints frozen for "
        f"{len(selected_ids)} targets from {len(sources)} eligible source tasks; "
        "no target ground truth or dev/test content loaded."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
