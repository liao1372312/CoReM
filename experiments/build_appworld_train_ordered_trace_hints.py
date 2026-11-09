#!/usr/bin/env python3
"""Build private, generator-disjoint hints from verified TRAIN API traces.

Target instructions determine retrieval. Only the chosen training source's
reference-execution log is read; target reference traces and dev/test content
never enter the hint. The output contains protected text and stays private.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from itertools import groupby
from pathlib import Path
from urllib.parse import urlparse

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = REPOSITORY_ROOT / "data" / "private_corem"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.build_appworld_train_exemplar_hints import select_source  # noqa: E402
from experiments.run_appworld_local_agent_pilot import _sample_task_ids  # noqa: E402


def route_pattern(path: str) -> re.Pattern[str]:
    """Match an AppWorld documented route, including dynamic path segments."""
    parts = path.strip("/").split("/")
    if not parts or not all(parts):
        raise ValueError("invalid documented API route")
    regex = "/".join(
        "[^/]+" if part.startswith("{") and part.endswith("}") else re.escape(part)
        for part in parts
    )
    return re.compile("^/" + regex + "$")


def ordered_api_names(
    log_path: Path, docs: list[tuple[str, re.Pattern[str], str]]
) -> list[str]:
    """Resolve each actual reference call to exactly one documented API."""
    result: list[str] = []
    for line in log_path.read_text().splitlines():
        call = json.loads(line)
        route = urlparse(call["url"]).path
        method = call["method"].upper()
        matches = [
            name
            for documented_method, pattern, name in docs
            if documented_method == method and pattern.fullmatch(route)
        ]
        if len(matches) != 1:
            raise ValueError("reference call does not resolve to exactly one API")
        result.append(matches[0])
    if not result:
        raise ValueError("empty reference API log")
    return result


def compact_api_sequence(names: list[str]) -> str:
    """Preserve order while compressing adjacent repeated calls."""
    if not names:
        raise ValueError("cannot compact an empty sequence")
    segments = []
    for name, group in groupby(names):
        count = sum(1 for _ in group)
        segments.append(f"{name} x{count}" if count > 1 else name)
    return " -> ".join(segments)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--num-tasks", type=int, default=9)
    parser.add_argument("--max-calls", type=int, default=128)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if not root.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error("AppWorld root must be private")
    if not output.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error("protected output must stay under data/private_corem")
    if output.exists():
        parser.error("refusing to overwrite protected output")
    if not 1 <= args.num_tasks <= 30 or not 1 <= args.max_calls <= 512:
        parser.error("invalid task count or trace cap")

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
    first = Task.load(train_ids[0], load_ground_truth=False)
    docs = [
        (str(doc["method"]).upper(), route_pattern(str(doc["path"])), f"{app}.{api}")
        for app, app_docs in first.api_docs.items()
        for api, doc in app_docs.items()
    ]
    hints: dict[str, str] = {}
    audits: list[dict[str, object]] = []
    for task_id in selected_ids:
        target = by_id[task_id]
        source = select_source(target, rows)
        source_root = (
            root / "experiments/outputs/verification/tasks" / source["task_id"]
        )
        log_path = source_root / "logs/api_calls.jsonl"
        report_path = source_root / "evaluation/report.md"
        report = report_path.read_text()
        failed = re.search(r"Failed Tests\s*:\s*(\d+)", report)
        passed = re.search(r"Passed Tests\s*:\s*(\d+)", report)
        if (
            failed is None
            or passed is None
            or int(failed[1]) != 0
            or int(passed[1]) < 1
        ):
            raise ValueError("selected source has no passing reference verification")
        names = ordered_api_names(log_path, docs)
        shown = names[: args.max_calls]
        target_hash = hashlib.sha256(task_id.encode()).hexdigest()
        hints[target_hash] = (
            "Verified reference from a DIFFERENT training task; this is an "
            "ordered API-call hint, not the answer to the current task. "
            "Inspect current API docs and actual return values before acting. "
            "Never reuse source entity IDs or credentials.\n"
            "Example task:\n"
            + source["instruction"]
            + "\nReference API-call order (adjacent repeats compressed):\n"
            + compact_api_sequence(shown)
            + (
                f" -> [remaining {len(names) - len(shown)} calls omitted]"
                if len(names) > len(shown)
                else ""
            )
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
                "source_api_calls": len(names),
                "shown_api_calls": len(shown),
                "reference_log_sha256": hashlib.sha256(
                    log_path.read_bytes()
                ).hexdigest(),
                "verification_report_sha256": hashlib.sha256(
                    report_path.read_bytes()
                ).hexdigest(),
                "verification_failed_tests": 0,
            }
        )
    if any(
        row["target_generator_sha256"] == row["source_generator_sha256"]
        for row in audits
    ):
        raise AssertionError("generator overlap")
    payload = {
        "status": "protected_local_only_train_exploratory",
        "split": "train",
        "method": "bm25_top1_other_generator_verified_runtime_api_order",
        "seed": args.seed,
        "num_tasks": len(hints),
        "max_calls": args.max_calls,
        "train_split_sha256": hashlib.sha256(
            (root / "data/datasets/train.txt").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection_audit": audits,
        "hints": hints,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        f"Private verified train API-order view frozen for {len(hints)} tasks; "
        "all sources are generator-disjoint and have zero failed verification tests."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
