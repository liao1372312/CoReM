#!/usr/bin/env python3
"""Print aggregate requirement counts from private AppWorld agent reports."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = REPOSITORY_ROOT / "data" / "private_corem"


def summarize_reports(experiment_root: Path, expected_tasks: int) -> dict[str, int]:
    paths = sorted((experiment_root / "tasks").glob("*/evaluation/report.md"))
    if len(paths) != expected_tasks:
        raise ValueError("unexpected number of task reports")
    passed_total = 0
    failed_total = 0
    success_reports = 0
    for path in paths:
        report = path.read_text()
        passed = re.search(r"Passed Tests\s*:\s*(\d+)", report)
        failed = re.search(r"Failed Tests\s*:\s*(\d+)", report)
        if passed is None or failed is None:
            raise ValueError("task report lacks test counts")
        passed_total += int(passed[1])
        failed_total += int(failed[1])
        success_reports += int(int(failed[1]) == 0)
    return {
        "reports": len(paths),
        "passed_requirements": passed_total,
        "failed_requirements": failed_total,
        "success_reports": success_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    if not root.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error("report path must stay under data/private_corem")
    if args.expected_tasks < 1:
        parser.error("expected task count must be positive")
    print(json.dumps(summarize_reports(root, args.expected_tasks), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
