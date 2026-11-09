#!/usr/bin/env python3
"""Verify deterministic task-level outcomes across two isolated replays."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_jsonl, write_checksums, write_json


def outcome(row: dict[str, Any]) -> dict[str, Any]:
    checker = row.get("checker") or {}
    execution = row.get("execution") or {}
    response = execution.get("response") or {}
    steps = execution.get("steps") or []
    return {
        "selected_memory_ids": row.get("selected_memory_ids"),
        "execution_error": row.get("execution_error"),
        "checker_error": row.get("checker_error"),
        "http_status": response.get("status"),
        "workflow_http_statuses": [
            step.get("response", {}).get("status")
            for step in steps
            if isinstance(step, dict) and isinstance(step.get("response"), dict)
        ],
        "goal_satisfied": checker.get("goal_satisfied"),
        "binding_checks": checker.get("binding_checks"),
        "evidence_checks": checker.get("evidence_checks"),
        "communication_checks": checker.get("communication_checks"),
        "reuse_attempts": checker.get("reuse_attempts"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-first-superset", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    first_rows = load_jsonl(args.first)
    second_rows = load_jsonl(args.second)
    first = {
        (str(row["method"]), int(row["seed"]), str(row["event_id"])): outcome(row)
        for row in first_rows
    }
    second = {
        (str(row["method"]), int(row["seed"]), str(row["event_id"])): outcome(row)
        for row in second_rows
    }
    compared_keys = set(first).intersection(second)
    key_errors = []
    if args.allow_first_superset:
        key_errors.extend(
            f"missing from first: {key}"
            for key in sorted(set(second).difference(first))
        )
    else:
        key_errors.extend(
            f"row-set mismatch: {key}"
            for key in sorted(set(first).symmetric_difference(second))
        )
    mismatches = []
    for key in sorted(compared_keys):
        if first.get(key) != second.get(key):
            mismatches.append(
                {"key": list(key), "first": first.get(key), "second": second.get(key)}
            )
    args.output.mkdir(parents=True)
    write_json(
        args.output / "result.json",
        {
            "status": "reproducible"
            if not mismatches and not key_errors
            else "mismatch",
            "first_sha256": hashlib.sha256(args.first.read_bytes()).hexdigest(),
            "second_sha256": hashlib.sha256(args.second.read_bytes()).hexdigest(),
            "first_rows": len(first_rows),
            "second_rows": len(second_rows),
            "matched_task_outcomes": len(compared_keys) - len(mismatches),
            "mismatch_count": len(mismatches) + len(key_errors),
            "key_errors": key_errors,
            "mismatches": mismatches,
            "comparison_scope": "Selections, errors, single-request or per-step HTTP statuses, and every independent checker decision; latency, ports, HTTP dates, dynamic bindings, and state hashes are intentionally excluded.",
        },
    )
    write_checksums(args.output, ("result.json",))
    total_mismatches = len(mismatches) + len(key_errors)
    print(
        f"compared {len(first_rows)} and {len(second_rows)} rows: {total_mismatches} mismatches"
    )
    return 0 if not total_mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
