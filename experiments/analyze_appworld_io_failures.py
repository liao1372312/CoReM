#!/usr/bin/env python3
"""Summarize private AppWorld interaction logs without exporting task content."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = REPOSITORY_ROOT / "data" / "private_corem"
INTERACTION = re.compile(r"(?m)^### Environment Interaction \d+\n")
BODY = re.compile(
    r"^-+\n```python\n(?P<code>.*?)\n```\n\n```\n(?P<output>.*?)\n```\s*$",
    re.DOTALL,
)
EXCEPTION = re.compile(r"\b([A-Za-z_][\w.]*(?:Error|Exception))(?=:|\b)")


def summarize_io(
    experiment_root: Path,
    expected_tasks: int,
    aggregate_file: Path | None = None,
) -> dict[str, object]:
    paths = sorted((experiment_root / "tasks").glob("*/logs/environment_io.md"))
    if len(paths) != expected_tasks:
        raise ValueError("unexpected number of task logs")
    counts: Counter[str] = Counter()
    exception_counts: Counter[str] = Counter()
    logged_by_hash: dict[str, Counter[str]] = {}
    for path in paths:
        task_hash = hashlib.sha256(path.parents[1].name.encode()).hexdigest()
        task_counts: Counter[str] = Counter()
        logged_by_hash[task_hash] = task_counts
        parts = INTERACTION.split(path.read_text())
        if len(parts) < 2 or parts[0].strip():
            raise ValueError("unrecognized interaction log format")
        for part in parts[1:]:
            match = BODY.fullmatch(part.strip())
            if match is None:
                raise ValueError("unrecognized interaction body format")
            code = match["code"]
            output = match["output"]
            counts["interactions"] += 1
            task_counts["steps"] += 1
            counts["outputs_over_2000_chars"] += len(output) > 2000
            try:
                response = json.loads(output)
            except json.JSONDecodeError:
                response = None
            if isinstance(response, dict) and set(response) == {"message"}:
                counts["message_only_json_outputs"] += 1
            if "apis.api_docs." in code:
                counts["api_doc_interactions"] += 1
                counts["api_doc_outputs_over_2000_chars"] += len(output) > 2000
            if "Execution failed." in output:
                counts["execution_failures"] += 1
                task_counts["failures"] += 1
                counts["failure_outputs_over_2000_chars"] += len(output) > 2000
                if "Syntax error in line:" in output:
                    counts["syntax_failures"] += 1
                    task_counts["syntax_failures"] += 1
                elif "Usage of the following" in output and "not allowed" in output:
                    counts["safety_failures"] += 1
                else:
                    counts["runtime_or_other_failures"] += 1
                matches = EXCEPTION.findall(output)
                exception_counts[matches[-1] if matches else "unclassified"] += 1
    summary: dict[str, object] = {
        "task_logs": len(paths),
        "interactions": counts["interactions"],
        "outputs_over_2000_chars": counts["outputs_over_2000_chars"],
        "message_only_json_outputs": counts["message_only_json_outputs"],
        "api_doc_interactions": counts["api_doc_interactions"],
        "api_doc_outputs_over_2000_chars": counts[
            "api_doc_outputs_over_2000_chars"
        ],
        "execution_failures": counts["execution_failures"],
        "syntax_failures": counts["syntax_failures"],
        "safety_failures": counts["safety_failures"],
        "runtime_or_other_failures": counts["runtime_or_other_failures"],
        "failure_outputs_over_2000_chars": counts[
            "failure_outputs_over_2000_chars"
        ],
        "failures_by_exception_class": dict(sorted(exception_counts.items())),
        "protected_text_exported": False,
    }
    if aggregate_file is not None:
        aggregate = json.loads(aggregate_file.read_text())
        outcomes = aggregate.get("private_task_outcomes")
        if aggregate.get("num_tasks") != expected_tasks or not isinstance(
            outcomes, list
        ) or len(outcomes) != expected_tasks:
            raise ValueError("aggregate task count does not match logs")
        outcome_by_hash = {row["task_sha256"]: row for row in outcomes}
        if len(outcome_by_hash) != expected_tasks or set(outcome_by_hash) != set(
            logged_by_hash
        ):
            raise ValueError("aggregate task identities do not match logs")
        recorded_steps = sum(int(row["steps"]) for row in outcomes)
        logged_steps = counts["interactions"]
        recorded_failures = int(aggregate["execution_failure_messages"])
        logged_failures = counts["execution_failures"]
        if recorded_steps < logged_steps or recorded_failures < logged_failures:
            raise ValueError("interaction log exceeds aggregate counts")
        task_gaps = []
        syntax_gaps = []
        for task_hash, row in outcome_by_hash.items():
            logged = logged_by_hash[task_hash]
            gap = int(row["steps"]) - logged["steps"]
            syntax_gap = int(row["syntax_failure_messages"]) - logged[
                "syntax_failures"
            ]
            if gap < 0 or syntax_gap < 0:
                raise ValueError("task log exceeds aggregate counts")
            task_gaps.append(gap)
            syntax_gaps.append(syntax_gap)
        summary.update(
            {
                "aggregate_recorded_steps": recorded_steps,
                "steps_missing_from_logs": recorded_steps - logged_steps,
                "aggregate_recorded_failures": recorded_failures,
                "failures_missing_from_logs": recorded_failures - logged_failures,
                "aggregate_recorded_syntax_failures": int(
                    aggregate["syntax_failure_messages"]
                ),
                "tasks_with_missing_log_steps": sum(gap > 0 for gap in task_gaps),
                "missing_steps_accounted_for_by_syntax": task_gaps == syntax_gaps,
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--aggregate-file", type=Path)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    if not root.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error("interaction logs must stay under data/private_corem")
    if args.expected_tasks < 1:
        parser.error("expected task count must be positive")
    aggregate_file = args.aggregate_file.resolve() if args.aggregate_file else None
    if aggregate_file and not aggregate_file.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error("aggregate must stay under data/private_corem")
    print(
        json.dumps(
            summarize_io(root, args.expected_tasks, aggregate_file), sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
