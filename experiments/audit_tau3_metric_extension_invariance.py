#!/usr/bin/env python3
"""Prove that frozen CoReM selections remain valid for the metric extension."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("data/tau3_corem_counterexamples")
    )
    parser.add_argument(
        "--extension", type=Path, default=Path("data/tau3_corem_metric_extension")
    )
    parser.add_argument(
        "--selections",
        type=Path,
        default=Path(
            "experiments/results_tau3_counterexample_retrieval/"
            "retrieval_ablation_selections.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/results_tau3_metric_extension_corem/"
            "selection_invariance_audit.json"
        ),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_banks(
    root: Path, *, filter_active_versions: bool
) -> dict[str, frozenset[str]]:
    tasks = {
        row["task_id"]: row
        for row in load_jsonl(root / "retrieval_tasks.jsonl")
        if row["split"] == "test"
    }
    memories = {
        row["memory_id"]: row for row in load_jsonl(root / "memory_records.jsonl")
    }
    raw: dict[str, set[str]] = defaultdict(set)
    for row in load_jsonl(root / "set_outcomes.jsonl"):
        if row["split"] == "test":
            raw[row["task_id"]].update(row["selected_memory_ids"])
    output = {}
    for task_id, memory_ids in raw.items():
        active = frozenset(tasks[task_id].get("active_versions", []))
        output[task_id] = frozenset(
            memory_id
            for memory_id in memory_ids
            if not filter_active_versions
            or (
                bool(memories[memory_id].get("admissible", True))
                and bool(memories[memory_id].get("provenance_verified", True))
                and (
                    not memories[memory_id].get("version_scope")
                    or bool(active.intersection(memories[memory_id]["version_scope"]))
                )
            )
        )
    return output


def main() -> None:
    args = parse_args()
    exact_files = (
        "retrieval_tasks.jsonl",
        "dependency_hyperedges.jsonl",
        "operations.jsonl",
    )
    exact_equal = {
        filename: (args.source / filename).read_bytes()
        == (args.extension / filename).read_bytes()
        for filename in exact_files
    }
    prefix_equal = {}
    for filename in (
        "memory_records.jsonl",
        "set_outcomes.jsonl",
        "duplicate_groups.jsonl",
    ):
        source_rows = load_jsonl(args.source / filename)
        extension_rows = load_jsonl(args.extension / filename)
        prefix_equal[filename] = source_rows == extension_rows[: len(source_rows)]

    source_banks = candidate_banks(args.source, filter_active_versions=False)
    extension_banks = candidate_banks(args.extension, filter_active_versions=True)
    bank_equal = source_banks == extension_banks

    selection_artifact = json.loads(args.selections.read_text(encoding="utf-8"))
    full_rows = [
        row
        for run in selection_artifact["runs"]
        for row in run["selections"]
        if row["method"] == "full_corem"
    ]
    unknown = sorted(
        {
            memory_id
            for row in full_rows
            for memory_id in row["selected_memory_ids"]
            if memory_id not in extension_banks[row["task_id"]]
        }
    )
    checks = {
        "task_edge_operation_files_unchanged": all(exact_equal.values()),
        "source_rows_are_unchanged_prefix": all(prefix_equal.values()),
        "admissible_test_candidate_banks_equal": bank_equal,
        "frozen_full_corem_selections_remain_admissible": not unknown,
        "five_seed_43_task_coverage": len(full_rows) == 5 * 43,
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "purpose": (
            "justify carrying forward frozen CoReM selections while freshly "
            "replaying them with evaluation-only metric annotations"
        ),
        "checks": checks,
        "details": {
            "exact_file_equality": exact_equal,
            "source_prefix_equality": prefix_equal,
            "source_test_tasks": len(source_banks),
            "extension_test_tasks": len(extension_banks),
            "full_corem_selection_rows": len(full_rows),
            "unknown_or_incompatible_selected_ids": unknown,
        },
        "source_hashes": {
            "source_manifest": sha256(args.source / "manifest.json"),
            "extension_manifest": sha256(args.extension / "manifest.json"),
            "frozen_selections": sha256(args.selections),
            "audit_code": sha256(Path(__file__)),
        },
        "boundary": (
            "The audit supports selection carry-forward only for CoReM, whose "
            "version gate excludes all added stale records before selection."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
