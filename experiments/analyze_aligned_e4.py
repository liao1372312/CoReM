#!/usr/bin/env python3
"""Verify and consolidate the nine HGT-aligned E4 online budget runs."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import (  # noqa: E402
    load_json,
    load_jsonl,
    write_checksums,
    write_json,
)

RETRIEVAL_BUDGETS = (160, 200, 240)
STORAGE_BUDGETS = (256, 640, 960)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_checksums(directory: Path) -> None:
    checksum_path = directory / "checksums.sha256"
    if not checksum_path.is_file():
        raise ValueError(f"missing checksums: {directory}")
    for line in checksum_path.read_text().splitlines():
        digest, filename = line.split("  ", 1)
        if sha256(directory / filename) != digest:
            raise ValueError(f"checksum mismatch: {directory / filename}")


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")

    cells: list[dict[str, Any]] = []
    retrieval_model: dict[str, Any] | None = None
    seen: set[tuple[int, int]] = set()
    source_runs = []
    for directory in args.runs:
        verify_checksums(directory)
        result = load_json(directory / "result.json")
        key = (int(result["retrieval_budget"]), int(result["storage_budget"]))
        if (
            result.get("status") != "complete_online_retention"
            or result.get("events") != 320
            or result.get("methods") != ["group_aware_online"]
            or result.get("seed") != 2027
            or result.get("workload_window") != 16
            or key in seen
        ):
            raise ValueError(f"invalid or duplicate E4 run: {directory}")
        seen.add(key)
        if retrieval_model is None:
            retrieval_model = result["retrieval_model"]
        elif result["retrieval_model"] != retrieval_model:
            raise ValueError(f"retrieval model mismatch: {directory}")

        scored = load_jsonl(directory / "scored_executions.jsonl")
        banks = load_jsonl(directory / "bank_history.jsonl")
        executions = load_jsonl(directory / "execution_results.jsonl")
        if len(scored) != 320 or len(banks) != 320 or len(executions) != 320:
            raise ValueError(f"incomplete E4 trajectory: {directory}")
        if any(
            row.get("execution_error") is not None
            or row.get("checker_error") is not None
            or row.get("reset_verified") is not True
            or row.get("checker_independent") is not True
            or int(row["token_cost"]) > key[0]
            or not set(row["selected_memory_ids"]).issubset(row["bank_before_ids"])
            for row in executions
        ):
            raise ValueError(f"execution or retrieval audit failed: {directory}")
        if any(int(row["bank_after_cost"]) > key[1] for row in banks):
            raise ValueError(f"storage-budget audit failed: {directory}")

        summary = result["summaries"][0]
        cells.append(
            {
                "retrieval_budget": key[0],
                "storage_budget": key[1],
                "vsr": float(summary["vsr"]),
                "ecr": float(summary["ecr"]),
                "bs": float(summary["bs"]),
                "selection_p95_ms": percentile(
                    [float(row["selection_latency_ms"]) for row in executions], 0.95
                ),
                "update_p95_ms": percentile(
                    [float(row["update_latency_ms"]) for row in banks], 0.95
                ),
                "churn": sum(
                    int(row["activated_records"]) + int(row["evicted_records"])
                    for row in banks
                ),
                "physical_executions": int(result["physical_executions_this_run"]),
                "exact_source_reuses": int(result["exact_prior_executions_used"]),
                "peak_bank_cost": max(int(row["bank_after_cost"]) for row in banks),
            }
        )
        source_runs.append(
            {
                "directory": str(directory.resolve()),
                "result_sha256": sha256(directory / "result.json"),
            }
        )

    expected = {
        (retrieval, storage)
        for retrieval in RETRIEVAL_BUDGETS
        for storage in STORAGE_BUDGETS
    }
    if seen != expected:
        raise ValueError(f"budget grid mismatch: missing={sorted(expected - seen)}")
    cells.sort(key=lambda row: (row["retrieval_budget"], row["storage_budget"]))
    args.output.mkdir(parents=True)
    write_json(
        args.output / "result.json",
        {
            "status": "complete_hgt_aligned_e4",
            "schema_version": "corem-hgt-aligned-e4-v1",
            "events_per_cell": 320,
            "logical_decisions": 320 * len(cells),
            "retrieval_model": retrieval_model,
            "cells": cells,
            "total_physical_executions": sum(
                row["physical_executions"] for row in cells
            ),
            "total_exact_source_reuses": sum(
                row["exact_source_reuses"] for row in cells
            ),
            "source_runs": source_runs,
            "claim_boundary": (
                "Post-hoc budget sensitivity on one controlled 320-event stream; "
                "cells are not independent samples and were not used for tuning."
            ),
        },
    )
    write_checksums(args.output, ("result.json",))
    print("wrote aligned E4 analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
