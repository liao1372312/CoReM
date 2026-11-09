#!/usr/bin/env python3
"""Prepare past-only candidate outcomes for two additional rolling HGT windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import (  # noqa: E402
    load_jsonl,
    parse_utc,
    write_checksums,
    write_json,
    write_jsonl,
)

WINDOWS = {
    "window_1": {
        "train": range(1, 17),
        "tune": range(17, 21),
        "validation": range(21, 25),
        "test": range(25, 41),
    },
    "window_2": {
        "train": range(1, 25),
        "tune": range(25, 33),
        "validation": range(33, 41),
        "test": range(41, 57),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--train-executions", type=Path, required=True)
    parser.add_argument("--tune-executions", type=Path, required=True)
    parser.add_argument("--validation-executions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    events = load_jsonl(args.dataset / "events.jsonl")
    by_service: dict[str, list[dict[str, Any]]] = {}
    for service in sorted({str(row["service_id"]) for row in events}):
        by_service[service] = sorted(
            (row for row in events if row["service_id"] == service),
            key=lambda row: (
                parse_utc(row["event_time"], "event_time"),
                str(row["event_id"]),
            ),
        )
        if len(by_service[service]) != 80:
            raise ValueError(f"{service} does not have exactly 80 chronological events")
    event_by_id = {str(row["event_id"]): row for row in events}
    split_by_id = {
        str(row["event_id"]): row for row in load_jsonl(args.splits)
    }
    execution_rows = []
    for path in (
        args.train_executions,
        args.tune_executions,
        args.validation_executions,
    ):
        execution_rows.extend(load_jsonl(path))
    executions_by_event: dict[str, list[dict[str, Any]]] = {}
    for row in execution_rows:
        executions_by_event.setdefault(str(row["event_id"]), []).append(row)

    args.output.mkdir(parents=True)
    manifest_windows = {}
    for window, partitions in WINDOWS.items():
        window_dir = args.output / window
        window_dir.mkdir()
        partition_ids: dict[str, set[str]] = {}
        for partition, indices in partitions.items():
            ids = {
                str(rows[index - 1]["event_id"])
                for rows in by_service.values()
                for index in indices
            }
            partition_ids[partition] = ids
            selected = [
                row
                for event_id in sorted(ids)
                for row in executions_by_event.get(event_id, [])
            ]
            if partition != "test" and not selected:
                raise ValueError(f"{window} {partition} has no executed candidates")
            if partition != "test":
                write_jsonl(window_dir / f"{partition}_executions.jsonl", selected)
        test_ids = partition_ids["test"]
        missing = test_ids.difference(event_by_id)
        if missing:
            raise ValueError(f"unknown rolling test events: {sorted(missing)[:3]}")
        split_rows = [
            {**split_by_id[event_id], "split": "test", "rolling_window": window}
            for event_id in sorted(test_ids)
        ]
        write_jsonl(window_dir / "test_splits.jsonl", split_rows)
        write_json(
            window_dir / "manifest.json",
            {
                "schema_version": "corem-rolling-hgt-fold-v1",
                "window": window,
                "per_service_indices": {
                    key: [min(value), max(value)] for key, value in partitions.items()
                },
                "counts": {
                    key: len(value) for key, value in partition_ids.items()
                },
                "time_contract": "Every fitting/tuning/validation event precedes every test event within service.",
            },
        )
        write_checksums(
            window_dir,
            (
                "manifest.json",
                "test_splits.jsonl",
                "train_executions.jsonl",
                "tune_executions.jsonl",
                "validation_executions.jsonl",
            ),
        )
        manifest_windows[window] = {
            key: len(value) for key, value in partition_ids.items()
        }
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": "corem-rolling-hgt-suite-v1",
            "dataset": str(args.dataset.resolve()),
            "windows": manifest_windows,
        },
    )
    write_checksums(args.output, ("manifest.json",))
    print(f"wrote rolling folds to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
