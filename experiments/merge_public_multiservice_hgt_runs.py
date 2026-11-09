#!/usr/bin/env python3
"""Merge independently trained v5 HGT seeds into one frozen test artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_jsonl, write_checksums, write_json, write_jsonl


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2

    rows: list[dict[str, Any]] = []
    sources = []
    runs = []
    configurations = []
    for directory in args.inputs:
        result_path = directory / "result.json"
        selections_path = directory / "selections.jsonl"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") != "trained_and_test_selections_frozen":
            raise ValueError(f"source run is incomplete: {directory}")
        configurations.append(payload["configuration"])
        runs.extend(payload["runs"])
        rows.extend(load_jsonl(selections_path))
        sources.append(
            {
                "directory": str(directory.resolve()),
                "result_sha256": sha256(result_path),
                "selections_sha256": sha256(selections_path),
            }
        )
    comparable = [
        {key: value for key, value in config.items() if key != "seeds"}
        for config in configurations
    ]
    if any(config != comparable[0] for config in comparable[1:]):
        raise ValueError("source configurations do not match")
    run_ids = [str(row["run_id"]) for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("merged selections contain duplicate run IDs")
    methods = sorted({str(row["method"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    event_ids = sorted({str(row["event_id"]) for row in rows})
    expected = len(methods) * len(seeds) * len(event_ids)
    if len(rows) != expected:
        raise ValueError(
            f"incomplete method/seed/event product: {len(rows)} != {expected}"
        )

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for run in runs:
        for split in ("tune", "validation"):
            for metric in (
                "top1_vsr",
                "mean_reciprocal_rank",
                "roc_auc",
                "brier",
            ):
                grouped[(str(run["variant"]), split, metric)].append(
                    float(run[split][metric])
                )
    aggregate = {
        method: {
            split: {
                metric: describe(grouped[(method, split, metric)])
                for metric in (
                    "top1_vsr",
                    "mean_reciprocal_rank",
                    "roc_auc",
                    "brier",
                )
            }
            for split in ("tune", "validation")
        }
        for method in methods
    }

    rows.sort(
        key=lambda row: (str(row["method"]), int(row["seed"]), str(row["event_id"]))
    )
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "selections.jsonl", rows)
    write_json(
        args.output / "result.json",
        {
            "status": "merged_test_selections_frozen",
            "methods": methods,
            "seeds": seeds,
            "test_events": len(event_ids),
            "selection_rows": len(rows),
            "configuration": comparable[0],
            "candidate_metrics": aggregate,
            "variant_contracts": {
                "hgt_corem": "Full trained HGT with complement set head.",
                "hgt_additive": (
                    "Encoder- and candidate-matched A1; complement and redundancy "
                    "set terms are disabled."
                ),
                "hgt_no_hyperedges": (
                    "Candidate-matched representation control; hyperedges are removed "
                    "from HGT messages and set utility, while the frozen candidate "
                    "sets are retained. This is not the stricter no-bundle-exposure A2."
                ),
            },
            "source_runs": sources,
        },
    )
    write_checksums(args.output, ("result.json", "selections.jsonl"))
    print(
        f"merged {len(rows)} selections for {len(methods)} methods, "
        f"{len(seeds)} seeds, and {len(event_ids)} events"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
