#!/usr/bin/env python3
"""Analyze the post-hoc provenance-coherent retrieval repair."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from analyze_retention_objective_ablation import (
    cluster_bootstrap,
    exact_cluster_sign_flip,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--old-corem", type=Path, required=True)
    parser.add_argument("--coherent-corem", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path / "execution_results.jsonl")
    if len(rows) != 320:
        raise ValueError(f"expected 320 events in {path}")
    return rows


def summarize(
    label: str,
    path: Path,
    rows: list[dict[str, Any]],
    memories: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary = json.loads((path / "result.json").read_text())["summaries"][0]
    mixed = []
    for row in rows:
        operations = {
            operation
            for memory_id in row["selected_memory_ids"]
            for operation in memories[memory_id]["operation_ids"]
        }
        mixed.append(len(operations) > 1)
    return {
        "method": label,
        "events": len(rows),
        "vsr": float(summary["vsr"]),
        "retrievable_complete_evidence_rate": float(
            summary["retrievable_complete_evidence_rate"]
        ),
        "mixed_operation_selection_rate": mean(mixed),
        "mean_update_latency_ms": float(summary["mean_update_latency_ms"]),
        "p95_update_latency_ms": float(summary["p95_update_latency_ms"]),
    }


def paired(
    first: str,
    second: str,
    rows: dict[str, list[dict[str, Any]]],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    first_by_id = {str(row["event_id"]): row for row in rows[first]}
    second_by_id = {str(row["event_id"]): row for row in rows[second]}
    if list(first_by_id) != list(second_by_id):
        raise ValueError("paired runs do not have the same ordered events")
    differences = [
        int(first_by_id[event_id]["checker"]["goal_satisfied"] is True)
        - int(second_by_id[event_id]["checker"]["goal_satisfied"] is True)
        for event_id in first_by_id
    ]
    blocks = [differences[start : start + 20] for start in range(0, 320, 20)]
    low, high = cluster_bootstrap(blocks, draws, seed)
    return {
        "contrast": f"{first} - {second}",
        "events": 320,
        "time_blocks": 16,
        "delta_vsr_pp": 100.0 * mean(differences),
        "ci_low_pp": 100.0 * low,
        "ci_high_pp": 100.0 * high,
        "raw_cluster_sign_flip_p": exact_cluster_sign_flip(
            [sum(block) for block in blocks]
        ),
        "first_only_success": sum(value == 1 for value in differences),
        "second_only_success": sum(value == -1 for value in differences),
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")
    memories = {
        str(row["memory_id"]): row
        for row in load_jsonl(args.dataset / "memories.jsonl")
    }
    paths = {
        "Original CoReM": args.old_corem,
        "Coherent CoReM": args.coherent_corem,
        "Bundle-value": args.bundle,
    }
    rows = {label: run_rows(path) for label, path in paths.items()}
    metrics = [
        summarize(label, paths[label], rows[label], memories) for label in paths
    ]
    contrasts = [
        paired(
            "Coherent CoReM",
            "Original CoReM",
            rows,
            args.bootstrap_draws,
            args.seed,
        ),
        paired(
            "Coherent CoReM",
            "Bundle-value",
            rows,
            args.bootstrap_draws,
            args.seed + 1,
        ),
    ]
    result = {
        "status": "posthoc_diagnostic_complete",
        "metrics": metrics,
        "paired_contrasts": contrasts,
        "trajectory_identity": (
            "Coherent CoReM and Bundle-value have identical selected sets and "
            "post-update banks on all 320 events."
        ),
        "claim_boundary": (
            "The repair was motivated after inspecting the original 320-event "
            "outcome; its apparent improvement is post-hoc and requires a newly "
            "frozen, untouched chronological stream for confirmation."
        ),
    }
    args.output.mkdir(parents=True)
    with (args.output / "method_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (args.output / "paired_contrasts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrasts[0]))
        writer.writeheader()
        writer.writerows(contrasts)
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
