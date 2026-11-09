#!/usr/bin/env python3
"""Analyze adaptive forecast retention on chronological and shuffled streams."""

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
    parser.add_argument("--chronological-forecast", type=Path, required=True)
    parser.add_argument("--chronological-bundle", type=Path, required=True)
    parser.add_argument("--shuffled-forecast", type=Path, required=True)
    parser.add_argument("--shuffled-bundle", type=Path, required=True)
    parser.add_argument("--confirm-shuffled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def method_rows(run: Path, method: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in load_jsonl(run / "execution_results.jsonl")
        if row["method"] == method
    ]
    if len(rows) != 320:
        raise ValueError(f"expected 320 {method} rows in {run}")
    return rows


def method_summary(run: Path, method: str) -> dict[str, Any]:
    result = json.loads((run / "result.json").read_text())
    summary = next(row for row in result["summaries"] if row["method"] == method)
    return {
        "vsr": float(summary["vsr"]),
        "rce": float(summary["retrievable_complete_evidence_rate"]),
        "churn": float(summary["mean_churn_records_per_update"]),
        "mean_update_ms": float(summary["mean_update_latency_ms"]),
        "p95_update_ms": float(summary["p95_update_latency_ms"]),
    }


def paired(
    scenario: str,
    forecast: list[dict[str, Any]],
    bundle: list[dict[str, Any]],
    start: int,
    stop: int,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    forecast_slice = forecast[start:stop]
    bundle_slice = bundle[start:stop]
    if [row["event_id"] for row in forecast_slice] != [
        row["event_id"] for row in bundle_slice
    ]:
        raise ValueError(f"event mismatch in {scenario}")
    differences = [
        int(left["checker"]["goal_satisfied"] is True)
        - int(right["checker"]["goal_satisfied"] is True)
        for left, right in zip(forecast_slice, bundle_slice)
    ]
    if len(differences) % 20:
        raise ValueError("paired segment must contain complete 20-event blocks")
    blocks = [
        differences[index : index + 20]
        for index in range(0, len(differences), 20)
    ]
    low, high = cluster_bootstrap(blocks, draws, seed)
    return {
        "scenario": scenario,
        "events": len(differences),
        "time_blocks": len(blocks),
        "forecast_vsr": mean(
            int(row["checker"]["goal_satisfied"] is True)
            for row in forecast_slice
        ),
        "bundle_vsr": mean(
            int(row["checker"]["goal_satisfied"] is True) for row in bundle_slice
        ),
        "delta_vsr_pp": 100.0 * mean(differences),
        "ci_low_pp": 100.0 * low,
        "ci_high_pp": 100.0 * high,
        "raw_cluster_sign_flip_p": exact_cluster_sign_flip(
            [sum(block) for block in blocks]
        ),
        "forecast_only_success": sum(value == 1 for value in differences),
        "bundle_only_success": sum(value == -1 for value in differences),
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")
    chronological_forecast = method_rows(
        args.chronological_forecast, "forecast_corem"
    )
    chronological_bundle = method_rows(
        args.chronological_bundle, "bundle_value_retention"
    )
    shuffled_forecast = method_rows(args.shuffled_forecast, "forecast_corem")
    shuffled_bundle = method_rows(
        args.shuffled_bundle, "bundle_value_retention"
    )
    confirm_forecast = method_rows(args.confirm_shuffled, "forecast_corem")
    confirm_bundle = method_rows(
        args.confirm_shuffled, "bundle_value_retention"
    )
    comparisons = [
        paired(
            "Chronological development prefix",
            chronological_forecast,
            chronological_bundle,
            0,
            80,
            args.bootstrap_draws,
            args.seed,
        ),
        paired(
            "Chronological cross-service remainder",
            chronological_forecast,
            chronological_bundle,
            80,
            320,
            args.bootstrap_draws,
            args.seed + 1,
        ),
        paired(
            "Chronological full stream",
            chronological_forecast,
            chronological_bundle,
            0,
            320,
            args.bootstrap_draws,
            args.seed + 2,
        ),
        paired(
            "Seeded shuffled-arrival stress",
            shuffled_forecast,
            shuffled_bundle,
            0,
            320,
            args.bootstrap_draws,
            args.seed + 3,
        ),
        paired(
            "Second-seed shuffled-arrival confirmation",
            confirm_forecast,
            confirm_bundle,
            0,
            320,
            args.bootstrap_draws,
            args.seed + 4,
        ),
    ]
    metrics = []
    for scenario, run, method in (
        ("chronological", args.chronological_forecast, "forecast_corem"),
        ("chronological", args.chronological_bundle, "bundle_value_retention"),
        ("shuffled", args.shuffled_forecast, "forecast_corem"),
        ("shuffled", args.shuffled_bundle, "bundle_value_retention"),
        ("confirm-shuffled", args.confirm_shuffled, "forecast_corem"),
        (
            "confirm-shuffled",
            args.confirm_shuffled,
            "bundle_value_retention",
        ),
    ):
        metrics.append(
            {
                "scenario": scenario,
                "method": method,
                **method_summary(run, method),
            }
        )
    result = {
        "status": "posthoc_development_and_controlled_stress_complete",
        "method_metrics": metrics,
        "paired_comparisons": comparisons,
        "selected_policy": {
            "forecast_min_margin": 0.015,
            "forecast_min_support": 2,
            "forecast_min_confidence": 0.75,
            "selection_basis": (
                "The margin was selected on the first 80 chronological events. "
                "Minimum support and confidence were introduced after inspecting "
                "the first shuffled stress and then frozen before generating the "
                "second shuffled order with seed 20271025."
            ),
        },
        "claim_boundary": (
            "All streams derive from the already inspected v5 benchmark. The "
            "cross-service remainder tests transfer within v5. The second seeded "
            "shuffle is an unseen randomization of known events, not independent "
            "data or natural-traffic generalization."
        ),
    }
    args.output.mkdir(parents=True)
    for name, rows in (
        ("method_metrics.csv", metrics),
        ("paired_comparisons.csv", comparisons),
    ):
        with (args.output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
