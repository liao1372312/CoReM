#!/usr/bin/env python3
"""Summarize frozen fresh-query confirmation runs with clustered inference."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import defaultdict
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

METHODS = (
    "hgt_rank_ensemble",
    "version_bm25_bundle",
    "corem_query_only",
)
DISPLAY = {
    "hgt_rank_ensemble": "Frozen HGT ensemble + BM25 residual",
    "version_bm25_bundle": "Version BM25 + verified closure",
    "corem_query_only": "CoReM query-only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    return parser.parse_args()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cluster_ci(
    grouped: dict[tuple[str, ...], list[float]], draws: int, seed: int
) -> tuple[float, float]:
    keys = sorted(grouped)
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sampled = [rng.choice(keys) for _ in keys]
        values = [value for key in sampled for value in grouped[key]]
        estimates.append(sum(values) / len(values))
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / 2**n)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    if args.bootstrap_draws < 100:
        raise ValueError("bootstrap draws must be at least 100")

    events = {
        str(row["event_id"]): row for row in load_jsonl(args.dataset / "events.jsonl")
    }
    rows = []
    source_status = []
    for analysis in args.analysis:
        result = load_json(analysis / "result.json")
        source_status.append(
            {
                "analysis": str(analysis.resolve()),
                "status": result["status"],
                "execution_sha256": result["execution_sha256"],
            }
        )
        rows.extend(load_jsonl(analysis / "scored_executions.jsonl"))
    present = {str(row["method"]) for row in rows}
    if present != set(METHODS):
        raise ValueError(f"expected methods {METHODS}, observed {sorted(present)}")

    values: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        method = str(row["method"])
        event_id = str(row["event_id"])
        if event_id in values[method]:
            raise ValueError(f"duplicate method/event row: {method}/{event_id}")
        values[method][event_id] = int(row["verified_success"])
    event_ids = set(values[METHODS[0]])
    if any(set(values[method]) != event_ids for method in METHODS):
        raise ValueError("methods do not cover identical events")

    family = {
        event_id: (
            str(events[event_id]["service_id"]),
            str(events[event_id]["version_id"]),
            str(events[event_id]["task_key"]),
        )
        for event_id in event_ids
    }
    method_rows = []
    service_rows = []
    family_rows = []
    for method_index, method in enumerate(METHODS):
        grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
        for event_id in sorted(event_ids):
            grouped[family[event_id]].append(float(values[method][event_id]))
        low, high = cluster_ci(grouped, args.bootstrap_draws, 202700 + method_index)
        method_rows.append(
            {
                "method": method,
                "display_name": DISPLAY[method],
                "events": len(event_ids),
                "task_version_families": len(grouped),
                "verified_successes": sum(values[method].values()),
                "vsr": sum(values[method].values()) / len(event_ids),
                "family_cluster_ci_low": low,
                "family_cluster_ci_high": high,
            }
        )
        by_service: dict[str, list[int]] = defaultdict(list)
        for event_id in event_ids:
            by_service[str(events[event_id]["service_id"])].append(
                values[method][event_id]
            )
        for service, observed in sorted(by_service.items()):
            service_rows.append(
                {
                    "method": method,
                    "service_id": service,
                    "events": len(observed),
                    "verified_successes": sum(observed),
                    "vsr": sum(observed) / len(observed),
                }
            )
        for key, observed in sorted(grouped.items()):
            family_rows.append(
                {
                    "method": method,
                    "service_id": key[0],
                    "version_id": key[1],
                    "task_key": key[2],
                    "events": len(observed),
                    "verified_successes": int(sum(observed)),
                    "vsr": sum(observed) / len(observed),
                }
            )

    comparison_rows = []
    for comparison_index, baseline in enumerate(METHODS[1:]):
        grouped_differences: dict[tuple[str, ...], list[float]] = defaultdict(list)
        wins = losses = ties = 0
        for event_id in sorted(event_ids):
            difference = (
                values["hgt_rank_ensemble"][event_id] - values[baseline][event_id]
            )
            grouped_differences[family[event_id]].append(float(difference))
            wins += difference > 0
            losses += difference < 0
            ties += difference == 0
        low, high = cluster_ci(
            grouped_differences,
            args.bootstrap_draws,
            202710 + comparison_index,
        )
        family_means = [
            sum(observed) / len(observed)
            for observed in grouped_differences.values()
        ]
        family_wins = sum(value > 0 for value in family_means)
        family_losses = sum(value < 0 for value in family_means)
        comparison_rows.append(
            {
                "contrast": f"hgt_rank_ensemble_minus_{baseline}",
                "events": len(event_ids),
                "task_version_families": len(grouped_differences),
                "delta_vsr": sum(map(sum, grouped_differences.values()))
                / len(event_ids),
                "family_cluster_ci_low": low,
                "family_cluster_ci_high": high,
                "event_wins": wins,
                "event_losses": losses,
                "event_ties": ties,
                "event_exact_sign_p": exact_sign_p(wins, losses),
                "family_wins": family_wins,
                "family_losses": family_losses,
                "family_ties": len(family_means) - family_wins - family_losses,
                "family_exact_sign_p": exact_sign_p(family_wins, family_losses),
            }
        )

    args.output.mkdir(parents=True)
    write_csv(args.output / "method_metrics.csv", method_rows)
    write_csv(args.output / "per_service_metrics.csv", service_rows)
    write_csv(args.output / "per_family_metrics.csv", family_rows)
    write_csv(args.output / "paired_comparisons.csv", comparison_rows)
    write_json(
        args.output / "result.json",
        {
            "status": "complete_fresh_query_confirmation",
            "events": len(event_ids),
            "providers": len(
                {str(events[event_id]["provider_id"]) for event_id in event_ids}
            ),
            "task_version_families": len(set(family.values())),
            "bootstrap_draws": args.bootstrap_draws,
            "method_metrics": method_rows,
            "paired_comparisons": comparison_rows,
            "sources": source_status,
            "inference_unit": (
                "Query events are paired; 95% intervals resample the 16 latest-"
                "version task families. Exact sign tests are descriptive because "
                "the families share four providers and benchmark construction."
            ),
            "claim_boundary": (
                "All test queries are absent from v5 and weights were frozen before "
                "execution, but providers, versions, tasks, memory bank, and checker "
                "are reused. This confirms query robustness, not independent provider "
                "or task-family generalization."
            ),
        },
    )
    lookup = {str(row["method"]): row for row in method_rows}
    comparisons = {str(row["contrast"]): row for row in comparison_rows}
    hgt = lookup["hgt_rank_ensemble"]
    matched = lookup["version_bm25_bundle"]
    query_only = lookup["corem_query_only"]
    versus_matched = comparisons[
        "hgt_rank_ensemble_minus_version_bm25_bundle"
    ]
    versus_query = comparisons["hgt_rank_ensemble_minus_corem_query_only"]
    summary = f"""# Fresh-Query Confirmation Summary

The frozen 0.9 HGT / 0.1 BM25 ensemble succeeds on
{int(hgt['verified_successes'])}/{len(event_ids)} events ({float(hgt['vsr']) * 100:.2f}%).
The candidate-matched BM25 closure reaches {float(matched['vsr']) * 100:.2f}%, and
query-only CoReM reaches {float(query_only['vsr']) * 100:.2f}%.

Against matched BM25, the ensemble difference is
{float(versus_matched['delta_vsr']) * 100:+.2f} points (task-family cluster 95% CI
[{float(versus_matched['family_cluster_ci_low']) * 100:.2f},
{float(versus_matched['family_cluster_ci_high']) * 100:.2f}]). Against query-only
CoReM it is {float(versus_query['delta_vsr']) * 100:+.2f} points (95% CI
[{float(versus_query['family_cluster_ci_low']) * 100:.2f},
{float(versus_query['family_cluster_ci_high']) * 100:.2f}]).

All 128 queries are absent from v5, and no confirmation outcome selected a model
or residual weight. The providers, task families, memories, and checker are reused,
so these results support fresh-query robustness only. The next experiment should
add independently sourced providers and task families.
"""
    (args.output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    write_checksums(
        args.output,
        (
            "SUMMARY.md",
            "method_metrics.csv",
            "paired_comparisons.csv",
            "per_family_metrics.csv",
            "per_service_metrics.csv",
            "result.json",
        ),
    )
    print(f"wrote confirmation summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
