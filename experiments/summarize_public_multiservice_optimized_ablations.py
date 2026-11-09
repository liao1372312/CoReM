#!/usr/bin/env python3
"""Summarize matched optimized HGT ablations on the fresh-query test set."""

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

from corem.real_eval import load_json, load_jsonl, write_checksums, write_json  # noqa: E402
from experiments.common import holm_adjust  # noqa: E402

FULL = "hgt_rank_ensemble"
ADDITIVE = "hgt_rank_ensemble_additive"
NO_EDGE = "hgt_rank_ensemble_no_hyperedges"
HEAD_OFF = "hgt_rank_ensemble_head_off"
RANKING_OFF = "hgt_rank_ensemble_ranking_off"
PURE = "hgt_rank_ensemble_pure"
ADDITIVE_PURE = "hgt_rank_ensemble_additive_pure"
NO_EDGE_PURE = "hgt_rank_ensemble_no_hyperedges_pure"

METHODS = (
    FULL,
    ADDITIVE,
    NO_EDGE,
    HEAD_OFF,
    RANKING_OFF,
    PURE,
    ADDITIVE_PURE,
    NO_EDGE_PURE,
)
DISPLAY = {
    FULL: "Full HGT + 0.1 BM25",
    ADDITIVE: "Retrained additive + 0.1 BM25",
    NO_EDGE: "Retrained no-edge + 0.1 BM25",
    HEAD_OFF: "Full weights, complement off + 0.1 BM25",
    RANKING_OFF: "Retrained ranking-loss-off + 0.1 BM25",
    PURE: "Full HGT, neural only",
    ADDITIVE_PURE: "Retrained additive, neural only",
    NO_EDGE_PURE: "Retrained no-edge, neural only",
}
CONTRASTS = (
    (FULL, ADDITIVE, True, "primary_complement"),
    (FULL, NO_EDGE, True, "primary_representation"),
    (FULL, HEAD_OFF, False, "diagnostic_head_off"),
    (FULL, RANKING_OFF, False, "diagnostic_ranking_loss"),
    (FULL, PURE, False, "diagnostic_residual"),
    (ADDITIVE, ADDITIVE_PURE, False, "diagnostic_additive_residual"),
    (NO_EDGE, NO_EDGE_PURE, False, "diagnostic_noedge_residual"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, action="append", required=True)
    parser.add_argument("--selection", type=Path, action="append", required=True)
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
    observations = wins + losses
    if observations == 0:
        return 1.0
    tail = sum(
        math.comb(observations, index)
        for index in range(min(wins, losses) + 1)
    )
    return min(1.0, 2.0 * tail / 2**observations)


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
    rows: list[dict[str, Any]] = []
    sources = []
    for analysis in args.analysis:
        result = load_json(analysis / "result.json")
        if result["status"] != "complete_real_track":
            raise ValueError(f"incomplete analysis: {analysis}")
        sources.append(
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
    by_method: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        method = str(row["method"])
        event_id = str(row["event_id"])
        if event_id in by_method[method]:
            raise ValueError(f"duplicate method/event row: {method}/{event_id}")
        by_method[method][event_id] = row
    event_ids = set(by_method[FULL])
    if any(set(by_method[method]) != event_ids for method in METHODS):
        raise ValueError("methods do not cover identical events")
    selections: dict[str, dict[str, tuple[str, ...]]] = defaultdict(dict)
    for path in args.selection:
        for row in load_jsonl(path):
            method = str(row["method"])
            event_id = str(row["event_id"])
            if method not in METHODS or event_id not in event_ids:
                continue
            if event_id in selections[method]:
                raise ValueError(f"duplicate selection: {method}/{event_id}")
            selections[method][event_id] = tuple(row["selected_memory_ids"])
    if set(selections) != set(METHODS) or any(
        set(selections[method]) != event_ids for method in METHODS
    ):
        raise ValueError("selection artifacts do not cover every method/event")

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
            grouped[family[event_id]].append(
                float(by_method[method][event_id]["verified_success"])
            )
        low, high = cluster_ci(grouped, args.bootstrap_draws, 202730 + method_index)
        successes = sum(
            int(by_method[method][event_id]["verified_success"])
            for event_id in event_ids
        )
        agreement = sum(
            selections[method][event_id] == selections[FULL][event_id]
            for event_id in event_ids
        ) / len(event_ids)
        method_rows.append(
            {
                "method": method,
                "display_name": DISPLAY[method],
                "events": len(event_ids),
                "task_version_families": len(grouped),
                "verified_successes": successes,
                "vsr": successes / len(event_ids),
                "family_cluster_ci_low": low,
                "family_cluster_ci_high": high,
                "bundle_agreement_with_full": agreement,
            }
        )
        by_service: dict[str, list[int]] = defaultdict(list)
        for event_id in event_ids:
            by_service[str(events[event_id]["service_id"])].append(
                int(by_method[method][event_id]["verified_success"])
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
    for contrast_index, (left, right, primary, role) in enumerate(CONTRASTS):
        grouped_differences: dict[tuple[str, ...], list[float]] = defaultdict(list)
        wins = losses = ties = agreements = 0
        for event_id in sorted(event_ids):
            difference = int(by_method[left][event_id]["verified_success"]) - int(
                by_method[right][event_id]["verified_success"]
            )
            grouped_differences[family[event_id]].append(float(difference))
            wins += difference > 0
            losses += difference < 0
            ties += difference == 0
            agreements += (
                selections[left][event_id] == selections[right][event_id]
            )
        low, high = cluster_ci(
            grouped_differences, args.bootstrap_draws, 202750 + contrast_index
        )
        family_means = [
            sum(observed) / len(observed)
            for observed in grouped_differences.values()
        ]
        family_wins = sum(value > 0 for value in family_means)
        family_losses = sum(value < 0 for value in family_means)
        comparison_rows.append(
            {
                "contrast": f"{left}_minus_{right}",
                "role": role,
                "primary": primary,
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
                "holm_family_p": None,
                "bundle_agreement": agreements / len(event_ids),
            }
        )
    primary_adjusted = holm_adjust(
        {
            str(row["contrast"]): float(row["family_exact_sign_p"])
            for row in comparison_rows
            if row["primary"]
        }
    )
    for row in comparison_rows:
        if row["primary"]:
            row["holm_family_p"] = primary_adjusted[str(row["contrast"])]

    args.output.mkdir(parents=True)
    write_csv(args.output / "method_metrics.csv", method_rows)
    write_csv(args.output / "paired_comparisons.csv", comparison_rows)
    write_csv(args.output / "per_service_metrics.csv", service_rows)
    write_csv(args.output / "per_family_metrics.csv", family_rows)
    write_json(
        args.output / "result.json",
        {
            "status": "complete_optimized_ablation_confirmation",
            "events": len(event_ids),
            "task_version_families": len(set(family.values())),
            "bootstrap_draws": args.bootstrap_draws,
            "method_metrics": method_rows,
            "paired_comparisons": comparison_rows,
            "sources": sources,
            "frozen_contract": (
                "All retrained variants use the same old-v5 train split, five seeds, "
                "architecture, optimization settings, candidate bundles, and frozen "
                "0.9 neural/0.1 BM25 residual. Pure-neural, head-off, and ranking-"
                "loss-off rows are predeclared diagnostics, not newly selected methods."
            ),
            "inference": (
                "Intervals resample the 16 service/version/task families. Family sign "
                "p-values for the two primary contrasts use Holm correction. A primary "
                "absolute difference below five points is practically small; an interval "
                "containing zero is treated as inconclusive."
            ),
            "claim_boundary": (
                "The no-edge row keeps the same pre-enumerated bundles, so it is a "
                "representation control rather than strict end-to-end search ablation."
            ),
        },
    )
    lookup = {str(row["method"]): row for row in method_rows}
    comparisons = {str(row["role"]): row for row in comparison_rows}
    complement = comparisons["primary_complement"]
    representation = comparisons["primary_representation"]
    ranking = comparisons["diagnostic_ranking_loss"]
    residual = comparisons["diagnostic_residual"]
    summary = f"""# Optimized Ablation Confirmation

The frozen full method reaches {int(lookup[FULL]['verified_successes'])}/{len(event_ids)}
({float(lookup[FULL]['vsr']) * 100:.2f}%) verified success. The matched retrained
additive variant reaches {float(lookup[ADDITIVE]['vsr']) * 100:.2f}% and the
candidate-matched no-edge representation control reaches
{float(lookup[NO_EDGE]['vsr']) * 100:.2f}%.

Full minus additive is {float(complement['delta_vsr']) * 100:+.2f} points (family
cluster 95% CI [{float(complement['family_cluster_ci_low']) * 100:.2f},
{float(complement['family_cluster_ci_high']) * 100:.2f}]; Holm-adjusted family sign
p={float(complement['holm_family_p']):.3f}). Full minus no-edge is
{float(representation['delta_vsr']) * 100:+.2f} points (95% CI
[{float(representation['family_cluster_ci_low']) * 100:.2f},
{float(representation['family_cluster_ci_high']) * 100:.2f}]; adjusted p=
{float(representation['holm_family_p']):.3f}). Neither primary contrast demonstrates
a complement or graph-edge gain on this confirmation set.

The pure full neural ensemble reaches {float(lookup[PURE]['vsr']) * 100:.2f}%; the
frozen 0.1 BM25 residual therefore contributes
{float(residual['delta_vsr']) * 100:+.2f} points in this diagnostic. The no-edge
experiment is candidate matched and must not be described as a strict search-space
ablation. Turning off ranking supervision reaches
{float(lookup[RANKING_OFF]['vsr']) * 100:.2f}%, or
{float(ranking['delta_vsr']) * 100:+.2f} full-minus-ranking-off points. All methods
were frozen before these executions.
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
    print(f"wrote optimized ablation summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
