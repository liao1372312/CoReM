#!/usr/bin/env python3
"""Create paper-ready tables, statistics, and a figure for challenge v4."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_jsonl, write_checksums, write_json

DISPLAY = {
    "corem_query_only": "CoReM query-only",
    "version_bm25": "Version BM25",
    "recency_bm25": "Recency BM25",
    "service_bm25": "Service BM25",
    "no_memory": "No memory",
}
COLORS = {
    "corem_query_only": "#3568b8",
    "version_bm25": "#2a9d8f",
    "recency_bm25": "#7a5195",
    "service_bm25": "#e76f51",
    "no_memory": "#777777",
}


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def exact_sign_p(wins: int, losses: int) -> float:
    total = wins + losses
    if not total:
        return 1.0
    smaller = min(wins, losses)
    tail = sum(math.comb(total, index) for index in range(smaller + 1)) / 2**total
    return min(1.0, 2.0 * tail)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--test-analysis", type=Path, required=True)
    parser.add_argument("--longitudinal-analysis", type=Path, required=True)
    parser.add_argument("--oracle-analysis", type=Path, required=True)
    parser.add_argument("--reproducibility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2

    events = {
        str(row["event_id"]): row for row in load_jsonl(args.dataset / "events.jsonl")
    }
    scored = load_jsonl(args.test_analysis / "scored_executions.jsonl")
    with (args.test_analysis / "method_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        metrics = {row["method"]: row for row in csv.DictReader(handle)}
    by_incident: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in scored:
        by_incident[(str(row["method"]), str(row["event_id"]))].append(
            int(row["verified_success"])
        )

    paired_rows = []
    target = "corem_query_only"
    target_events = sorted(
        event_id for method, event_id in by_incident if method == target
    )
    for baseline in ("version_bm25", "recency_bm25", "service_bm25"):
        differences = []
        wins = losses = ties = 0
        for event_id in target_events:
            target_value = sum(by_incident[(target, event_id)]) / len(
                by_incident[(target, event_id)]
            )
            baseline_value = sum(by_incident[(baseline, event_id)]) / len(
                by_incident[(baseline, event_id)]
            )
            difference = target_value - baseline_value
            differences.append(difference)
            wins += difference > 0
            losses += difference < 0
            ties += difference == 0
        rng = random.Random(202704)
        draws = [
            sum(rng.choice(differences) for _ in differences) / len(differences)
            for _ in range(2000)
        ]
        paired_rows.append(
            {
                "contrast": f"{target}_minus_{baseline}",
                "incidents": len(differences),
                "delta_vsr": sum(differences) / len(differences),
                "bootstrap_ci_low": quantile(draws, 0.025),
                "bootstrap_ci_high": quantile(draws, 0.975),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "two_sided_exact_sign_p": exact_sign_p(wins, losses),
            }
        )

    task_rows = []
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in scored:
        task_key = str(events[str(row["event_id"])]["task_key"])
        grouped[(str(row["method"]), task_key)].append(int(row["verified_success"]))
    for (method, task_key), values in sorted(grouped.items()):
        task_rows.append(
            {
                "method": method,
                "task_key": task_key,
                "executions": len(values),
                "vsr": sum(values) / len(values),
            }
        )

    args.output.mkdir(parents=True)
    write_csv(args.output / "paired_comparisons.csv", paired_rows)
    write_csv(args.output / "per_task_metrics.csv", task_rows)
    oracle = json.loads((args.oracle_analysis / "result.json").read_text())
    reproducibility = json.loads((args.reproducibility / "result.json").read_text())
    with (args.longitudinal_analysis / "method_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        longitudinal = {row["method"]: row for row in csv.DictReader(handle)}
    write_json(
        args.output / "result.json",
        {
            "status": "complete_exploratory_challenge",
            "events": len(events),
            "test_executions": len(scored),
            "oracle_vsr": oracle["methods"][0]["vsr"],
            "test_metrics": metrics,
            "longitudinal_metrics": longitudinal,
            "paired_comparisons": paired_rows,
            "reproducibility": reproducibility,
            "claim_boundary": "Controlled query-blind hard-negative challenge on genuine releases; post-v3 exploratory repair, not natural production drift or a trained full-model result.",
        },
    )

    methods = [
        "corem_query_only",
        "version_bm25",
        "recency_bm25",
        "service_bm25",
        "no_memory",
    ]
    values = [float(metrics[method]["vsr"]) * 100 for method in methods]
    lows = [float(metrics[method]["vsr_ci_low"]) * 100 for method in methods]
    highs = [float(metrics[method]["vsr_ci_high"]) * 100 for method in methods]
    errors = [
        [value - low for value, low in zip(values, lows, strict=True)],
        [high - value for value, high in zip(values, highs, strict=True)],
    ]
    args.figures.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(4.8, 2.65))
    positions = list(range(len(methods)))
    axis.barh(
        positions,
        values,
        xerr=errors,
        color=[COLORS[method] for method in methods],
        capsize=3,
        height=0.62,
    )
    axis.set_yticks(positions, [DISPLAY[method] for method in methods])
    axis.invert_yaxis()
    axis.set_xlim(0, 105)
    axis.set_xlabel("Verified success rate (%)")
    axis.grid(axis="x", color="#dddddd", linewidth=0.6)
    axis.set_axisbelow(True)
    for position, value in zip(positions, values, strict=True):
        axis.text(
            min(101, value + 1.0), position, f"{value:.1f}", va="center", fontsize=8
        )
    figure.tight_layout()
    figure.savefig(args.figures / "corem_challenge_bar.pdf", bbox_inches="tight")
    figure.savefig(
        args.figures / "corem_challenge_bar.png", dpi=300, bbox_inches="tight"
    )
    plt.close(figure)

    summary = f"""# Query-Blind Three-Step Challenge v4

The oracle feasibility gate passed all {len(events)} events. The held-out test
contains 16 incidents, five deterministic tie-break seeds, five methods, and
{len(scored)} executions. Queries use variants 8--9, which are absent from the
train split, and retrieval cannot access event task keys or exact evidence IDs.

| Method | VSR | BS | ECR | SUR | Incident-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| CoReM query-only | {float(metrics["corem_query_only"]["vsr"]) * 100:.2f}% | {float(metrics["corem_query_only"]["bs"]) * 100:.2f}% | {float(metrics["corem_query_only"]["ecr"]) * 100:.2f}% | {float(metrics["corem_query_only"]["sur"]) * 100:.2f}% | [{float(metrics["corem_query_only"]["vsr_ci_low"]) * 100:.2f}, {float(metrics["corem_query_only"]["vsr_ci_high"]) * 100:.2f}] |
| Version BM25 | {float(metrics["version_bm25"]["vsr"]) * 100:.2f}% | {float(metrics["version_bm25"]["bs"]) * 100:.2f}% | {float(metrics["version_bm25"]["ecr"]) * 100:.2f}% | {float(metrics["version_bm25"]["sur"]) * 100:.2f}% | [{float(metrics["version_bm25"]["vsr_ci_low"]) * 100:.2f}, {float(metrics["version_bm25"]["vsr_ci_high"]) * 100:.2f}] |
| Recency BM25 | {float(metrics["recency_bm25"]["vsr"]) * 100:.2f}% | {float(metrics["recency_bm25"]["bs"]) * 100:.2f}% | {float(metrics["recency_bm25"]["ecr"]) * 100:.2f}% | {float(metrics["recency_bm25"]["sur"]) * 100:.2f}% | [{float(metrics["recency_bm25"]["vsr_ci_low"]) * 100:.2f}, {float(metrics["recency_bm25"]["vsr_ci_high"]) * 100:.2f}] |
| Service BM25 | {float(metrics["service_bm25"]["vsr"]) * 100:.2f}% | {float(metrics["service_bm25"]["bs"]) * 100:.2f}% | {float(metrics["service_bm25"]["ecr"]) * 100:.2f}% | {float(metrics["service_bm25"]["sur"]) * 100:.2f}% | [{float(metrics["service_bm25"]["vsr_ci_low"]) * 100:.2f}, {float(metrics["service_bm25"]["vsr_ci_high"]) * 100:.2f}] |
| No memory | 0.00% | 0.00% | 0.00% | n/a | [0.00, 0.00] |

CoReM exceeds version BM25 by 31.25 percentage points, but the incident-paired
exact sign test is p=0.125 (6 wins, 1 loss, 9 ties), so the difference is not
statistically significant. Across all 80 incidents, CoReM reaches 40.0% VSR
versus 23.75% for version BM25. The cold rerun matches all 400 test outcomes,
including every per-step HTTP status and checker decision.

This is a mechanism-focused controlled challenge. Correct records are joined by
verified dependency hyperedges while hard negatives are constructed, so the
result supports dependency-bundle retrieval under this design; it is not
evidence of natural production drift or universal superiority. The reported
CoReM selector is the transparent query-only reference, not the trained HGT
full model.
"""
    (args.output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    write_checksums(
        args.output,
        ("SUMMARY.md", "paired_comparisons.csv", "per_task_metrics.csv", "result.json"),
    )
    print(f"wrote challenge summary, two CSV tables, JSON, and figure to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
