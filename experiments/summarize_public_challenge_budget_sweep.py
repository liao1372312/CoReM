#!/usr/bin/env python3
"""Validate and summarize the frozen v4 retrieval-budget robustness sweep."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_jsonl, write_checksums, write_json
from experiments.generate_public_challenge_selections import METHODS

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


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def artifact_paths(
    root: Path, sweep_root: Path, budget: int, reference_budget: int
) -> tuple[Path, Path, Path]:
    if budget == reference_budget:
        selection_dir = root / "experiments/public_challenge_v4_selections"
    else:
        selection_dir = (
            root / f"experiments/public_challenge_v4_budget_{budget}_selections"
        )
    return (
        selection_dir,
        sweep_root / f"budget_{budget:03d}" / "executions",
        sweep_root / f"budget_{budget:03d}" / "analysis",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--primary-execution", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[80, 120, 160, 200, 240]
    )
    parser.add_argument("--reference-budget", type=int, default=160)
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    budgets = sorted(set(args.budgets))
    if budgets != sorted(args.budgets) or args.reference_budget not in budgets:
        print("budgets must be unique, sorted, and contain the reference budget")
        return 2

    assignments = load_jsonl(args.splits)
    test_ids = {
        str(row["event_id"]) for row in assignments if row.get("split") == "test"
    }
    root = REPOSITORY_ROOT
    metrics: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    reference_executions: list[dict[str, Any]] = []
    initial_vsr_mismatches: list[str] = []
    for budget in budgets:
        selection_dir, execution_dir, analysis_dir = artifact_paths(
            root, args.sweep_root, budget, args.reference_budget
        )
        selection_manifest = json.loads(
            (selection_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if int(selection_manifest["token_budget"]) != budget:
            raise ValueError(f"budget {budget} selection manifest disagrees")
        selections = [
            row
            for row in load_jsonl(selection_dir / "selections.jsonl")
            if str(row["event_id"]) in test_ids
        ]
        executions = load_jsonl(execution_dir / "execution_results.jsonl")
        selection_keys = {
            (str(row["method"]), int(row["seed"]), str(row["event_id"]))
            for row in selections
        }
        execution_keys = {
            (str(row["method"]), int(row["seed"]), str(row["event_id"]))
            for row in executions
        }
        budget_violations = sum(int(row["token_cost"]) > budget for row in selections)
        runner_errors = sum(
            row.get("execution_error") is not None for row in executions
        )
        checker_errors = sum(row.get("checker_error") is not None for row in executions)
        expected_rows = len(METHODS) * 5 * len(test_ids)
        if (
            len(selections) != expected_rows
            or len(selection_keys) != expected_rows
            or len(executions) != expected_rows
            or len(execution_keys) != expected_rows
            or selection_keys != execution_keys
            or budget_violations
            or runner_errors
            or checker_errors
        ):
            raise RuntimeError(f"budget {budget} failed the completeness gate")
        analysis = json.loads(
            (analysis_dir / "result.json").read_text(encoding="utf-8")
        )
        method_metrics = {row["method"]: row for row in analysis["methods"]}
        if set(method_metrics) != set(METHODS):
            raise RuntimeError(f"budget {budget} has an incomplete method table")
        initial_analysis_dir = (
            root / "experiments/results_public_challenge_v4_test_analysis"
            if budget == args.reference_budget
            else root
            / f"experiments/results_public_challenge_v4_budget_{budget}_test_analysis"
        )
        initial_analysis = json.loads(
            (initial_analysis_dir / "result.json").read_text(encoding="utf-8")
        )
        initial_by_method = {row["method"]: row for row in initial_analysis["methods"]}
        for method in METHODS:
            if float(method_metrics[method]["vsr"]) != float(
                initial_by_method[method]["vsr"]
            ):
                initial_vsr_mismatches.append(f"{budget}:{method}")
        for method in METHODS:
            row = method_metrics[method]
            metrics.append(
                {
                    "budget": budget,
                    "method": method,
                    "n": row["n"],
                    "incidents": row["incidents"],
                    "vsr": row["vsr"],
                    "vsr_ci_low": row["vsr_ci_low"],
                    "vsr_ci_high": row["vsr_ci_high"],
                    "bs": row["bs"],
                    "ecr": row["ecr"],
                    "sur": row["sur"],
                    "mean_tokens": row["mean_tokens"],
                }
            )
        gate_rows.append(
            {
                "budget": budget,
                "selection_rows": len(selections),
                "execution_rows": len(executions),
                "unique_keys": len(execution_keys),
                "budget_violations": budget_violations,
                "runner_errors": runner_errors,
                "checker_errors": checker_errors,
            }
        )
        if budget == args.reference_budget:
            reference_executions = executions

    primary_rows = load_jsonl(args.primary_execution)
    primary_by_key = {
        (str(row["method"]), int(row["seed"]), str(row["event_id"])): row
        for row in primary_rows
    }
    reference_by_key = {
        (str(row["method"]), int(row["seed"]), str(row["event_id"])): row
        for row in reference_executions
    }
    primary_checker_mismatches = [
        key
        for key in sorted(primary_by_key)
        if key not in reference_by_key
        or primary_by_key[key].get("selected_memory_ids")
        != reference_by_key[key].get("selected_memory_ids")
        or primary_by_key[key].get("checker") != reference_by_key[key].get("checker")
    ]
    if initial_vsr_mismatches or primary_checker_mismatches:
        raise RuntimeError(
            "normalized runner changed a scored outcome or the frozen 160-token check"
        )

    profile = json.loads((args.profile / "result.json").read_text(encoding="utf-8"))
    if (
        profile.get("status") != "passed"
        or int(profile["budget"]) != args.reference_budget
    ):
        raise RuntimeError("retrieval profile did not pass at the reference budget")
    profile_by_method = {row["method"]: row for row in profile["methods"]}
    reference_metric = {
        str(row["method"]): row
        for row in metrics
        if int(row["budget"]) == args.reference_budget
    }
    execution_by_method: dict[str, list[float]] = {method: [] for method in METHODS}
    for row in reference_executions:
        execution_by_method[str(row["method"])].append(
            float(row["execution_latency_ms"])
        )
    efficiency: list[dict[str, Any]] = []
    for method in METHODS:
        values = execution_by_method[method]
        profiled = profile_by_method[method]
        efficiency.append(
            {
                "method": method,
                "vsr": reference_metric[method]["vsr"],
                "retrieval_median_ms": profiled["median_latency_ms"],
                "retrieval_p95_ms": profiled["p95_latency_ms"],
                "execution_median_ms": statistics.median(values),
                "execution_p95_ms": percentile(values, 0.95),
                "profile_measurements": profiled["measurements"],
                "offline_training_included": False,
            }
        )

    args.output.mkdir(parents=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "budget_metrics.csv", metrics)
    write_csv(args.output / "completeness_gates.csv", gate_rows)
    write_csv(args.output / "efficiency_metrics.csv", efficiency)

    order = [
        "corem_query_only",
        "version_bm25",
        "recency_bm25",
        "service_bm25",
        "no_memory",
    ]
    fig, ax = plt.subplots(figsize=(7.1, 4.1))
    for method in order:
        rows = [row for row in metrics if row["method"] == method]
        ax.plot(
            [row["budget"] for row in rows],
            [100.0 * float(row["vsr"]) for row in rows],
            marker="o",
            linewidth=2,
            color=COLORS[method],
            label=DISPLAY[method],
        )
    ax.axvline(args.reference_budget, color="#444444", linestyle="--", linewidth=1)
    ax.set_xlabel("Retrieval budget (tokens)")
    ax.set_ylabel("Verified success rate (%)")
    ax.set_ylim(-2, 102)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.figures / "corem_challenge_budget.pdf", bbox_inches="tight")
    fig.savefig(
        args.figures / "corem_challenge_budget.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    for method in order:
        row = next(item for item in efficiency if item["method"] == method)
        x_value = max(0.001, float(row["retrieval_p95_ms"]))
        y_value = 100.0 * float(row["vsr"])
        ax.scatter(
            x_value,
            y_value,
            s=65,
            color=COLORS[method],
            label=DISPLAY[method],
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Complete retrieval p95 latency (ms, log scale)")
    ax.set_ylabel("Verified success rate (%)")
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.figures / "corem_challenge_efficiency.pdf", bbox_inches="tight")
    fig.savefig(
        args.figures / "corem_challenge_efficiency.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    result = {
        "status": "passed",
        "scope": "post-hoc frozen-selector retrieval-budget robustness",
        "budgets": budgets,
        "reference_budget": args.reference_budget,
        "test_incidents": len(test_ids),
        "gate_rows": gate_rows,
        "initial_vsr_mismatches": initial_vsr_mismatches,
        "primary_checker_mismatches": len(primary_checker_mismatches),
        "budget_metrics": metrics,
        "efficiency_metrics": efficiency,
        "claim_boundary": "One-dimensional retrieval-token sensitivity on the controlled v4 challenge; not a storage-budget or natural-production result.",
    }
    write_json(args.output / "result.json", result)
    corem_rows = [row for row in metrics if row["method"] == "corem_query_only"]
    version_rows = [row for row in metrics if row["method"] == "version_bm25"]
    summary = """# Challenge v4 Retrieval-Budget Robustness

All declared budget points pass the 400-row completeness gate with no budget,
runner, or checker violations. Selectors, checker, runner, data, and seeds are
frozen; the 160-token point matches every checker decision in the
cold-rerun-verified primary artifact.

| Budget | CoReM query-only VSR | Version BM25 VSR |
|---:|---:|---:|
"""
    for corem, version in zip(corem_rows, version_rows, strict=True):
        summary += (
            f"| {corem['budget']} | {100 * float(corem['vsr']):.2f}% | "
            f"{100 * float(version['vsr']):.2f}% |\n"
        )
    summary += """

This is an exploratory, one-dimensional retrieval-token sweep over already
inspected controlled test incidents. It does not measure the active-memory
storage budget or production traffic. Efficiency uses a dedicated complete
query-to-selection profile and reports service execution separately; offline
training is excluded.
"""
    (args.output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    write_checksums(
        args.output,
        (
            "SUMMARY.md",
            "budget_metrics.csv",
            "completeness_gates.csv",
            "efficiency_metrics.csv",
            "result.json",
        ),
    )
    print("wrote validated budget and efficiency summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
