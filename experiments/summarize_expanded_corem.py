#!/usr/bin/env python3
"""Validate and consolidate CoReM-only E1--E7 paper measurements."""

from __future__ import annotations

import argparse
import hashlib
import math
import statistics
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_json, load_jsonl, write_checksums, write_json  # noqa: E402


def verify(directory: Path) -> None:
    checksums = directory / "checksums.sha256"
    if not checksums.is_file():
        raise ValueError(f"missing checksums: {directory}")
    for line in checksums.read_text().splitlines():
        digest, filename = line.split("  ", 1)
        observed = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        if digest != observed:
            raise ValueError(f"checksum mismatch: {directory / filename}")


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def online_summary(directory: Path, expected_method: str) -> dict[str, Any]:
    verify(directory)
    result = load_json(directory / "result.json")
    if (
        result.get("status") != "complete_online_retention"
        or result.get("events") != 320
    ):
        raise ValueError(f"incomplete online result: {directory}")
    summaries = [
        row for row in result["summaries"] if row["method"] == expected_method
    ]
    if len(summaries) != 1 or summaries[0]["n"] != 320:
        raise ValueError(f"missing online method summary: {directory}")
    executions = [
        row
        for row in load_jsonl(directory / "execution_results.jsonl")
        if row["method"] == expected_method
    ]
    if len(executions) != 320:
        raise ValueError(f"online execution coverage mismatch: {directory}")
    if any(
        row.get("checker_independent") is not True
        or row.get("reset_verified") is not True
        or row.get("execution_error") is not None
        or row.get("checker_error") is not None
        for row in executions
    ):
        raise ValueError(f"unverified online execution: {directory}")
    summary = summaries[0]
    return {
        "method": expected_method,
        "retrieval_budget": int(result["retrieval_budget"]),
        "storage_budget": int(result["storage_budget"]),
        "vsr": float(summary["vsr"]),
        "bs": float(summary["bs"]),
        "ecr": float(summary["ecr"]),
        "churn": int(summary["total_activated_records"])
        + int(summary["total_evicted_records"]),
        "mean_update_ms": float(summary["mean_update_latency_ms"]),
        "p95_update_ms": float(summary["p95_update_latency_ms"]),
        "p95_selection_ms": p95(
            [float(row["selection_latency_ms"]) for row in executions]
        ),
        "physical_executions": int(result["physical_executions_this_run"]),
        "exact_cached_executions": int(result["exact_prior_executions_used"]),
        "source": str(directory.resolve()),
        "source_result_sha256": hashlib.sha256(
            (directory / "result.json").read_bytes()
        ).hexdigest(),
    }


def tau3_e2_summary(directory: Path) -> dict[str, Any]:
    """Extract the preregistered ten-seed set-utility comparison for E2."""

    verify(directory)
    analysis = load_json(directory / "final_analysis.json")
    if analysis.get("status") != "complete":
        raise ValueError("incomplete tau3 E2 analysis")
    dataset = analysis.get("dataset", {})
    if dataset.get("outcome_examples", {}).get("test") != 340:
        raise ValueError("unexpected tau3 E2 test coverage")
    if dataset.get("tasks", {}).get("test") != 43:
        raise ValueError("unexpected tau3 E2 task coverage")
    methods = (
        ("Full CoReM", "hgt_corem"),
        ("w/o non-additive set terms", "hgt_additive_encoder_matched"),
        ("w/o hyperedge message passing", "hgt_no_hyperedges"),
        ("w/o redundancy term", "hgt_no_redundancy"),
    )
    metrics = analysis.get("method_metrics", {})
    rows = []
    for display_name, method in methods:
        if method not in metrics:
            raise ValueError(f"missing tau3 E2 method: {method}")
        row = metrics[method]
        rows.append(
            {
                "variant": display_name,
                "method": method,
                "roc_auc_mean": float(row["roc_auc"]["mean"]),
                "roc_auc_sd": float(row["roc_auc"]["sample_std"]),
                "pairwise_accuracy_mean": float(
                    row["task_pairwise_accuracy"]["mean"]
                ),
                "pairwise_accuracy_sd": float(
                    row["task_pairwise_accuracy"]["sample_std"]
                ),
                "brier_mean": float(row["brier"]["mean"]),
                "brier_sd": float(row["brier"]["sample_std"]),
            }
        )
    return {
        "rows": rows,
        "test_candidate_sets": 340,
        "test_tasks": 43,
        "seeds": 10,
        "fresh_environment_replays": int(dataset["fresh_environment_replays"]),
        "primary_contrast": analysis["primary_hgt_corem_vs_additive_ablation"],
        "claim_boundary": str(analysis["claim_scope"]),
        "source": str(directory.resolve()),
        "source_analysis_sha256": hashlib.sha256(
            (directory / "final_analysis.json").read_bytes()
        ).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hsc", type=Path, required=True)
    parser.add_argument("--e2-tau3", type=Path, required=True)
    parser.add_argument("--e4-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--e5", type=Path, required=True)
    parser.add_argument("--e6", type=Path, required=True)
    parser.add_argument("--group-bm25", type=Path, required=True)
    parser.add_argument("--itemwise", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    for directory in (args.hsc, args.e2_tau3, args.e5, args.e6):
        verify(directory)
    hsc = load_json(args.hsc / "result.json")
    e5 = load_json(args.e5 / "result.json")
    e6 = load_json(args.e6 / "result.json")
    if hsc.get("status") != "complete_corem_only_e1_e2_e7":
        raise ValueError("incomplete HSC experiment suite")
    if e5.get("status") != "complete_corem_e5":
        raise ValueError("incomplete E5 experiment")
    if (
        e5.get("schema_version") != "corem-expanded-e5-v2-attested"
        or e5.get("split") != "test"
    ):
        raise ValueError("E5 must be the frozen attested test run")
    if e6.get("status") != "complete_corem_e6":
        raise ValueError("incomplete E6 experiment")

    e4 = [online_summary(directory, "group_aware_online") for directory in args.e4_dirs]
    keys = {(row["retrieval_budget"], row["storage_budget"]) for row in e4}
    expected = {(q, m) for q in (160, 200, 240) for m in (256, 640, 960)}
    if keys != expected or len(e4) != len(expected):
        raise ValueError(f"E4 budget grid mismatch: {sorted(keys)}")
    full = next(
        row for row in e4 if row["retrieval_budget"] == 160 and row["storage_budget"] == 640
    )
    group_bm25 = online_summary(args.group_bm25, "group_bm25_online")
    itemwise = online_summary(args.itemwise, "independent_relevance_online")
    e2 = tau3_e2_summary(args.e2_tau3)
    result = {
        "status": "complete_corem_only_expanded_tables",
        "schema_version": "corem-expanded-paper-results-v3",
        "scope": (
            "Only CoReM and its internal/hybrid diagnostic cells are populated; "
            "external baseline cells remain unmeasured."
        ),
        "e1": hsc["e1"],
        "e2": e2,
        "e3": [
            {"retrieval": "CoReM", "retention": "CoReM group", **full},
            {"retrieval": "CoReM", "retention": "Group BM25", **group_bm25},
            {"retrieval": "CoReM", "retention": "CoReM itemwise", **itemwise},
        ],
        "e4": sorted(
            e4, key=lambda row: (row["retrieval_budget"], row["storage_budget"])
        ),
        "e5": e5["summaries"],
        "e6": e6["summary"],
        "e7": hsc["e7"],
        "measurement_boundaries": {
            "e1_e7": hsc["claim_boundary"],
            "e2": (
                "Execution-verified candidate-set prediction on a controlled tau3 "
                "derivative with complete-but-failing counterexamples; this is not "
                "open-ended composition or natural production traffic."
            ),
            "e5": e5["claim_boundary"],
            "e6": e6["claim_boundary"],
            "e3_e4": (
                "Controlled resettable four-provider stream; exact prior executions "
                "are reused only under exact event/selection identity."
            ),
        },
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "result.json", result)
    write_checksums(args.output, ("result.json",))
    print("validated and consolidated CoReM E1--E7 measurements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
