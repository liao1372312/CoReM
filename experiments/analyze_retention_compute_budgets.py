#!/usr/bin/env python3
"""Build the matched anytime-maintenance budget table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


LABELS = {
    "bundle_value_retention": "Bundle-value retention",
    "corem_retention": "CoReM retention",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs=6, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-events", type=int, default=8)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def summarize(run: Path, calibration_events: int) -> dict[str, Any]:
    result = json.loads((run / "result.json").read_text())
    if result.get("status") != "complete_online_retention":
        raise ValueError(f"incomplete run: {run}")
    methods = result["methods"]
    if len(methods) != 1 or methods[0] not in LABELS:
        raise ValueError(f"expected one matched method: {run}")
    method = methods[0]
    budget = result.get("update_time_limit_ms")
    if budget is None:
        raise ValueError(f"run has no maintenance deadline: {run}")
    all_scored = load_jsonl(run / "scored_executions.jsonl")
    event_indices = {
        str(row["event_id"]): int(row["event_index"]) for row in all_scored
    }
    scored = [
        row for row in all_scored if int(row["event_index"]) > calibration_events
    ]
    executions = [
        row
        for row in load_jsonl(run / "execution_results.jsonl")
        if event_indices[str(row["event_id"])] > calibration_events
    ]
    banks = [
        row
        for row in load_jsonl(run / "bank_history.jsonl")
        if int(row["event_index"]) > calibration_events
    ]
    if not (len(scored) == len(executions) == len(banks) == 320 - calibration_events):
        raise ValueError(f"unexpected evaluation-event count: {run}")
    latencies = [float(row["update_latency_ms"]) for row in banks]
    return {
        "time_limit_ms": float(budget),
        "method": method,
        "label": LABELS[method],
        "evaluation_events": len(scored),
        "vsr": mean(float(row["verified_success"]) for row in scored),
        "retrievable_complete_evidence_rate": mean(
            row["retrieval_status"] == "ok" for row in executions
        ),
        "mean_update_latency_ms": mean(latencies),
        "p95_update_latency_ms": percentile(latencies, 0.95),
        "mean_churn_records_per_update": mean(
            int(row["activated_records"]) + int(row["evicted_records"])
            for row in banks
        ),
        "timeout_rate": mean(bool(row["update_timed_out"]) for row in banks),
        "source": str(run),
        "result_sha256": hashlib.sha256((run / "result.json").read_bytes()).hexdigest(),
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")
    rows = [summarize(run, args.calibration_events) for run in args.runs]
    keys = {(row["time_limit_ms"], row["method"]) for row in rows}
    budgets = sorted({row["time_limit_ms"] for row in rows})
    expected = {(budget, method) for budget in budgets for method in LABELS}
    if len(budgets) != 3 or keys != expected:
        raise ValueError("runs do not form a complete 3 x 2 budget grid")
    rows.sort(key=lambda row: (row["time_limit_ms"], row["method"]))
    args.output.mkdir(parents=True)
    fields = [
        "time_limit_ms",
        "label",
        "evaluation_events",
        "vsr",
        "retrievable_complete_evidence_rate",
        "mean_update_latency_ms",
        "p95_update_latency_ms",
        "mean_churn_records_per_update",
        "timeout_rate",
        "source",
        "result_sha256",
    ]
    with (args.output / "table4_compute_budget.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tex = [
        r"\begin{tabular}{rlrrrrrr}",
        r"\toprule",
        r"Limit & Method & VSR & RCE & Mean ms & p95 ms & Churn & Timeout \\",
        r"\midrule",
    ]
    for row in rows:
        tex.append(
            f"{row['time_limit_ms']:.0f} ms & {row['label']} & "
            f"{100*row['vsr']:.1f} & "
            f"{100*row['retrievable_complete_evidence_rate']:.1f} & "
            f"{row['mean_update_latency_ms']:.1f} & "
            f"{row['p95_update_latency_ms']:.1f} & "
            f"{row['mean_churn_records_per_update']:.2f} & "
            f"{100*row['timeout_rate']:.1f} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (args.output / "table4_compute_budget.tex").write_text(
        "\n".join(tex) + "\n"
    )
    result = {
        "status": "complete",
        "rows": rows,
        "calibration_contract": (
            f"T1/T2/T3 were fixed as {budgets} ms from the first "
            f"{args.calibration_events}-event development prefix; table metrics "
            "exclude that prefix while retaining its online bank warm-up."
        ),
        "timing_contract": (
            "Each cell ran in a separate cold process on CPU cores 0-11 with one "
            "Torch thread and identical deterministic model/caching configuration. "
            "Deadlines are checked at request-scoring boundaries; each in-flight "
            "request score is atomic, so "
            "actual latency and timeout rate are reported."
        ),
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote compute-budget table to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
