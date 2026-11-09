#!/usr/bin/env python3
"""Export every data-bearing figure in the current paper as tabular data."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "generated" / "figure_tables"

DISPLAY = {
    "corem_query_only": "CoReM query-only",
    "version_bm25": "Version BM25",
    "recency_bm25": "Recency BM25",
    "service_bm25": "Service BM25",
    "no_memory": "No memory",
    "group_aware_retention": "Group-aware retention",
    "independent_relevance": "Independent relevance",
    "item_recency": "Item recency",
    "group_aware_online": "Group-aware",
    "independent_relevance_online": "Independent relevance",
    "lru_online": "LRU",
    "lfu_online": "LFU",
    "hgt_rolling_ensemble": "HGT/BM25 ensemble",
    "hgt_leave_provider_out": "HGT leave-provider-out",
    "version_bm25_bundle": "BM25 + closure",
    "hgt_corem": "HGT CoReM",
    "hgt_additive": "HGT additive",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {name}")
    path = OUTPUT / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: str | float | int) -> float:
    return round(100.0 * float(value), 6)


def export_retrieval_budget() -> None:
    source = (
        ROOT
        / "experiments/results_public_challenge_v4_budget_summary/budget_metrics.csv"
    )
    rows = []
    for row in read_csv(source):
        rows.append(
            {
                "retrieval_budget_tokens": int(row["budget"]),
                "method": row["method"],
                "display": DISPLAY[row["method"]],
                "verified_success_rate_pct": pct(row["vsr"]),
                "vsr_ci_low_pct": pct(row["vsr_ci_low"]),
                "vsr_ci_high_pct": pct(row["vsr_ci_high"]),
                "executions": int(row["n"]),
                "independent_incidents": int(row["incidents"]),
                "mean_retrieved_tokens": float(row["mean_tokens"]),
            }
        )
    write_csv("fig_retrieval_budget.csv", rows)


def export_storage_budget() -> None:
    source = (
        ROOT
        / "experiments/results_public_challenge_v4_retention_summary/storage_metrics.csv"
    )
    rows = []
    for row in read_csv(source):
        rows.append(
            {
                "storage_budget_cost_units": int(row["storage_budget"]),
                "method": row["method"],
                "display": DISPLAY[row["method"]],
                "verified_success_rate_pct": pct(row["vsr"]),
                "vsr_ci_low_pct": pct(row["vsr_ci_low"]),
                "vsr_ci_high_pct": pct(row["vsr_ci_high"]),
                "executions": int(row["n"]),
                "independent_incidents": int(row["incidents"]),
                "mean_bank_cost": float(row["mean_bank_cost"]),
                "mean_retained_items": float(row["mean_retained_items"]),
                "mean_complete_groups": float(row["mean_complete_groups"]),
            }
        )
    write_csv("fig_storage_budget.csv", rows)


def export_efficiency() -> None:
    source = (
        ROOT
        / "experiments/results_public_challenge_v4_budget_summary/efficiency_metrics.csv"
    )
    rows = []
    for row in read_csv(source):
        rows.append(
            {
                "method": row["method"],
                "display": DISPLAY[row["method"]],
                "verified_success_rate_pct": pct(row["vsr"]),
                "retrieval_median_ms": float(row["retrieval_median_ms"]),
                "retrieval_p95_ms": float(row["retrieval_p95_ms"]),
                "execution_median_ms": float(row["execution_median_ms"]),
                "execution_p95_ms": float(row["execution_p95_ms"]),
                "retrieval_profile_measurements": int(row["profile_measurements"]),
                "offline_training_included": row["offline_training_included"],
            }
        )
    write_csv("fig_efficiency.csv", rows)


def export_rolling_lpo() -> None:
    base = ROOT / "experiments/results_longitudinal_extensions_v5_strict_semantic_cache"
    rolling_rows = []
    labels = {"1": "late old", "2": "early new", "3": "late new"}
    for row in read_csv(base / "rolling_window_metrics.csv"):
        rolling_rows.append(
            {
                "window": int(row["window"]),
                "window_label": labels[row["window"]],
                "test_indices_per_service": row["test_indices_per_service"],
                "method": row["method"],
                "display": row["display"],
                "verified_success_rate_pct": pct(row["vsr"]),
                "vsr_ci_low_pct": pct(row["vsr_ci_low"]),
                "vsr_ci_high_pct": pct(row["vsr_ci_high"]),
                "events": int(row["events"]),
                "executions": int(row["executions"]),
                "seeds": int(row["seeds"]),
            }
        )
    write_csv("fig_rolling_windows.csv", rolling_rows)

    lpo_rows = []
    for row in read_csv(base / "leave_provider_out_metrics.csv"):
        if row["provider_id"] == "ALL":
            continue
        lpo_rows.append(
            {
                "held_out_provider": row["provider_id"],
                "method": row["method"],
                "display": row["display"],
                "verified_success_rate_pct": pct(row["vsr"]),
                "vsr_ci_low_pct": pct(row["vsr_ci_low"]),
                "vsr_ci_high_pct": pct(row["vsr_ci_high"]),
                "events": int(row["events"]),
                "executions": int(row["executions"]),
                "seeds": int(row["seeds"]),
            }
        )
    write_csv("fig_leave_provider_out.csv", lpo_rows)


def export_online() -> None:
    source = ROOT / "experiments/results_online_retention_v5_b640_strict_semantic_cache"
    methods = (
        "group_aware_online",
        "independent_relevance_online",
        "lru_online",
        "lfu_online",
    )
    scored = read_jsonl(source / "scored_executions.jsonl")
    cumulative_rows = []
    for method in methods:
        members = sorted(
            (row for row in scored if row["method"] == method),
            key=lambda row: int(row["event_index"]),
        )
        successes = 0
        for incident, row in enumerate(members, start=1):
            successes += int(row["verified_success"])
            cumulative_rows.append(
                {
                    "chronological_incident": incident,
                    "event_index": int(row["event_index"]),
                    "event_id": row["event_id"],
                    "method": method,
                    "display": DISPLAY[method],
                    "event_verified_success": int(row["verified_success"]),
                    "cumulative_successes": successes,
                    "cumulative_vsr_pct": round(100.0 * successes / incident, 6),
                }
            )
    write_csv("fig_online_retention_cumulative_vsr.csv", cumulative_rows)

    banks = read_jsonl(source / "bank_history.jsonl")
    churn_rows = []
    for method in methods:
        members = sorted(
            (row for row in banks if row["method"] == method),
            key=lambda row: int(row["event_index"]),
        )
        churn = [
            int(row["activated_records"]) + int(row["evicted_records"])
            for row in members
        ]
        for start in range(0, len(churn), 20):
            churn_rows.append(
                {
                    "arrival_window": start // 20 + 1,
                    "incident_start": start + 1,
                    "incident_end": min(start + 20, len(churn)),
                    "method": method,
                    "display": DISPLAY[method],
                    "activations_plus_evictions": sum(churn[start : start + 20]),
                }
            )
    write_csv("fig_online_retention_churn.csv", churn_rows)


def export_release_drift() -> None:
    source = (
        ROOT
        / "experiments/results_longitudinal_extensions_v5_strict_semantic_cache/public_release_drift.csv"
    )
    raw = read_csv(source)
    methods = (
        "group_aware_online",
        "independent_relevance_online",
        "lru_online",
        "lfu_online",
    )
    phases = (
        ("pre_change_late", "pre-change"),
        ("post_change_early", "early post"),
        ("post_change_late", "late post"),
    )
    rows = []
    for method in methods:
        for phase, label in phases:
            members = [
                row
                for row in raw
                if row["method"] == method and row["phase"] == phase
            ]
            rows.append(
                {
                    "phase": phase,
                    "phase_label": label,
                    "method": method,
                    "display": DISPLAY[method],
                    "mean_vsr_across_four_releases_pct": round(
                        100.0 * mean(float(row["vsr"]) for row in members), 6
                    ),
                    "providers": len(members),
                    "events_per_provider": int(members[0]["events"]),
                    "total_events": sum(int(row["events"]) for row in members),
                }
            )
    write_csv("fig_public_release_drift.csv", rows)

    detailed = []
    for row in raw:
        if row["method"] not in methods:
            continue
        detailed.append(
            {
                "phase": row["phase"],
                "method": row["method"],
                "display": DISPLAY[row["method"]],
                "service_id": row["service_id"],
                "events": int(row["events"]),
                "verified_success_rate_pct": pct(row["vsr"]),
            }
        )
    write_csv("fig_public_release_drift_by_service.csv", detailed)


def export_multiservice() -> None:
    dataset = ROOT / "data/public_multiservice_challenge_v5/events.jsonl"
    events = {row["event_id"]: row for row in read_jsonl(dataset)}
    sources = (
        ROOT
        / "experiments/results_public_multiservice_challenge_v5_matched_baseline_test_analysis/scored_executions.jsonl",
        ROOT
        / "experiments/results_public_multiservice_challenge_v5_test_analysis/scored_executions.jsonl",
        ROOT
        / "experiments/results_public_multiservice_challenge_v5_hgt_final_test_analysis/scored_executions.jsonl",
    )
    selected = {
        "version_bm25_bundle",
        "corem_query_only",
        "hgt_corem",
        "hgt_additive",
    }
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    counts: dict[tuple[str, str], set[str]] = defaultdict(set)
    seeds: dict[tuple[str, str], set[int]] = defaultdict(set)
    for source in sources:
        for row in read_jsonl(source):
            method = str(row["method"])
            if method not in selected:
                continue
            service = str(events[str(row["event_id"])]["service_id"])
            key = (method, service)
            grouped[key].append(float(row["verified_success"]))
            counts[key].add(str(row["event_id"]))
            seeds[key].add(int(row["seed"]))
    method_order = (
        "version_bm25_bundle",
        "corem_query_only",
        "hgt_corem",
        "hgt_additive",
    )
    service_order = ("json-server", "go-httpbin", "wiremock", "mockserver")
    rows = []
    for method in method_order:
        for service in service_order:
            key = (method, service)
            rows.append(
                {
                    "service": service,
                    "method": method,
                    "display": DISPLAY[method],
                    "verified_success_rate_pct": round(
                        100.0 * mean(grouped[key]), 6
                    ),
                    "events": len(counts[key]),
                    "executions": len(grouped[key]),
                    "seeds": len(seeds[key]),
                }
            )
    write_csv("fig_multiservice_v5.csv", rows)


def markdown_table(
    headers: tuple[str, ...], rows: Iterable[tuple[Any, ...]]
) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return lines


def write_readme() -> None:
    budget = read_csv(OUTPUT / "fig_retrieval_budget.csv")
    storage = read_csv(OUTPUT / "fig_storage_budget.csv")
    efficiency = read_csv(OUTPUT / "fig_efficiency.csv")
    rolling = read_csv(OUTPUT / "fig_rolling_windows.csv")
    lpo = read_csv(OUTPUT / "fig_leave_provider_out.csv")
    drift = read_csv(OUTPUT / "fig_public_release_drift.csv")
    multiservice = read_csv(OUTPUT / "fig_multiservice_v5.csv")
    online = read_csv(OUTPUT / "fig_online_retention_cumulative_vsr.csv")
    churn = read_csv(OUTPUT / "fig_online_retention_churn.csv")

    latest_online = [row for row in online if row["chronological_incident"] == "320"]
    churn_totals: dict[str, int] = defaultdict(int)
    for row in churn:
        churn_totals[row["method"]] += int(row["activations_plus_evictions"])

    lines = [
        "# Current-paper experimental figure data",
        "",
        "These tables reproduce all seven data-bearing experimental figures referenced by `CoReM_WWW2027_Main.tex`. The method overview is a conceptual diagram and has no measurement table. Percentages are reported on a 0--100 scale. Full curve points and supporting fields are in the linked CSV files.",
        "",
        "## Online retention",
        "",
        "Full 320-incident curves: [cumulative VSR](fig_online_retention_cumulative_vsr.csv); [20-incident churn windows](fig_online_retention_churn.csv).",
        "",
        *markdown_table(
            ("Method", "Final VSR (%)", "Total churn"),
            (
                (
                    row["display"],
                    row["cumulative_vsr_pct"],
                    churn_totals[row["method"]],
                )
                for row in latest_online
            ),
        ),
        "",
        "## Rolling windows",
        "",
        "CSV: [fig_rolling_windows.csv](fig_rolling_windows.csv)",
        "",
        *markdown_table(
            ("Window", "Method", "VSR (%)"),
            (
                (row["window_label"], row["display"], row["verified_success_rate_pct"])
                for row in rolling
            ),
        ),
        "",
        "## Leave-provider-out",
        "",
        "CSV: [fig_leave_provider_out.csv](fig_leave_provider_out.csv)",
        "",
        *markdown_table(
            ("Provider", "Method", "VSR (%)"),
            (
                (row["held_out_provider"], row["display"], row["verified_success_rate_pct"])
                for row in lpo
            ),
        ),
        "",
        "## Public-release drift",
        "",
        "Plotted means: [fig_public_release_drift.csv](fig_public_release_drift.csv); provider-level source: [fig_public_release_drift_by_service.csv](fig_public_release_drift_by_service.csv).",
        "",
        *markdown_table(
            ("Phase", "Method", "Mean VSR (%)"),
            (
                (row["phase_label"], row["display"], row["mean_vsr_across_four_releases_pct"])
                for row in drift
            ),
        ),
        "",
        "## Retrieval-budget sensitivity",
        "",
        "CSV: [fig_retrieval_budget.csv](fig_retrieval_budget.csv)",
        "",
        *markdown_table(
            ("Budget", "Method", "VSR (%)"),
            (
                (row["retrieval_budget_tokens"], row["display"], row["verified_success_rate_pct"])
                for row in budget
            ),
        ),
        "",
        "## Storage-budget sensitivity",
        "",
        "CSV: [fig_storage_budget.csv](fig_storage_budget.csv)",
        "",
        *markdown_table(
            ("Budget", "Method", "VSR (%)"),
            (
                (row["storage_budget_cost_units"], row["display"], row["verified_success_rate_pct"])
                for row in storage
            ),
        ),
        "",
        "## Multiservice v5",
        "",
        "CSV: [fig_multiservice_v5.csv](fig_multiservice_v5.csv)",
        "",
        *markdown_table(
            ("Service", "Method", "VSR (%)"),
            (
                (row["service"], row["display"], row["verified_success_rate_pct"])
                for row in multiservice
            ),
        ),
        "",
        "## Retrieval efficiency",
        "",
        "CSV: [fig_efficiency.csv](fig_efficiency.csv)",
        "",
        *markdown_table(
            ("Method", "VSR (%)", "Retrieval p95 (ms)"),
            (
                (row["display"], row["verified_success_rate_pct"], row["retrieval_p95_ms"])
                for row in efficiency
            ),
        ),
        "",
        "## Scope notes",
        "",
        "- Retrieval/storage sweeps and multiservice v5 are controlled, post-hoc analyses.",
        "- Release drift uses observed public software releases with controlled benchmark queries; it is not natural production-traffic drift.",
        "- Seeds are robustness repetitions, not independent task observations.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    export_retrieval_budget()
    export_storage_budget()
    export_efficiency()
    export_rolling_lpo()
    export_online()
    export_release_drift()
    export_multiservice()
    write_readme()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
