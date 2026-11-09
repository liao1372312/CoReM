#!/usr/bin/env python3
"""Analyze online retention, rolling windows, LPO, and public-release drift."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import matplotlib

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import (  # noqa: E402
    load_jsonl,
    score_execution_row,
    write_checksums,
    write_json,
)

DISPLAY = {
    "group_aware_online": "Group-aware",
    "independent_relevance_online": "Independent relevance",
    "lru_online": "LRU",
    "lfu_online": "LFU",
    "no_memory": "No memory",
    "hgt_rolling_ensemble": "HGT/BM25 ensemble",
    "hgt_leave_provider_out": "HGT leave-provider-out",
    "version_bm25_bundle": "BM25 + closure",
}
COLORS = {
    "group_aware_online": "#3568b8",
    "independent_relevance_online": "#e76f51",
    "lru_online": "#2a9d8f",
    "lfu_online": "#7a5195",
    "no_memory": "#7f7f7f",
    "hgt_rolling_ensemble": "#3568b8",
    "hgt_leave_provider_out": "#3568b8",
    "version_bm25_bundle": "#2a9d8f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--online", type=Path, required=True)
    parser.add_argument("--lpo-executions", type=Path, required=True)
    parser.add_argument("--matched-test-executions", type=Path, required=True)
    parser.add_argument("--rolling-hgt", type=Path, nargs=3, required=True)
    parser.add_argument("--rolling-matched", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty measurements: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("mean requires observations")
    return statistics.mean(materialized)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def family_bootstrap(
    event_rates: dict[str, float],
    events: dict[str, dict[str, Any]],
    draws: int,
    seed: int,
) -> tuple[float, float]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for event_id, value in event_rates.items():
        event = events[event_id]
        groups[
            (
                str(event["service_id"]),
                str(event["version_id"]),
                str(event["task_key"]),
            )
        ].append(value)
    keys = sorted(groups)
    if not keys:
        raise ValueError("bootstrap has no task families")
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sample = [rng.choice(keys) for _ in keys]
        values = [value for key in sample for value in groups[key]]
        estimates.append(mean(values))
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def score_raw(
    path: Path, events: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"no execution rows in {path}")
    output = []
    seen = set()
    for row in rows:
        event_id = str(row["event_id"])
        key = (str(row["method"]), int(row["seed"]), event_id)
        if key in seen:
            raise ValueError(f"duplicate execution key in {path}: {key}")
        seen.add(key)
        if row.get("execution_error") is not None or row.get("checker_error") is not None:
            raise ValueError(f"infrastructure error in {path}: {row.get('run_id')}")
        output.append(
            {
                "event_id": event_id,
                "provider_id": events[event_id]["provider_id"],
                "service_id": events[event_id]["service_id"],
                "version_id": events[event_id]["version_id"],
                "task_key": events[event_id]["task_key"],
                "event_time": events[event_id]["event_time"],
                "method": row["method"],
                "seed": row["seed"],
                "token_cost": row["token_cost"],
                "latency_ms": row["latency_ms"],
                **score_execution_row(row, events[event_id]),
            }
        )
    return output


def aggregate(
    rows: list[dict[str, Any]],
    events: dict[str, dict[str, Any]],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    by_event: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_event[str(row["event_id"])].append(float(row["verified_success"]))
    event_rates = {key: mean(values) for key, values in by_event.items()}
    ci_low, ci_high = family_bootstrap(event_rates, events, draws, seed)
    attempted = sum(int(row["attempted_reuses"]) for row in rows)
    stale = sum(int(row["stale_attempts"]) for row in rows)
    return {
        "events": len(by_event),
        "executions": len(rows),
        "seeds": len({int(row["seed"]) for row in rows}),
        "vsr": mean(float(row["verified_success"]) for row in rows),
        "bs": sum(int(row["binding_passed"]) for row in rows)
        / sum(int(row["binding_total"]) for row in rows),
        "ecr": mean(float(row["evidence_complete"]) for row in rows),
        "sur": stale / attempted if attempted else None,
        "vsr_ci_low": ci_low,
        "vsr_ci_high": ci_high,
    }


def paired_delta(
    target: list[dict[str, Any]],
    comparator: list[dict[str, Any]],
    events: dict[str, dict[str, Any]],
    draws: int,
    seed: int,
) -> tuple[int, float, float, float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for label, rows in (("target", target), ("comparator", comparator)):
        for row in rows:
            grouped[(label, str(row["event_id"]))].append(
                float(row["verified_success"])
            )
    target_ids = {event_id for label, event_id in grouped if label == "target"}
    comparator_ids = {
        event_id for label, event_id in grouped if label == "comparator"
    }
    if target_ids != comparator_ids:
        raise ValueError("paired comparison event sets differ")
    differences = {
        event_id: mean(grouped[("target", event_id)])
        - mean(grouped[("comparator", event_id)])
        for event_id in target_ids
    }
    low, high = family_bootstrap(differences, events, draws, seed)
    return len(differences), mean(differences.values()), low, high


def rolling_rows(
    hgt_paths: list[Path],
    matched_paths: list[Path],
    events: dict[str, dict[str, Any]],
    draws: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = []
    contrasts = []
    for index, (hgt_path, matched_path) in enumerate(
        zip(hgt_paths, matched_paths, strict=True), start=1
    ):
        hgt = score_raw(hgt_path, events)
        matched = score_raw(matched_path, events)
        methods = {str(row["method"]) for row in hgt}
        if len(methods) != 1:
            raise ValueError(f"rolling window {index} has multiple HGT methods")
        for method, rows in (
            ("hgt_rolling_ensemble", hgt),
            ("version_bm25_bundle", matched),
        ):
            metrics.append(
                {
                    "window": index,
                    "test_indices_per_service": (
                        "25-40" if index == 1 else "41-56" if index == 2 else "65-80"
                    ),
                    "method": method,
                    "display": DISPLAY[method],
                    **aggregate(rows, events, draws, 20274000 + 10 * index),
                }
            )
        count, delta, low, high = paired_delta(
            hgt, matched, events, draws, 20274100 + index
        )
        contrasts.append(
            {
                "window": index,
                "events": count,
                "contrast": "hgt_rolling_ensemble_minus_version_bm25_bundle",
                "delta_vsr": delta,
                "delta_vsr_ci_low": low,
                "delta_vsr_ci_high": high,
            }
        )
    return metrics, contrasts


def lpo_rows(
    lpo_path: Path,
    matched_path: Path,
    events: dict[str, dict[str, Any]],
    draws: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lpo = score_raw(lpo_path, events)
    matched = score_raw(matched_path, events)
    output = []
    contrasts = []
    providers = sorted({str(row["provider_id"]) for row in lpo})
    for index, provider in enumerate(providers):
        left = [row for row in lpo if row["provider_id"] == provider]
        right = [row for row in matched if row["provider_id"] == provider]
        for method, rows in (
            ("hgt_leave_provider_out", left),
            ("version_bm25_bundle", right),
        ):
            output.append(
                {
                    "provider_id": provider,
                    "method": method,
                    "display": DISPLAY[method],
                    **aggregate(rows, events, draws, 20275000 + 10 * index),
                }
            )
        count, delta, low, high = paired_delta(
            left, right, events, draws, 20275100 + index
        )
        contrasts.append(
            {
                "provider_id": provider,
                "events": count,
                "contrast": "hgt_leave_provider_out_minus_version_bm25_bundle",
                "delta_vsr": delta,
                "delta_vsr_ci_low": low,
                "delta_vsr_ci_high": high,
            }
        )
    for method, rows in (
        ("hgt_leave_provider_out", lpo),
        ("version_bm25_bundle", matched),
    ):
        output.append(
            {
                "provider_id": "ALL",
                "method": method,
                "display": DISPLAY[method],
                **aggregate(rows, events, draws, 20275999),
            }
        )
    count, delta, low, high = paired_delta(lpo, matched, events, draws, 20275998)
    contrasts.append(
        {
            "provider_id": "ALL",
            "events": count,
            "contrast": "hgt_leave_provider_out_minus_version_bm25_bundle",
            "delta_vsr": delta,
            "delta_vsr_ci_low": low,
            "delta_vsr_ci_high": high,
        }
    )
    return output, contrasts


def online_rows(
    online_dir: Path, events: dict[str, dict[str, Any]], draws: int
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    rows = load_jsonl(online_dir / "scored_executions.jsonl")
    banks = load_jsonl(online_dir / "bank_history.jsonl")
    if not rows or not banks:
        raise ValueError("online retention measurements are empty")
    methods = sorted({str(row["method"]) for row in rows})
    metrics = []
    for index, method in enumerate(methods):
        members = [row for row in rows if row["method"] == method]
        summary = aggregate(members, events, draws, 20276000 + index)
        method_banks = [row for row in banks if row["method"] == method]
        metrics.append(
            {
                "method": method,
                "display": DISPLAY[method],
                **summary,
                "peak_bank_cost": max(int(row["bank_after_cost"]) for row in method_banks),
                "activated_records": sum(int(row["activated_records"]) for row in method_banks),
                "evicted_records": sum(int(row["evicted_records"]) for row in method_banks),
                "mean_update_latency_ms": mean(
                    float(row["update_latency_ms"]) for row in method_banks
                ),
            }
        )

    target = [row for row in rows if row["method"] == "group_aware_online"]
    contrasts = []
    for index, method in enumerate(
        ("independent_relevance_online", "lru_online", "lfu_online", "no_memory")
    ):
        comparator = [row for row in rows if row["method"] == method]
        count, delta, low, high = paired_delta(
            target, comparator, events, draws, 20276100 + index
        )
        contrasts.append(
            {
                "contrast": f"group_aware_online_minus_{method}",
                "events": count,
                "delta_vsr": delta,
                "delta_vsr_ci_low": low,
                "delta_vsr_ci_high": high,
            }
        )

    phase_ranges = {
        "pre_change_late": range(33, 41),
        "post_change_early": range(41, 49),
        "post_change_late": range(65, 81),
    }
    drift = []
    for method in methods:
        for service in sorted({str(row["service_id"]) for row in rows}):
            service_rows = sorted(
                (
                    row
                    for row in rows
                    if row["method"] == method and row["service_id"] == service
                ),
                key=lambda row: int(row["service_event_index"]),
            )
            for phase, indices in phase_ranges.items():
                members = [
                    row
                    for row in service_rows
                    if int(row["service_event_index"]) in indices
                ]
                if not members:
                    raise ValueError(f"empty drift phase: {method} {service} {phase}")
                drift.append(
                    {
                        "method": method,
                        "service_id": service,
                        "phase": phase,
                        "events": len(members),
                        "vsr": mean(float(row["verified_success"]) for row in members),
                    }
                )

    recovery = []
    for method in methods:
        for service in sorted({str(row["service_id"]) for row in rows}):
            post = sorted(
                (
                    row
                    for row in rows
                    if row["method"] == method
                    and row["service_id"] == service
                    and int(row["service_event_index"]) >= 41
                ),
                key=lambda row: int(row["service_event_index"]),
            )
            first = next(
                (
                    int(row["service_event_index"]) - 40
                    for row in post
                    if row["verified_success"]
                ),
                None,
            )
            recovery.append(
                {
                    "method": method,
                    "service_id": service,
                    "attempts_to_first_post_change_success": first,
                    "censored": first is None,
                    "max_post_change_attempts": len(post),
                }
            )
    return metrics, contrasts, drift, recovery


def plot_online(
    online_dir: Path, figures: Path
) -> None:
    rows = load_jsonl(online_dir / "scored_executions.jsonl")
    banks = load_jsonl(online_dir / "bank_history.jsonl")
    methods = [
        "group_aware_online",
        "independent_relevance_online",
        "lru_online",
        "lfu_online",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))
    for method in methods:
        members = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["event_index"]),
        )
        cumulative = []
        total = 0
        for index, row in enumerate(members, start=1):
            total += int(row["verified_success"])
            cumulative.append(100 * total / index)
        axes[0].plot(
            range(1, len(cumulative) + 1),
            cumulative,
            color=COLORS[method],
            label=DISPLAY[method],
            linewidth=1.5,
        )
        bank_members = sorted(
            (row for row in banks if row["method"] == method),
            key=lambda row: int(row["event_index"]),
        )
        churn = [
            int(row["activated_records"]) + int(row["evicted_records"])
            for row in bank_members
        ]
        windowed = [sum(churn[start : start + 20]) for start in range(0, len(churn), 20)]
        axes[1].plot(
            range(1, len(windowed) + 1),
            windowed,
            marker="o",
            markersize=2.5,
            color=COLORS[method],
            linewidth=1.2,
        )
    axes[0].set_xlabel("Chronological incident")
    axes[0].set_ylabel("Cumulative VSR (%)")
    axes[0].set_ylim(0, 100)
    axes[1].set_xlabel("20-incident arrival window")
    axes[1].set_ylabel("Bank activations + evictions")
    for axis in axes:
        axis.grid(color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper right")
    figure.tight_layout()
    figure.savefig(figures / "corem_online_retention.pdf", bbox_inches="tight")
    figure.savefig(figures / "corem_online_retention.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_extensions(
    rolling: list[dict[str, Any]],
    lpo: list[dict[str, Any]],
    figures: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))
    for method in ("hgt_rolling_ensemble", "version_bm25_bundle"):
        members = sorted(
            (row for row in rolling if row["method"] == method),
            key=lambda row: int(row["window"]),
        )
        axes[0].plot(
            [int(row["window"]) for row in members],
            [100 * float(row["vsr"]) for row in members],
            marker="o",
            color=COLORS[method],
            label=DISPLAY[method],
        )
    axes[0].set_xticks((1, 2, 3), ("late old", "early new", "late new"))
    axes[0].set_ylabel("Verified success rate (%)")
    axes[0].set_xlabel("Rolling test window")
    axes[0].set_ylim(0, 100)
    axes[0].legend(frameon=False, fontsize=7)

    providers = sorted(
        {str(row["provider_id"]) for row in lpo if row["provider_id"] != "ALL"}
    )
    width = 0.36
    for method_index, method in enumerate(
        ("hgt_leave_provider_out", "version_bm25_bundle")
    ):
        values = [
            100
            * float(
                next(
                    row["vsr"]
                    for row in lpo
                    if row["method"] == method and row["provider_id"] == provider
                )
            )
            for provider in providers
        ]
        offsets = [index + (method_index - 0.5) * width for index in range(len(providers))]
        axes[1].bar(
            offsets,
            values,
            width=width,
            color=COLORS[method],
            label=DISPLAY[method],
        )
    axes[1].set_xticks(range(len(providers)), providers, rotation=18, ha="right")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Verified success rate (%)")
    axes[1].set_xlabel("Held-out provider")
    axes[1].legend(frameon=False, fontsize=6.5)
    for axis in axes:
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(figures / "corem_rolling_lpo.pdf", bbox_inches="tight")
    figure.savefig(figures / "corem_rolling_lpo.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_drift(drift: list[dict[str, Any]], figures: Path) -> None:
    methods = [
        "group_aware_online",
        "independent_relevance_online",
        "lru_online",
        "lfu_online",
    ]
    phases = ("pre_change_late", "post_change_early", "post_change_late")
    labels = ("pre-change", "early post", "late post")
    figure, axis = plt.subplots(figsize=(5.8, 2.8))
    for method in methods:
        values = []
        for phase in phases:
            members = [
                row for row in drift if row["method"] == method and row["phase"] == phase
            ]
            values.append(100 * mean(float(row["vsr"]) for row in members))
        axis.plot(
            range(3),
            values,
            marker="o",
            color=COLORS[method],
            label=DISPLAY[method],
        )
    axis.set_xticks(range(3), labels)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Mean VSR across releases (%)")
    axis.grid(color="#dddddd", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=7, ncols=2)
    figure.tight_layout()
    figure.savefig(figures / "corem_public_release_drift.pdf", bbox_inches="tight")
    figure.savefig(figures / "corem_public_release_drift.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    if args.bootstrap_draws < 100:
        raise ValueError("bootstrap draws must be at least 100")
    online_manifest = json.loads((args.online / "result.json").read_text())
    if (
        online_manifest.get("status") != "complete_online_retention"
        or online_manifest.get("schema_version")
        != "corem-online-retention-v2-strict-prequential"
    ):
        raise ValueError("online source must be a complete strict-prequential run")
    online_executions = load_jsonl(args.online / "execution_results.jsonl")
    if any(row.get("activated_pre_ids") for row in online_executions):
        raise ValueError("online source activated memory before an event")
    for method in sorted({str(row["method"]) for row in online_executions}):
        # The execution file is written in chronological event order, which
        # need not match lexicographic IDs across services.
        trajectory = [row for row in online_executions if row["method"] == method]
        if trajectory[0].get("bank_before_ids"):
            raise ValueError(f"{method} did not start with an empty bank")
        for previous, current in zip(trajectory, trajectory[1:]):
            if not set(current["bank_before_ids"]).issubset(
                previous["bank_after_ids"]
            ):
                raise ValueError(f"{method} activated memory before an event")
        if any(
            not set(row["selected_memory_ids"]).issubset(row["bank_before_ids"])
            for row in trajectory
        ):
            raise ValueError(f"{method} retrieved outside its active bank")
    events = {
        str(row["event_id"]): row for row in load_jsonl(args.dataset / "events.jsonl")
    }
    online_metrics, online_contrasts, drift, recovery = online_rows(
        args.online, events, args.bootstrap_draws
    )
    rolling, rolling_contrasts = rolling_rows(
        list(args.rolling_hgt),
        list(args.rolling_matched),
        events,
        args.bootstrap_draws,
    )
    lpo, lpo_contrasts = lpo_rows(
        args.lpo_executions,
        args.matched_test_executions,
        events,
        args.bootstrap_draws,
    )
    args.output.mkdir(parents=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "online_retention_metrics.csv", online_metrics)
    write_csv(args.output / "online_retention_contrasts.csv", online_contrasts)
    write_csv(args.output / "rolling_window_metrics.csv", rolling)
    write_csv(args.output / "rolling_window_contrasts.csv", rolling_contrasts)
    write_csv(args.output / "leave_provider_out_metrics.csv", lpo)
    write_csv(args.output / "leave_provider_out_contrasts.csv", lpo_contrasts)
    write_csv(args.output / "public_release_drift.csv", drift)
    write_csv(args.output / "public_release_recovery.csv", recovery)
    plot_online(args.online, args.figures)
    plot_extensions(rolling, lpo, args.figures)
    plot_drift(drift, args.figures)
    measurement_inputs = [
        args.dataset / "events.jsonl",
        args.online / "result.json",
        args.online / "execution_results.jsonl",
        args.online / "scored_executions.jsonl",
        args.online / "bank_history.jsonl",
        args.lpo_executions,
        args.matched_test_executions,
        *args.rolling_hgt,
        *args.rolling_matched,
    ]
    result = {
        "status": "complete_longitudinal_extensions",
        "online_retention": online_metrics,
        "online_retention_contrasts": online_contrasts,
        "rolling_windows": rolling,
        "rolling_contrasts": rolling_contrasts,
        "leave_provider_out": lpo,
        "leave_provider_out_contrasts": lpo_contrasts,
        "drift_claim_boundary": (
            "The version changes are independently observed public releases; queries "
            "are controlled benchmark construction, not natural production traffic."
        ),
        "natural_traffic_track": {
            "status": "not_measured_missing_eligible_source",
            "audit": "experiments/NATURAL_TRAFFIC_AUDIT.md",
            "required": (
                "Observed request arrivals and queries, joined service deployment versions, "
                "replayable workflow contracts, and independent outcome checks."
            ),
        },
        "figures": [
            "corem_online_retention.pdf/png",
            "corem_rolling_lpo.pdf/png",
            "corem_public_release_drift.pdf/png",
        ],
        "reproducibility_stamp": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "matplotlib": matplotlib.__version__,
            "bootstrap_draws": args.bootstrap_draws,
            "bootstrap_seed_ranges": {
                "rolling": "20274010--20274030",
                "leave_provider_out": "20275000 + 10 * sorted-provider-index; overall 20275999",
                "online": "20276000 + sorted-method-index",
                "paired_online": "20276100 + comparator-index",
                "paired_rolling": "20274100 + window-index",
                "paired_leave_provider_out": "20275100 + sorted-provider-index; overall 20275998",
            },
            "measurement_input_sha256": {
                str(path.resolve().relative_to(REPOSITORY_ROOT)): sha256(path)
                for path in measurement_inputs
            },
        },
    }
    write_json(args.output / "result.json", result)
    summary_lines = [
        "# Longitudinal Extension Results",
        "",
        "All measurements use independently checked reset-environment executions. ",
        "Public-release drift is not labeled as natural production-traffic drift.",
        "",
        "## Online retention",
        "",
    ]
    for row in online_metrics:
        summary_lines.append(
            f"- {row['display']}: VSR {100*float(row['vsr']):.1f}% "
            f"(95% CI {100*float(row['vsr_ci_low']):.1f}--"
            f"{100*float(row['vsr_ci_high']):.1f}%), peak bank "
            f"{row['peak_bank_cost']}/640."
        )
    for row in online_contrasts:
        summary_lines.append(
            f"- {row['contrast']}: paired VSR delta "
            f"{100*float(row['delta_vsr']):+.1f} points "
            f"(95% family CI {100*float(row['delta_vsr_ci_low']):+.1f} to "
            f"{100*float(row['delta_vsr_ci_high']):+.1f})."
        )
    summary_lines.extend(["", "## Rolling and provider holdout", ""])
    for row in rolling_contrasts:
        summary_lines.append(
            f"- Rolling window {row['window']}: HGT ensemble minus BM25 + closure "
            f"{100*float(row['delta_vsr']):+.1f} points "
            f"(95% family CI {100*float(row['delta_vsr_ci_low']):+.1f} to "
            f"{100*float(row['delta_vsr_ci_high']):+.1f}) "
            f"over {row['events']} incidents."
        )
    for row in lpo_contrasts:
        summary_lines.append(
            f"- Held-out {row['provider_id']}: HGT minus BM25 + closure "
            f"{100*float(row['delta_vsr']):+.1f} points "
            f"(95% family CI {100*float(row['delta_vsr_ci_low']):+.1f} to "
            f"{100*float(row['delta_vsr_ci_high']):+.1f}) "
            f"over {row['events']} incidents."
        )
    (args.output / "SUMMARY.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    write_checksums(
        args.output,
        (
            "SUMMARY.md",
            "leave_provider_out_contrasts.csv",
            "leave_provider_out_metrics.csv",
            "online_retention_contrasts.csv",
            "online_retention_metrics.csv",
            "public_release_drift.csv",
            "public_release_recovery.csv",
            "result.json",
            "rolling_window_contrasts.csv",
            "rolling_window_metrics.csv",
        ),
    )
    print(f"wrote longitudinal extension analysis to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
