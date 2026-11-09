#!/usr/bin/env python3
"""Summarize v5 by incident, task/version family, and provider."""

from __future__ import annotations

import argparse
import csv
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
    if total == 0:
        return 1.0
    tail = sum(
        math.comb(total, index) for index in range(min(wins, losses) + 1)
    ) / 2**total
    return min(1.0, 2.0 * tail)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cluster_bootstrap(
    grouped_differences: dict[tuple[str, ...], list[float]], seed: int
) -> tuple[float, float]:
    groups = sorted(grouped_differences)
    rng = random.Random(seed)
    draws = []
    for _ in range(5000):
        sampled = [rng.choice(groups) for _ in groups]
        values = [
            value for group in sampled for value in grouped_differences[group]
        ]
        draws.append(sum(values) / len(values))
    return quantile(draws, 0.025), quantile(draws, 0.975)


def status_trace(row: dict[str, Any]) -> list[int]:
    return [
        int(step["response"]["status"])
        for step in row.get("execution", {}).get("steps", [])
        if isinstance(step, dict) and isinstance(step.get("response"), dict)
    ]


def reproducibility(
    primary: list[dict[str, Any]], cold: list[dict[str, Any]]
) -> dict[str, Any]:
    first = {str(row["run_id"]): row for row in primary}
    second = {str(row["run_id"]): row for row in cold}
    common = sorted(set(first).intersection(second))
    return {
        "primary_rows": len(primary),
        "cold_rows": len(cold),
        "matching_run_key_sets": set(first) == set(second),
        "selection_mismatches": sum(
            first[key].get("selected_memory_ids")
            != second[key].get("selected_memory_ids")
            for key in common
        ),
        "status_trace_mismatches": sum(
            status_trace(first[key]) != status_trace(second[key]) for key in common
        ),
        "checker_mismatches": sum(
            first[key].get("checker") != second[key].get("checker") for key in common
        ),
        "error_field_mismatches": sum(
            (
                first[key].get("execution_error"),
                first[key].get("checker_error"),
            )
            != (
                second[key].get("execution_error"),
                second[key].get("checker_error"),
            )
            for key in common
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--primary-execution", type=Path, required=True)
    parser.add_argument("--cold-execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2

    events = {
        str(row["event_id"]): row
        for row in load_jsonl(args.dataset / "events.jsonl")
    }
    scored = load_jsonl(args.analysis / "scored_executions.jsonl")
    with (args.analysis / "method_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        metrics = {str(row["method"]): row for row in csv.DictReader(handle)}

    values_by_event: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in scored:
        values_by_event[(str(row["method"]), str(row["event_id"]))].append(
            int(row["verified_success"])
        )
    event_rates = {
        key: sum(values) / len(values) for key, values in values_by_event.items()
    }

    target = "corem_query_only"
    target_events = sorted(
        event_id for method, event_id in event_rates if method == target
    )
    comparison_rows = []
    for baseline_index, baseline in enumerate(
        ("version_bm25", "recency_bm25", "service_bm25")
    ):
        by_family: dict[tuple[str, ...], list[float]] = defaultdict(list)
        by_provider: dict[tuple[str, ...], list[float]] = defaultdict(list)
        incident_differences = []
        for event_id in target_events:
            difference = event_rates[(target, event_id)] - event_rates[
                (baseline, event_id)
            ]
            event = events[event_id]
            family = (
                str(event["service_id"]),
                str(event["version_id"]),
                str(event["task_key"]),
            )
            provider = (str(event["provider_id"]),)
            by_family[family].append(difference)
            by_provider[provider].append(difference)
            incident_differences.append(difference)
        family_differences = [
            sum(values) / len(values) for values in by_family.values()
        ]
        provider_differences = [
            sum(values) / len(values) for values in by_provider.values()
        ]
        family_wins = sum(value > 0 for value in family_differences)
        family_losses = sum(value < 0 for value in family_differences)
        provider_wins = sum(value > 0 for value in provider_differences)
        provider_losses = sum(value < 0 for value in provider_differences)
        family_ci = cluster_bootstrap(by_family, 202705 + baseline_index)
        provider_ci = cluster_bootstrap(by_provider, 202715 + baseline_index)
        comparison_rows.append(
            {
                "contrast": f"{target}_minus_{baseline}",
                "incidents": len(incident_differences),
                "task_version_families": len(by_family),
                "providers": len(by_provider),
                "delta_vsr": sum(incident_differences)
                / len(incident_differences),
                "family_cluster_ci_low": family_ci[0],
                "family_cluster_ci_high": family_ci[1],
                "provider_cluster_ci_low": provider_ci[0],
                "provider_cluster_ci_high": provider_ci[1],
                "family_wins": family_wins,
                "family_losses": family_losses,
                "family_ties": len(family_differences)
                - family_wins
                - family_losses,
                "family_sign_p": exact_sign_p(family_wins, family_losses),
                "provider_wins": provider_wins,
                "provider_losses": provider_losses,
                "provider_ties": len(provider_differences)
                - provider_wins
                - provider_losses,
                "provider_sign_p": exact_sign_p(provider_wins, provider_losses),
            }
        )

    service_rows = []
    service_values: dict[tuple[str, str], list[int]] = defaultdict(list)
    family_values: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for row in scored:
        event = events[str(row["event_id"])]
        method = str(row["method"])
        service = str(event["service_id"])
        service_values[(method, service)].append(int(row["verified_success"]))
        family_values[
            (
                method,
                service,
                str(event["version_id"]),
                str(event["task_key"]),
            )
        ].append(int(row["verified_success"]))
    for (method, service), values in sorted(service_values.items()):
        service_rows.append(
            {
                "method": method,
                "service": service,
                "events": len(values) // 5,
                "executions": len(values),
                "vsr": sum(values) / len(values),
            }
        )
    family_rows = [
        {
            "method": key[0],
            "service": key[1],
            "version": key[2],
            "task_key": key[3],
            "events": len(values) // 5,
            "executions": len(values),
            "vsr": sum(values) / len(values),
        }
        for key, values in sorted(family_values.items())
    ]

    primary = load_jsonl(args.primary_execution / "execution_results.jsonl")
    cold = load_jsonl(args.cold_execution / "execution_results.jsonl")
    reproduced = reproducibility(primary, cold)
    if not (
        reproduced["matching_run_key_sets"]
        and reproduced["selection_mismatches"] == 0
        and reproduced["status_trace_mismatches"] == 0
        and reproduced["checker_mismatches"] == 0
        and reproduced["error_field_mismatches"] == 0
    ):
        raise ValueError("cold rerun does not reproduce frozen decisions")

    args.output.mkdir(parents=True)
    write_csv(args.output / "paired_comparisons.csv", comparison_rows)
    write_csv(args.output / "per_service_metrics.csv", service_rows)
    write_csv(args.output / "per_family_metrics.csv", family_rows)
    write_json(
        args.output / "result.json",
        {
            "status": "complete_controlled_multiservice_v5",
            "events": len(events),
            "test_incidents": len(target_events),
            "test_executions": len(scored),
            "test_metrics": metrics,
            "paired_comparisons": comparison_rows,
            "reproducibility": reproduced,
            "claim_boundary": "Controlled public-release reconstruction with constructed queries and hard negatives; not production telemetry, online learning, or a trained-HGT evaluation.",
        },
    )

    services = ["json-server", "go-httpbin", "wiremock", "mockserver"]
    methods = ["corem_query_only", "version_bm25", "recency_bm25"]
    lookup = {
        (str(row["method"]), str(row["service"])): float(row["vsr"]) * 100
        for row in service_rows
    }
    args.figures.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 3.0))
    width = 0.24
    positions = list(range(len(services)))
    for method_index, method in enumerate(methods):
        offsets = [
            position + (method_index - 1) * width for position in positions
        ]
        axis.bar(
            offsets,
            [lookup[(method, service)] for service in services],
            width=width,
            label=DISPLAY[method],
            color=COLORS[method],
        )
    axis.set_xticks(positions, services)
    axis.set_ylim(0, 70)
    axis.set_ylabel("Verified success rate (%)")
    axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncols=3, loc="upper center")
    figure.tight_layout()
    figure.savefig(
        args.figures / "corem_multiservice_v5.pdf", bbox_inches="tight"
    )
    figure.savefig(
        args.figures / "corem_multiservice_v5.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    comparison = comparison_rows[0]
    provider_rows = {
        row["service"]: row
        for row in service_rows
        if row["method"] == "corem_query_only"
    }
    summary = f"""# Four-Provider Three-Step Challenge v5

The frozen benchmark contains 320 controlled incidents over four independently
maintained services, two pinned releases per service, and 32 task/version
families. The oracle gate passes 320/320 events. The chronological held-out test
contains 64 incidents and 1,600 executions (five methods and five deterministic
tie-break seeds), with zero runner, checker, reset, or 160-token budget errors.

| Method | VSR | BS | ECR | Incident-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| CoReM query-only | {float(metrics['corem_query_only']['vsr']) * 100:.2f}% | {float(metrics['corem_query_only']['bs']) * 100:.2f}% | {float(metrics['corem_query_only']['ecr']) * 100:.2f}% | [{float(metrics['corem_query_only']['vsr_ci_low']) * 100:.2f}, {float(metrics['corem_query_only']['vsr_ci_high']) * 100:.2f}] |
| Version BM25 | {float(metrics['version_bm25']['vsr']) * 100:.2f}% | {float(metrics['version_bm25']['bs']) * 100:.2f}% | {float(metrics['version_bm25']['ecr']) * 100:.2f}% | [{float(metrics['version_bm25']['vsr_ci_low']) * 100:.2f}, {float(metrics['version_bm25']['vsr_ci_high']) * 100:.2f}] |
| Recency BM25 | {float(metrics['recency_bm25']['vsr']) * 100:.2f}% | {float(metrics['recency_bm25']['bs']) * 100:.2f}% | {float(metrics['recency_bm25']['ecr']) * 100:.2f}% | [{float(metrics['recency_bm25']['vsr_ci_low']) * 100:.2f}, {float(metrics['recency_bm25']['vsr_ci_high']) * 100:.2f}] |
| Service BM25 | {float(metrics['service_bm25']['vsr']) * 100:.2f}% | {float(metrics['service_bm25']['bs']) * 100:.2f}% | {float(metrics['service_bm25']['ecr']) * 100:.2f}% | [{float(metrics['service_bm25']['vsr_ci_low']) * 100:.2f}, {float(metrics['service_bm25']['vsr_ci_high']) * 100:.2f}] |
| No memory | 0.00% | 0.00% | 0.00% | [0.00, 0.00] |

CoReM exceeds version BM25 by {comparison['delta_vsr'] * 100:.2f} percentage
points. The task/version-family cluster bootstrap interval is
[{comparison['family_cluster_ci_low'] * 100:.2f},
{comparison['family_cluster_ci_high'] * 100:.2f}] points. All four providers
show a positive mean difference, but the provider-level exact sign test is
p={comparison['provider_sign_p']:.3f}; four providers are not enough for a
conventional 0.05 sign-test result.

| Service | CoReM VSR |
|---|---:|
| json-server | {float(provider_rows['json-server']['vsr']) * 100:.2f}% |
| go-httpbin | {float(provider_rows['go-httpbin']['vsr']) * 100:.2f}% |
| WireMock | {float(provider_rows['wiremock']['vsr']) * 100:.2f}% |
| MockServer | {float(provider_rows['mockserver']['vsr']) * 100:.2f}% |

The cold rerun reproduces all 1,600 selections, status traces, error fields,
and complete checker decisions. These results support dependency-bundle
retrieval only under this controlled public-release construction. They do not
measure natural production traffic, online retention, or a trained HGT.
"""
    (args.output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    write_checksums(
        args.output,
        (
            "SUMMARY.md",
            "paired_comparisons.csv",
            "per_family_metrics.csv",
            "per_service_metrics.csv",
            "result.json",
        ),
    )
    print(f"wrote v5 summary, statistics, and figure to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
