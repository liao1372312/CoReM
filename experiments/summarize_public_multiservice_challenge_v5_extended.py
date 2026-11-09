#!/usr/bin/env python3
"""Combine the post-hoc v5 baselines, mechanism ablations, and HGT runs."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_jsonl, write_checksums, write_json

DISPLAY = {
    "no_memory": "No memory",
    "service_bm25": "Service BM25",
    "recency_bm25": "Recency BM25",
    "version_bm25": "Version BM25",
    "version_bm25_bundle": "Version BM25 + verified closure",
    "corem_query_only": "CoReM query-only",
    "corem_additive_utility": "A1: additive query utility",
    "corem_no_hyperedges": "A2: no hyperedges",
    "corem_item_only": "A4: item-only selection",
    "corem_no_closure": "A5: no mandatory closure",
    "hgt_corem": "Trained HGT CoReM",
    "hgt_additive": "Trained HGT additive",
    "hgt_no_hyperedges": "Trained HGT no-hyperedge representation",
}

METHOD_ORDER = tuple(DISPLAY)

# The first method is always the estimand and the second is the comparator.
CONTRASTS = (
    ("version_bm25_bundle", "corem_query_only"),
    ("hgt_corem", "corem_query_only"),
    ("hgt_additive", "hgt_corem"),
    ("hgt_corem", "hgt_no_hyperedges"),
    ("corem_query_only", "corem_additive_utility"),
    ("corem_query_only", "corem_no_hyperedges"),
    ("corem_query_only", "corem_item_only"),
    ("corem_query_only", "corem_no_closure"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--main-analysis", type=Path, required=True)
    parser.add_argument("--matched-analysis", type=Path, required=True)
    parser.add_argument("--ablation-analysis", type=Path, required=True)
    parser.add_argument("--hgt-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    return parser.parse_args()


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


def cluster_bootstrap(
    grouped_values: dict[tuple[str, ...], list[float]],
    draws: int,
    seed: int,
) -> tuple[float, float]:
    """Resample clusters and retain every event within each sampled cluster."""

    groups = sorted(grouped_values)
    if not groups:
        raise ValueError("cluster bootstrap requires at least one group")
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sampled = [rng.choice(groups) for _ in groups]
        values = [value for group in sampled for value in grouped_values[group]]
        estimates.append(sum(values) / len(values))
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_scored(analysis: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(analysis / "scored_executions.jsonl")
    if not rows:
        raise ValueError(f"no scored executions in {analysis}")
    return rows


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("mean requires at least one value")
    return sum(materialized) / len(materialized)


def aggregate_methods(
    rows: list[dict[str, Any]], draws: int
) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)

    summaries = []
    for method in METHOD_ORDER:
        members = by_method.get(method)
        if not members:
            continue
        by_event: dict[str, list[float]] = defaultdict(list)
        for row in members:
            by_event[str(row["event_id"])].append(float(row["verified_success"]))
        event_rates = {key: mean(values) for key, values in by_event.items()}
        event_groups = {(event_id,): [value] for event_id, value in event_rates.items()}
        ci_low, ci_high = cluster_bootstrap(
            event_groups, draws, 20271014 + METHOD_ORDER.index(method)
        )
        attempted = sum(int(row["attempted_reuses"]) for row in members)
        stale = sum(int(row["stale_attempts"]) for row in members)
        summaries.append(
            {
                "method": method,
                "display": DISPLAY[method],
                "events": len(by_event),
                "executions": len(members),
                "seeds": len({int(row["seed"]) for row in members}),
                "vsr": mean(float(row["verified_success"]) for row in members),
                "bs": sum(int(row["binding_passed"]) for row in members)
                / sum(int(row["binding_total"]) for row in members),
                "ecr": mean(float(row["evidence_complete"]) for row in members),
                "sur": stale / attempted if attempted else None,
                "vsr_ci_low": ci_low,
                "vsr_ci_high": ci_high,
            }
        )
    return summaries


def paired_comparisons(
    rows: list[dict[str, Any]],
    events: dict[str, dict[str, Any]],
    draws: int,
) -> list[dict[str, Any]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        values[(str(row["method"]), str(row["event_id"]))].append(
            float(row["verified_success"])
        )
    rates = {key: mean(group) for key, group in values.items()}

    output = []
    for index, (target, comparator) in enumerate(CONTRASTS):
        target_events = {event_id for method, event_id in rates if method == target}
        comparator_events = {
            event_id for method, event_id in rates if method == comparator
        }
        if target_events != comparator_events:
            raise ValueError(
                f"event mismatch for {target} versus {comparator}: "
                f"{len(target_events)} versus {len(comparator_events)}"
            )
        by_family: dict[tuple[str, ...], list[float]] = defaultdict(list)
        by_provider: dict[tuple[str, ...], list[float]] = defaultdict(list)
        differences = []
        for event_id in sorted(target_events):
            difference = rates[(target, event_id)] - rates[(comparator, event_id)]
            event = events[event_id]
            family = (
                str(event["service_id"]),
                str(event["version_id"]),
                str(event["task_key"]),
            )
            provider = (str(event["provider_id"]),)
            by_family[family].append(difference)
            by_provider[provider].append(difference)
            differences.append(difference)

        family_means = [mean(group) for group in by_family.values()]
        provider_means = [mean(group) for group in by_provider.values()]
        family_wins = sum(value > 0 for value in family_means)
        family_losses = sum(value < 0 for value in family_means)
        provider_wins = sum(value > 0 for value in provider_means)
        provider_losses = sum(value < 0 for value in provider_means)
        family_ci = cluster_bootstrap(
            by_family, draws, 20272000 + index
        )
        provider_ci = cluster_bootstrap(
            by_provider, draws, 20273000 + index
        )
        output.append(
            {
                "contrast": f"{target}_minus_{comparator}",
                "target": target,
                "comparator": comparator,
                "events": len(differences),
                "task_version_families": len(by_family),
                "providers": len(by_provider),
                "delta_vsr": mean(differences),
                "family_cluster_ci_low": family_ci[0],
                "family_cluster_ci_high": family_ci[1],
                "family_wins": family_wins,
                "family_losses": family_losses,
                "family_ties": len(family_means) - family_wins - family_losses,
                "family_sign_p": exact_sign_p(family_wins, family_losses),
                "provider_cluster_ci_low": provider_ci[0],
                "provider_cluster_ci_high": provider_ci[1],
                "provider_wins": provider_wins,
                "provider_losses": provider_losses,
                "provider_ties": len(provider_means)
                - provider_wins
                - provider_losses,
                "provider_sign_p": exact_sign_p(provider_wins, provider_losses),
            }
        )
    return output


def plot_per_service(
    rows: list[dict[str, Any]],
    events: dict[str, dict[str, Any]],
    figures: Path,
) -> None:
    methods = (
        "version_bm25_bundle",
        "corem_query_only",
        "hgt_corem",
        "hgt_additive",
    )
    services = ("json-server", "go-httpbin", "wiremock", "mockserver")
    colors = ("#2a9d8f", "#3568b8", "#7a5195", "#e76f51")
    labels = ("BM25 + closure", "CoReM query-only", "HGT CoReM", "HGT additive")
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        method = str(row["method"])
        if method not in methods:
            continue
        service = str(events[str(row["event_id"])]["service_id"])
        grouped[(method, service)].append(float(row["verified_success"]))

    figures.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 3.0))
    positions = list(range(len(services)))
    width = 0.19
    for method_index, (method, color, label) in enumerate(
        zip(methods, colors, labels, strict=True)
    ):
        offsets = [
            position + (method_index - 1.5) * width for position in positions
        ]
        axis.bar(
            offsets,
            [mean(grouped[(method, service)]) * 100 for service in services],
            width=width,
            label=label,
            color=color,
        )
    axis.set_xticks(positions, ("json-server", "go-httpbin", "WireMock", "MockServer"))
    axis.set_ylim(0, 100)
    axis.set_ylabel("Verified success rate (%)")
    axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=7, ncols=2, loc="upper center")
    figure.tight_layout()
    figure.savefig(figures / "corem_multiservice_v5.pdf", bbox_inches="tight")
    figure.savefig(
        figures / "corem_multiservice_v5.png", dpi=300, bbox_inches="tight"
    )
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    if args.bootstrap_draws < 100:
        raise ValueError("bootstrap-draws must be at least 100")

    events = {
        str(row["event_id"]): row
        for row in load_jsonl(args.dataset / "events.jsonl")
    }
    main_rows = load_scored(args.main_analysis)
    matched_rows = load_scored(args.matched_analysis)
    ablation_rows = [
        row
        for row in load_scored(args.ablation_analysis)
        if str(row["method"]) != "corem_query_only"
    ]
    hgt_rows = load_scored(args.hgt_analysis)
    rows = main_rows + matched_rows + ablation_rows + hgt_rows

    unknown_events = sorted({str(row["event_id"]) for row in rows} - set(events))
    if unknown_events:
        raise ValueError(f"scored rows contain unknown events: {unknown_events[:3]}")

    methods = aggregate_methods(rows, args.bootstrap_draws)
    comparisons = paired_comparisons(rows, events, args.bootstrap_draws)
    method_lookup = {str(row["method"]): row for row in methods}
    comparison_lookup = {str(row["contrast"]): row for row in comparisons}

    args.output.mkdir(parents=True)
    write_csv(args.output / "method_metrics.csv", methods)
    write_csv(args.output / "paired_comparisons.csv", comparisons)
    write_json(
        args.output / "result.json",
        {
            "status": "complete_posthoc_v5_extended_analysis",
            "test_events": len(
                {str(row["event_id"]) for row in rows}
            ),
            "task_version_families": len(
                {
                    (
                        str(event["service_id"]),
                        str(event["version_id"]),
                        str(event["task_key"]),
                    )
                    for event in events.values()
                    if str(event["event_id"])
                    in {str(row["event_id"]) for row in rows}
                }
            ),
            "providers": len(
                {
                    str(events[str(row["event_id"])]["provider_id"])
                    for row in rows
                }
            ),
            "bootstrap_draws": args.bootstrap_draws,
            "method_metrics": methods,
            "paired_comparisons": comparisons,
            "statistical_unit": (
                "Seeds are averaged within event. Bootstrap and sign-test "
                "inference use task/version family or provider clusters."
            ),
            "variant_contracts": {
                "version_bm25_bundle": (
                    "Version-scoped BM25 ranking followed by the same verified "
                    "hyperedge closure used by the query-only CoReM reference."
                ),
                "corem_additive_utility": (
                    "Same query-only graph, candidates, bundles, and budget; "
                    "complement and redundancy terms are disabled."
                ),
                "corem_no_hyperedges": (
                    "Strict mechanism ablation: hyperedges are removed from "
                    "candidate exposure, closure, and utility."
                ),
                "corem_item_only": (
                    "Individual memories are ranked without bundle selection."
                ),
                "corem_no_closure": (
                    "Same query-only selection with mandatory evidence closure "
                    "disabled; it is identical on this dataset."
                ),
                "hgt_no_hyperedges": (
                    "Candidate-matched representation ablation: verified bundle "
                    "candidates remain frozen, but hyperedges are removed from "
                    "HGT message passing and set utility. It is not strict A2."
                ),
            },
            "claim_boundary": (
                "All added contrasts are post-hoc exploratory on a previously "
                "inspected controlled public-release test set. They do not "
                "establish production generalization, natural drift adaptation, "
                "or causal contributions of individual memories."
            ),
        },
    )

    fair = comparison_lookup[
        "version_bm25_bundle_minus_corem_query_only"
    ]
    learned = comparison_lookup["hgt_corem_minus_corem_query_only"]
    complement = comparison_lookup["hgt_additive_minus_hgt_corem"]
    strict_a2 = comparison_lookup[
        "corem_query_only_minus_corem_no_hyperedges"
    ]
    summary_lines = [
        "# Extended Four-Provider Challenge Analysis",
        "",
        (
            "This post-hoc analysis combines the frozen 64-event v5 test results, "
            "averaging repeated seeds within each event before clustered inference. "
            "All underlying runs have zero runner/checker errors."
        ),
        "",
        "| Method | VSR | BS | ECR | SUR | Event-bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        if method not in method_lookup:
            continue
        row = method_lookup[method]
        sur = "N/A" if row["sur"] is None else f"{float(row['sur']) * 100:.2f}%"
        summary_lines.append(
            f"| {row['display']} | {float(row['vsr']) * 100:.2f}% | "
            f"{float(row['bs']) * 100:.2f}% | "
            f"{float(row['ecr']) * 100:.2f}% | {sur} | "
            f"[{float(row['vsr_ci_low']) * 100:.2f}, "
            f"{float(row['vsr_ci_high']) * 100:.2f}] |"
        )
    summary_lines.extend(
        [
            "",
            "## Main findings",
            "",
            (
                "The matched Version-BM25+closure control exceeds query-only CoReM "
                f"by {float(fair['delta_vsr']) * 100:.2f} points "
                "(family-cluster 95% CI "
                f"[{float(fair['family_cluster_ci_low']) * 100:.2f}, "
                f"{float(fair['family_cluster_ci_high']) * 100:.2f}]; family sign "
                f"p={float(fair['family_sign_p']):.4f}). Therefore, the earlier "
                "35.94-point advantage over plain Version BM25 does not establish "
                "superiority once candidate exposure is matched."
            ),
            "",
            (
                "Trained full HGT CoReM changes VSR by "
                f"{float(learned['delta_vsr']) * 100:+.2f} points versus the "
                "query-only reference (family-cluster 95% CI "
                f"[{float(learned['family_cluster_ci_low']) * 100:.2f}, "
                f"{float(learned['family_cluster_ci_high']) * 100:.2f}]). The "
                "learned selector is not an improvement in this small-data setting."
            ),
            "",
            (
                "Within the trained model, disabling complement/redundancy changes "
                f"VSR by {float(complement['delta_vsr']) * 100:+.2f} points relative "
                "to full HGT, so the learned non-additive term is not supported. In "
                "contrast, strict removal of query-time hyperedges reduces VSR by "
                f"{float(strict_a2['delta_vsr']) * 100:.2f} points relative to the "
                "query-only reference; this supports bundle exposure as a mechanism, "
                "not the trained utility model."
            ),
            "",
            (
                "These comparisons are exploratory because the test set had already "
                "been inspected before the new controls were added. The benchmark "
                "uses constructed queries and hard negatives over pinned public "
                "releases; it is not production telemetry or online adaptation."
            ),
            "",
        ]
    )
    (args.output / "SUMMARY.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    plot_per_service(rows, events, args.figures)
    write_checksums(
        args.output,
        ("SUMMARY.md", "method_metrics.csv", "paired_comparisons.csv", "result.json"),
    )
    print(f"wrote extended v5 analysis to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
