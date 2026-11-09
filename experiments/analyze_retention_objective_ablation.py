#!/usr/bin/env python3
"""Analyze matched retention objectives and the requested paired contrasts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.generate_online_retention_paper_table import (
    verify_run_files,
    verify_run_metrics,
)


METHOD_LABELS = {
    "item_value_retention": "Item-value retention",
    "bundle_value_retention": "Bundle-value retention",
    "corem_retention": "CoReM retention",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--independent-contrast", type=Path)
    parser.add_argument("--e2-full", type=Path)
    parser.add_argument("--e2-additive", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def verify_external_source(path: Path) -> None:
    manifest = path.parent / "checksums.sha256"
    if not manifest.is_file():
        raise ValueError(f"external contrast source has no checksums: {path}")
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or name in listed:
            raise ValueError(f"malformed external checksum manifest: {manifest}")
        listed[name] = digest
    if path.name not in listed or hashlib.sha256(path.read_bytes()).hexdigest() != listed[path.name]:
        raise ValueError(f"external contrast source checksum mismatch: {path}")


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cluster_bootstrap(
    cluster_differences: list[list[float]], draws: int, seed: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        selected = [rng.choice(cluster_differences) for _ in cluster_differences]
        flattened = [value for cluster in selected for value in cluster]
        samples.append(mean(flattened))
    return quantile(samples, 0.025), quantile(samples, 0.975)


def exact_cluster_sign_flip(cluster_sums: list[float]) -> float:
    observed = abs(sum(cluster_sums))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(cluster_sums)):
        total += 1
        if abs(sum(sign * value for sign, value in zip(signs, cluster_sums))) >= observed - 1e-12:
            exceed += 1
    return exceed / total


def retention_table(
    runs: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results = [json.loads(verify_run_files(run)) for run in runs]
    for run, result in zip(runs, results):
        if result.get("status") != "complete_online_retention":
            raise ValueError("retention input is not a complete run")
        if result.get("events") != 320:
            raise ValueError("matched retention table requires all 320 events")
        model = result.get("retrieval_model")
        composer = result.get("composer_contract")
        if (
            not isinstance(model, dict)
            or model.get("kind") != "fixed_online_logistic"
            or model.get("dimension") != 256
            or not isinstance(composer, dict)
            or composer.get("model") != "gpt-4.1-2025-04-14"
            or not composer.get("catalog_sha256")
            or result.get("checker_contract")
            != "public-multiservice-challenge-checker-v1"
        ):
            raise ValueError("matched retention run lacks paper execution contracts")
        matched_methods = tuple(
            (method, label)
            for method, label in METHOD_LABELS.items()
            if method in result.get("methods", [])
        )
        if not matched_methods:
            raise ValueError("retention run has no matched objective method")
        verify_run_metrics(run, result, matched_methods)
    comparable = (
        "dataset", "seed", "storage_budget", "retrieval_budget",
        "workload_window", "retrieval_model", "composer_contract",
        "checker_contract", "source_executions_sha256",
    )
    if any(
        result.get(key) != results[0].get(key)
        for result in results[1:]
        for key in comparable
    ):
        raise ValueError("retention runs have unmatched dataset or execution settings")
    summaries = {
        row["method"]: row
        for result in results
        for row in result["summaries"]
        if row["method"] in METHOD_LABELS
    }
    if sum(
        row["method"] in METHOD_LABELS
        for result in results for row in result["summaries"]
    ) != len(summaries):
        raise ValueError("retention objective appears in multiple runs")
    if set(METHOD_LABELS).difference(summaries):
        raise ValueError("retention run is missing a matched objective")
    rows = []
    objectives = {
        "item_value_retention": "Sum of singleton marginal utilities",
        "bundle_value_retention": "Best fixed complete-bundle utility",
        "corem_retention": "Deployed-retriever selected-set utility",
    }
    for method in METHOD_LABELS:
        summary = summaries[method]
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "objective": objectives[method],
                "n": int(summary["n"]),
                "vsr": float(summary["vsr"]),
                "retrievable_complete_evidence_rate": float(
                    summary["retrievable_complete_evidence_rate"]
                ),
                "mean_churn_records_per_update": float(
                    summary["mean_churn_records_per_update"]
                ),
                "mean_update_latency_ms": float(summary["mean_update_latency_ms"]),
                "p95_update_latency_ms": float(summary["p95_update_latency_ms"]),
            }
        )
    return rows, results


def online_contrast(runs: list[Path], draws: int, seed: int) -> dict[str, Any]:
    scored = [
        row
        for run in runs
        for row in load_jsonl(run / "scored_executions.jsonl")
        if row["method"] in {"corem_retention", "bundle_value_retention"}
    ]
    by_method: dict[str, dict[int, int]] = defaultdict(dict)
    for row in scored:
        method = str(row["method"])
        if method in {"corem_retention", "bundle_value_retention"}:
            by_method[method][int(row["event_index"])] = int(row["verified_success"])
    indices = sorted(by_method["corem_retention"])
    if indices != list(range(1, 321)) or indices != sorted(by_method["bundle_value_retention"]):
        raise ValueError("online contrast does not contain the same 320 events")
    differences = [
        by_method["corem_retention"][index]
        - by_method["bundle_value_retention"][index]
        for index in indices
    ]
    blocks = [differences[start : start + 20] for start in range(0, 320, 20)]
    low, high = cluster_bootstrap(blocks, draws, seed)
    return {
        "scenario": "Matched retention-objective ablation",
        "contrast": "CoReM retention - Bundle-value retention",
        "tasks": 320,
        "clusters": 16,
        "cluster_unit": "non-overlapping chronological 20-event blocks",
        "delta_vsr_pp": 100.0 * mean(differences),
        "ci_low_pp": 100.0 * low,
        "ci_high_pp": 100.0 * high,
        "raw_p": exact_cluster_sign_flip([sum(block) for block in blocks]),
    }


def independent_contrast(path: Path) -> dict[str, Any]:
    verify_external_source(path)
    result = load_json(path)
    contrast = result["paired_contrasts"][
        "graph_complement_minus_candidate_bm25_sum"
    ]
    family_values = list(contrast["by_family_net_successes"].values())
    return {
        "scenario": "Independent new-task",
        "contrast": "CoReM - Candidate BM25-sum",
        "tasks": 128,
        "clusters": 16,
        "cluster_unit": "task family",
        "delta_vsr_pp": 100.0 * float(contrast["difference"]),
        "ci_low_pp": 100.0 * float(contrast["family_cluster_95ci"][0]),
        "ci_high_pp": 100.0 * float(contrast["family_cluster_95ci"][1]),
        "raw_p": exact_cluster_sign_flip([float(value) for value in family_values]),
        "source": str(path),
    }


def _e2_family(event_id: str) -> str:
    parts = event_id.split(":")
    if len(parts) < 3 or parts[0] != "v10-e2":
        raise ValueError(f"unexpected E2 event id: {event_id}")
    return parts[1]


def e2_contrast(
    draws: int, seed: int, full: Path, additive: Path
) -> dict[str, Any]:
    paths = {"full": full, "additive": additive}
    outcomes: dict[str, dict[str, int]] = {}
    for method, path in paths.items():
        verify_external_source(path)
        outcomes[method] = {
            str(row["event_id"]): int(row["checker"]["goal_satisfied"] is True)
            for row in load_jsonl(path)
        }
    event_ids = sorted(outcomes["full"])
    if len(event_ids) != 128 or event_ids != sorted(outcomes["additive"]):
        raise ValueError("E2 methods do not contain the same 128 events")
    clustered: dict[str, list[float]] = defaultdict(list)
    for event_id in event_ids:
        clustered[_e2_family(event_id)].append(
            outcomes["full"][event_id] - outcomes["additive"][event_id]
        )
    clusters = [clustered[key] for key in sorted(clustered)]
    if len(clusters) != 16:
        raise ValueError("E2 does not contain 16 families")
    differences = [value for cluster in clusters for value in cluster]
    low, high = cluster_bootstrap(clusters, draws, seed)
    return {
        "scenario": "E2",
        "contrast": "Full - w/o set interactions",
        "tasks": 128,
        "clusters": 16,
        "cluster_unit": "task family",
        "delta_vsr_pp": 100.0 * mean(differences),
        "ci_low_pp": 100.0 * low,
        "ci_high_pp": 100.0 * high,
        "raw_p": exact_cluster_sign_flip([sum(cluster) for cluster in clusters]),
        "sources": {key: str(value) for key, value in paths.items()},
    }


def pending_rows() -> list[dict[str, Any]]:
    return [
        {
            "scenario": "Enhanced-tau3",
            "contrast": "CoReM - HGMem reference-code adapter",
            "tasks": 43,
            "clusters": None,
            "status": "pending_reference_code_adapter",
            "reason": "Only the method-inspired adapted HGMem result is available.",
        },
        {
            "scenario": "E1, 20 candidates",
            "contrast": "CoReM - Learned rerank + closure",
            "tasks": 36,
            "clusters": 36,
            "status": "pending_comparator",
            "reason": "The CoReM five-seed run exists; no matched reranker output exists.",
        },
    ]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: Any, digits: int = 1) -> str:
    return "Pending" if value is None else f"{float(value):.{digits}f}"


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")
    if args.bootstrap_draws < 1000:
        raise ValueError("use at least 1,000 bootstrap draws")
    table1, retention_results = retention_table(args.retention_runs)
    if (args.e2_full is None) != (args.e2_additive is None):
        raise ValueError("provide both E2 execution files or neither")
    table2 = pending_rows() + [
        online_contrast(args.retention_runs, args.bootstrap_draws, args.seed + 2),
    ]
    table2.append(
        independent_contrast(args.independent_contrast)
        if args.independent_contrast is not None
        else {
            "scenario": "Independent new-task", "contrast": "CoReM - Candidate BM25-sum",
            "tasks": 128, "clusters": 16, "status": "pending_source",
            "reason": "A verified independent contrast was not supplied.",
        }
    )
    table2.append(
        e2_contrast(args.bootstrap_draws, args.seed + 1, args.e2_full, args.e2_additive)
        if args.e2_full is not None and args.e2_additive is not None
        else {
            "scenario": "E2", "contrast": "Full - w/o set interactions",
            "tasks": 128, "clusters": 16, "status": "pending_source",
            "reason": "Matched E2 execution files were not supplied.",
        }
    )
    order = {
        "Enhanced-tau3": 0,
        "Independent new-task": 1,
        "E1, 20 candidates": 2,
        "E2": 3,
        "Matched retention-objective ablation": 4,
    }
    table2.sort(key=lambda row: order[row["scenario"]])
    for row in table2:
        row["corrected_p"] = (
            min(1.0, 5.0 * float(row["raw_p"]))
            if row.get("raw_p") is not None
            else None
        )
    args.output.mkdir(parents=True)
    write_csv(
        args.output / "table1_retention_objectives.csv",
        table1,
        [
            "label",
            "objective",
            "n",
            "vsr",
            "retrievable_complete_evidence_rate",
            "mean_churn_records_per_update",
            "mean_update_latency_ms",
            "p95_update_latency_ms",
        ],
    )
    write_csv(
        args.output / "table2_paired_statistics.csv",
        table2,
        [
            "scenario",
            "contrast",
            "tasks",
            "clusters",
            "cluster_unit",
            "delta_vsr_pp",
            "ci_low_pp",
            "ci_high_pp",
            "raw_p",
            "corrected_p",
            "status",
            "reason",
        ],
    )
    table1_tex = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Retention policy & Bank-value objective & VSR $\uparrow$ & RCE $\uparrow$ & Churn $\downarrow$ & Update ms $\downarrow$ \\",
        r"\midrule",
    ]
    for row in table1:
        table1_tex.append(
            f"{row['label']} & {row['objective']} & "
            f"{100*row['vsr']:.1f} & "
            f"{100*row['retrievable_complete_evidence_rate']:.1f} & "
            f"{row['mean_churn_records_per_update']:.2f} & "
            f"{row['mean_update_latency_ms']:.1f} \\\\"
        )
    table1_tex += [r"\bottomrule", r"\end{tabular}"]
    (args.output / "table1_retention_objectives.tex").write_text(
        "\n".join(table1_tex) + "\n"
    )
    table2_tex = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Scenario & Paired contrast & Tasks/clusters & $\Delta$VSR & 95\% CI & Corrected $p$ \\",
        r"\midrule",
    ]
    for row in table2:
        count = f"{row['tasks']}/{row.get('clusters') or 'Pending'}"
        delta = format_number(row.get("delta_vsr_pp"))
        interval = (
            "Pending"
            if row.get("ci_low_pp") is None
            else f"[{row['ci_low_pp']:.1f}, {row['ci_high_pp']:.1f}]"
        )
        corrected = (
            "Pending"
            if row.get("corrected_p") is None
            else f"{row['corrected_p']:.4f}"
        )
        table2_tex.append(
            f"{row['scenario']} & {row['contrast']} & {count} & "
            f"{delta} & {interval} & {corrected} \\\\"
        )
    table2_tex += [r"\bottomrule", r"\end{tabular}"]
    (args.output / "table2_paired_statistics.tex").write_text(
        "\n".join(table2_tex) + "\n"
    )
    result = {
        "status": "complete_with_two_external_comparators_pending",
        "table1": table1,
        "table2": table2,
        "statistical_contract": {
            "bootstrap_draws": args.bootstrap_draws,
            "online_blocks": "16 fixed non-overlapping chronological blocks of 20 events",
            "offline_clusters": "task families where available",
            "model_ensemble": "one frozen policy, never five independent test samples",
            "multiplicity": (
                "Two-sided cluster sign-flip p-values use Bonferroni correction "
                "for the five prespecified comparisons; missing comparisons "
                "remain pending without changing the factor of five."
            ),
        },
        "retention_source_sha256": {
            str(run): hashlib.sha256((run / "result.json").read_bytes()).hexdigest()
            for run in args.retention_runs
        },
        "retention_protocol": retention_results[0]["retention_objective_contract"],
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    hashes = []
    for path in sorted(args.output.iterdir()):
        if path.name == "checksums.sha256":
            continue
        hashes.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (args.output / "checksums.sha256").write_text("\n".join(hashes) + "\n")
    print(f"wrote matched retention and paired-statistics tables to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
