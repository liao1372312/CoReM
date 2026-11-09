#!/usr/bin/env python3
"""Merge compatible tau3 multi-seed shards and recompute all summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.dataset import PublicCoremDataset
from experiments.run_tau3_multistep import (
    brier,
    log_loss,
    metric_summary,
    paired_task_bootstrap,
    roc_auc,
    rounded_metrics,
    sign_flip_pvalue,
    task_pairwise_accuracy,
)


METRICS = (
    "roc_auc",
    "brier",
    "ece_10",
    "balanced_accuracy",
    "log_loss",
    "task_pairwise_accuracy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    candidate = path / "tau3_multistep_multiseed.json" if path.is_dir() else path
    return json.loads(candidate.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    shards = [load_result(path) for path in args.inputs]
    first = shards[0]
    compatibility_keys = (
        "epochs",
        "dimension",
        "text_hash_dimension",
        "message_layers",
        "learning_rate",
        "calibration",
        "outcome_derived_oracle_edges_used",
        "gold_operation_ids_used_as_request_features",
        "duplicate_pairs_used",
    )
    for shard in shards[1:]:
        if shard["dataset"] != first["dataset"]:
            raise ValueError("dataset metadata differs across shards")
        for key in compatibility_keys:
            if shard["configuration"].get(key) != first["configuration"].get(key):
                raise ValueError(f"configuration differs for {key}")

    by_seed: dict[int, dict[str, Any]] = {}
    for shard in shards:
        for run in shard["per_seed"]:
            seed = int(run["seed"])
            if seed in by_seed:
                raise ValueError(f"duplicate seed {seed}")
            by_seed[seed] = run
    per_seed = [by_seed[seed] for seed in sorted(by_seed)]
    methods = tuple(first["aggregate"])
    if any(tuple(shard["aggregate"]) != methods for shard in shards):
        raise ValueError("method lists differ across shards")

    aggregate = {
        method: {
            metric: metric_summary(
                [float(run["methods"][method]["test"][metric]) for run in per_seed]
            )
            for metric in METRICS
        }
        for method in methods
    }
    references = tuple(method for method in methods if method != "hgt_corem")
    comparisons: dict[str, Any] = {}
    for reference in references:
        comparisons[reference] = {}
        for metric in METRICS:
            differences = [
                float(run["methods"]["hgt_corem"]["test"][metric])
                - float(run["methods"][reference]["test"][metric])
                for run in per_seed
            ]
            comparisons[reference][metric] = {
                "hgt_corem_minus_reference": metric_summary(differences),
                "exact_sign_flip_pvalue": sign_flip_pvalue(differences),
            }

    dataset = PublicCoremDataset(args.dataset)
    test = list(dataset.iter_outcome_examples(splits={"test"}))
    ensemble_rows: list[dict[str, Any]] = []
    for index, example in enumerate(test):
        row: dict[str, Any] = {
            "task_id": example.request.request_id,
            "label": example.outcome,
        }
        for method in methods:
            row[method] = statistics.mean(
                float(run["methods"][method]["test_probabilities"][index])
                for run in per_seed
            )
        ensemble_rows.append(row)

    ensemble: dict[str, Any] = {}
    for method in methods:
        probabilities = [float(row[method]) for row in ensemble_rows]
        threshold = statistics.mean(
            float(run["methods"][method]["threshold"]) for run in per_seed
        )
        ensemble[method] = rounded_metrics(test, probabilities, threshold)
        ensemble[method]["task_pairwise_accuracy"] = round(
            task_pairwise_accuracy(test, probabilities), 8
        )

    statistics_map = {"roc_auc": roc_auc, "brier": brier, "log_loss": log_loss}
    bootstrap = {
        reference: {
            metric: paired_task_bootstrap(
                ensemble_rows,
                "hgt_corem",
                reference,
                statistic,
                draws=args.bootstrap_draws,
                seed=2027,
            )
            for metric, statistic in statistics_map.items()
        }
        for reference in references
    }

    configuration = dict(first["configuration"])
    configuration["seeds"] = sorted(by_seed)
    configuration["workers"] = max(
        int(shard["configuration"].get("workers", 1)) for shard in shards
    )
    configuration["bootstrap_draws"] = args.bootstrap_draws
    configuration["merged_seed_shards"] = len(shards)
    limitations = [
        text
        for text in first.get("limitations", [])
        if "near-sufficient structural rule" not in text
    ]
    limitations.append(
        "The augmented dataset includes execution-verified complete-but-failing counterexamples; it remains a controlled tau3 derivative."
    )
    result = {
        **first,
        "configuration": configuration,
        "aggregate": aggregate,
        "paired_seed_comparisons": comparisons,
        "ensemble_test": ensemble,
        "task_clustered_bootstrap_95ci_hgt_minus_reference": bootstrap,
        "per_seed": per_seed,
        "elapsed_seconds": sum(float(shard.get("elapsed_seconds", 0.0)) for shard in shards),
        "limitations": limitations,
        "merge_input_sha256": {
            str(path): hashlib.sha256(
                (path / "tau3_multistep_multiseed.json" if path.is_dir() else path).read_bytes()
            ).hexdigest()
            for path in args.inputs
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "tau3_multistep_multiseed.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum = hashlib.sha256(output_path.read_bytes()).hexdigest()
    (args.output / "checksums.sha256").write_text(
        f"{checksum}  {output_path.name}\n", encoding="utf-8"
    )
    print(json.dumps({"seeds": sorted(by_seed), "aggregate": aggregate}, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
