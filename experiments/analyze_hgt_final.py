"""Analyze final multi-seed HGT results and paired task-level predictions."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.common import (
    binary_metrics,
    holm_adjust,
    paired_randomization_pvalue,
    percentile,
)

METRICS = ("roc_auc", "brier", "ece_10", "balanced_accuracy", "log_loss")
HIGHER_IS_BETTER = frozenset(("roc_auc", "balanced_accuracy"))
PROBABILITY_METRICS = ("roc_auc", "brier", "ece_10", "log_loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("experiments/results_hgt_final")
    )
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    summary = json.loads(
        (args.results / "endpoint_hgt_multiseed.json").read_text(encoding="utf-8")
    )
    seeds = summary["seeds"]
    results = {
        seed: json.loads(
            (args.results / f"seed_{seed}" / "endpoint_hgt_sample.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in seeds
    }

    raw_seed_table = []
    unadjusted: dict[str, float] = {}
    seed_comparison = {}
    for seed in seeds:
        row: dict[str, object] = {"seed": seed}
        for metric in METRICS:
            row[f"hgt_{metric}"] = results[seed]["hgt"]["test"][metric]
            row[f"reference_{metric}"] = results[seed]["stable_hash_reference"][
                "test"
            ][metric]
        raw_seed_table.append(row)
    for metric in METRICS:
        differences = [
            float(row[f"hgt_{metric}"]) - float(row[f"reference_{metric}"])
            for row in raw_seed_table
        ]
        improvements = [
            value if metric in HIGHER_IS_BETTER else -value
            for value in differences
        ]
        pvalue = paired_randomization_pvalue(improvements)
        unadjusted[metric] = pvalue
        seed_comparison[metric] = {
            "mean_hgt_minus_reference": statistics.mean(differences),
            "sample_std": statistics.stdev(differences),
            "all_seeds_favor_hgt": all(value > 0.0 for value in improvements),
            "paired_randomization_p_two_sided": pvalue,
        }
    adjusted = holm_adjust(unadjusted)
    for metric, value in adjusted.items():
        seed_comparison[metric]["holm_adjusted_p"] = value

    per_seed_predictions: dict[int, dict[str, dict[str, object]]] = {}
    for seed in seeds:
        rows = read_jsonl(args.results / f"seed_{seed}" / "test_predictions.jsonl")
        indexed = {str(row["request_id"]): row for row in rows}
        if len(indexed) != summary["sample_sizes_per_seed"]["test"]:
            raise ValueError(f"duplicate or missing predictions for seed {seed}")
        per_seed_predictions[seed] = indexed
    request_ids = sorted(per_seed_predictions[seeds[0]])
    if any(set(rows) != set(request_ids) for rows in per_seed_predictions.values()):
        raise ValueError("test request ids differ across seeds")
    labels = []
    hgt_probabilities = []
    reference_probabilities = []
    for request_id in request_ids:
        seed_rows = [per_seed_predictions[seed][request_id] for seed in seeds]
        seed_labels = {int(row["label"]) for row in seed_rows}
        if len(seed_labels) != 1:
            raise ValueError(f"label mismatch for {request_id}")
        labels.append(seed_labels.pop())
        hgt_probabilities.append(
            statistics.mean(float(row["hgt_probability"]) for row in seed_rows)
        )
        reference_probabilities.append(
            statistics.mean(
                float(row["stable_hash_probability"]) for row in seed_rows
            )
        )

    hgt_ensemble = binary_metrics(labels, hgt_probabilities, 0.5)
    reference_ensemble = binary_metrics(labels, reference_probabilities, 0.5)
    rng = random.Random(args.seed)
    bootstrap_differences = {metric: [] for metric in PROBABILITY_METRICS}
    for _ in range(args.bootstrap_draws):
        positions = rng.choices(range(len(labels)), k=len(labels))
        sampled_labels = [labels[index] for index in positions]
        sampled_hgt = [hgt_probabilities[index] for index in positions]
        sampled_reference = [reference_probabilities[index] for index in positions]
        hgt_metrics = binary_metrics(sampled_labels, sampled_hgt, 0.5)
        reference_metrics = binary_metrics(
            sampled_labels, sampled_reference, 0.5
        )
        for metric in PROBABILITY_METRICS:
            bootstrap_differences[metric].append(
                hgt_metrics[metric] - reference_metrics[metric]
            )
    task_bootstrap = {
        metric: {
            "ensemble_hgt": hgt_ensemble[metric],
            "ensemble_reference": reference_ensemble[metric],
            "hgt_minus_reference": hgt_ensemble[metric]
            - reference_ensemble[metric],
            "ci_95": [
                percentile(values, 0.025),
                percentile(values, 0.975),
            ],
        }
        for metric, values in bootstrap_differences.items()
    }

    analysis = {
        "status": "complete",
        "scope": "singleton endpoint prediction only",
        "seeds": seeds,
        "test_tasks": len(labels),
        "raw_seed_table": raw_seed_table,
        "seed_level_paired_comparison": seed_comparison,
        "task_bootstrap_on_seed_ensemble": {
            "draws": args.bootstrap_draws,
            "metrics": task_bootstrap,
            "note": (
                "Task bootstrap quantifies held-out task uncertainty for the "
                "seed-averaged predictor; it is not a workflow-composition test."
            ),
        },
    }
    output = args.results / "final_analysis.json"
    output.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(analysis, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
