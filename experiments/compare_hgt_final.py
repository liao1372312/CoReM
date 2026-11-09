"""Compare two frozen multi-seed HGT result directories."""

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
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def load_seed_results(root: Path, seeds: list[int]) -> dict[int, dict]:
    return {
        seed: json.loads(
            (root / f"seed_{seed}" / "endpoint_hgt_sample.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in seeds
    }


def ensemble(root: Path, seeds: list[int]) -> tuple[list[int], list[float]]:
    per_seed = {}
    for seed in seeds:
        path = root / f"seed_{seed}" / "test_predictions.jsonl"
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        per_seed[seed] = {str(row["request_id"]): row for row in rows}
    request_ids = sorted(per_seed[seeds[0]])
    if any(set(per_seed[seed]) != set(request_ids) for seed in seeds):
        raise ValueError(f"test request ids differ across seeds in {root}")
    labels = []
    probabilities = []
    for request_id in request_ids:
        rows = [per_seed[seed][request_id] for seed in seeds]
        observed = {int(row["label"]) for row in rows}
        if len(observed) != 1:
            raise ValueError(f"label mismatch for {request_id} in {root}")
        labels.append(observed.pop())
        probabilities.append(
            statistics.mean(float(row["hgt_probability"]) for row in rows)
        )
    return labels, probabilities


def main() -> None:
    args = parse_args()
    baseline_summary = json.loads(
        (args.baseline / "endpoint_hgt_multiseed.json").read_text(encoding="utf-8")
    )
    candidate_summary = json.loads(
        (args.candidate / "endpoint_hgt_multiseed.json").read_text(encoding="utf-8")
    )
    seeds = [int(seed) for seed in candidate_summary["seeds"]]
    if seeds != [int(seed) for seed in baseline_summary["seeds"]]:
        raise ValueError("baseline and candidate seeds differ")
    baseline = load_seed_results(args.baseline, seeds)
    candidate = load_seed_results(args.candidate, seeds)

    raw_seed_table = []
    pvalues = {}
    comparisons = {}
    for seed in seeds:
        row: dict[str, int | float] = {"seed": seed}
        for metric in METRICS:
            old = float(baseline[seed]["hgt"]["test"][metric])
            new = float(candidate[seed]["hgt"]["test"][metric])
            row[f"baseline_{metric}"] = old
            row[f"candidate_{metric}"] = new
            row[f"candidate_minus_baseline_{metric}"] = new - old
        raw_seed_table.append(row)
    for metric in METRICS:
        differences = [
            float(row[f"candidate_minus_baseline_{metric}"])
            for row in raw_seed_table
        ]
        improvements = [
            value if metric in HIGHER_IS_BETTER else -value
            for value in differences
        ]
        pvalues[metric] = paired_randomization_pvalue(improvements)
        comparisons[metric] = {
            "mean_candidate_minus_baseline": statistics.mean(differences),
            "sample_std": statistics.stdev(differences),
            "seeds_favoring_candidate": sum(value > 0.0 for value in improvements),
            "paired_randomization_p_two_sided": pvalues[metric],
        }
    for metric, adjusted in holm_adjust(pvalues).items():
        comparisons[metric]["holm_adjusted_p"] = adjusted

    baseline_labels, baseline_probabilities = ensemble(args.baseline, seeds)
    candidate_labels, candidate_probabilities = ensemble(args.candidate, seeds)
    if baseline_labels != candidate_labels:
        raise ValueError("baseline and candidate test labels differ")
    labels = baseline_labels
    baseline_metrics = binary_metrics(labels, baseline_probabilities, 0.5)
    candidate_metrics = binary_metrics(labels, candidate_probabilities, 0.5)
    rng = random.Random(args.seed)
    draws = {metric: [] for metric in PROBABILITY_METRICS}
    for _ in range(args.bootstrap_draws):
        positions = rng.choices(range(len(labels)), k=len(labels))
        sampled_labels = [labels[index] for index in positions]
        sampled_baseline = [baseline_probabilities[index] for index in positions]
        sampled_candidate = [candidate_probabilities[index] for index in positions]
        old = binary_metrics(sampled_labels, sampled_baseline, 0.5)
        new = binary_metrics(sampled_labels, sampled_candidate, 0.5)
        for metric in PROBABILITY_METRICS:
            draws[metric].append(new[metric] - old[metric])

    result = {
        "status": "complete",
        "scope": "frozen singleton endpoint predictors only",
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "seeds": seeds,
        "raw_seed_table": raw_seed_table,
        "paired_seed_comparison": comparisons,
        "task_bootstrap_on_seed_ensembles": {
            "draws": args.bootstrap_draws,
            "metrics": {
                metric: {
                    "baseline": baseline_metrics[metric],
                    "candidate": candidate_metrics[metric],
                    "candidate_minus_baseline": candidate_metrics[metric]
                    - baseline_metrics[metric],
                    "ci_95": [
                        percentile(draws[metric], 0.025),
                        percentile(draws[metric], 0.975),
                    ],
                }
                for metric in PROBABILITY_METRICS
            },
        },
    }
    output = args.candidate / "optimization_comparison.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
