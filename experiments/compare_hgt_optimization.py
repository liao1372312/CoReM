"""Compare validation tuning and paired pre/post HGT test results."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.common import paired_randomization_pvalue

METRICS = ("roc_auc", "brier", "ece_10", "balanced_accuracy", "log_loss")
HIGHER_IS_BETTER = frozenset(("roc_auc", "balanced_accuracy"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("experiments/results_hgt"))
    parser.add_argument(
        "--optimized", type=Path, default=Path("experiments/results_hgt_optimized")
    )
    parser.add_argument(
        "--tuning-pattern",
        default="experiments/results_hgt/tuning_hgt_validation*.json",
    )
    return parser.parse_args()


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    tuning_paths = sorted(Path().glob(args.tuning_pattern))
    if not tuning_paths:
        raise FileNotFoundError(args.tuning_pattern)
    tuning = [json.loads(path.read_text(encoding="utf-8")) for path in tuning_paths]
    configurations: dict[tuple[int, float], list[float]] = {}
    for run in tuning:
        if run["test_split_accessed"]:
            raise ValueError("tuning artifact reports test access")
        for trial in run["trials"]:
            key = (trial["epochs"], trial["class_balance_power"])
            configurations.setdefault(key, []).append(trial["selection"]["roc_auc"])
    tuning_summary = [
        {
            "epochs": key[0],
            "class_balance_power": key[1],
            "selection_roc_auc": describe(values),
        }
        for key, values in sorted(configurations.items())
    ]
    chosen = max(
        tuning_summary,
        key=lambda row: (
            row["selection_roc_auc"]["mean"],
            -row["epochs"],
            -row["class_balance_power"],
        ),
    )

    optimized_summary = json.loads(
        (args.optimized / "endpoint_hgt_multiseed.json").read_text(encoding="utf-8")
    )
    seeds = optimized_summary["seeds"]
    paired: dict[str, dict[str, object]] = {}
    for metric in METRICS:
        baseline_values = []
        optimized_values = []
        for seed in seeds:
            relative = Path(f"seed_{seed}/endpoint_hgt_sample.json")
            baseline = json.loads((args.baseline / relative).read_text(encoding="utf-8"))
            optimized = json.loads((args.optimized / relative).read_text(encoding="utf-8"))
            baseline_values.append(float(baseline["hgt"]["test"][metric]))
            optimized_values.append(float(optimized["hgt"]["test"][metric]))
        raw_differences = [
            current - previous
            for previous, current in zip(baseline_values, optimized_values)
        ]
        improvements = [
            difference if metric in HIGHER_IS_BETTER else -difference
            for difference in raw_differences
        ]
        paired[metric] = {
            "baseline": describe(baseline_values),
            "optimized": describe(optimized_values),
            "optimized_minus_baseline": describe(raw_differences),
            "improvement_direction": (
                "positive" if metric in HIGHER_IS_BETTER else "negative"
            ),
            "paired_randomization_p_two_sided": paired_randomization_pvalue(
                improvements
            ),
        }

    result = {
        "status": "complete",
        "tuning": {
            "test_split_accessed": False,
            "seeds": [run["seed"] for run in tuning],
            "chosen": {
                "epochs": chosen["epochs"],
                "class_balance_power": chosen["class_balance_power"],
            },
            "configurations": tuning_summary,
        },
        "paired_test_comparison": {
            "seeds": seeds,
            "metrics": paired,
            "warning": (
                "Exploratory five-seed comparison; no metric reaches a two-sided "
                "paired-randomization p-value below 0.05."
            ),
        },
    }
    output = args.optimized / "optimization_comparison.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
