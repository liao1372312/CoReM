"""Run and summarize the sampled private endpoint HGT experiment over seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/private_corem"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results_hgt"))
    parser.add_argument("--seeds", default="17,29,41,53,67,79,97,109,127,149")
    parser.add_argument("--train-size", type=int, default=3138)
    parser.add_argument(
        "--train-sampling",
        choices=("uniform", "balanced"),
        default="balanced",
    )
    parser.add_argument("--train-negative-fraction", type=float, default=0.5)
    parser.add_argument("--val-size", type=int, default=1627)
    parser.add_argument("--test-size", type=int, default=1706)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--text-hash-dimension", type=int, default=32)
    parser.add_argument(
        "--text-feature-mode", choices=("tokens", "subword"), default="tokens"
    )
    parser.add_argument("--message-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--relation-weight", type=float, default=0.1)
    parser.add_argument("--class-balance-power", type=float, default=0.0)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="recompute only the aggregate from existing seed result files",
    )
    return parser.parse_args()


def metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    script = Path(__file__).with_name("run_hgt_endpoint.py")
    args.output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    )
    started = time.perf_counter()
    results = []
    for index, seed in enumerate(seeds, start=1):
        seed_output = args.output / f"seed_{seed}"
        result_path = seed_output / "endpoint_hgt_sample.json"
        if not args.reuse_existing:
            command = [
                sys.executable,
                str(script),
                "--dataset",
                str(args.dataset),
                "--output",
                str(seed_output),
                "--train-size",
                str(args.train_size),
                "--train-sampling",
                args.train_sampling,
                "--train-negative-fraction",
                str(args.train_negative_fraction),
                "--val-size",
                str(args.val_size),
                "--test-size",
                str(args.test_size),
                "--epochs",
                str(args.epochs),
                "--dimension",
                str(args.dimension),
                "--text-hash-dimension",
                str(args.text_hash_dimension),
                "--text-feature-mode",
                args.text_feature_mode,
                "--message-layers",
                str(args.message_layers),
                "--learning-rate",
                str(args.learning_rate),
                "--relation-weight",
                str(args.relation_weight),
                "--class-balance-power",
                str(args.class_balance_power),
                "--seed",
                str(seed),
            ]
            print(f"[{index}/{len(seeds)}] seed={seed}", flush=True)
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            if not result_path.is_file():
                raise RuntimeError(completed.stdout + completed.stderr)
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        expected = {
            "seed": seed,
            "dimension": args.dimension,
            "text_hash_dimension": args.text_hash_dimension,
            "text_feature_mode": args.text_feature_mode,
            "message_layers": args.message_layers,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "relation_weight": args.relation_weight,
            "class_balance_power": args.class_balance_power,
            "train_sampling": args.train_sampling,
            "train_negative_fraction": args.train_negative_fraction,
        }
        mismatches = {
            key: {"expected": value, "found": result.get(key)}
            for key, value in expected.items()
            if result.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"seed artifact configuration mismatch in {result_path}: "
                f"{mismatches}"
            )
        results.append(result)

    metrics = ("roc_auc", "brier", "ece_10", "balanced_accuracy", "log_loss")
    methods = ("hgt", "stable_hash_reference")
    summary = {
        "status": "complete",
        "scope": "observed singleton endpoint prediction only",
        "seeds": seeds,
        "sample_sizes_per_seed": {
            "train": args.train_size,
            "validation": args.val_size,
            "test": args.test_size,
        },
        "epochs": args.epochs,
        "dimension": args.dimension,
        "text_hash_dimension": args.text_hash_dimension,
        "text_feature_mode": args.text_feature_mode,
        "message_layers": args.message_layers,
        "learning_rate": args.learning_rate,
        "relation_weight": args.relation_weight,
        "class_balance_power": args.class_balance_power,
        "train_sampling": args.train_sampling,
        "train_negative_fraction": args.train_negative_fraction,
        "reused_existing_seed_runs": args.reuse_existing,
        "code_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path("corem/dataset.py"),
                Path("corem/encoding.py"),
                Path("corem/model.py"),
                Path("corem/torch_encoding.py"),
                Path("corem/torch_model.py"),
                Path("experiments/analyze_hgt_final.py"),
                Path("experiments/compare_hgt_final.py"),
                Path("experiments/run_hgt_endpoint.py"),
                Path("experiments/run_hgt_endpoint_multiseed.py"),
                Path("experiments/tune_hgt_endpoint.py"),
            )
        },
        "methods": {
            method: {
                metric: metric_summary(
                    [float(result[method]["test"][metric]) for result in results]
                )
                for metric in metrics
            }
            for method in methods
        },
        "paired_hgt_minus_reference": {
            metric: metric_summary(
                [
                    float(result["hgt"]["test"][metric])
                    - float(result["stable_hash_reference"]["test"][metric])
                    for result in results
                ]
            )
            for metric in metrics
        },
        "per_seed_files": [
            str(Path(f"seed_{seed}") / "endpoint_hgt_sample.json") for seed in seeds
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "warning": (
            "Repeated sampled singleton prediction, not verified multi-operation "
            "composition or interaction evaluation."
        ),
    }
    summary_path = args.output / "endpoint_hgt_multiseed.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_files = sorted(
        path
        for path in args.output.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    (args.output / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(args.output)}\n"
            for path in checksum_files
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
