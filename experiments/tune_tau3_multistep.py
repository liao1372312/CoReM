#!/usr/bin/env python3
"""Select tau3 multi-step HGT epochs using validation tasks only."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.dataset import PublicCoremDataset  # noqa: E402
from corem.model import TrainingConfig  # noqa: E402
from corem.torch_encoding import TorchHGTEncoder  # noqa: E402
from corem.torch_model import TorchSetUtilityModel  # noqa: E402
from corem.types import OutcomeExample  # noqa: E402
from experiments.common import binary_metrics  # noqa: E402
from experiments.run_tau3_multistep import build_graph, predict  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/tau3_corem"))
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/results_tau3_tuning")
    )
    parser.add_argument("--seeds", default="2027,2029,2039")
    parser.add_argument("--epochs", default="4,8,12,20")
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cuda")
    parser.add_argument("--include-duplicates", action="store_true")
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def labels(examples: list[OutcomeExample]) -> list[int]:
    return [example.outcome for example in examples]


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    epoch_grid = [int(value) for value in args.epochs.split(",") if value.strip()]
    dataset = PublicCoremDataset(args.dataset)
    train = list(dataset.iter_outcome_examples(splits={"train"}))
    validation = list(dataset.iter_outcome_examples(splits={"val"}))
    train_graph = build_graph(
        dataset,
        train,
        include_hyperedges=True,
        include_duplicates=args.include_duplicates,
    )
    validation_graph = build_graph(
        dataset,
        validation,
        include_hyperedges=True,
        include_duplicates=args.include_duplicates,
    )
    runs = []
    started = time.perf_counter()
    for epochs in epoch_grid:
        for seed in seeds:
            print(f"epochs={epochs} seed={seed}", flush=True)
            model = TorchSetUtilityModel(
                TorchHGTEncoder(
                    dimension=32,
                    text_hash_dimension=32,
                    text_feature_mode="tokens",
                    message_layers=2,
                    attention_heads=4,
                    seed=seed,
                    device=args.device,
                )
            )
            fit = model.fit(
                train,
                train_graph,
                TrainingConfig(
                    learning_rate=0.01,
                    epochs=epochs,
                    calibration_weight=0.1,
                    relation_weight=0.1,
                    interaction_weight=0.0,
                    class_balance_power=1.0,
                ),
            )
            probabilities = predict(model, validation, validation_graph)
            metrics = binary_metrics(labels(validation), probabilities, 0.5)
            runs.append(
                {
                    "epochs": epochs,
                    "seed": seed,
                    "final_loss": fit.losses[-1],
                    "validation": metrics,
                }
            )
    aggregate = {}
    for epochs in epoch_grid:
        members = [run for run in runs if run["epochs"] == epochs]
        aggregate[str(epochs)] = {
            metric: summarize(
                [float(run["validation"][metric]) for run in members]
            )
            for metric in ("roc_auc", "brier", "log_loss")
        }
    selected = max(
        epoch_grid,
        key=lambda value: (
            aggregate[str(value)]["roc_auc"]["mean"],
            -aggregate[str(value)]["brier"]["mean"],
            -value,
        ),
    )
    result = {
        "status": "complete",
        "selection_data": "validation only; test outcomes were not loaded",
        "dataset": {
            "name": dataset.summary.name,
            "schema_version": dataset.summary.schema_version,
            "integrity": dataset.validate_integrity(),
            "examples": {"train": len(train), "validation": len(validation)},
            "tasks": {
                "train": len({item.request.request_id for item in train}),
                "validation": len(
                    {item.request.request_id for item in validation}
                ),
            },
        },
        "seeds": seeds,
        "epoch_grid": epoch_grid,
        "fixed_configuration": {
            "dimension": 32,
            "text_hash_dimension": 32,
            "message_layers": 2,
            "learning_rate": 0.01,
            "device": args.device,
            "include_duplicates": args.include_duplicates,
        },
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": args.device,
            "gpu_name": (
                torch.cuda.get_device_name(torch.device(args.device))
                if args.device.startswith("cuda")
                else None
            ),
        },
        "aggregate": aggregate,
        "selected_epochs": selected,
        "runs": runs,
        "elapsed_seconds": time.perf_counter() - started,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "tau3_validation_tuning.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output / "checksums.sha256").write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps({"aggregate": aggregate, "selected_epochs": selected}, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
