#!/usr/bin/env python3
"""Run multi-seed CoReM experiments on execution-verified tau3-bench sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import random
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.dataset import PublicCoremDataset  # noqa: E402
from corem.encoding import StableHashEncoder  # noqa: E402
from corem.model import SetUtilityModel, TrainingConfig  # noqa: E402
from corem.torch_encoding import TorchHGTEncoder  # noqa: E402
from corem.torch_model import TorchSetUtilityModel  # noqa: E402
from corem.types import EvidenceGraph, OutcomeExample  # noqa: E402
from experiments.common import binary_metrics, roc_auc, select_threshold  # noqa: E402
from experiments.run_hgt_endpoint import (  # noqa: E402
    apply_affine_calibration,
    select_affine_calibrator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/tau3_corem"))
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/results_tau3_multistep")
    )
    parser.add_argument("--seeds", default="17,29,41,53,67,79,97,109,127,149")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--text-hash-dimension", type=int, default=32)
    parser.add_argument("--message-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cuda")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="independent seed workers; use several workers for a large GPU",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument(
        "--calibration", choices=("none", "affine"), default="none"
    )
    parser.add_argument(
        "--include-duplicates",
        action="store_true",
        help="load duplicate pairs and add the A3 no-redundancy variant",
    )
    return parser.parse_args()


def build_graph(
    dataset: PublicCoremDataset,
    examples: list[OutcomeExample],
    *,
    include_hyperedges: bool,
    include_duplicates: bool = False,
) -> EvidenceGraph:
    memory_ids = frozenset(
        record_id for example in examples for record_id in example.selected_ids
    )
    return dataset.build_graph(
        memory_ids=memory_ids,
        include_operation_entities=True,
        include_duplicates=include_duplicates,
        include_hyperedges=include_hyperedges,
    )


def split_validation_by_task(
    examples: list[OutcomeExample], seed: int
) -> tuple[list[OutcomeExample], list[OutcomeExample]]:
    task_ids = sorted({example.request.request_id for example in examples})
    random.Random(seed).shuffle(task_ids)
    midpoint = len(task_ids) // 2
    selection_ids = set(task_ids[:midpoint])
    selection = [
        example
        for example in examples
        if example.request.request_id in selection_ids
    ]
    calibration = [
        example
        for example in examples
        if example.request.request_id not in selection_ids
    ]
    if not selection or not calibration:
        raise ValueError("validation requires at least two task clusters")
    return selection, calibration


def predict(
    model: TorchSetUtilityModel | SetUtilityModel,
    examples: list[OutcomeExample],
    graph: EvidenceGraph,
) -> list[float]:
    return [
        model.probability(example.request, example.selected_ids, graph)
        for example in examples
    ]


def rounded_metrics(
    examples: list[OutcomeExample], probabilities: list[float], threshold: float
) -> dict[str, float]:
    labels = [example.outcome for example in examples]
    return {
        key: round(value, 8)
        for key, value in binary_metrics(labels, probabilities, threshold).items()
    }


def task_pairwise_accuracy(
    examples: list[OutcomeExample], probabilities: list[float]
) -> float:
    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for example, probability in zip(examples, probabilities):
        groups[example.request.request_id].append((example.outcome, probability))
    scores = []
    for rows in groups.values():
        positives = [probability for label, probability in rows if label == 1]
        negatives = [probability for label, probability in rows if label == 0]
        for positive in positives:
            for negative in negatives:
                scores.append(
                    1.0 if positive > negative else 0.5 if positive == negative else 0.0
                )
    return sum(scores) / len(scores) if scores else float("nan")


def fit_method(
    method: str,
    seed: int,
    examples: dict[str, list[OutcomeExample]],
    dataset: PublicCoremDataset,
    args: argparse.Namespace,
) -> dict[str, Any]:
    include_hyperedges = method != "hgt_no_hyperedges"
    additive_encoder_matched = method == "hgt_additive_encoder_matched"
    no_redundancy = method == "hgt_no_redundancy"
    graphs = {
        split: build_graph(
            dataset,
            split_examples,
            include_hyperedges=include_hyperedges,
            include_duplicates=args.include_duplicates,
        )
        for split, split_examples in examples.items()
    }
    config = TrainingConfig(
        learning_rate=args.learning_rate if method != "stable_hash_reference" else 0.08,
        epochs=args.epochs if method != "stable_hash_reference" else 80,
        calibration_weight=0.1,
        relation_weight=0.1,
        interaction_weight=0.0,
        class_balance_power=1.0,
    )
    if method == "stable_hash_reference":
        model: TorchSetUtilityModel | SetUtilityModel = SetUtilityModel(
            encoder=StableHashEncoder()
        )
    else:
        model = TorchSetUtilityModel(
            TorchHGTEncoder(
                dimension=args.dimension,
                text_hash_dimension=args.text_hash_dimension,
                text_feature_mode="tokens",
                message_layers=args.message_layers,
                attention_heads=4,
                seed=seed,
                device=args.device,
                enable_complements=not additive_encoder_matched,
                enable_redundancy=(
                    not additive_encoder_matched and not no_redundancy
                ),
            )
        )
    started = time.perf_counter()
    fit = model.fit(examples["train"], graphs["train"], config)
    raw = {
        split: predict(model, split_examples, graphs[split])
        for split, split_examples in examples.items()
    }
    if args.calibration == "affine":
        scale, bias = select_affine_calibrator(
            examples["calibration"], raw["calibration"]
        )
        probabilities = {
            split: apply_affine_calibration(values, scale, bias)
            for split, values in raw.items()
        }
    else:
        scale, bias = 1.0, 0.0
        probabilities = raw
    threshold = select_threshold(
        [example.outcome for example in examples["selection"]],
        probabilities["selection"],
    )
    test_metrics = rounded_metrics(examples["test"], probabilities["test"], threshold)
    test_metrics["task_pairwise_accuracy"] = round(
        task_pairwise_accuracy(examples["test"], probabilities["test"]), 8
    )
    return {
        "fit": {
            "epochs": fit.epochs,
            "converged": fit.converged,
            "final_loss": fit.losses[-1],
            "elapsed_seconds": time.perf_counter() - started,
        },
        "calibration": {"scale": scale, "bias": bias},
        "threshold": threshold,
        "test": test_metrics,
        "test_probabilities": probabilities["test"],
    }


def fit_neural_seed(
    seed: int,
    dataset_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Fit both neural variants for one independent seed."""
    dataset = PublicCoremDataset(dataset_path)
    train = list(dataset.iter_outcome_examples(splits={"train"}))
    validation = list(dataset.iter_outcome_examples(splits={"val"}))
    test = list(dataset.iter_outcome_examples(splits={"test"}))
    if args.calibration == "affine":
        selection, calibration = split_validation_by_task(validation, 2027)
    else:
        selection, calibration = validation, []
    partitions = {
        "train": train,
        "selection": selection,
        "calibration": calibration,
        "test": test,
    }
    neural_methods = {
        "hgt_corem": fit_method(
            "hgt_corem", seed, partitions, dataset, args
        ),
        "hgt_additive_encoder_matched": fit_method(
            "hgt_additive_encoder_matched", seed, partitions, dataset, args
        ),
        "hgt_no_hyperedges": fit_method(
            "hgt_no_hyperedges", seed, partitions, dataset, args
        ),
    }
    if args.include_duplicates:
        neural_methods["hgt_no_redundancy"] = fit_method(
            "hgt_no_redundancy", seed, partitions, dataset, args
        )
    return {
        "seed": seed,
        "validation_task_counts": {
            "selection": len({item.request.request_id for item in selection}),
            "calibration": len({item.request.request_id for item in calibration}),
        },
        "methods": neural_methods,
    }


def metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def sign_flip_pvalue(differences: list[float]) -> float:
    observed = abs(statistics.mean(differences))
    count = len(differences)
    if count > 20:
        raise ValueError("exact sign-flip test is limited to 20 pairs")
    extreme = 0
    for mask in range(1 << count):
        value = statistics.mean(
            difference if mask & (1 << index) else -difference
            for index, difference in enumerate(differences)
        )
        extreme += abs(value) >= observed - 1e-15
    return extreme / (1 << count)


def paired_task_bootstrap(
    rows: list[dict[str, Any]],
    left: str,
    right: str,
    statistic: Callable[[list[int], list[float]], float],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["task_id"]].append(row)
    task_ids = sorted(groups)
    rng = random.Random(seed)
    differences = []
    for _ in range(draws):
        sampled = [rng.choice(task_ids) for _ in task_ids]
        batch = [row for task_id in sampled for row in groups[task_id]]
        labels = [int(row["label"]) for row in batch]
        differences.append(
            statistic(labels, [float(row[left]) for row in batch])
            - statistic(labels, [float(row[right]) for row in batch])
        )
    differences.sort()
    lower = differences[math.floor(0.025 * (draws - 1))]
    upper = differences[math.ceil(0.975 * (draws - 1))]
    return lower, upper


def brier(labels: list[int], probabilities: list[float]) -> float:
    return statistics.mean(
        (probability - label) ** 2
        for label, probability in zip(labels, probabilities)
    )


def log_loss(labels: list[int], probabilities: list[float]) -> float:
    clipped = [min(1.0 - 1e-12, max(1e-12, value)) for value in probabilities]
    return -statistics.mean(
        label * math.log(probability) + (1 - label) * math.log(1 - probability)
        for label, probability in zip(labels, clipped)
    )


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    dataset = PublicCoremDataset(args.dataset)
    train = list(dataset.iter_outcome_examples(splits={"train"}))
    validation = list(dataset.iter_outcome_examples(splits={"val"}))
    test = list(dataset.iter_outcome_examples(splits={"test"}))
    methods = (
        "hgt_corem",
        "hgt_additive_encoder_matched",
        "hgt_no_hyperedges",
        *(("hgt_no_redundancy",) if args.include_duplicates else ()),
        "stable_hash_reference",
    )
    per_seed = []
    started = time.perf_counter()
    if args.calibration == "affine":
        selection, calibration = split_validation_by_task(validation, 2027)
    else:
        selection, calibration = validation, []
    partitions = {
        "train": train,
        "selection": selection,
        "calibration": calibration,
        "test": test,
    }
    print("fitting deterministic stable-hash reference", flush=True)
    stable_result = fit_method(
        "stable_hash_reference", seeds[0], partitions, dataset, args
    )
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.workers == 1:
        for position, seed in enumerate(seeds, start=1):
            print(f"[{position}/{len(seeds)}] seed={seed}", flush=True)
            run = fit_neural_seed(seed, args.dataset, args)
            run["methods"]["stable_hash_reference"] = stable_result
            per_seed.append(run)
    else:
        completed: dict[int, dict[str, Any]] = {}
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(seeds)),
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(fit_neural_seed, seed, args.dataset, args): seed
                for seed in seeds
            }
            for position, future in enumerate(as_completed(futures), start=1):
                seed = futures[future]
                run = future.result()
                completed[seed] = run
                print(
                    f"[{position}/{len(seeds)}] completed seed={seed}",
                    flush=True,
                )
        for seed in seeds:
            run = completed[seed]
            run["methods"]["stable_hash_reference"] = stable_result
            per_seed.append(run)

    metric_names = (
        "roc_auc",
        "brier",
        "ece_10",
        "balanced_accuracy",
        "log_loss",
        "task_pairwise_accuracy",
    )
    aggregate = {
        method: {
            metric: metric_summary(
                [float(run["methods"][method]["test"][metric]) for run in per_seed]
            )
            for metric in metric_names
        }
        for method in methods
    }
    comparisons = {}
    references = (
        "hgt_additive_encoder_matched",
        "hgt_no_hyperedges",
        *(("hgt_no_redundancy",) if args.include_duplicates else ()),
        "stable_hash_reference",
    )
    for reference in references:
        comparisons[reference] = {}
        for metric in metric_names:
            differences = [
                float(run["methods"]["hgt_corem"]["test"][metric])
                - float(run["methods"][reference]["test"][metric])
                for run in per_seed
            ]
            comparisons[reference][metric] = {
                "hgt_corem_minus_reference": metric_summary(differences),
                "exact_sign_flip_pvalue": sign_flip_pvalue(differences),
            }

    ensemble_rows = []
    for index, example in enumerate(test):
        row: dict[str, Any] = {
            "task_id": example.request.request_id,
            "label": example.outcome,
        }
        for method in methods:
            row[method] = statistics.mean(
                run["methods"][method]["test_probabilities"][index]
                for run in per_seed
            )
        ensemble_rows.append(row)
    ensemble = {}
    for method in methods:
        probabilities = [float(row[method]) for row in ensemble_rows]
        threshold = statistics.mean(
            run["methods"][method]["threshold"] for run in per_seed
        )
        ensemble[method] = rounded_metrics(test, probabilities, threshold)
        ensemble[method]["task_pairwise_accuracy"] = round(
            task_pairwise_accuracy(test, probabilities), 8
        )
    bootstrap = {}
    statistics_map = {"roc_auc": roc_auc, "brier": brier, "log_loss": log_loss}
    for reference in references:
        bootstrap[reference] = {
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

    result = {
        "status": "complete",
        "scope": "execution-verified multi-operation deterministic final state",
        "dataset": {
            "name": dataset.summary.name,
            "schema_version": dataset.summary.schema_version,
            "integrity": dataset.validate_integrity(),
            "examples": {
                "train": len(train),
                "validation": len(validation),
                "test": len(test),
            },
            "tasks": {
                "train": len({item.request.request_id for item in train}),
                "validation": len(
                    {item.request.request_id for item in validation}
                ),
                "test": len({item.request.request_id for item in test}),
            },
        },
        "configuration": {
            "seeds": seeds,
            "epochs": args.epochs,
            "dimension": args.dimension,
            "text_hash_dimension": args.text_hash_dimension,
            "message_layers": args.message_layers,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "workers": args.workers,
            "bootstrap_draws": args.bootstrap_draws,
            "calibration": args.calibration,
            "task_clustered_splits_and_bootstrap": True,
            "validation_partition_seed": 2027,
            "deterministic_reference_fit_once": True,
            "outcome_derived_oracle_edges_used": False,
            "gold_operation_ids_used_as_request_features": False,
            "duplicate_pairs_used": args.include_duplicates,
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
        "paired_seed_comparisons": comparisons,
        "ensemble_test": ensemble,
        "task_clustered_bootstrap_95ci_hgt_minus_reference": bootstrap,
        "per_seed": per_seed,
        "elapsed_seconds": time.perf_counter() - started,
        "code_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path("corem/dataset.py"),
                Path("corem/torch_encoding.py"),
                Path("corem/torch_model.py"),
                Path("experiments/run_tau3_multistep.py"),
            )
        },
        "limitations": [
            (
                "Labels cover deterministic final-state criteria (database hashes "
                "and/or environment assertions), not communication quality."
            ),
            (
                f"{sum(len({item.request.request_id for item in rows}) for rows in (train, validation, test))} "
                "strict multi-operation tasks survive filtering."
            ),
            "Candidate sets are perturbations of one official reference trajectory.",
            "Declared mutation groups are a near-sufficient structural rule in this release.",
            "Environment-assertion and database-state labels are pooled across domains.",
        ],
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
    print(json.dumps({"aggregate": aggregate, "ensemble": ensemble}, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
