"""Tune HGT training settings without constructing or reading the test split."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem import TrainingConfig
from corem.dataset import PrivateCoremDataset
from corem.torch_encoding import TorchHGTEncoder
from corem.torch_model import TorchSetUtilityModel
from corem.types import EvidenceGraph, OutcomeExample
from experiments.common import binary_metrics, select_threshold
from experiments.run_hgt_endpoint import (
    build_split_graph,
    predict,
    sample_examples,
    sample_training_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/private_corem"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results_hgt/tuning_hgt_validation.json"),
    )
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument(
        "--train-sampling",
        choices=("uniform", "balanced"),
        default="uniform",
    )
    parser.add_argument("--train-negative-fraction", type=float, default=0.5)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--epochs", default="8,12,20")
    parser.add_argument("--balance-powers", default="0,0.5,1")
    parser.add_argument("--text-hash-dimensions", default="32")
    parser.add_argument("--text-feature-modes", default="tokens")
    parser.add_argument("--message-layers", default="2")
    parser.add_argument("--relation-weights", default="0.1")
    parser.add_argument("--learning-rates", default="0.01")
    parser.add_argument(
        "--selection-scope",
        choices=("full", "half"),
        default="full",
        help="use all validation examples for architecture selection or reserve half",
    )
    return parser.parse_args()


def parse_numbers(
    values: str, cast: type[int | float]
) -> list[int] | list[float]:
    parsed = [cast(value) for value in values.split(",") if value.strip()]
    if not parsed:
        raise ValueError("candidate list must not be empty")
    return parsed


def main() -> None:
    args = parse_args()
    epoch_candidates = parse_numbers(args.epochs, int)
    balance_candidates = parse_numbers(args.balance_powers, float)
    text_hash_candidates = parse_numbers(args.text_hash_dimensions, int)
    text_feature_candidates = [
        value.strip() for value in args.text_feature_modes.split(",") if value.strip()
    ]
    if not text_feature_candidates:
        raise ValueError("text feature mode list must not be empty")
    layer_candidates = parse_numbers(args.message_layers, int)
    relation_candidates = parse_numbers(args.relation_weights, float)
    learning_rate_candidates = parse_numbers(args.learning_rates, float)
    dataset = PrivateCoremDataset(args.dataset)
    train_examples = sample_training_examples(
        dataset,
        args.train_size,
        args.seed,
        args.train_sampling,
        args.train_negative_fraction,
    )
    validation_examples = sample_examples(dataset, "val", args.val_size, args.seed + 1)
    selection_examples = (
        validation_examples
        if args.selection_scope == "full"
        else validation_examples[: len(validation_examples) // 2]
    )
    train_graph = build_split_graph(dataset, train_examples)
    selection_graph = build_split_graph(dataset, selection_examples)
    labels = [example.outcome for example in selection_examples]
    normalize_query = lambda value: re.sub(r"\s+", " ", value.casefold()).strip()
    train_queries = {
        normalize_query(example.request.text)
        for example in dataset.iter_outcome_examples(splits={"train"})
    }
    novel_indices = [
        index
        for index, example in enumerate(selection_examples)
        if normalize_query(example.request.text) not in train_queries
    ]
    novel_labels = [labels[index] for index in novel_indices]

    trials = []
    for epochs in epoch_candidates:
        for balance_power in balance_candidates:
            for text_hash_dimension in text_hash_candidates:
                for text_feature_mode in text_feature_candidates:
                    for message_layers in layer_candidates:
                        for relation_weight in relation_candidates:
                            for learning_rate in learning_rate_candidates:
                                trials.append(
                                    run_trial(
                                        args=args,
                                        epochs=epochs,
                                        balance_power=balance_power,
                                        text_hash_dimension=text_hash_dimension,
                                        text_feature_mode=text_feature_mode,
                                        message_layers=message_layers,
                                        relation_weight=relation_weight,
                                        learning_rate=learning_rate,
                                        train_examples=train_examples,
                                        train_graph=train_graph,
                                        selection_examples=selection_examples,
                                        selection_graph=selection_graph,
                                        labels=labels,
                                        novel_indices=novel_indices,
                                        novel_labels=novel_labels,
                                    )
                                )

    chosen = max(
        trials,
        key=lambda trial: (
            trial["novel_query_selection"]["roc_auc"],
            trial["selection"]["roc_auc"],
            -trial["selection"]["log_loss"],
            -trial["epochs"],
            -trial["text_hash_dimension"],
        ),
    )
    chosen_fields = (
        "epochs",
        "class_balance_power",
        "text_hash_dimension",
        "text_feature_mode",
        "message_layers",
        "relation_weight",
        "learning_rate",
    )
    result = {
        "status": "complete",
        "scope": "training and validation-selection partitions only",
        "test_split_accessed": False,
        "seed": args.seed,
        "sample_sizes": {
            "train": len(train_examples),
            "validation_selection": len(selection_examples),
            "validation_novel_query": len(novel_indices),
            "validation_calibration_reserved": len(validation_examples)
            - len(selection_examples),
        },
        "train_sampling": args.train_sampling,
        "train_negative_fraction": args.train_negative_fraction,
        "train_outcome_counts": {
            "negative": sum(example.outcome == 0 for example in train_examples),
            "positive": sum(example.outcome == 1 for example in train_examples),
        },
        "selection_rule": (
            "maximum novel-query ROC-AUC; overall ROC-AUC, log loss, fewer "
            "epochs, and smaller text hash dimension break ties"
        ),
        "chosen": {key: chosen[key] for key in chosen_fields},
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}")


def run_trial(
    *,
    args: argparse.Namespace,
    epochs: int,
    balance_power: float,
    text_hash_dimension: int,
    text_feature_mode: str,
    message_layers: int,
    relation_weight: float,
    learning_rate: float,
    train_examples: list[OutcomeExample],
    train_graph: EvidenceGraph,
    selection_examples: list[OutcomeExample],
    selection_graph: EvidenceGraph,
    labels: list[int],
    novel_indices: list[int],
    novel_labels: list[int],
) -> dict:
    started = time.perf_counter()
    model = TorchSetUtilityModel(
        TorchHGTEncoder(
            dimension=args.dimension,
            text_hash_dimension=text_hash_dimension,
            text_feature_mode=text_feature_mode,
            message_layers=message_layers,
            attention_heads=4,
            seed=args.seed,
        )
    )
    fit = model.fit(
        train_examples,
        train_graph,
        TrainingConfig(
            learning_rate=learning_rate,
            epochs=epochs,
            calibration_weight=0.1,
            relation_weight=relation_weight,
            interaction_weight=0.0,
            class_balance_power=balance_power,
        ),
    )
    probabilities = predict(model, selection_examples, selection_graph)
    threshold = select_threshold(labels, probabilities)
    trial_metrics = binary_metrics(labels, probabilities, threshold)
    novel_probabilities = [probabilities[index] for index in novel_indices]
    novel_threshold = select_threshold(novel_labels, novel_probabilities)
    novel_metrics = binary_metrics(novel_labels, novel_probabilities, novel_threshold)
    trial = {
        "epochs": epochs,
        "class_balance_power": balance_power,
        "text_hash_dimension": text_hash_dimension,
        "text_feature_mode": text_feature_mode,
        "message_layers": message_layers,
        "relation_weight": relation_weight,
        "learning_rate": learning_rate,
        "loss_first": fit.losses[0],
        "loss_last": fit.losses[-1],
        "selection": {
            key: round(value, 6) for key, value in trial_metrics.items()
        },
        "novel_query_selection": {
            key: round(value, 6) for key, value in novel_metrics.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(trial, sort_keys=True), flush=True)
    del model
    gc.collect()
    return trial


if __name__ == "__main__":
    main()
