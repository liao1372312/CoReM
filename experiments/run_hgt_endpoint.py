"""Evaluate the end-to-end HGT backend on observed singleton outcomes.

This experiment is deliberately scoped to endpoint prediction. The private
dataset has no verified multi-operation outcomes, so the run does not estimate
composition success or high-order interaction effects.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem import SetUtilityModel, StableHashEncoder, TrainingConfig
from corem.dataset import PrivateCoremDataset
from corem.torch_encoding import TorchHGTEncoder
from corem.torch_model import TorchSetUtilityModel
from corem.types import EvidenceGraph, OutcomeExample
from experiments.common import binary_metrics, select_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/private_corem"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results_hgt"))
    parser.add_argument("--train-size", type=int, default=3138)
    parser.add_argument(
        "--train-sampling",
        choices=("uniform", "balanced"),
        default="balanced",
    )
    parser.add_argument(
        "--train-negative-fraction",
        type=float,
        default=0.5,
        help="negative fraction for stratified ('balanced') sampling",
    )
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
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def sample_examples(
    dataset: PrivateCoremDataset, split: str, size: int, seed: int
) -> list[OutcomeExample]:
    examples = list(dataset.iter_outcome_examples(splits={split}))
    if size <= 0 or size > len(examples):
        raise ValueError(
            f"invalid {split} sample size {size}; available={len(examples)}"
        )
    return random.Random(seed).sample(examples, size)


def sample_training_examples(
    dataset: PrivateCoremDataset,
    size: int,
    seed: int,
    strategy: str,
    negative_fraction: float = 0.5,
) -> list[OutcomeExample]:
    if strategy == "uniform":
        return sample_examples(dataset, "train", size, seed)
    if strategy != "balanced":
        raise ValueError(f"unknown training sampling strategy: {strategy}")
    if not 0.0 < negative_fraction < 1.0:
        raise ValueError("train negative fraction must be between zero and one")
    examples = list(dataset.iter_outcome_examples(splits={"train"}))
    by_outcome = {
        outcome: [example for example in examples if example.outcome == outcome]
        for outcome in (0, 1)
    }
    negative_count = round(size * negative_fraction)
    positive_count = size - negative_count
    if negative_count > len(by_outcome[0]) or positive_count > len(by_outcome[1]):
        raise ValueError(
            "balanced training sample exceeds an outcome class: "
            f"requested positive={positive_count}, negative={negative_count}; "
            f"available positive={len(by_outcome[1])}, negative={len(by_outcome[0])}"
        )
    rng = random.Random(seed)
    sampled = rng.sample(by_outcome[0], negative_count) + rng.sample(
        by_outcome[1], positive_count
    )
    rng.shuffle(sampled)
    return sampled


def build_split_graph(
    dataset: PrivateCoremDataset, examples: list[OutcomeExample]
) -> EvidenceGraph:
    memory_ids = frozenset(
        record_id for example in examples for record_id in example.selected_ids
    )
    return dataset.build_graph(
        memory_ids=memory_ids,
        include_operation_entities=True,
        sanitize_outcome_text=True,
        include_duplicates=False,
    )


def predict(
    model: TorchSetUtilityModel | SetUtilityModel,
    examples: list[OutcomeExample],
    graph: EvidenceGraph,
) -> list[float]:
    return [
        model.probability(example.request, example.selected_ids, graph)
        for example in examples
    ]


def metrics(
    examples: list[OutcomeExample], probabilities: list[float], threshold: float
) -> dict[str, float]:
    labels = [example.outcome for example in examples]
    return {
        key: round(value, 6)
        for key, value in binary_metrics(labels, probabilities, threshold).items()
    }


def apply_affine_calibration(
    probabilities: list[float], scale: float, bias: float
) -> list[float]:
    calibrated = []
    for probability in probabilities:
        clipped = min(1.0 - 1e-7, max(1e-7, probability))
        logit = scale * math.log(clipped / (1.0 - clipped)) + bias
        calibrated.append(
            1.0 / (1.0 + math.exp(-logit))
            if logit >= 0.0
            else math.exp(logit) / (1.0 + math.exp(logit))
        )
    return calibrated


def select_affine_calibrator(
    examples: list[OutcomeExample], probabilities: list[float]
) -> tuple[float, float]:
    """Fit positive-scale affine logit calibration with damped Newton steps."""
    labels = [example.outcome for example in examples]
    logits = [
        math.log(clipped / (1.0 - clipped))
        for probability in probabilities
        for clipped in [min(1.0 - 1e-7, max(1e-7, probability))]
    ]
    regularization = 1e-4

    def objective(scale: float, bias: float) -> float:
        total = 0.0
        for label, logit in zip(labels, logits):
            value = scale * logit + bias
            total += max(value, 0.0) - label * value + math.log1p(
                math.exp(-abs(value))
            )
        return total / len(labels) + 0.5 * regularization * (
            (scale - 1.0) ** 2 + bias**2
        )

    scale, bias = 1.0, 0.0
    current = objective(scale, bias)
    for _ in range(50):
        gradient_scale = regularization * (scale - 1.0)
        gradient_bias = regularization * bias
        hessian_ss = regularization
        hessian_sb = 0.0
        hessian_bb = regularization
        for label, logit in zip(labels, logits):
            value = scale * logit + bias
            probability = (
                1.0 / (1.0 + math.exp(-value))
                if value >= 0.0
                else math.exp(value) / (1.0 + math.exp(value))
            )
            residual = probability - label
            curvature = probability * (1.0 - probability)
            gradient_scale += residual * logit / len(labels)
            gradient_bias += residual / len(labels)
            hessian_ss += curvature * logit * logit / len(labels)
            hessian_sb += curvature * logit / len(labels)
            hessian_bb += curvature / len(labels)
        determinant = hessian_ss * hessian_bb - hessian_sb * hessian_sb
        if determinant <= 1e-12:
            break
        step_scale = (
            hessian_bb * gradient_scale - hessian_sb * gradient_bias
        ) / determinant
        step_bias = (
            hessian_ss * gradient_bias - hessian_sb * gradient_scale
        ) / determinant
        if max(abs(step_scale), abs(step_bias)) < 1e-8:
            break
        step_size = 1.0
        accepted = False
        for _ in range(25):
            candidate_scale = min(100.0, max(0.01, scale - step_size * step_scale))
            candidate_bias = min(50.0, max(-50.0, bias - step_size * step_bias))
            candidate = objective(candidate_scale, candidate_bias)
            if candidate < current:
                scale, bias, current = candidate_scale, candidate_bias, candidate
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            break
    return scale, bias


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    dataset = PrivateCoremDataset(args.dataset)
    examples = {
        "train": sample_training_examples(
            dataset,
            args.train_size,
            args.seed,
            args.train_sampling,
            args.train_negative_fraction,
        ),
        "val": sample_examples(dataset, "val", args.val_size, args.seed + 1),
        "test": sample_examples(dataset, "test", args.test_size, args.seed + 2),
    }
    if len(examples["val"]) < 4:
        raise ValueError("validation sample must contain at least four examples")
    selection_size = len(examples["val"]) // 2
    selection_examples = examples["val"][:selection_size]
    calibration_examples = examples["val"][selection_size:]
    partitions = {
        "train": examples["train"],
        "selection": selection_examples,
        "calibration": calibration_examples,
        "test": examples["test"],
    }
    graphs = {
        split: build_split_graph(dataset, split_examples)
        for split, split_examples in partitions.items()
    }

    encoder = TorchHGTEncoder(
        dimension=args.dimension,
        text_hash_dimension=args.text_hash_dimension,
        text_feature_mode=args.text_feature_mode,
        message_layers=args.message_layers,
        attention_heads=4,
        seed=args.seed,
    )
    hgt = TorchSetUtilityModel(encoder)
    hgt_fit = hgt.fit(
        examples["train"],
        graphs["train"],
        TrainingConfig(
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            calibration_weight=0.1,
            relation_weight=args.relation_weight,
            interaction_weight=0.0,
            class_balance_power=args.class_balance_power,
        ),
    )
    hgt_probabilities = {
        split: predict(hgt, split_examples, graphs[split])
        for split, split_examples in partitions.items()
    }
    hgt_scale, hgt_bias = select_affine_calibrator(
        calibration_examples,
        hgt_probabilities["calibration"],
    )
    hgt_probabilities = {
        split: apply_affine_calibration(probabilities, hgt_scale, hgt_bias)
        for split, probabilities in hgt_probabilities.items()
    }
    hgt_threshold = select_threshold(
        [example.outcome for example in selection_examples],
        hgt_probabilities["selection"],
    )

    reference = SetUtilityModel(encoder=StableHashEncoder())
    reference_fit = reference.fit(
        examples["train"],
        graphs["train"],
        TrainingConfig(learning_rate=0.08, epochs=60, calibration_weight=0.1),
    )
    reference_probabilities = {
        split: predict(reference, split_examples, graphs[split])
        for split, split_examples in partitions.items()
    }
    reference_scale, reference_bias = select_affine_calibrator(
        calibration_examples,
        reference_probabilities["calibration"],
    )
    reference_probabilities = {
        split: apply_affine_calibration(probabilities, reference_scale, reference_bias)
        for split, probabilities in reference_probabilities.items()
    }
    reference_threshold = select_threshold(
        [example.outcome for example in selection_examples],
        reference_probabilities["selection"],
    )

    graph_counts = {
        split: {
            "records": len(graph.records),
            "entities": len(graph.entities),
            "typed_edges": len(graph.edges),
        }
        for split, graph in graphs.items()
    }
    result = {
        "status": "complete",
        "scope": "observed singleton endpoint prediction only",
        "prohibited_claims": [
            "verified multi-operation composition success",
            "learned high-order interaction effectiveness",
            "causal memory contribution",
        ],
        "seed": args.seed,
        "device": encoder.device_name,
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
        "train_outcome_counts": {
            "negative": sum(example.outcome == 0 for example in examples["train"]),
            "positive": sum(example.outcome == 1 for example in examples["train"]),
        },
        "graph_counts": graph_counts,
        "validation_partition": {
            "threshold_selection_n": len(selection_examples),
            "probability_calibration_n": len(calibration_examples),
        },
        "calibration_method": "affine_logit_newton_l2_1e-4",
        "test_predictions": "test_predictions.jsonl",
        "hgt": {
            "loss_first": hgt_fit.losses[0],
            "loss_last": hgt_fit.losses[-1],
            "threshold_selected_on_selection": hgt_threshold,
            "calibrator_selected_on_calibration": {
                "scale": hgt_scale,
                "bias": hgt_bias,
            },
            "selection": metrics(
                selection_examples,
                hgt_probabilities["selection"],
                hgt_threshold,
            ),
            "calibration": metrics(
                calibration_examples,
                hgt_probabilities["calibration"],
                hgt_threshold,
            ),
            "test": metrics(examples["test"], hgt_probabilities["test"], hgt_threshold),
        },
        "stable_hash_reference": {
            "epochs": reference_fit.epochs,
            "loss_first": reference_fit.losses[0],
            "loss_last": reference_fit.losses[-1],
            "threshold_selected_on_selection": reference_threshold,
            "calibrator_selected_on_calibration": {
                "scale": reference_scale,
                "bias": reference_bias,
            },
            "selection": metrics(
                selection_examples,
                reference_probabilities["selection"],
                reference_threshold,
            ),
            "calibration": metrics(
                calibration_examples,
                reference_probabilities["calibration"],
                reference_threshold,
            ),
            "test": metrics(
                examples["test"],
                reference_probabilities["test"],
                reference_threshold,
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "endpoint_hgt_sample.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prediction_path = args.output / "test_predictions.jsonl"
    prediction_path.write_text(
        "".join(
            json.dumps(
                {
                    "request_id": example.request.request_id,
                    "label": example.outcome,
                    "hgt_probability": hgt_probability,
                    "stable_hash_probability": reference_probability,
                },
                sort_keys=True,
            )
            + "\n"
            for example, hgt_probability, reference_probability in zip(
                examples["test"],
                hgt_probabilities["test"],
                reference_probabilities["test"],
            )
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {output_path}")
    print(f"wrote {prediction_path}")


if __name__ == "__main__":
    main()
