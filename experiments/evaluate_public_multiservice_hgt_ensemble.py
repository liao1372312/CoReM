#!/usr/bin/env python3
"""Tune a seed-ensemble HGT/BM25 residual ranker without using test outcomes."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch  # noqa: E402

from corem.real_eval import (  # noqa: E402
    load_jsonl,
    write_checksums,
    write_json,
    write_jsonl,
)
from corem.torch_model import TorchSetUtilityModel  # noqa: E402
from corem.types import EvidenceGraph, OutcomeExample  # noqa: E402
from experiments.generate_public_workflow_selections import (  # noqa: E402
    bm25_scores,
    make_graph,
)
from experiments.train_public_multiservice_hgt import (  # noqa: E402
    VARIANTS,
    build_model,
    load_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tune-executions", type=Path, required=True)
    parser.add_argument("--validation-executions", type=Path, required=True)
    parser.add_argument("--model-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-variant", choices=VARIANTS, default="hgt_corem")
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--text-hash-dimension", type=int, default=64)
    parser.add_argument("--message-layers", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--lexical-features", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--neural-weights",
        type=float,
        nargs="+",
        default=[0.9],
        help="Frozen paper mixture; pass multiple values only for development tuning.",
    )
    return parser.parse_args()


def load_models(
    model_dirs: list[Path],
    graph: EvidenceGraph,
    *,
    dimension: int,
    text_hash_dimension: int,
    message_layers: int,
    device: str,
    lexical_features: bool,
    state_variant: str = "hgt_corem",
    model_variant: str | None = None,
) -> list[tuple[int, TorchSetUtilityModel, Path]]:
    if state_variant not in VARIANTS:
        raise ValueError(f"unknown state variant: {state_variant}")
    effective_model_variant = model_variant or state_variant
    if effective_model_variant not in VARIANTS:
        raise ValueError(f"unknown model variant: {effective_model_variant}")
    output = []
    observed_seeds: set[int] = set()
    for directory in model_dirs:
        states = sorted(directory.glob(f"{state_variant}_seed_*.pt"))
        if not states:
            raise ValueError(
                f"expected at least one {state_variant} state in {directory}"
            )
        for state_path in states:
            seed = int(state_path.stem.rsplit("_", maxsplit=1)[1])
            if seed in observed_seeds:
                raise ValueError(f"duplicate model seed across directories: {seed}")
            observed_seeds.add(seed)
            model = build_model(
                effective_model_variant,
                seed=seed,
                dimension=dimension,
                text_hash_dimension=text_hash_dimension,
                message_layers=message_layers,
                device=device,
                lexical_features=lexical_features,
            )
            model.encoder._forward_graph(graph)
            state = torch.load(state_path, map_location=device, weights_only=True)
            incompatible = model.encoder.load_state_dict(state, strict=False)
            allowed_unexpected = {
                key
                for key in incompatible.unexpected_keys
                if key.startswith("relation_classifier.")
            }
            if set(incompatible.unexpected_keys) != allowed_unexpected:
                raise ValueError(
                    f"unexpected state keys in {state_path}: {incompatible.unexpected_keys}"
                )
            if incompatible.missing_keys:
                raise ValueError(
                    f"missing state keys in {state_path}: {incompatible.missing_keys}"
                )
            model.encoder.encode_graph(graph)
            output.append((seed, model, state_path))
    return output


def candidate_rows(
    examples: list[OutcomeExample],
    models: list[tuple[int, TorchSetUtilityModel, Path]],
    graph: EvidenceGraph,
    memory_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_request: dict[str, list[OutcomeExample]] = defaultdict(list)
    for example in examples:
        by_request[example.request.request_id].append(example)
    output = []
    for event_id, group in sorted(by_request.items()):
        request = group[0].request
        active = [
            row
            for row in memory_rows
            if graph.is_admissible(str(row["memory_id"]), request)
        ]
        lexical = bm25_scores(request.text, active)
        raw_bm25 = [
            max(lexical[memory_id] for memory_id in example.selected_ids)
            for example in group
        ]
        low, high = min(raw_bm25), max(raw_bm25)
        for example, bm25 in zip(group, raw_bm25, strict=True):
            probabilities = [
                model.probability(request, example.selected_ids, graph)
                for _, model, _ in models
            ]
            output.append(
                {
                    "event_id": event_id,
                    "selected_memory_ids": sorted(example.selected_ids),
                    "label": example.outcome,
                    "ensemble_probability": statistics.mean(probabilities),
                    "seed_probability_std": (
                        statistics.stdev(probabilities)
                        if len(probabilities) > 1
                        else 0.0
                    ),
                    "bm25_max": bm25,
                    "bm25_normalized": (bm25 - low) / (high - low)
                    if high > low
                    else 0.0,
                }
            )
    return output


def rank_metrics(rows: list[dict[str, Any]], neural_weight: float) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        enriched = dict(row)
        enriched["score"] = neural_weight * float(row["ensemble_probability"]) + (
            1.0 - neural_weight
        ) * float(row["bm25_normalized"])
        grouped[str(row["event_id"])].append(enriched)
    top1 = []
    reciprocal_ranks = []
    selected_rows = []
    for event_id, candidates in sorted(grouped.items()):
        ranked = sorted(
            candidates,
            key=lambda row: (
                float(row["score"]),
                tuple(row["selected_memory_ids"]),
            ),
            reverse=True,
        )
        top1.append(int(ranked[0]["label"]))
        first_positive = next(
            index for index, row in enumerate(ranked, start=1) if int(row["label"])
        )
        reciprocal_ranks.append(1.0 / first_positive)
        selected_rows.append(
            {
                "event_id": event_id,
                "selected_memory_ids": ranked[0]["selected_memory_ids"],
                "label": ranked[0]["label"],
                "score": ranked[0]["score"],
                "ensemble_probability": ranked[0]["ensemble_probability"],
                "bm25_normalized": ranked[0]["bm25_normalized"],
            }
        )
    return {
        "neural_weight": neural_weight,
        "events": len(grouped),
        "top1_vsr": statistics.mean(top1),
        "mean_reciprocal_rank": statistics.mean(reciprocal_ranks),
        "selected_rows": selected_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    if any(not 0.0 <= value <= 1.0 for value in args.neural_weights):
        raise ValueError("neural weights must be between zero and one")
    if len(set(args.neural_weights)) != len(args.neural_weights):
        raise ValueError("neural weights must be unique")
    if args.torch_threads < 1:
        raise ValueError("torch threads must be positive")
    torch.set_num_threads(args.torch_threads)

    events = {
        str(row["event_id"]): row for row in load_jsonl(args.dataset / "events.jsonl")
    }
    memories = load_jsonl(args.dataset / "memories.jsonl")
    hyperedges = load_jsonl(args.dataset / "dependency_hyperedges.jsonl")
    graph = make_graph(memories, hyperedges)
    models = load_models(
        args.model_dirs,
        graph,
        dimension=args.dimension,
        text_hash_dimension=args.text_hash_dimension,
        message_layers=args.message_layers,
        device=args.device,
        lexical_features=args.lexical_features,
        state_variant=args.state_variant,
    )
    tune_examples = load_examples(args.tune_executions, events, graph)
    validation_examples = load_examples(args.validation_executions, events, graph)
    tune_candidates = candidate_rows(tune_examples, models, graph, memories)
    validation_candidates = candidate_rows(validation_examples, models, graph, memories)

    weight_rows = []
    tune_by_weight = {}
    validation_by_weight = {}
    for weight in sorted(args.neural_weights):
        tune = rank_metrics(tune_candidates, weight)
        validation = rank_metrics(validation_candidates, weight)
        tune_by_weight[weight] = tune
        validation_by_weight[weight] = validation
        weight_rows.append(
            {
                "neural_weight": weight,
                "tune_top1_vsr": tune["top1_vsr"],
                "tune_mrr": tune["mean_reciprocal_rank"],
                "validation_top1_vsr": validation["top1_vsr"],
                "validation_mrr": validation["mean_reciprocal_rank"],
            }
        )
    selected_weight = max(
        args.neural_weights,
        key=lambda weight: (
            tune_by_weight[weight]["top1_vsr"],
            tune_by_weight[weight]["mean_reciprocal_rank"],
            weight,
        ),
    )
    selected_tune = tune_by_weight[selected_weight]
    selected_validation = validation_by_weight[selected_weight]

    args.output.mkdir(parents=True)
    write_csv(args.output / "weight_sweep.csv", weight_rows)
    write_jsonl(args.output / "tune_candidate_scores.jsonl", tune_candidates)
    write_jsonl(
        args.output / "validation_candidate_scores.jsonl", validation_candidates
    )
    write_jsonl(args.output / "selected_tune.jsonl", selected_tune.pop("selected_rows"))
    write_jsonl(
        args.output / "selected_validation.jsonl",
        selected_validation.pop("selected_rows"),
    )
    write_json(
        args.output / "result.json",
        {
            "status": (
                "fixed_weight_validation_evaluated"
                if len(args.neural_weights) == 1
                else "tune_selected_validation_evaluated"
            ),
            "state_variant": args.state_variant,
            "seeds": [seed for seed, _, _ in models],
            "model_states": [str(path.resolve()) for _, _, path in models],
            "selection_rule": (
                "Fixed predeclared neural weight; no weight selection."
                if len(args.neural_weights) == 1
                else "Maximize tune Top-1 VSR, then tune MRR, then neural weight. "
                "Validation never selects the residual weight."
            ),
            "selected_neural_weight": selected_weight,
            "selected_bm25_weight": 1.0 - selected_weight,
            "torch_threads": args.torch_threads,
            "tune": selected_tune,
            "validation": selected_validation,
            "claim_boundary": (
                "This is internal model development on tune/validation only. "
                "No test outcomes are loaded; a fresh confirmation set is required."
            ),
        },
    )
    write_checksums(
        args.output,
        (
            "result.json",
            "selected_tune.jsonl",
            "selected_validation.jsonl",
            "tune_candidate_scores.jsonl",
            "validation_candidate_scores.jsonl",
            "weight_sweep.csv",
        ),
    )
    print(
        f"selected neural weight {selected_weight:.2f}: "
        f"tune={selected_tune['top1_vsr']:.4f}, "
        f"validation={selected_validation['top1_vsr']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
