#!/usr/bin/env python3
"""Train v5 HGT bundle selectors and freeze label-blind test selections."""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch  # noqa: E402

from corem.model import TrainingConfig  # noqa: E402
from corem.real_eval import (  # noqa: E402
    load_jsonl,
    score_execution_row,
    write_checksums,
    write_json,
    write_jsonl,
)
from corem.torch_encoding import TorchHGTEncoder  # noqa: E402
from corem.torch_model import TorchSetUtilityModel  # noqa: E402
from corem.types import EvidenceGraph, OutcomeExample  # noqa: E402
from experiments.common import binary_metrics  # noqa: E402
from experiments.generate_public_multiservice_challenge_ablations import (  # noqa: E402
    request_for,
)
from experiments.generate_public_workflow_selections import make_graph  # noqa: E402

VARIANTS = (
    "hgt_corem",
    "hgt_additive",
    "hgt_no_hyperedges",
    "text_complement",
    "text_additive",
)
FACTORIAL_VARIANTS = frozenset(
    {"hgt_corem", "hgt_additive", "text_complement", "text_additive"}
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_examples(
    path: Path,
    events: dict[str, dict[str, Any]],
    graph: EvidenceGraph,
    *,
    exclude_provider: str | None = None,
) -> list[OutcomeExample]:
    examples = []
    for row in load_jsonl(path):
        event_id = str(row["event_id"])
        event = events[event_id]
        if exclude_provider is not None and event["provider_id"] == exclude_provider:
            continue
        if (
            row.get("execution_error") is not None
            or row.get("checker_error") is not None
        ):
            raise ValueError(
                f"execution artifact contains an infrastructure error: {row['run_id']}"
            )
        selected = frozenset(map(str, row["selected_memory_ids"]))
        request = request_for(event)
        if any(not graph.is_admissible(memory_id, request) for memory_id in selected):
            raise ValueError(f"inadmissible training selection: {row['run_id']}")
        if not graph.is_evidence_complete(request, selected):
            raise ValueError(f"generic-role-incomplete selection: {row['run_id']}")
        outcome = int(score_execution_row(row, event)["verified_success"])
        examples.append(
            OutcomeExample(
                request=request,
                selected_ids=selected,
                outcome=outcome,
                actually_executed=True,
                independently_verified=True,
                eligible_for_training=True,
                eligible_for_interaction_training=True,
            )
        )
    return examples


def candidate_metrics(
    model: TorchSetUtilityModel,
    examples: list[OutcomeExample],
    graph: EvidenceGraph,
) -> dict[str, Any]:
    probabilities = [
        model.probability(example.request, example.selected_ids, graph)
        for example in examples
    ]
    labels = [example.outcome for example in examples]
    metrics = binary_metrics(labels, probabilities, 0.5)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[example.request.request_id].append(index)
    selected = []
    reciprocal_ranks = []
    for event_id, indices in sorted(grouped.items()):
        ranked = sorted(
            indices,
            key=lambda index: (
                probabilities[index],
                tuple(sorted(examples[index].selected_ids)),
            ),
            reverse=True,
        )
        selected.append(labels[ranked[0]])
        first_positive = next(
            (rank for rank, index in enumerate(ranked, start=1) if labels[index]),
            None,
        )
        if first_positive is None:
            raise ValueError(f"event has no positive candidate: {event_id}")
        reciprocal_ranks.append(1.0 / first_positive)
    return {
        **{key: float(value) for key, value in metrics.items()},
        "events": len(grouped),
        "candidates": len(examples),
        "top1_vsr": statistics.mean(selected),
        "mean_reciprocal_rank": statistics.mean(reciprocal_ranks),
    }


def build_model(
    variant: str,
    *,
    seed: int,
    dimension: int,
    text_hash_dimension: int,
    message_layers: int,
    device: str,
    lexical_features: bool,
) -> TorchSetUtilityModel:
    text_only = variant in {"text_complement", "text_additive"}
    return TorchSetUtilityModel(
        TorchHGTEncoder(
            dimension=dimension,
            text_hash_dimension=text_hash_dimension,
            text_feature_mode="subword",
            message_layers=0 if text_only else message_layers,
            attention_heads=4,
            seed=seed,
            device=device,
            enable_complements=variant not in {"hgt_additive", "text_additive"},
            enable_redundancy=variant not in {"hgt_additive", "text_additive"},
            enable_lexical_features=lexical_features,
            enable_graph_context=not text_only,
        )
    )


def test_candidates(
    selections_path: Path,
    split_path: Path,
    events: dict[str, dict[str, Any]],
    graph: EvidenceGraph,
    *,
    test_provider: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    test_ids = {
        str(row["event_id"])
        for row in load_jsonl(split_path)
        if row["split"] == "test"
        and (
            test_provider is None
            or events[str(row["event_id"])]["provider_id"] == test_provider
        )
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(selections_path):
        event_id = str(row["event_id"])
        if event_id not in test_ids:
            continue
        request = request_for(events[event_id])
        selected = frozenset(map(str, row["selected_memory_ids"]))
        if not graph.is_evidence_complete(request, selected):
            raise ValueError(f"test candidate is incomplete: {row['run_id']}")
        grouped[event_id].append(row)
    if set(grouped) != test_ids:
        raise ValueError("test candidate enumeration is incomplete")
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--candidate-selections", type=Path, required=True)
    parser.add_argument("--train-executions", type=Path, required=True)
    parser.add_argument("--tune-executions", type=Path, required=True)
    parser.add_argument("--validation-executions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41, 53, 67])
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=["hgt_corem"]
    )
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--text-hash-dimension", type=int, default=64)
    parser.add_argument("--message-layers", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--calibration-weight", type=float, default=0.1)
    parser.add_argument("--relation-weight", type=float, default=0.0)
    parser.add_argument("--interaction-weight", type=float, default=0.001)
    parser.add_argument("--class-balance-power", type=float, default=0.5)
    parser.add_argument("--ranking-weight", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--lexical-features", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--exclude-provider",
        help="Exclude this provider from train, tune, and validation outcomes.",
    )
    parser.add_argument(
        "--test-provider",
        help="Freeze selections only for this provider's test incidents.",
    )
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    if len(set(args.seeds)) != len(args.seeds):
        print("seeds must be unique")
        return 2
    if args.torch_threads < 1:
        print("torch threads must be positive")
        return 2
    torch.set_num_threads(args.torch_threads)
    if (args.exclude_provider is None) != (args.test_provider is None):
        print("--exclude-provider and --test-provider must be supplied together")
        return 2
    if args.exclude_provider != args.test_provider:
        print("leave-provider-out training must test the excluded provider")
        return 2
    if (
        any(variant.startswith("text_") for variant in args.variants)
        and args.relation_weight
    ):
        print(
            "text-only factorial cells require --relation-weight 0 so relation "
            "supervision cannot reintroduce graph information"
        )
        return 2

    events = {
        str(row["event_id"]): row for row in load_jsonl(args.dataset / "events.jsonl")
    }
    memories = load_jsonl(args.dataset / "memories.jsonl")
    hyperedges = load_jsonl(args.dataset / "dependency_hyperedges.jsonl")
    full_graph = make_graph(memories, hyperedges)
    no_hyperedge_graph = make_graph(memories, [])
    training_memories = (
        memories
        if args.exclude_provider is None
        else [
            row for row in memories if row.get("provider_id") != args.exclude_provider
        ]
    )
    training_memory_ids = {str(row["memory_id"]) for row in training_memories}
    training_hyperedges = [
        row
        for row in hyperedges
        if set(map(str, row["members"])).issubset(training_memory_ids)
    ]
    training_full_graph = make_graph(training_memories, training_hyperedges)
    training_no_hyperedge_graph = make_graph(training_memories, [])
    train = load_examples(
        args.train_executions,
        events,
        training_full_graph,
        exclude_provider=args.exclude_provider,
    )
    tune = load_examples(
        args.tune_executions,
        events,
        training_full_graph,
        exclude_provider=args.exclude_provider,
    )
    validation = load_examples(
        args.validation_executions,
        events,
        training_full_graph,
        exclude_provider=args.exclude_provider,
    )
    test = test_candidates(
        args.candidate_selections,
        args.splits,
        events,
        full_graph,
        test_provider=args.test_provider,
    )

    args.output.mkdir(parents=True)
    result_runs = []
    selection_rows = []
    started = time.perf_counter()
    for variant in args.variants:
        training_graph = (
            training_no_hyperedge_graph
            if variant == "hgt_no_hyperedges"
            else training_full_graph
        )
        inference_graph = (
            no_hyperedge_graph if variant == "hgt_no_hyperedges" else full_graph
        )
        for seed in args.seeds:
            model = build_model(
                variant,
                seed=seed,
                dimension=args.dimension,
                text_hash_dimension=args.text_hash_dimension,
                message_layers=args.message_layers,
                device=args.device,
                lexical_features=args.lexical_features,
            )
            fit_started = time.perf_counter()
            fit = model.fit(
                train,
                training_graph,
                TrainingConfig(
                    learning_rate=args.learning_rate,
                    epochs=args.epochs,
                    calibration_weight=args.calibration_weight,
                    relation_weight=args.relation_weight,
                    interaction_weight=args.interaction_weight,
                    class_balance_power=args.class_balance_power,
                    ranking_weight=args.ranking_weight,
                ),
            )
            tune_metrics = candidate_metrics(model, tune, training_graph)
            validation_metrics = candidate_metrics(model, validation, training_graph)
            state_name = f"{variant}_seed_{seed}.pt"
            torch.save(model.encoder.state_dict(), args.output / state_name)
            for event_id, candidates in sorted(test.items()):
                request = request_for(events[event_id])
                ranked = []
                for row in candidates:
                    selected = frozenset(map(str, row["selected_memory_ids"]))
                    probability = model.probability(request, selected, inference_graph)
                    ranked.append((probability, tuple(sorted(selected)), row))
                probability, _, selected_row = max(ranked, key=lambda item: item[:2])
                method = variant
                selection_rows.append(
                    {
                        "run_id": f"{method}:{seed}:{event_id}",
                        "event_id": event_id,
                        "seed": seed,
                        "method": method,
                        "selected_memory_ids": selected_row["selected_memory_ids"],
                        "token_cost": selected_row["token_cost"],
                        "latency_ms": 0.0,
                        "llm_calls": 0,
                        "active_bank_cost": selected_row["active_bank_cost"],
                        "retrieval_status": "ok",
                        "evaluations": len(candidates),
                        "predicted_probability": probability,
                        "source_candidate_run_id": selected_row["run_id"],
                    }
                )
            result_runs.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "device": model.encoder.device_name,
                    "fit_seconds": time.perf_counter() - fit_started,
                    "fit_epochs": fit.epochs,
                    "loss_first": fit.losses[0],
                    "loss_last": fit.losses[-1],
                    "tune": tune_metrics,
                    "validation": validation_metrics,
                    "state_file": state_name,
                    "state_sha256": sha256(args.output / state_name),
                }
            )
            print(
                f"{variant} seed={seed} tune_top1={tune_metrics['top1_vsr']:.4f} "
                f"validation_top1={validation_metrics['top1_vsr']:.4f}",
                flush=True,
            )

    write_jsonl(args.output / "selections.jsonl", selection_rows)
    write_json(
        args.output / "result.json",
        {
            "status": "trained_and_test_selections_frozen",
            "scope": "controlled v5 candidate-bundle outcome prediction and selection",
            "configuration": {
                "seeds": args.seeds,
                "variants": args.variants,
                "epochs": args.epochs,
                "dimension": args.dimension,
                "text_hash_dimension": args.text_hash_dimension,
                "message_layers": args.message_layers,
                "torch_threads": args.torch_threads,
                "learning_rate": args.learning_rate,
                "calibration_weight": args.calibration_weight,
                "relation_weight": args.relation_weight,
                "interaction_weight": args.interaction_weight,
                "class_balance_power": args.class_balance_power,
                "ranking_weight": args.ranking_weight,
                "device": args.device,
                "lexical_features": args.lexical_features,
                "exclude_provider": args.exclude_provider,
                "test_provider": args.test_provider,
            },
            "examples": {
                "train": len(train),
                "tune": len(tune),
                "validation": len(validation),
                "test_candidates": sum(len(rows) for rows in test.values()),
            },
            "graph_boundary": {
                "training_records": len(training_memories),
                "training_hyperedges": len(training_hyperedges),
                "inference_records": len(memories),
                "inference_hyperedges": len(hyperedges),
                "heldout_provider_removed_from_training_graph": (
                    args.exclude_provider is not None
                ),
                "factorial_contract": (
                    "text_* disables HGT message layers, graph-neighbor request "
                    "conditioning, and graph-degree input while retaining the "
                    "same full graph for candidate bundles and, in text_complement, "
                    "the complement head. For a clean 2x2 comparison, train all "
                    "four cells with relation_weight=0."
                ),
                "factorial_four_cells_complete": set(args.variants)
                == FACTORIAL_VARIANTS,
            },
            "label_boundary": (
                "Only train execution outcomes update parameters. Tune and validation "
                "are scored without fitting; test candidates have no execution labels "
                "at selection time. When exclude_provider is set, no outcome from that "
                "provider enters fitting, tuning, or validation. Public specifications "
                "and memory text remain visible at inference."
            ),
            "retrieval_inputs": {
                "query": True,
                "service_and_active_version": True,
                "generic_roles": True,
                "event_task_key": False,
                "exact_required_evidence_atoms": False,
                "test_checker_outcomes": False,
            },
            "runs": result_runs,
            "elapsed_seconds": time.perf_counter() - started,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
    )
    files = ["result.json", "selections.jsonl"] + [
        str(row["state_file"]) for row in result_runs
    ]
    write_checksums(args.output, files)
    print(f"wrote {len(selection_rows)} frozen HGT selections to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
