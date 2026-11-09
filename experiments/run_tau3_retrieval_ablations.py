#!/usr/bin/env python3
"""Run encoder-matched A4/A5 retrieval ablations on tau3 candidate banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.dataset import PublicCoremDataset  # noqa: E402
from corem.model import TrainingConfig  # noqa: E402
from corem.retrieval import BundleBeamRetriever, RetrievalConfig  # noqa: E402
from corem.torch_encoding import TorchHGTEncoder  # noqa: E402
from corem.torch_model import TorchSetUtilityModel  # noqa: E402
from corem.types import Request  # noqa: E402
from experiments.run_experiments import ItemOnlyRetriever, NoClosureRetriever  # noqa: E402
from experiments.run_tau3_multistep import build_graph  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/tau3_corem"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results_tau3_retrieval_ablations"),
    )
    parser.add_argument("--seeds", default="17,29,41,53,67")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--include-duplicates", action="store_true")
    parser.add_argument(
        "--methods",
        default="full_corem,a4_item_only,a5_no_evidence_closure",
        help="comma-separated subset of full_corem,a4_item_only,a5_no_evidence_closure",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def task_request(row: dict[str, Any]) -> Request:
    """Build the observable request without answer-side gold operation IDs."""
    return Request(
        request_id=row["task_id"],
        text=row["query"],
        required_atoms=frozenset(row["required_atoms"]),
        verified_atoms=frozenset(row.get("verified_atoms", [])),
        timestamp=float(row.get("event_time") or 0.0),
        active_versions=frozenset(row.get("active_versions", [])),
    )


def run_seed(
    seed: int,
    dataset_path: Path,
    epochs: int,
    device: str,
    budget: int,
    beam_width: int,
    include_duplicates: bool,
    methods: tuple[str, ...],
) -> dict[str, Any]:
    dataset = PublicCoremDataset(dataset_path)
    train = [
        example
        for example in dataset.iter_outcome_examples(splits={"train"})
        if example.eligible_for_training
    ]
    train_graph = build_graph(
        dataset,
        train,
        include_hyperedges=True,
        include_duplicates=include_duplicates,
    )
    model = TorchSetUtilityModel(
        TorchHGTEncoder(
            dimension=32,
            text_hash_dimension=32,
            text_feature_mode="tokens",
            message_layers=2,
            attention_heads=4,
            seed=seed,
            device=device,
        )
    )
    started = time.perf_counter()
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
    tasks = [
        row
        for row in load_rows(dataset_path / "retrieval_tasks.jsonl")
        if row["split"] == "test"
    ]
    outcomes = [
        row
        for row in load_rows(dataset_path / "set_outcomes.jsonl")
        if row["split"] == "test"
    ]
    memory_ids_by_task: dict[str, set[str]] = {}
    for row in outcomes:
        memory_ids_by_task.setdefault(row["task_id"], set()).update(
            row["selected_memory_ids"]
        )
    memory_rows = {
        row["memory_id"]: row
        for row in load_rows(dataset_path / "memory_records.jsonl")
    }

    selections: list[dict[str, Any]] = []
    for task in tasks:
        task_id = task["task_id"]
        active_versions = frozenset(task.get("active_versions", []))
        bank = frozenset(
            memory_id
            for memory_id in memory_ids_by_task[task_id]
            if bool(memory_rows[memory_id].get("admissible", True))
            and bool(memory_rows[memory_id].get("provenance_verified", True))
            and (
                not memory_rows[memory_id].get("version_scope")
                or bool(
                    active_versions.intersection(
                        memory_rows[memory_id]["version_scope"]
                    )
                )
            )
        )
        graph = dataset.build_graph(
            memory_ids=bank,
            task_id=task_id,
            include_operation_entities=True,
            include_duplicates=include_duplicates,
            include_hyperedges=True,
        )
        request = task_request(task)
        config = RetrievalConfig(
            retrieval_budget=budget,
            beam_width=beam_width,
            max_depth=min(20, max(2, len(bank))),
            candidate_pool_size=min(20, len(bank)),
            dependency_hops=2,
            max_closure_size=20,
            context_cost_weight=0.05,
            uncovered_penalty=0.75,
        )
        all_retrievers = {
            "full_corem": BundleBeamRetriever(model, config),
            "a4_item_only": ItemOnlyRetriever(model, config),
            "a5_no_evidence_closure": NoClosureRetriever(model, config),
        }
        for method in methods:
            retriever = all_retrievers[method]
            retrieval_started = time.perf_counter()
            result = retriever.retrieve(request, bank, graph)
            selections.append(
                {
                    "seed": seed,
                    "task_id": task_id,
                    "source_task_id": task["source_task_id"],
                    "domain": task["domain"],
                    "method": method,
                    "selected_memory_ids": sorted(result.selected_ids),
                    "status": result.status,
                    "predicted_probability": result.probability,
                    "objective": result.objective,
                    "token_cost": result.token_cost,
                    "evaluations": result.evaluations,
                    "latency_ms": (time.perf_counter() - retrieval_started) * 1000.0,
                }
            )
    return {
        "seed": seed,
        "fit": {
            "epochs": fit.epochs,
            "final_loss": fit.losses[-1],
            "elapsed_seconds": time.perf_counter() - started,
        },
        "selections": selections,
    }


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    allowed_methods = {
        "full_corem",
        "a4_item_only",
        "a5_no_evidence_closure",
    }
    if not methods or not set(methods).issubset(allowed_methods):
        raise ValueError(f"invalid methods: {methods}")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    started = time.perf_counter()
    if args.workers == 1:
        runs = [
            run_seed(
                seed,
                args.dataset,
                args.epochs,
                args.device,
                args.budget,
                args.beam_width,
                args.include_duplicates,
                methods,
            )
            for seed in seeds
        ]
    else:
        completed: dict[int, dict[str, Any]] = {}
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(seeds)), mp_context=context
        ) as executor:
            futures = {
                executor.submit(
                    run_seed,
                    seed,
                    args.dataset,
                    args.epochs,
                    args.device,
                    args.budget,
                    args.beam_width,
                    args.include_duplicates,
                    methods,
                ): seed
                for seed in seeds
            }
            for position, future in enumerate(as_completed(futures), start=1):
                seed = futures[future]
                completed[seed] = future.result()
                print(f"[{position}/{len(seeds)}] completed seed={seed}", flush=True)
        runs = [completed[seed] for seed in seeds]

    result = {
        "status": "selection_complete_execution_pending",
        "scope": "encoder-matched retrieval mechanism ablations",
        "configuration": {
            "seeds": seeds,
            "epochs": args.epochs,
            "device": args.device,
            "workers": args.workers,
            "retrieval_budget": args.budget,
            "beam_width": args.beam_width,
            "duplicate_pairs_used": args.include_duplicates,
            "methods": list(methods),
            "version_incompatible_nodes_excluded_before_graph_encoding": True,
            "gold_operation_ids_used_as_request_features": False,
            "outcome_derived_oracle_edges_used": False,
        },
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(torch.device(args.device))
                if args.device.startswith("cuda")
                else None
            ),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "runs": runs,
        "limitations": [
            "The task-local bank contains only actions from the official reference trajectory.",
            "Selected sets require independent environment replay before VSR is assigned.",
            "Every record individually covers the coarse task goal atom, so A5 may be a no-op on this release.",
        ],
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "retrieval_ablation_selections.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output / "checksums.sha256").write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n",
        encoding="utf-8",
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
