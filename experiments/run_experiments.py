#!/usr/bin/env python3
"""Run the honest, locally supported CoReM experiment suite.

Observed RapidAPI endpoint outcomes train and evaluate only unary prediction.
EMP/five-domain/HSC composition labels are not execution outcomes; this runner
uses the exact EMP subset only for clearly labelled reconstruction and retention
proxy experiments. It never reports proxy reconstruction as verified success.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import platform
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.encoding import StableHashEncoder, Vector
from corem.model import SetUtilityModel, TrainingConfig
from corem.retention import GroupAwareRetention, RetentionConfig
from corem.retrieval import Bundle, BundleBeamRetriever, RetrievalConfig
from corem.types import (
    DependencyHyperedge,
    EvidenceGraph,
    MemoryRecord,
    OutcomeExample,
    Request,
    RetrievalResult,
)
from experiments.common import (
    binary_metrics,
    calibration_bins,
    clustered_bootstrap_ci,
    compile_figures,
    holm_adjust,
    json_dump,
    mean,
    paired_randomization_pvalue,
    percentile,
    select_threshold,
    write_bar_figure,
    write_csv,
    write_line_figure,
    write_table_bundle,
)


SEEDS = (17, 29, 41, 53, 67)
RETRIEVAL_BUDGETS = (256, 512, 1024, 2048)
BEAM_WIDTHS = (1, 4, 8, 16)
BOOTSTRAP_DRAWS = 2_000
RANDOMIZATION_DRAWS = 20_000
OUTCOME_SUFFIX = re.compile(r"\s*Observed endpoint test (?:succeeded|failed)\.\s*$", re.I)


class CachedHashEncoder:
    """Request-vector cache around the deterministic reference encoder."""

    def __init__(self) -> None:
        self.base = StableHashEncoder()
        self.dimension = self.base.dimension
        self._requests: dict[Request, Vector] = {}

    def encode_request(self, request: Request) -> Vector:
        if request not in self._requests:
            self._requests[request] = self.base.encode_request(request)
        return self._requests[request]

    def encode_graph(self, graph: EvidenceGraph) -> dict[str, Vector]:
        return self.base.encode_graph(graph)

    def condition_on_request(
        self,
        request: Request,
        record_id: str,
        graph: EvidenceGraph,
        record_vectors: dict[str, Vector],
    ) -> Vector:
        return self.base.condition_on_request(request, record_id, graph, record_vectors)

    @property
    def _all_vectors(self) -> dict[str, Vector]:
        return self.base._all_vectors


class ItemOnlyRetriever(BundleBeamRetriever):
    """A4 ablation: keep the objective but prohibit dependency-bundle moves."""

    def build_bundles(
        self, request: Request, bank: frozenset[str], graph: EvidenceGraph
    ) -> tuple[Bundle, ...]:
        eligible = [key for key in bank if graph.is_admissible(key, request)]
        ranked = sorted(
            eligible,
            key=lambda key: (self.model.semantic_similarity(request, key, graph), key),
            reverse=True,
        )[: self.config.candidate_pool_size]
        return tuple(Bundle(f"record:{key}", frozenset((key,))) for key in ranked)

    def _dependency_closure(
        self,
        request: Request,
        seed: frozenset[str],
        bank: frozenset[str],
        graph: EvidenceGraph,
    ) -> frozenset[str]:
        del request, bank, graph
        return seed


class NoClosureRetriever(BundleBeamRetriever):
    """A5 ablation: optimize a nonempty set without mandatory evidence closure."""

    def _is_feasible(
        self, request: Request, selected: frozenset[str], graph: EvidenceGraph
    ) -> bool:
        return (
            bool(selected)
            and self._within_budget(selected, graph)
            and all(graph.is_admissible(key, request) for key in selected)
        )

    def _dependency_closure(
        self,
        request: Request,
        seed: frozenset[str],
        bank: frozenset[str],
        graph: EvidenceGraph,
    ) -> frozenset[str]:
        del request, bank, graph
        return seed


class SingletonProposalRetention(GroupAwareRetention):
    """A7 ablation: retention may insert new memories only one at a time."""

    def _bundle_proposals(
        self,
        new_ids: frozenset[str],
        universe: frozenset[str],
        graph: EvidenceGraph,
    ) -> list[frozenset[str]]:
        del universe, graph
        return [frozenset((key,)) for key in sorted(new_ids)]


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rng(*parts: object) -> random.Random:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return random.Random(int(hashlib.sha256(payload).hexdigest()[:16], 16))


def row_to_record(row: dict[str, Any], *, sanitize_outcome: bool = False) -> MemoryRecord:
    text = OUTCOME_SUFFIX.sub("", row["text"]) if sanitize_outcome else row["text"]
    return MemoryRecord(
        record_id=row["memory_id"],
        text=text,
        evidence_atoms=frozenset(row["evidence_atoms"]),
        operation_ids=frozenset(row["operation_ids"]),
        token_cost=int(row["token_cost"]),
        storage_cost=int(row["storage_cost"]),
        timestamp=float(row.get("event_time") or 0.0),
        provenance=str(row["provenance"]),
        service_scope=frozenset(row["operation_ids"]),
        version_scope=frozenset(row.get("version_scope", [])),
        valid_from=float(row.get("valid_from") or 0.0),
        valid_until=(
            float(row["valid_until"])
            if row.get("valid_until") is not None
            else None
        ),
        execution_outcome=(
            int(row["observed_outcome"])
            if row.get("observed_outcome") is not None
            else None
        ),
        provenance_verified=bool(
            row.get("provenance_verified", row.get("admissible", True))
        ),
        admissible=bool(row["admissible"]),
    )


def outcome_example(row: dict[str, Any]) -> OutcomeExample:
    return OutcomeExample(
        request=Request(
            request_id=row["task_id"],
            text=row["query"],
            required_atoms=frozenset(row["required_atoms"]),
            timestamp=float(row.get("event_time") or 0.0),
            active_versions=frozenset(row.get("active_versions", [])),
        ),
        selected_ids=frozenset(row["selected_memory_ids"]),
        outcome=int(row["outcome"]),
        actually_executed=bool(row.get("actually_executed", True)),
        independently_verified=bool(row.get("independently_verified", True)),
        eligible_for_training=bool(row.get("eligible_for_training", True)),
        eligible_for_interaction_training=bool(
            row.get("eligible_for_interaction_training", False)
        ),
    )


def task_request(row: dict[str, Any]) -> Request:
    # gold_operation_ids are intentionally omitted: including them would leak the answer.
    return Request(
        request_id=row["task_id"],
        text=row["query"],
        required_atoms=frozenset(row["required_atoms"]),
        verified_atoms=frozenset(row.get("verified_atoms", [])),
        timestamp=float(row.get("event_time") or 0.0),
        active_versions=frozenset(row.get("active_versions", [])),
    )


def clone_model(source: SetUtilityModel, *, complement: bool = True, redundancy: bool = True) -> SetUtilityModel:
    complement_weights = list(source.complement_weights) if complement else [-40.0, 0.0, 0.0, 0.0, 0.0]
    redundancy_weights = list(source.redundancy_weights) if redundancy else [-40.0, 0.0, 0.0, 0.0]
    return SetUtilityModel(
        encoder=CachedHashEncoder(),
        base_weights=list(source.base_weights),
        unary_weights=list(source.unary_weights),
        complement_weights=complement_weights,
        redundancy_weights=redundancy_weights,
        relation_heads={key: list(value) for key, value in source.relation_heads.items()},
    )


def model_weights(model: SetUtilityModel) -> dict[str, Any]:
    return {
        "base": list(model.base_weights),
        "unary": list(model.unary_weights),
        "complement": list(model.complement_weights),
        "redundancy": list(model.redundancy_weights),
        "relations": {
            key: list(value) for key, value in sorted(model.relation_heads.items())
        },
    }


def prepare_data(dataset: Path) -> dict[str, Any]:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    quality = json.loads((dataset / "quality_report.json").read_text(encoding="utf-8"))
    outcomes = list(load_jsonl(dataset / "set_outcomes.jsonl"))
    outcome_memory_ids = {key for row in outcomes for key in row["selected_memory_ids"]}

    exact_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(dataset / "dependency_hyperedges.jsonl"):
        if row.get("admissible_default") and row.get("grounding") == "output_input_exact":
            exact_edges[row["task_id"]].append(row)

    tasks: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(dataset / "retrieval_tasks.jsonl"):
        if row["task_id"] in exact_edges:
            tasks[row["task_id"]] = row

    emp_ids = {key for row in tasks.values() for key in row["gold_memory_ids"]}
    rapid_memories: dict[str, dict[str, Any]] = {}
    emp_memories: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(dataset / "memory_records.jsonl"):
        key = row["memory_id"]
        if key in outcome_memory_ids:
            rapid_memories[key] = row
        if row["source"] == "emp_synthetic":
            emp_memories[key] = row
    if not emp_ids.issubset(emp_memories):
        raise ValueError("exact EMP tasks reference missing memories")

    operation_services: dict[str, str] = {}
    rapid_operations = {
        operation
        for row in rapid_memories.values()
        for operation in row["operation_ids"]
    }
    for row in load_jsonl(dataset / "operations.jsonl"):
        if row["operation_id"] in rapid_operations:
            operation_services[row["operation_id"]] = str(row.get("service_id") or row["operation_id"])

    duplicate_groups = []
    for row in load_jsonl(dataset / "duplicate_groups.jsonl"):
        members = frozenset(emp_memories).intersection(row["memory_ids"])
        if len(members) > 1:
            duplicate_groups.append(frozenset(members))

    return {
        "manifest": manifest,
        "quality": quality,
        "outcomes": outcomes,
        "rapid_memories": rapid_memories,
        "operation_services": operation_services,
        "exact_edges": dict(exact_edges),
        "tasks": tasks,
        "emp_memories": emp_memories,
        "duplicate_groups": duplicate_groups,
    }


def build_outcome_graph(
    outcome_rows: Sequence[dict[str, Any]],
    memory_rows: dict[str, dict[str, Any]],
    *,
    sanitize_outcome: bool,
) -> EvidenceGraph:
    graph = EvidenceGraph()
    ids = {key for row in outcome_rows for key in row["selected_memory_ids"]}
    graph.add_records(row_to_record(memory_rows[key], sanitize_outcome=sanitize_outcome) for key in sorted(ids))
    return graph


def predict_examples(
    model: SetUtilityModel, examples: Sequence[OutcomeExample], graph: EvidenceGraph
) -> list[float]:
    return [model.probability(example.request, example.selected_ids, graph) for example in examples]


def run_endpoint_prediction(data: dict[str, Any], output: Path, log: list[str]) -> dict[str, Any]:
    rows_by_split = {
        split: [row for row in data["outcomes"] if row["split"] == split]
        for split in ("train", "val", "test")
    }
    examples = {split: [outcome_example(row) for row in rows] for split, rows in rows_by_split.items()}
    sanitized_graphs = {
        split: build_outcome_graph(rows, data["rapid_memories"], sanitize_outcome=True)
        for split, rows in rows_by_split.items()
    }

    trained = SetUtilityModel(encoder=CachedHashEncoder())
    config = TrainingConfig(
        learning_rate=0.08,
        epochs=60,
        calibration_weight=0.10,
        interaction_weight=0.001,
        l2_weight=0.0001,
        tolerance=1e-7,
        patience=8,
    )
    started = time.perf_counter()
    fit = trained.fit(examples["train"], sanitized_graphs["train"], config)
    train_seconds = time.perf_counter() - started
    log.append(
        f"Unary training: n={len(examples['train'])}, epochs={fit.epochs}, "
        f"loss={fit.losses[0]:.6f}->{fit.losses[-1]:.6f}, seconds={train_seconds:.3f}."
    )

    untrained = SetUtilityModel(encoder=CachedHashEncoder())
    probabilities: dict[str, dict[str, list[float]]] = {
        "trained_sanitized": {
            split: predict_examples(trained, split_examples, sanitized_graphs[split])
            for split, split_examples in examples.items()
        },
        "untrained_sanitized": {
            split: predict_examples(untrained, split_examples, sanitized_graphs[split])
            for split, split_examples in examples.items()
        },
    }

    # Leakage diagnostic: the generated memory text literally contains succeeded/failed.
    leaky_graphs = {
        split: build_outcome_graph(rows, data["rapid_memories"], sanitize_outcome=False)
        for split, rows in rows_by_split.items()
    }
    leaky = SetUtilityModel(encoder=CachedHashEncoder())
    leaky_fit = leaky.fit(
        examples["train"],
        leaky_graphs["train"],
        copy.copy(config),
    )
    probabilities["leaky_text_diagnostic"] = {
        split: predict_examples(leaky, split_examples, leaky_graphs[split])
        for split, split_examples in examples.items()
    }
    train_rate = mean(float(example.outcome) for example in examples["train"])
    probabilities["constant_train_rate"] = {
        split: [train_rate] * len(split_examples) for split, split_examples in examples.items()
    }

    thresholds: dict[str, float] = {}
    for method, split_probabilities in probabilities.items():
        val_labels = [example.outcome for example in examples["val"]]
        thresholds[method] = select_threshold(val_labels, split_probabilities["val"])

    summary_rows: list[dict[str, Any]] = []
    for method, split_probabilities in probabilities.items():
        for split in ("train", "val", "test"):
            labels = [example.outcome for example in examples[split]]
            metrics = binary_metrics(labels, split_probabilities[split], thresholds[method])
            summary_rows.append(
                {
                    "method": method,
                    "split": split,
                    **{key: round(value, 6) for key, value in metrics.items()},
                }
            )

    test_rows = []
    for index, (source_row, example) in enumerate(zip(rows_by_split["test"], examples["test"])):
        memory_id = next(iter(example.selected_ids))
        operation_id = data["rapid_memories"][memory_id]["operation_ids"][0]
        test_rows.append(
            {
                "task_id": example.request.request_id,
                "service_id": data["operation_services"].get(operation_id, operation_id),
                "label": example.outcome,
                "trained_sanitized_probability": probabilities["trained_sanitized"]["test"][index],
                "untrained_sanitized_probability": probabilities["untrained_sanitized"]["test"][index],
                "leaky_text_probability": probabilities["leaky_text_diagnostic"]["test"][index],
            }
        )
    write_csv(output / "raw" / "endpoint_test_predictions.csv", test_rows)

    brier_ci = clustered_bootstrap_ci(
        test_rows,
        cluster_key="service_id",
        statistic=lambda sample: mean(
            (float(row["trained_sanitized_probability"]) - int(row["label"])) ** 2
            for row in sample
        ),
        draws=BOOTSTRAP_DRAWS,
        seed=2027,
    )
    test_labels = [example.outcome for example in examples["test"]]
    bins = calibration_bins(test_labels, probabilities["trained_sanitized"]["test"])
    write_csv(output / "raw" / "endpoint_calibration_bins.csv", bins)
    json_dump(
        output / "models" / "unary_model.json",
        {
            "weights": model_weights(trained),
            "training_config": asdict(config),
            "fit": {
                "epochs": fit.epochs,
                "converged": fit.converged,
                "initial_loss": fit.losses[0],
                "final_loss": fit.losses[-1],
                "seconds": train_seconds,
            },
            "scope": "observed singleton endpoint outcomes only; interaction weights are not outcome-trained",
        },
    )
    json_dump(
        output / "raw" / "endpoint_prediction.json",
        {
            "thresholds_selected_on_validation": thresholds,
            "metrics": summary_rows,
            "trained_test_brier_cluster_bootstrap_ci95": brier_ci,
            "leaky_diagnostic_epochs": leaky_fit.epochs,
        },
    )
    return {
        "model": trained,
        "summary_rows": summary_rows,
        "calibration_bins": bins,
        "brier_ci": brier_ci,
        "counts": {split: len(rows) for split, rows in rows_by_split.items()},
    }


def build_emp_candidate_graph(
    data: dict[str, Any],
    task: dict[str, Any],
    *,
    seed: int,
    candidate_count: int,
) -> tuple[Request, EvidenceGraph, frozenset[str]]:
    gold = frozenset(task["gold_memory_ids"])
    available = sorted(set(data["emp_memories"]).difference(gold))
    need = max(0, candidate_count - len(gold))
    rng = stable_rng(task["task_id"], seed, candidate_count)
    distractors = rng.sample(available, min(need, len(available)))
    candidate_ids = gold.union(distractors)
    graph = EvidenceGraph()
    graph.add_records(row_to_record(data["emp_memories"][key]) for key in sorted(candidate_ids))
    for row in data["exact_edges"].get(task["task_id"], []):
        members = frozenset(row["members"])
        if members.issubset(candidate_ids):
            graph.add_hyperedge(
                DependencyHyperedge(
                    edge_id=row["edge_id"],
                    members=members,
                    evidence_atoms=frozenset(row["evidence_atoms"]),
                    relation_type=row["relation_type"],
                    provenance=row["provenance"],
                )
            )
    for group in data["duplicate_groups"]:
        members = sorted(group.intersection(candidate_ids))
        for left, right in itertools.combinations(members, 2):
            graph.add_duplicate(left, right)
    return task_request(task), graph, gold


def retrieval_config(*, budget: int, beam_width: int, candidate_count: int, max_depth: int = 6) -> RetrievalConfig:
    return RetrievalConfig(
        retrieval_budget=budget,
        beam_width=beam_width,
        max_depth=max_depth,
        candidate_pool_size=candidate_count,
        dependency_hops=2,
        context_cost_weight=0.05,
        uncovered_penalty=0.75,
        fallback_value=0.0,
    )


def semantic_topk(
    model: SetUtilityModel,
    request: Request,
    graph: EvidenceGraph,
    *,
    budget: int,
    max_items: int,
) -> RetrievalResult:
    ranked = sorted(
        graph.records,
        key=lambda key: (model.semantic_similarity(request, key, graph), key),
        reverse=True,
    )
    selected: set[str] = set()
    cost = 0
    for key in ranked:
        record_cost = graph.records[key].token_cost
        if len(selected) >= max_items:
            break
        if cost + record_cost <= budget:
            selected.add(key)
            cost += record_cost
    frozen = frozenset(selected)
    probability = model.probability(request, frozen, graph)
    objective = probability - 0.05 * cost / budget
    return RetrievalResult(
        selected_ids=frozen,
        probability=probability,
        objective=objective,
        token_cost=cost,
        status="ok_unconstrained" if frozen else "fallback",
        uncovered_atoms=graph.uncovered_atoms(request, frozen),
        evaluations=len(ranked),
    )


def retrieval_row(
    *,
    task: dict[str, Any],
    seed: int,
    method: str,
    graph: EvidenceGraph,
    request: Request,
    gold: frozenset[str],
    result: RetrievalResult,
    elapsed_ms: float,
    budget: int,
    beam_width: int,
    candidate_count: int,
) -> dict[str, Any]:
    covered = len(request.required_atoms.difference(graph.uncovered_atoms(request, result.selected_ids)))
    required = max(1, len(request.required_atoms))
    duplicates = sum(left in result.selected_ids and right in result.selected_ids for left, right in graph.duplicate_pairs)
    gold_cost = sum(graph.records[key].token_cost for key in gold)
    return {
        "task_id": task["task_id"],
        "split": task["split"],
        "seed": seed,
        "method": method,
        "budget": budget,
        "beam_width": beam_width,
        "candidate_count": candidate_count,
        "gold_size": len(gold),
        "gold_cost": gold_cost,
        "gold_budget_feasible": int(gold_cost <= budget),
        "status": result.status,
        "ecr_proxy": int(graph.is_evidence_complete(request, result.selected_ids)),
        "binding_coverage_proxy": covered / required,
        "gold_recall": len(gold.intersection(result.selected_ids)) / len(gold),
        "exact_set_match": int(result.selected_ids == gold),
        "selected_size": len(result.selected_ids),
        "token_cost": result.token_cost,
        "probability": result.probability,
        "objective": result.objective,
        "evaluations": result.evaluations,
        "duplicate_pairs_selected": duplicates,
        "latency_ms": elapsed_ms,
        "selected_ids": json.dumps(sorted(result.selected_ids), ensure_ascii=False),
    }


def evaluate_one_graph(
    *,
    data: dict[str, Any],
    task: dict[str, Any],
    seed: int,
    candidate_count: int,
    budget: int,
    beam_width: int,
    learned_model: SetUtilityModel,
    methods: Sequence[str],
) -> list[dict[str, Any]]:
    request, graph, gold = build_emp_candidate_graph(
        data, task, seed=seed, candidate_count=candidate_count
    )
    config = retrieval_config(
        budget=budget,
        beam_width=beam_width,
        candidate_count=candidate_count,
        max_depth=max(6, len(gold)),
    )
    models = {
        "full_corem_proxy": clone_model(learned_model),
        "additive_utility": clone_model(learned_model, complement=False, redundancy=False),
        "no_complement": clone_model(learned_model, complement=False, redundancy=True),
        "no_redundancy": clone_model(learned_model, complement=True, redundancy=False),
        "item_only_search": clone_model(learned_model),
        "no_evidence_closure": clone_model(learned_model),
        "semantic_topk": clone_model(learned_model),
    }
    output: list[dict[str, Any]] = []
    for method in methods:
        started = time.perf_counter()
        if method == "semantic_topk":
            result = semantic_topk(
                models[method], request, graph, budget=budget, max_items=config.max_depth
            )
        elif method == "item_only_search":
            result = ItemOnlyRetriever(models[method], copy.copy(config)).retrieve(
                request, graph.records, graph
            )
        elif method == "no_evidence_closure":
            result = NoClosureRetriever(models[method], copy.copy(config)).retrieve(
                request, graph.records, graph
            )
        else:
            result = BundleBeamRetriever(models[method], copy.copy(config)).retrieve(
                request, graph.records, graph
            )
        elapsed = (time.perf_counter() - started) * 1000.0
        output.append(
            retrieval_row(
                task=task,
                seed=seed,
                method=method,
                graph=graph,
                request=request,
                gold=gold,
                result=result,
                elapsed_ms=elapsed,
                budget=budget,
                beam_width=beam_width,
                candidate_count=candidate_count,
            )
        )
    return output


def choose_beam_width(data: dict[str, Any], model: SetUtilityModel) -> tuple[int, list[dict[str, Any]]]:
    val_tasks = sorted(
        (task for task in data["tasks"].values() if task["split"] == "val"),
        key=lambda row: row["task_id"],
    )
    rows: list[dict[str, Any]] = []
    for width in BEAM_WIDTHS:
        for task in val_tasks:
            for seed in SEEDS:
                rows.extend(
                    evaluate_one_graph(
                        data=data,
                        task=task,
                        seed=seed,
                        candidate_count=20,
                        budget=1024,
                        beam_width=width,
                        learned_model=model,
                        methods=("full_corem_proxy",),
                    )
                )
    candidates = []
    for width in BEAM_WIDTHS:
        subset = [row for row in rows if row["beam_width"] == width]
        candidates.append(
            (
                mean(float(row["ecr_proxy"]) for row in subset),
                -mean(float(row["latency_ms"]) for row in subset),
                -width,
                width,
            )
        )
    return max(candidates)[-1], rows


def aggregate_retrieval(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in sorted({str(row["method"]) for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        ci = clustered_bootstrap_ci(
            subset,
            cluster_key="task_id",
            statistic=lambda sample: mean(float(row["ecr_proxy"]) for row in sample),
            draws=BOOTSTRAP_DRAWS,
            seed=2027,
        )
        seed_means = [
            mean(float(row["ecr_proxy"]) for row in subset if row["seed"] == seed)
            for seed in SEEDS
        ]
        output.append(
            {
                "method": method,
                "task_seed_n": len(subset),
                "tasks": len({row["task_id"] for row in subset}),
                "ecr_proxy": mean(float(row["ecr_proxy"]) for row in subset),
                "ecr_ci_low": ci[0],
                "ecr_ci_high": ci[1],
                "seed_sd": statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0,
                "binding_coverage_proxy": mean(float(row["binding_coverage_proxy"]) for row in subset),
                "gold_recall": mean(float(row["gold_recall"]) for row in subset),
                "exact_set_match": mean(float(row["exact_set_match"]) for row in subset),
                "fallback_rate": mean(float(row["status"] == "fallback") for row in subset),
                "mean_tokens": mean(float(row["token_cost"]) for row in subset),
                "mean_latency_ms": mean(float(row["latency_ms"]) for row in subset),
                "p95_latency_ms": percentile([float(row["latency_ms"]) for row in subset], 0.95),
                "mean_evaluations": mean(float(row["evaluations"]) for row in subset),
            }
        )
    return output


def retrieval_significance(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    task_method: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        task_method[(str(row["task_id"]), str(row["method"]))].append(float(row["ecr_proxy"]))
    contrasts = ("semantic_topk", "additive_utility", "no_complement", "item_only_search")
    raw: dict[str, float] = {}
    differences_by_method: dict[str, list[float]] = {}
    tasks = sorted({task for task, method in task_method if method == "full_corem_proxy"})
    for method in contrasts:
        differences = [
            mean(task_method[(task, "full_corem_proxy")]) - mean(task_method[(task, method)])
            for task in tasks
        ]
        differences_by_method[method] = differences
        raw[method] = paired_randomization_pvalue(
            differences, draws=RANDOMIZATION_DRAWS, seed=2027
        )
    adjusted = holm_adjust(raw)
    return [
        {
            "contrast": f"full_corem_proxy - {method}",
            "mean_paired_difference": mean(differences_by_method[method]),
            "randomization_p": raw[method],
            "holm_p": adjusted[method],
            "clusters": len(differences_by_method[method]),
            "interpretation": "proxy ECR only; not verified workflow success",
        }
        for method in contrasts
    ]


def exhaustive_reference(
    model: SetUtilityModel,
    request: Request,
    graph: EvidenceGraph,
    *,
    budget: int,
) -> tuple[RetrievalResult, float]:
    keys = sorted(graph.records)
    best: tuple[float, frozenset[str], float, int] | None = None
    started = time.perf_counter()
    evaluations = 0
    for mask in range(1 << len(keys)):
        selected = frozenset(keys[index] for index in range(len(keys)) if mask & (1 << index))
        cost = sum(graph.records[key].token_cost for key in selected)
        if cost > budget or not graph.is_evidence_complete(request, selected):
            continue
        probability = model.probability(request, selected, graph)
        objective = probability - 0.05 * cost / budget
        evaluations += 1
        candidate = (objective, selected, probability, cost)
        if best is None or (candidate[0], -candidate[3]) > (best[0], -best[3]):
            best = candidate
    elapsed = (time.perf_counter() - started) * 1000.0
    if best is None:
        return (
            RetrievalResult(
                selected_ids=frozenset(), probability=0.0, objective=0.0, token_cost=0,
                status="fallback", uncovered_atoms=request.required_atoms, evaluations=evaluations,
            ),
            elapsed,
        )
    return (
        RetrievalResult(
            selected_ids=best[1], probability=best[2], objective=best[0], token_cost=best[3],
            status="ok", evaluations=evaluations,
        ),
        elapsed,
    )


def run_retrieval_experiments(
    data: dict[str, Any], model: SetUtilityModel, output: Path, log: list[str]
) -> dict[str, Any]:
    selected_width, selection_rows = choose_beam_width(data, model)
    write_csv(output / "raw" / "beam_selection_validation.csv", selection_rows)
    log.append(f"Beam width selected on validation proxy ECR: {selected_width}.")

    test_tasks = sorted(
        (task for task in data["tasks"].values() if task["split"] == "test"),
        key=lambda row: row["task_id"],
    )
    methods = (
        "full_corem_proxy",
        "semantic_topk",
        "additive_utility",
        "no_complement",
        "no_redundancy",
        "item_only_search",
        "no_evidence_closure",
    )
    main_rows: list[dict[str, Any]] = []
    for task in test_tasks:
        for seed in SEEDS:
            main_rows.extend(
                evaluate_one_graph(
                    data=data,
                    task=task,
                    seed=seed,
                    candidate_count=20,
                    budget=1024,
                    beam_width=selected_width,
                    learned_model=model,
                    methods=methods,
                )
            )
    write_csv(output / "raw" / "retrieval_test_task_seed.csv", main_rows)
    summary = aggregate_retrieval(main_rows)
    significance = retrieval_significance(main_rows)
    write_csv(output / "raw" / "retrieval_paired_tests.csv", significance)

    budget_rows: list[dict[str, Any]] = []
    for budget in RETRIEVAL_BUDGETS:
        for task in test_tasks:
            for seed in SEEDS:
                budget_rows.extend(
                    evaluate_one_graph(
                        data=data,
                        task=task,
                        seed=seed,
                        candidate_count=20,
                        budget=budget,
                        beam_width=selected_width,
                        learned_model=model,
                        methods=("full_corem_proxy",),
                    )
                )
    write_csv(output / "raw" / "retrieval_budget_grid.csv", budget_rows)
    budget_summary = []
    for budget in RETRIEVAL_BUDGETS:
        subset = [row for row in budget_rows if row["budget"] == budget]
        feasible = [row for row in subset if row["gold_budget_feasible"]]
        budget_summary.append(
            {
                "budget": budget,
                "task_seed_n": len(subset),
                "gold_feasible_rate": mean(float(row["gold_budget_feasible"]) for row in subset),
                "ecr_proxy_all": mean(float(row["ecr_proxy"]) for row in subset),
                "ecr_proxy_feasible": mean(float(row["ecr_proxy"]) for row in feasible),
                "gold_recall": mean(float(row["gold_recall"]) for row in subset),
                "mean_latency_ms": mean(float(row["latency_ms"]) for row in subset),
                "p95_latency_ms": percentile([float(row["latency_ms"]) for row in subset], 0.95),
                "mean_evaluations": mean(float(row["evaluations"]) for row in subset),
            }
        )

    efficiency_rows: list[dict[str, Any]] = []
    for candidate_count in (8, 12, 20):
        for width in BEAM_WIDTHS:
            for task in test_tasks:
                efficiency_rows.extend(
                    evaluate_one_graph(
                        data=data,
                        task=task,
                        seed=SEEDS[0],
                        candidate_count=max(candidate_count, len(task["gold_memory_ids"])),
                        budget=1024,
                        beam_width=width,
                        learned_model=model,
                        methods=("full_corem_proxy",),
                    )
                )
    write_csv(output / "raw" / "retrieval_efficiency_grid.csv", efficiency_rows)
    efficiency_summary = []
    for candidate_count in (8, 12, 20):
        for width in BEAM_WIDTHS:
            subset = [
                row for row in efficiency_rows
                if row["candidate_count"] == candidate_count and row["beam_width"] == width
            ]
            efficiency_summary.append(
                {
                    "candidate_count": candidate_count,
                    "beam_width": width,
                    "ecr_proxy": mean(float(row["ecr_proxy"]) for row in subset),
                    "mean_latency_ms": mean(float(row["latency_ms"]) for row in subset),
                    "p95_latency_ms": percentile([float(row["latency_ms"]) for row in subset], 0.95),
                    "mean_evaluations": mean(float(row["evaluations"]) for row in subset),
                }
            )

    exhaustive_rows = []
    for task in test_tasks:
        request, graph, gold = build_emp_candidate_graph(
            data, task, seed=SEEDS[0], candidate_count=max(12, len(task["gold_memory_ids"]))
        )
        beam_model = clone_model(model)
        beam = BundleBeamRetriever(
            beam_model,
            retrieval_config(
                budget=1024,
                beam_width=selected_width,
                candidate_count=len(graph.records),
                max_depth=max(6, len(gold)),
            ),
        )
        started = time.perf_counter()
        approximate = beam.retrieve(request, graph.records, graph)
        beam_ms = (time.perf_counter() - started) * 1000.0
        oracle, oracle_ms = exhaustive_reference(clone_model(model), request, graph, budget=1024)
        exhaustive_rows.append(
            {
                "task_id": task["task_id"],
                "candidate_count": len(graph.records),
                "beam_objective": approximate.objective,
                "exhaustive_objective": oracle.objective,
                "objective_gap": max(0.0, oracle.objective - approximate.objective),
                "same_selected_set": int(approximate.selected_ids == oracle.selected_ids),
                "beam_evaluations": approximate.evaluations,
                "exhaustive_feasible_evaluations": oracle.evaluations,
                "beam_latency_ms": beam_ms,
                "exhaustive_latency_ms": oracle_ms,
            }
        )
    write_csv(output / "raw" / "retrieval_exhaustive_reference.csv", exhaustive_rows)
    exhaustive_summary = {
        "tasks": len(exhaustive_rows),
        "mean_objective_gap": mean(float(row["objective_gap"]) for row in exhaustive_rows),
        "max_objective_gap": max(float(row["objective_gap"]) for row in exhaustive_rows),
        "same_selected_set_rate": mean(float(row["same_selected_set"]) for row in exhaustive_rows),
        "mean_beam_latency_ms": mean(float(row["beam_latency_ms"]) for row in exhaustive_rows),
        "mean_exhaustive_latency_ms": mean(float(row["exhaustive_latency_ms"]) for row in exhaustive_rows),
    }
    log.append(
        f"Retrieval proxy: {len(test_tasks)} exact-edge test tasks x {len(SEEDS)} seeds; "
        f"candidate bank=20, Bq=1024, selected beam={selected_width}."
    )
    return {
        "selected_beam_width": selected_width,
        "main_rows": main_rows,
        "summary": summary,
        "significance": significance,
        "budget_summary": budget_summary,
        "efficiency_summary": efficiency_summary,
        "exhaustive_summary": exhaustive_summary,
    }


def greedy_item_bank(
    universe: Iterable[str], scores: dict[str, float], budget: int, graph: EvidenceGraph
) -> frozenset[str]:
    retained: set[str] = set()
    cost = 0
    for key in sorted(universe, key=lambda item: (scores.get(item, 0.0), item), reverse=True):
        item_cost = graph.records[key].storage_cost
        if cost + item_cost <= budget:
            retained.add(key)
            cost += item_cost
    return frozenset(retained)


def run_retention_proxy(
    data: dict[str, Any], learned_model: SetUtilityModel, beam_width: int, output: Path, log: list[str]
) -> dict[str, Any]:
    tasks = sorted(
        (
            task for task in data["tasks"].values()
            if len(task["gold_memory_ids"]) == 2 and data["exact_edges"].get(task["task_id"])
        ),
        key=lambda row: row["task_id"],
    )
    graph = EvidenceGraph()
    record_ids = sorted({key for task in tasks for key in task["gold_memory_ids"]})
    graph.add_records(row_to_record(data["emp_memories"][key]) for key in record_ids)
    for task in tasks:
        for row in data["exact_edges"][task["task_id"]]:
            members = frozenset(row["members"])
            if members.issubset(graph.records):
                graph.add_hyperedge(
                    DependencyHyperedge(
                        row["edge_id"], members, frozenset(row["evidence_atoms"]),
                        row["relation_type"], row["provenance"],
                    )
                )
    total_storage = sum(record.storage_cost for record in graph.records.values())
    max_group = max(
        sum(graph.records[key].storage_cost for key in task["gold_memory_ids"])
        for task in tasks
    )
    storage_budget = max(max_group, int(total_storage * 0.40))
    retrieval = BundleBeamRetriever(
        clone_model(learned_model),
        retrieval_config(
            budget=1024,
            beam_width=beam_width,
            candidate_count=max(1, len(graph.records)),
            max_depth=2,
        ),
    )
    group_retainer = GroupAwareRetention(
        retrieval,
        RetentionConfig(
            storage_budget=storage_budget,
            max_proposals=12,
            eviction_shortlist_size=12,
            max_eviction_group=4,
            relaxation_steps=5,
        ),
    )
    singleton_retainer = SingletonProposalRetention(
        retrieval,
        RetentionConfig(
            storage_budget=storage_budget,
            max_proposals=12,
            eviction_shortlist_size=12,
            max_eviction_group=4,
            relaxation_steps=5,
        ),
    )
    uniform_retainer = GroupAwareRetention(
        retrieval,
        RetentionConfig(
            storage_budget=storage_budget,
            max_proposals=12,
            eviction_shortlist_size=12,
            max_eviction_group=4,
            relaxation_steps=5,
        ),
    )
    methods = ("group_aware", "item_lru", "independent_score", "singleton_swaps", "uniform_forecast")
    banks = {method: frozenset() for method in methods}
    last_seen: dict[str, int] = {}
    seen: set[str] = set()
    history: list[Request] = []
    rows: list[dict[str, Any]] = []
    stream = [task for _cycle in range(2) for task in tasks for _repeat in range(3)]

    for event, task in enumerate(stream):
        request = task_request(task)
        new_ids = frozenset(task["gold_memory_ids"])
        seen.update(new_ids)
        for key in new_ids:
            last_seen[key] = event
        recent_workload = tuple((history + [request])[-5:])
        unique_history = {item.request_id: item for item in history + [request]}
        uniform_workload = tuple(unique_history[key] for key in sorted(unique_history)[:5])

        for method in methods:
            before = banks[method]
            started = time.perf_counter()
            pre = retrieval.retrieve(request, before, graph)
            if method == "group_aware":
                after = group_retainer.optimize(before, new_ids, recent_workload, graph).retained_ids
            elif method == "singleton_swaps":
                after = singleton_retainer.optimize(before, new_ids, recent_workload, graph).retained_ids
            elif method == "uniform_forecast":
                after = uniform_retainer.optimize(before, new_ids, uniform_workload, graph).retained_ids
            elif method == "item_lru":
                after = greedy_item_bank(seen, {key: float(last_seen[key]) for key in seen}, storage_budget, graph)
            else:
                scores = {}
                for key in seen:
                    utility = mean(
                        retrieval.model.probability(workload_request, {key}, graph)
                        for workload_request in recent_workload
                    )
                    scores[key] = utility / graph.records[key].storage_cost
                after = greedy_item_bank(seen, scores, storage_budget, graph)
            update_ms = (time.perf_counter() - started) * 1000.0
            banks[method] = after
            post_complete = mean(
                float(retrieval.retrieve(workload_request, after, graph).status == "ok")
                for workload_request in recent_workload
            )
            rows.append(
                {
                    "event": event,
                    "task_id": task["task_id"],
                    "method": method,
                    "prequential_ecr_proxy": int(pre.status == "ok"),
                    "recent_complete_group_availability": post_complete,
                    "bank_storage": sum(graph.records[key].storage_cost for key in after),
                    "bank_records": len(after),
                    "churn_records": len(before.symmetric_difference(after)),
                    "update_latency_ms": update_ms,
                }
            )
        history.append(request)
    write_csv(output / "raw" / "retention_controlled_proxy.csv", rows)
    warmup = len(tasks) * 3
    summary = []
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        post_warmup = [row for row in subset if int(row["event"]) >= warmup]
        summary.append(
            {
                "method": method,
                "events": len(subset),
                "prequential_ecr_proxy": mean(float(row["prequential_ecr_proxy"]) for row in subset),
                "post_warmup_ecr_proxy": mean(float(row["prequential_ecr_proxy"]) for row in post_warmup),
                "complete_group_availability": mean(
                    float(row["recent_complete_group_availability"]) for row in subset
                ),
                "mean_bank_occupancy": mean(float(row["bank_storage"]) / storage_budget for row in subset),
                "mean_churn_records": mean(float(row["churn_records"]) for row in subset),
                "mean_update_latency_ms": mean(float(row["update_latency_ms"]) for row in subset),
            }
        )
    log.append(
        f"Controlled retention proxy: {len(tasks)} two-record exact groups, {len(stream)} synthetic-order events, "
        f"Bm={storage_budget}; no chronological claim."
    )
    return {
        "summary": summary,
        "rows": rows,
        "task_groups": len(tasks),
        "events": len(stream),
        "storage_budget": storage_budget,
    }


def build_inventory_table(data: dict[str, Any]) -> list[dict[str, Any]]:
    quality = data["quality"]
    rows = []
    task_map = quality["task_counts_by_source"]
    memory_map = quality["memory_counts_by_source"]
    source_pairs = (
        ("rapidapi", "rapidapi_selection", "rapidapi_processed"),
        ("emp", "emp_synthetic", "emp_synthetic"),
        ("five_domain", "five_domain_weak_gold", "five_domain_weak_gold"),
        ("hsc", "hsc_optimizer_gold", "hsc_optimizer_gold"),
    )
    for label, task_key, memory_key in source_pairs:
        rows.append(
            {
                "source": label,
                "tasks": task_map[task_key],
                "memories": memory_map[memory_key],
                "observed_execution_outcomes": len(data["outcomes"]) if label == "rapidapi" else 0,
                "exact_admissible_hyperedges": 117 if label == "emp" else 0,
                "role": "unary outcome" if label == "rapidapi" else "weak proxy only",
            }
        )
    return rows


def write_analysis_documents(
    output: Path, data: dict[str, Any], endpoint: dict[str, Any], retrieval: dict[str, Any], retention: dict[str, Any]
) -> None:
    json_dump(
        output / "artifacts" / "data_contract.json",
        {
            "unit_observed_outcome": "one RapidAPI endpoint feedback event",
            "unit_retrieval_proxy": "one non-executed EMP composition task x distractor seed",
            "observed_label": "singleton endpoint test success/failure",
            "proxy_labels": ["synthetic composition", "LLM/fallback selection", "QoS optimizer solution"],
            "forbidden_conversion": "proxy labels must not enter execution-outcome BCE or VSR",
            "splits": "source-provided group split; event_time is unavailable",
            "primary_supported_estimand": "out-of-split predictive performance for singleton endpoint outcomes",
            "unsupported_estimands": ["verified composed-workflow success", "causal memory contribution", "natural drift response"],
        },
    )
    split_counts = Counter(row["split"] for row in data["outcomes"])
    positive_counts = Counter()
    for row in data["outcomes"]:
        positive_counts[row["split"]] += int(row["outcome"])
    json_dump(
        output / "artifacts" / "sample_construction.json",
        {
            "observed_outcomes": {
                split: {
                    "n": split_counts[split],
                    "positive": positive_counts[split],
                    "negative": split_counts[split] - positive_counts[split],
                }
                for split in ("train", "val", "test")
            },
            "exact_emp_tasks": Counter(task["split"] for task in data["tasks"].values()),
            "retrieval_test_tasks": len({row["task_id"] for row in retrieval["main_rows"]}),
            "retrieval_seeds": list(SEEDS),
            "retention_task_groups": retention["task_groups"],
            "retention_events": retention["events"],
            "exclusions": [
                "declared synthetic and heterogeneous hyperedges excluded from default evidence graph",
                "weak composition labels excluded from outcome training",
                "records without trustworthy timestamps excluded from chronological claims",
            ],
        },
    )
    plan = """# Analysis Plan

## SkillOpt Route Card

- **best_skill:** Python full empirical analysis, adapted to algorithmic predictive evaluation.
- **train_signal:** observed singleton RapidAPI endpoint outcomes only.
- **selection_split:** validation split selects probability thresholds and beam width.
- **heldout_gate:** test metrics are computed only after the corresponding choices are frozen.
- **accepted_patterns:** calibration, retrieval reconstruction, matched ablations, budget/efficiency profiling, bounded exhaustive reference, controlled retention proxy.
- **rejected_patterns:** causal estimators, verified workflow success, stale-use/drift claims, and BCE training on weak composition labels.
- **patch_scope:** experiment runner, generated artifacts, cache-correctness fix, and documentation.
- **reject_if:** weak labels enter outcome BCE; gold operation IDs enter query features; proxy ECR is renamed VSR; unsupported external baselines receive fabricated values.

## Identification Boundary

This is predictive and algorithmic evaluation, not causal inference. The dataset has no trustworthy event timestamps or verified multi-operation execution outcomes. Consequently, the run estimates singleton predictive metrics and static/controlled proxy metrics only.
"""
    (output / "artifacts" / "analysis_plan.md").write_text(plan, encoding="utf-8")

    skipped = """# Experiments Not Run

The following paper-protocol experiments cannot be validly run from the current repository and private dataset:

- **Verified Success Rate (VSR), Binding Satisfaction (BS), and runtime failure analysis:** no executable multi-service composer/checker or independently verified composition outcomes are present.
- **Stale Use Rate (SUR), natural drift, and chronological prequential replay:** source event times and version snapshots are missing.
- **MAGMA, ACGM, HyperMem/HGMem, and AgeMem comparisons:** no pinned implementations, compatible environments, or matched candidate exports are vendored.
- **Learned high-order interaction calibration:** the only observed outcomes are singletons; weak composition selections are deliberately excluded from BCE.
- **Five stochastic execution runs:** the local algorithms are deterministic; five seeds vary only controlled distractor sampling in the proxy retrieval benchmark.

These omissions are design limitations, not zero-valued results. The generated tables never fill their cells with fabricated observations.
"""
    (output / "artifacts" / "skipped_experiments.md").write_text(skipped, encoding="utf-8")


def create_tables_and_figures(
    output: Path,
    inventory: list[dict[str, Any]],
    endpoint: dict[str, Any],
    retrieval: dict[str, Any],
    retention: dict[str, Any],
) -> list[dict[str, Any]]:
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    write_table_bundle(tables, "table1_data_inventory", inventory, caption="Private dataset inventory and evidence scope")
    endpoint_test = [row for row in endpoint["summary_rows"] if row["split"] == "test"]
    write_table_bundle(
        tables,
        "table2_endpoint_prediction",
        endpoint_test,
        columns=("method", "n", "positive_rate", "threshold", "balanced_accuracy", "roc_auc", "brier", "ece_10", "log_loss"),
        caption="Held-out singleton endpoint prediction (not composition success)",
    )
    write_table_bundle(
        tables,
        "table3_retrieval_proxy",
        retrieval["summary"],
        columns=("method", "tasks", "task_seed_n", "ecr_proxy", "ecr_ci_low", "ecr_ci_high", "gold_recall", "exact_set_match", "p95_latency_ms", "mean_tokens"),
        caption="Exact-EMP gold-bank reconstruction proxy on the held-out split",
    )
    write_table_bundle(
        tables,
        "table4_component_ablations",
        [row for row in retrieval["summary"] if row["method"] != "semantic_topk"],
        columns=("method", "ecr_proxy", "binding_coverage_proxy", "gold_recall", "fallback_rate", "mean_evaluations", "p95_latency_ms"),
        caption="Encoder-matched retrieval ablations; proxy metrics only",
    )
    write_table_bundle(
        tables,
        "table5_budget_robustness",
        retrieval["budget_summary"],
        caption="Retrieval budget sensitivity on exact-EMP proxy tasks",
    )
    write_table_bundle(
        tables,
        "table6_efficiency",
        retrieval["efficiency_summary"],
        caption="Beam width and candidate-bank efficiency grid",
    )
    write_table_bundle(
        tables,
        "table7_retention_proxy",
        retention["summary"],
        caption="Controlled synthetic-order retention proxy; not longitudinal evidence",
    )
    write_table_bundle(
        tables,
        "table8_paired_tests",
        retrieval["significance"],
        caption="Paired task-level randomization tests with Holm correction",
    )

    source_labels = [row["source"] for row in inventory]
    write_bar_figure(
        figures / "fig1_dataset_tasks.tex",
        title="Private dataset task inventory",
        ylabel="Tasks",
        labels=source_labels,
        values=[float(row["tasks"]) for row in inventory],
    )
    bins = endpoint["calibration_bins"]
    write_line_figure(
        figures / "fig2_endpoint_calibration.tex",
        title="Singleton endpoint calibration (held-out test)",
        xlabel="Mean predicted probability",
        ylabel="Observed success rate",
        series=(("Trained-sanitized", [(row["mean_probability"], row["observed_rate"]) for row in bins]),),
        unit_square=True,
    )
    method_order = [
        "full_corem_proxy", "semantic_topk", "additive_utility", "no_complement",
        "no_redundancy", "item_only_search", "no_evidence_closure",
    ]
    by_method = {row["method"]: row for row in retrieval["summary"]}
    abbreviations = {
        "full_corem_proxy": "Full", "semantic_topk": "Top-k", "additive_utility": "Additive",
        "no_complement": "NoComp", "no_redundancy": "NoRed", "item_only_search": "ItemOnly",
        "no_evidence_closure": "NoClosure",
    }
    write_bar_figure(
        figures / "fig3_retrieval_proxy.tex",
        title="Exact-EMP reconstruction proxy",
        ylabel="Evidence-complete recall",
        labels=[abbreviations[key] for key in method_order],
        values=[float(by_method[key]["ecr_proxy"]) for key in method_order],
        ymax=1.0,
    )
    write_line_figure(
        figures / "fig4_budget_sensitivity.tex",
        title="Retrieval budget sensitivity",
        xlabel="Retrieval budget (tokens)",
        ylabel="Proxy ECR",
        series=(("Full", [(float(row["budget"]), float(row["ecr_proxy_all"])) for row in retrieval["budget_summary"]]),),
    )
    retention_points = []
    for method in ("group_aware", "item_lru", "independent_score"):
        subset = [row for row in retention["rows"] if row["method"] == method]
        cumulative = []
        successes = 0.0
        for index, row in enumerate(subset, start=1):
            successes += float(row["prequential_ecr_proxy"])
            cumulative.append((float(index), successes / index))
        retention_points.append((method, cumulative))
    write_line_figure(
        figures / "fig5_retention_proxy.tex",
        title="Controlled retention proxy trajectory",
        xlabel="Synthetic-order event",
        ylabel="Cumulative proxy ECR",
        series=retention_points,
    )
    return compile_figures(figures)


def write_run_summary(
    output: Path,
    data: dict[str, Any],
    endpoint: dict[str, Any],
    retrieval: dict[str, Any],
    retention: dict[str, Any],
    figure_status: list[dict[str, Any]],
    log: list[str],
    started_at: datetime,
) -> None:
    endpoint_test = next(
        row for row in endpoint["summary_rows"]
        if row["method"] == "trained_sanitized" and row["split"] == "test"
    )
    full_retrieval = next(row for row in retrieval["summary"] if row["method"] == "full_corem_proxy")
    result = {
        "status": "complete_with_design_limited_omissions",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "seeds": list(SEEDS),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "randomization_draws": RANDOMIZATION_DRAWS,
        "dataset_manifest_sha256": sha256_file(Path(data["manifest_path"])),
        "code_sha256": {
            "runner": sha256_file(Path(__file__)),
            "helpers": sha256_file(Path(__file__).with_name("common.py")),
            "types": sha256_file(REPOSITORY_ROOT / "corem" / "types.py"),
            "encoding": sha256_file(REPOSITORY_ROOT / "corem" / "encoding.py"),
            "model": sha256_file(REPOSITORY_ROOT / "corem" / "model.py"),
            "retrieval": sha256_file(REPOSITORY_ROOT / "corem" / "retrieval.py"),
            "retention": sha256_file(REPOSITORY_ROOT / "corem" / "retention.py"),
        },
        "headline_supported_result": {
            "estimand": "singleton endpoint outcome prediction",
            "test_n": int(endpoint_test["n"]),
            "roc_auc": endpoint_test["roc_auc"],
            "brier": endpoint_test["brier"],
            "brier_cluster_bootstrap_ci95": endpoint["brier_ci"],
            "ece_10": endpoint_test["ece_10"],
        },
        "retrieval_proxy": {
            "test_tasks": full_retrieval["tasks"],
            "task_seed_n": full_retrieval["task_seed_n"],
            "ecr": full_retrieval["ecr_proxy"],
            "ci95": [full_retrieval["ecr_ci_low"], full_retrieval["ecr_ci_high"]],
            "beam_width_selected_on_validation": retrieval["selected_beam_width"],
            "exhaustive_reference": retrieval["exhaustive_summary"],
            "warning": "gold-bank reconstruction with task-specific evidence atoms; not VSR",
        },
        "retention_proxy": {
            "task_groups": retention["task_groups"],
            "events": retention["events"],
            "storage_budget": retention["storage_budget"],
            "warning": "controlled synthetic order; not natural longitudinal evidence",
        },
        "figure_status": figure_status,
        "elapsed_seconds": (datetime.now(timezone.utc) - started_at).total_seconds(),
        "analysis_plan": "artifacts/analysis_plan.md",
        "data_contract": "artifacts/data_contract.json",
        "sample_construction": "artifacts/sample_construction.json",
        "skipped_experiments": "artifacts/skipped_experiments.md",
    }
    json_dump(output / "artifacts" / "result.json", result)
    (output / "artifacts" / "diagnostic.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    readme = f"""# CoReM Experiment Results

Run completed at `{result['completed_at_utc']}` using Python `{result['python_version']}`.

The supported held-out result is singleton endpoint prediction: test N={int(endpoint_test['n'])}, ROC-AUC={endpoint_test['roc_auc']:.4f}, Brier={endpoint_test['brier']:.4f}, ECE={endpoint_test['ece_10']:.4f}. The composition retrieval and retention numbers are explicitly proxy diagnostics because this dataset does not contain verified composed-workflow executions or trustworthy timestamps.

- `tables/`: CSV, Markdown, and LaTeX tables.
- `figures/`: generated TeX, PDF, and 300-dpi PNG figures.
- `raw/`: task-level predictions, proxy retrieval runs, sensitivity grids, and retention traces.
- `models/`: learned unary model weights and fit metadata.
- `artifacts/`: analysis plan, data contract, sample log, diagnostic log, omissions, and reproducibility stamp.

Do not copy proxy ECR into the paper's VSR cells. Read `artifacts/skipped_experiments.md` before interpreting results.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def write_checksums(output: Path) -> None:
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    lines = [f"{sha256_file(path)}  {path.relative_to(output)}" for path in files]
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPOSITORY_ROOT / "data" / "private_corem")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "experiments" / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for child in ("artifacts", "figures", "models", "raw", "tables"):
        (output / child).mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    log = [f"Started UTC: {started_at.isoformat()}", f"Dataset: {dataset}", f"Output: {output}"]
    print("[1/6] Loading and validating experiment inputs...", flush=True)
    data = prepare_data(dataset)
    data["manifest_path"] = str(dataset / "manifest.json")
    log.append("Dataset weak/observed label separation invariant: true (builder quality report).")
    log.append("Causal identification: unsupported and not attempted.")

    print("[2/6] Training and evaluating observed singleton endpoint predictor...", flush=True)
    endpoint = run_endpoint_prediction(data, output, log)
    print("[3/6] Running exact-EMP retrieval proxy, ablations, budgets, and exhaustive reference...", flush=True)
    retrieval = run_retrieval_experiments(data, endpoint["model"], output, log)
    print("[4/6] Running controlled retention proxy...", flush=True)
    retention = run_retention_proxy(
        data, endpoint["model"], retrieval["selected_beam_width"], output, log
    )
    inventory = build_inventory_table(data)
    write_analysis_documents(output, data, endpoint, retrieval, retention)
    print("[5/6] Exporting publication-oriented tables and figures...", flush=True)
    figure_status = create_tables_and_figures(output, inventory, endpoint, retrieval, retention)
    log.extend(f"Figure {row['figure']}: PDF exit={row['pdf_exit_code']}, PNG exit={row['png_exit_code']}." for row in figure_status)
    write_run_summary(output, data, endpoint, retrieval, retention, figure_status, log, started_at)
    write_checksums(output)
    print("[6/6] Complete.", flush=True)
    print(json.dumps(json.loads((output / "artifacts" / "result.json").read_text(encoding="utf-8")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

