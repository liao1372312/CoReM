#!/usr/bin/env python3
"""Run auditable, budget-matched adaptations of external memory baselines.

These are method-inspired adapters over the public tau3 memory bank, not native
reproductions of the released systems.  Hyperparameters are selected on the
validation split and the emitted test selections receive no outcome labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.analyze_tau3_paper_suite import (  # noqa: E402
    item_scores,
    load_jsonl,
)


SEEDS = (17, 29, 41, 53, 67)
FEATURE_NAMES = (
    "bm25_mean",
    "dense_mean",
    "hybrid_mean",
    "bm25_gain",
    "dense_gain",
    "hybrid_gain",
    "size_fraction",
    "token_fraction",
    "operation_coverage",
    "declared_coverage",
    "declared_complete",
    "duplicate_collision",
    "provenance_fraction",
    "version_fraction",
    "operation_prior_mean",
    "controlled_fraction",
    "domain_airline",
    "domain_retail",
    "domain_telecom",
)
ACGM_FEATURE_NAMES = tuple(
    name
    for name in FEATURE_NAMES
    if name not in {"declared_coverage", "declared_complete"}
)
OFFICIAL_SOURCES = {
    "MAGMA": {
        "paper": "https://aclanthology.org/2026.acl-long.1709/",
        "code": "https://github.com/FredJiang0324/MAGMA",
    },
    "ACGM": {
        "paper": "https://arxiv.org/abs/2604.07863",
        "code": "https://github.com/S-Forouzandeh/ACGM-Agentic-Web----SIGIR-2026",
    },
    "HyperMem": {
        "paper": "https://aclanthology.org/2026.acl-long.1627/",
        "code": "https://github.com/EverMind-AI/HyperMem",
    },
    "HGMem": {
        "paper": "https://proceedings.mlr.press/v306/zhou26bk.html",
        "code": "https://github.com/Encyclomen/HGMem",
    },
    "AgeMem": {
        "paper": "https://aclanthology.org/2026.acl-long.981/",
        "code": "https://github.com/y1y5/AgeMem",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/tau3_corem_counterexamples")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results_tau3_external_baselines"),
    )
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if math.isclose(low, high):
        return {key: 1.0 if high > 0.0 else 0.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def operation_priors(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    successes: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    global_rate = statistics.mean(float(row["outcome"]) for row in rows)
    for row in rows:
        for operation_id in set(row["selected_operation_ids"]):
            successes[operation_id] += float(row["outcome"])
            counts[operation_id] += 1
    return {
        operation_id: (successes[operation_id] + 4.0 * global_rate)
        / (counts[operation_id] + 4.0)
        for operation_id in counts
    }


@dataclass(frozen=True)
class TaskContext:
    task: dict[str, Any]
    bank: tuple[str, ...]
    memories: dict[str, dict[str, Any]]
    item_values: dict[str, dict[str, float]]
    declared: frozenset[str]
    duplicate_groups: tuple[frozenset[str], ...]
    operation_priors: dict[str, float]
    bank_operations: frozenset[str]

    def cost(self, selected: Iterable[str]) -> int:
        return sum(int(self.memories[key]["token_cost"]) for key in selected)

    def feature_dict(self, selected: Sequence[str]) -> dict[str, float]:
        chosen = frozenset(selected)
        ordered = tuple(sorted(chosen))
        count = len(ordered)
        if count:
            means = {
                method: statistics.mean(scores[key] for key in ordered)
                for method, scores in self.item_values.items()
            }
            gains = {
                method: math.fsum(scores[key] for key in ordered) / math.sqrt(count)
                for method, scores in self.item_values.items()
            }
        else:
            means = {method: 0.0 for method in self.item_values}
            gains = {method: 0.0 for method in self.item_values}
        selected_operations = {
            operation_id
            for key in ordered
            for operation_id in self.memories[key]["operation_ids"]
        }
        declared_coverage = (
            len(chosen.intersection(self.declared)) / len(self.declared)
            if self.declared
            else 0.0
        )
        relevant_duplicate_groups = [
            group for group in self.duplicate_groups if group.intersection(self.bank)
        ]
        duplicate_collision = (
            sum(float(len(group.intersection(chosen)) > 1) for group in relevant_duplicate_groups)
            / len(relevant_duplicate_groups)
            if relevant_duplicate_groups
            else 0.0
        )
        active_versions = frozenset(self.task.get("active_versions", []))
        version_fraction = (
            statistics.mean(
                float(
                    not self.memories[key].get("version_scope")
                    or frozenset(self.memories[key]["version_scope"]).issubset(active_versions)
                )
                for key in ordered
            )
            if ordered
            else 0.0
        )
        prior_values = [
            self.operation_priors.get(operation_id, 0.5)
            for operation_id in sorted(selected_operations)
        ]
        domain = str(self.task["domain"])
        return {
            "bm25_mean": means["bm25"],
            "dense_mean": means["dense_hash"],
            "hybrid_mean": means["hybrid_rrf"],
            "bm25_gain": gains["bm25"],
            "dense_gain": gains["dense_hash"],
            "hybrid_gain": gains["hybrid_rrf"],
            "size_fraction": count / max(1, len(self.bank)),
            "token_fraction": self.cost(chosen) / 1024.0,
            "operation_coverage": len(selected_operations)
            / max(1, len(self.bank_operations)),
            "declared_coverage": declared_coverage,
            "declared_complete": float(bool(self.declared) and self.declared.issubset(chosen)),
            "duplicate_collision": duplicate_collision,
            "provenance_fraction": (
                statistics.mean(
                    float(self.memories[key].get("provenance_verified", False))
                    for key in ordered
                )
                if ordered
                else 0.0
            ),
            "version_fraction": version_fraction,
            "operation_prior_mean": (
                statistics.mean(prior_values) if prior_values else 0.5
            ),
            "controlled_fraction": (
                statistics.mean(float("controlled-extra" in key) for key in ordered)
                if ordered
                else 0.0
            ),
            "domain_airline": float(domain == "airline"),
            "domain_retail": float(domain == "retail"),
            "domain_telecom": float(domain == "telecom"),
        }

    def vector(
        self, selected: Sequence[str], names: Sequence[str] = FEATURE_NAMES
    ) -> list[float]:
        features = self.feature_dict(selected)
        return [features[name] for name in names]


def build_contexts(
    tasks: dict[str, dict[str, Any]],
    outcomes: Sequence[dict[str, Any]],
    memories: dict[str, dict[str, Any]],
    declared: dict[str, frozenset[str]],
    duplicate_groups: Sequence[frozenset[str]],
    priors: dict[str, float],
) -> dict[str, TaskContext]:
    banks: dict[str, set[str]] = defaultdict(set)
    for row in outcomes:
        banks[row["task_id"]].update(row["selected_memory_ids"])
    contexts = {}
    for task_id, bank_set in banks.items():
        bank = tuple(sorted(bank_set))
        raw = item_scores(tasks[task_id], bank, memories)
        normalized = {method: minmax(values) for method, values in raw.items()}
        contexts[task_id] = TaskContext(
            task=tasks[task_id],
            bank=bank,
            memories=memories,
            item_values=normalized,
            declared=declared.get(task_id, frozenset()),
            duplicate_groups=tuple(
                group for group in duplicate_groups if group.intersection(bank_set)
            ),
            operation_priors=priors,
            bank_operations=frozenset(
                operation_id
                for key in bank
                for operation_id in memories[key]["operation_ids"]
            ),
        )
    return contexts


def score_from_features(
    method: str, features: dict[str, float], parameters: dict[str, float]
) -> float:
    cost = features["token_fraction"]
    if method == "graph_traversal":
        return (
            features["hybrid_gain"]
            + parameters["graph"]
            * (features["declared_coverage"] + features["declared_complete"])
            - parameters["cost"] * cost
        )
    if method == "magma_adapted":
        semantic = 0.5 * (features["bm25_gain"] + features["dense_gain"])
        causal = features["declared_coverage"] + features["declared_complete"]
        return (
            semantic
            + parameters["causal"] * causal
            + parameters["entity"] * features["operation_coverage"]
            + 0.1 * features["version_fraction"]
            - parameters["cost"] * cost
        )
    if method == "hypermem_adapted":
        propagated = features["declared_coverage"] ** 2 + features["declared_complete"]
        return (
            features["hybrid_gain"]
            + parameters["propagation"] * propagated
            + parameters["hierarchy"] * features["operation_coverage"]
            - parameters["cost"] * cost
        )
    if method == "hgmem_adapted":
        high_order = features["declared_coverage"] ** 2 + 2.0 * features["declared_complete"]
        return (
            features["hybrid_gain"]
            + parameters["hyperedge"] * high_order
            + parameters["relation"] * features["operation_coverage"]
            - parameters["duplicate"] * features["duplicate_collision"]
            - parameters["cost"] * cost
        )
    if method == "agemem_adapted":
        return (
            parameters["relevance"] * features["hybrid_gain"]
            + parameters["prior"] * features["operation_prior_mean"]
            + 0.25 * features["operation_coverage"]
            - parameters["filter"] * features["controlled_fraction"]
            - parameters["cost"] * cost
        )
    raise KeyError(method)


def parameter_grid(method: str) -> list[dict[str, float]]:
    if method == "graph_traversal":
        return [
            {"graph": graph, "cost": cost}
            for graph in (0.25, 0.5, 1.0, 2.0)
            for cost in (0.0, 0.1, 0.25)
        ]
    if method == "magma_adapted":
        return [
            {"causal": causal, "entity": entity, "cost": cost}
            for causal in (0.5, 1.0, 2.0)
            for entity in (0.25, 0.5, 1.0)
            for cost in (0.0, 0.1)
        ]
    if method == "hypermem_adapted":
        return [
            {"propagation": propagation, "hierarchy": hierarchy, "cost": cost}
            for propagation in (0.5, 1.0, 2.0)
            for hierarchy in (0.25, 0.5, 1.0)
            for cost in (0.0, 0.1)
        ]
    if method == "hgmem_adapted":
        return [
            {
                "hyperedge": hyperedge,
                "relation": relation,
                "duplicate": duplicate,
                "cost": 0.1,
            }
            for hyperedge in (0.5, 1.0, 2.0)
            for relation in (0.25, 0.5)
            for duplicate in (0.5, 1.0)
        ]
    if method == "agemem_adapted":
        return [
            {
                "relevance": relevance,
                "prior": prior,
                "filter": filtering,
                "cost": cost,
            }
            for relevance in (0.5, 1.0)
            for prior in (0.5, 1.0, 2.0)
            for filtering in (0.5, 1.0)
            for cost in (0.1, 0.25)
        ]
    raise KeyError(method)


def validation_vsr(
    rows: Sequence[dict[str, Any]],
    contexts: dict[str, TaskContext],
    budget: int,
    scorer: Callable[[TaskContext, Sequence[str]], float],
) -> float:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)
    successes = []
    for task_id in sorted(grouped):
        context = contexts[task_id]
        feasible = [
            row
            for row in grouped[task_id]
            if context.cost(row["selected_memory_ids"]) <= budget
        ]
        if not feasible:
            successes.append(0.0)
            continue
        selected = max(
            feasible,
            key=lambda row: (
                scorer(context, row["selected_memory_ids"]),
                -context.cost(row["selected_memory_ids"]),
                row["example_id"],
            ),
        )
        successes.append(float(selected["outcome"]))
    return statistics.mean(successes)


def tune_symbolic_methods(
    validation: Sequence[dict[str, Any]],
    contexts: dict[str, TaskContext],
    budget: int,
) -> tuple[dict[str, dict[str, float]], dict[str, list[dict[str, Any]]]]:
    selected: dict[str, dict[str, float]] = {}
    traces: dict[str, list[dict[str, Any]]] = {}
    for method in (
        "graph_traversal",
        "magma_adapted",
        "hypermem_adapted",
        "hgmem_adapted",
        "agemem_adapted",
    ):
        candidates = []
        for parameters in parameter_grid(method):
            score = validation_vsr(
                validation,
                contexts,
                budget,
                lambda context, ids, m=method, p=parameters: score_from_features(
                    m, context.feature_dict(ids), p
                ),
            )
            candidates.append({"parameters": parameters, "validation_vsr": score})
        candidates.sort(
            key=lambda row: (
                float(row["validation_vsr"]),
                json.dumps(row["parameters"], sort_keys=True),
            ),
            reverse=True,
        )
        selected[method] = dict(candidates[0]["parameters"])
        traces[method] = candidates
    return selected, traces


def acgm_vectors(
    rows: Sequence[dict[str, Any]], contexts: dict[str, TaskContext]
) -> list[list[float]]:
    return [
        contexts[row["task_id"]].vector(row["selected_memory_ids"], ACGM_FEATURE_NAMES)
        for row in rows
    ]


def make_mlp(seed: int, alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(16, 8),
                    activation="relu",
                    solver="lbfgs",
                    alpha=alpha,
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def fit_acgm_adapter(
    seed: int,
    train: Sequence[dict[str, Any]],
    validation: Sequence[dict[str, Any]],
    contexts: dict[str, TaskContext],
    budget: int,
) -> tuple[Pipeline, dict[str, Any]]:
    train_x = acgm_vectors(train, contexts)
    train_y = [int(row["outcome"]) for row in train]
    candidates = []
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("ignore", ConvergenceWarning)
        for alpha in (0.001, 0.01, 0.1, 1.0):
            model = make_mlp(seed, alpha)
            model.fit(train_x, train_y)
            probabilities = model.predict_proba(acgm_vectors(validation, contexts))[:, 1]
            probability_by_id = {
                row["example_id"]: float(probability)
                for row, probability in zip(validation, probabilities)
            }
            score = validation_vsr(
                validation,
                contexts,
                budget,
                lambda _context, ids, lookup=probability_by_id: lookup[
                    next(
                        row["example_id"]
                        for row in validation
                        if row["task_id"] == _context.task["task_id"]
                        and row["selected_memory_ids"] == list(ids)
                    )
                ],
            )
            candidates.append({"alpha": alpha, "validation_vsr": score})
        candidates.sort(key=lambda row: (row["validation_vsr"], -row["alpha"]), reverse=True)
        selected_alpha = float(candidates[0]["alpha"])
        final_model = make_mlp(seed, selected_alpha)
        combined = list(train) + list(validation)
        final_model.fit(
            acgm_vectors(combined, contexts),
            [int(row["outcome"]) for row in combined],
        )
    return final_model, {
        "selected_alpha": selected_alpha,
        "validation_vsr": candidates[0]["validation_vsr"],
        "candidates": candidates,
        "refit_on_train_plus_validation": True,
    }


def topk_select(
    context: TaskContext, method: str, budget: int
) -> tuple[tuple[str, ...], float, int]:
    ranking = sorted(
        context.bank,
        key=lambda key: (context.item_values[method][key], key),
        reverse=True,
    )
    selected = []
    for key in ranking:
        if context.cost((*selected, key)) <= budget:
            selected.append(key)
    score = sum(context.item_values[method][key] for key in selected)
    return tuple(sorted(selected)), score, len(ranking)


def beam_select(
    context: TaskContext,
    budget: int,
    beam_width: int,
    score_many: Callable[[Sequence[tuple[str, ...]]], Sequence[float]],
) -> tuple[tuple[str, ...], float, int]:
    bank = context.bank
    index = {key: position for position, key in enumerate(bank)}
    beam: list[tuple[str, ...]] = [tuple()]
    best: tuple[str, ...] = tuple()
    best_score = float("-inf")
    evaluations = 0
    for _depth in range(1, len(bank) + 1):
        expanded: set[tuple[str, ...]] = set()
        for selected in beam:
            start = index[selected[-1]] + 1 if selected else 0
            for position in range(start, len(bank)):
                candidate = (*selected, bank[position])
                if context.cost(candidate) <= budget:
                    expanded.add(candidate)
        if not expanded:
            break
        candidates = sorted(expanded)
        scores = [float(value) for value in score_many(candidates)]
        evaluations += len(candidates)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda pair: (pair[1], -context.cost(pair[0]), pair[0]),
            reverse=True,
        )
        beam = [candidate for candidate, _ in ranked[:beam_width]]
        candidate, score = ranked[0]
        if (score, -context.cost(candidate), candidate) > (
            best_score,
            -context.cost(best),
            best,
        ):
            best, best_score = candidate, score
    return best, best_score, evaluations


def selection_row(
    context: TaskContext,
    seed: int,
    method: str,
    selected: Sequence[str],
    score: float,
    evaluations: int,
    latency_ms: float,
    adapter_level: str,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "task_id": context.task["task_id"],
        "source_task_id": context.task["source_task_id"],
        "domain": context.task["domain"],
        "method": method,
        "selected_memory_ids": sorted(selected),
        "status": "selected" if selected else "empty_control",
        "predicted_probability": sigmoid(score) if math.isfinite(score) else 0.0,
        "objective": score if math.isfinite(score) else None,
        "token_cost": context.cost(selected),
        "evaluations": evaluations,
        "latency_ms": latency_ms,
        "adapter_level": adapter_level,
    }


def main() -> None:
    args = parse_args()
    if args.budget <= 0 or args.beam_width <= 0:
        raise ValueError("budget and beam width must be positive")
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    started = time.perf_counter()
    tasks = {
        row["task_id"]: row
        for row in load_jsonl(args.dataset / "retrieval_tasks.jsonl")
    }
    outcomes = load_jsonl(args.dataset / "set_outcomes.jsonl")
    memories = {
        row["memory_id"]: row
        for row in load_jsonl(args.dataset / "memory_records.jsonl")
    }
    edges = load_jsonl(args.dataset / "dependency_hyperedges.jsonl")
    task_edge_ids = {
        task_id: frozenset(task.get("dependency_edge_ids", []))
        for task_id, task in tasks.items()
    }
    declared: dict[str, frozenset[str]] = {}
    for task_id in tasks:
        members = {
            member
            for edge in edges
            if edge["task_id"] == task_id
            and edge["edge_id"] in task_edge_ids[task_id]
            and edge["relation_type"] == "declared_state_transition_group"
            for member in edge["members"]
        }
        declared[task_id] = frozenset(members)
    duplicate_groups = [
        frozenset(row["memory_ids"])
        for row in load_jsonl(args.dataset / "duplicate_groups.jsonl")
    ]
    train = [
        row
        for row in outcomes
        if row["split"] == "train" and row.get("eligible_for_training", False)
    ]
    validation = [
        row
        for row in outcomes
        if row["split"] == "val" and row.get("eligible_for_training", False)
    ]
    priors = operation_priors(train)
    contexts = build_contexts(
        tasks, outcomes, memories, declared, duplicate_groups, priors
    )
    tuned, tuning_traces = tune_symbolic_methods(
        validation, contexts, args.budget
    )

    test_tasks = sorted(
        (task for task in tasks.values() if task["split"] == "test"),
        key=lambda task: task["task_id"],
    )
    runs = []
    acgm_tuning = {}
    for seed in seeds:
        model, model_trace = fit_acgm_adapter(
            seed, train, validation, contexts, args.budget
        )
        acgm_tuning[str(seed)] = model_trace
        selections = []
        for task in test_tasks:
            context = contexts[task["task_id"]]
            for method in (
                "no_memory",
                "bm25_topk",
                "dense_hash_topk",
                "hybrid_rrf_topk",
                "graph_traversal",
                "magma_adapted",
                "acgm_adapted",
                "hypermem_adapted",
                "hgmem_adapted",
                "agemem_adapted",
            ):
                retrieval_started = time.perf_counter()
                if method == "no_memory":
                    selected, score, evaluations = tuple(), float("-inf"), 1
                    level = "retrieval_control"
                elif method in {"bm25_topk", "dense_hash_topk", "hybrid_rrf_topk"}:
                    item_method = {
                        "bm25_topk": "bm25",
                        "dense_hash_topk": "dense_hash",
                        "hybrid_rrf_topk": "hybrid_rrf",
                    }[method]
                    selected, score, evaluations = topk_select(
                        context, item_method, args.budget
                    )
                    level = "retrieval_control"
                elif method == "acgm_adapted":
                    selected, score, evaluations = beam_select(
                        context,
                        args.budget,
                        args.beam_width,
                        lambda candidates, c=context, fitted=model: fitted.predict_proba(
                            [c.vector(candidate, ACGM_FEATURE_NAMES) for candidate in candidates]
                        )[:, 1],
                    )
                    level = "adapted_not_native"
                else:
                    parameters = tuned[method]
                    selected, score, evaluations = beam_select(
                        context,
                        args.budget,
                        args.beam_width,
                        lambda candidates, c=context, m=method, p=parameters: [
                            score_from_features(m, c.feature_dict(candidate), p)
                            for candidate in candidates
                        ],
                    )
                    level = (
                        "retrieval_control"
                        if method == "graph_traversal"
                        else "adapted_not_native"
                    )
                selections.append(
                    selection_row(
                        context,
                        seed,
                        method,
                        selected,
                        score,
                        evaluations,
                        (time.perf_counter() - retrieval_started) * 1000.0,
                        level,
                    )
                )
        runs.append({"seed": seed, "selections": selections})
        print(f"completed adapter selections for seed={seed}", flush=True)

    all_rows = [row for run in runs for row in run["selections"]]
    violations = [row for row in all_rows if int(row["token_cost"]) > args.budget]
    unknown = sorted(
        {
            memory_id
            for row in all_rows
            for memory_id in row["selected_memory_ids"]
            if memory_id not in memories
        }
    )
    if violations or unknown:
        raise RuntimeError(
            f"selection gate failed: {len(violations)} budget violations, "
            f"{len(unknown)} unknown memories"
        )
    result = {
        "status": "selection_complete_execution_pending",
        "scope": "budget-matched tau3 retrieval controls and explicitly adapted external methods",
        "configuration": {
            "seeds": list(seeds),
            "retrieval_budget": args.budget,
            "beam_width": args.beam_width,
            "selection_split": "val",
            "heldout_split": "test",
            "train_outcomes": len(train),
            "validation_outcomes": len(validation),
            "test_tasks": len(test_tasks),
            "gold_memory_ids_used": False,
            "gold_operation_ids_used": False,
            "test_outcome_labels_used_for_selection": False,
            "test_candidate_sets_used_to_define_task_bank": True,
            "outcome_derived_oracle_edges_used": False,
        },
        "tuning": {
            "symbolic": tuned,
            "symbolic_traces": tuning_traces,
            "acgm": acgm_tuning,
        },
        "official_sources": OFFICIAL_SOURCES,
        "adaptation_boundary": {
            "native_reproductions": False,
            "shared_components": [
                "tau3 task-local memory bank",
                "1,024-token retrieval budget",
                "fresh deterministic environment replay",
            ],
            "missing_native_components": {
                "magma_adapted": "native graph construction, learned traversal policy, and original benchmark stack",
                "acgm_adapted": "policy-gradient/EMA training, modality decay, and original multimodal web stack",
                "hypermem_adapted": "native LLM fact extraction, embedding/reranker endpoints, and conversational stack",
                "hgmem_adapted": "native BGE/GPT extraction and multi-step RAG generation stack",
                "agemem_adapted": "Trinity-RFT three-stage RL, Qwen policy, and tool-calling generation stack",
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "runs": runs,
        "validation_gates": {
            "budget_violations": len(violations),
            "unknown_memory_ids": len(unknown),
            "test_task_coverage_per_method_seed": len(test_tasks),
        },
        "limitations": [
            "All external methods are adaptations, not native full-system reproductions.",
            "The task-local bank contains only memories derived from the official reference trajectory plus controlled counterexamples.",
            "tau3 provides no natural timestamps; temporal terms use version compatibility only.",
            "Selections require fresh environment replay before VSR is assigned.",
        ],
        "dataset_manifest_sha256": sha256_file(args.dataset / "manifest.json"),
        "code_sha256": sha256_file(Path(__file__)),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "external_baseline_selections.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output / "checksums.sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
    )
    print(f"wrote {path} with {len(all_rows)} selections")


if __name__ == "__main__":
    main()
