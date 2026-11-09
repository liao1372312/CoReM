#!/usr/bin/env python3
"""Run prequential online retention against the resettable public v5 services."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import statistics
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.model import SetUtilityModel  # noqa: E402
from corem.online_predictor import OnlineLogisticPredictor  # noqa: E402
from corem.real_eval import (  # noqa: E402
    RealTrackError,
    audit_real_dataset,
    load_jsonl,
    parse_utc,
    score_execution_row,
    summarize_scored_rows,
    write_checksums,
    write_json,
    write_jsonl,
)
from corem.retention import GroupAwareRetention, RetentionConfig  # noqa: E402
from corem.retrieval import BundleBeamRetriever, RetrievalConfig  # noqa: E402
from corem.types import EvidenceGraph, Request  # noqa: E402
from experiments.generate_public_challenge_selections import (  # noqa: E402
    GENERIC_REQUIRED_ATOMS,
)
from experiments.generate_public_workflow_selections import (  # noqa: E402
    bm25_scores,
    make_graph,
)
from experiments.public_multiservice_challenge_checker import (  # noqa: E402
    CHECKER_CONTRACT_ID,
    create_checker,
)
from experiments.public_multiservice_challenge_runner import (  # noqa: E402
    create_runner,
)

METHODS = (
    "group_aware_online",
    "item_value_retention",
    "bundle_value_retention",
    "corem_retention",
    "forecast_corem",
    "group_bm25_online",
    "group_bm25_residual_online",
    "independent_relevance_online",
    "lru_online",
    "lfu_online",
    "no_memory",
)


class FrozenHGTBM25Ensemble:
    """Five-seed HGT mean with a fixed, request-local BM25 residual.

    Unlike the sealed two-candidate experiments, online beam search does not
    know its final candidate family in advance.  The lexical term is therefore
    normalized once over all records admissible for the request.  This keeps a
    selected set's utility invariant across beam depths and retention-bank
    comparisons.  The same hybrid utility is exposed to the relaxed retention
    surrogate through a piecewise-linear max-BM25 extension.
    """

    def __init__(self, models: list[tuple[int, Any, Path]], neural_weight: float) -> None:
        if not models:
            raise ValueError("at least one frozen HGT model is required")
        if not 0.0 <= neural_weight <= 1.0:
            raise ValueError("neural weight must be between zero and one")
        self.models = tuple(models)
        self.neural_weight = neural_weight
        self._lexical_cache: dict[
            tuple[Request, int, int], dict[str, float]
        ] = {}
        self._probability_cache: dict[
            tuple[Request, frozenset[str], int, int], float
        ] = {}
        self._semantic_cache: dict[tuple[Request, str, int, int], float] = {}
        self._relaxed_cache: dict[
            tuple[Request, tuple[tuple[str, float], ...], int, int],
            tuple[float, dict[str, float]],
        ] = {}

    def _normalized_lexical(
        self, request: Request, graph: EvidenceGraph
    ) -> dict[str, float]:
        key = (request, id(graph), graph.revision)
        cached = self._lexical_cache.get(key)
        if cached is not None:
            return cached
        compatible = [
            {"memory_id": memory_id, "text": record.text}
            for memory_id, record in graph.records.items()
            if graph.is_admissible(memory_id, request)
        ]
        raw = bm25_scores(request.text, compatible)
        if not raw:
            normalized: dict[str, float] = {}
        else:
            low, high = min(raw.values()), max(raw.values())
            normalized = {
                memory_id: (value - low) / (high - low) if high > low else 0.0
                for memory_id, value in raw.items()
            }
        self._lexical_cache[key] = normalized
        if len(self._lexical_cache) > 1024:
            self._lexical_cache.pop(next(iter(self._lexical_cache)))
        return normalized

    def _lexical_set_value(
        self,
        request: Request,
        selected_ids: frozenset[str],
        graph: EvidenceGraph,
    ) -> float:
        lexical = self._normalized_lexical(request, graph)
        return max((lexical.get(item, 0.0) for item in selected_ids), default=0.0)

    def semantic_similarity(
        self, request: Request, record_id: str, graph: EvidenceGraph
    ) -> float:
        key = (request, record_id, id(graph), graph.revision)
        cached = self._semantic_cache.get(key)
        if cached is not None:
            return cached
        value = statistics.mean(
            model.semantic_similarity(request, record_id, graph)
            for _, model, _ in self.models
        )
        self._semantic_cache[key] = value
        if len(self._semantic_cache) > 32768:
            self._semantic_cache.pop(next(iter(self._semantic_cache)))
        return value

    def probability(
        self, request: Request, selected_ids: frozenset[str], graph: EvidenceGraph
    ) -> float:
        selected = frozenset(selected_ids)
        return self.probabilities(request, [selected], graph)[0]

    def probabilities(
        self,
        request: Request,
        selected_sets: list[frozenset[str]],
        graph: EvidenceGraph,
    ) -> tuple[float, ...]:
        if not selected_sets:
            return ()
        keys = [
            (request, selected, id(graph), graph.revision)
            for selected in selected_sets
        ]
        missing_indices = [
            index for index, key in enumerate(keys) if key not in self._probability_cache
        ]
        if missing_indices:
            missing_sets = [selected_sets[index] for index in missing_indices]
            per_model = [
                model.probabilities(request, missing_sets, graph)
                for _, model, _ in self.models
            ]
            for local_index, global_index in enumerate(missing_indices):
                selected = selected_sets[global_index]
                neural = statistics.mean(
                    values[local_index] for values in per_model
                )
                lexical = self._lexical_set_value(request, selected, graph)
                self._probability_cache[keys[global_index]] = (
                    self.neural_weight * neural
                    + (1.0 - self.neural_weight) * lexical
                )
        while len(self._probability_cache) > 131072:
            self._probability_cache.pop(next(iter(self._probability_cache)))
        return tuple(self._probability_cache[key] for key in keys)

    def relaxed_probability_and_gradient(
        self,
        request: Request,
        inclusion: dict[str, float],
        graph: EvidenceGraph,
    ) -> tuple[float, dict[str, float]]:
        cache_key = (
            request,
            tuple(sorted(inclusion.items())),
            id(graph),
            graph.revision,
        )
        cached = self._relaxed_cache.get(cache_key)
        if cached is not None:
            value, cached_gradient = cached
            return value, dict(cached_gradient)
        model_results = [
            model.relaxed_probability_and_gradient(request, inclusion, graph)
            for _, model, _ in self.models
        ]
        neural = statistics.mean(value for value, _ in model_results)
        gradient = {
            memory_id: self.neural_weight
            * statistics.mean(values[memory_id] for _, values in model_results)
            for memory_id in inclusion
        }
        lexical = self._normalized_lexical(request, graph)
        lexical_value = 0.0
        if inclusion:
            lexical_choice = max(
                inclusion,
                key=lambda memory_id: (
                    inclusion[memory_id] * lexical.get(memory_id, 0.0),
                    memory_id,
                ),
            )
            lexical_value = (
                inclusion[lexical_choice] * lexical.get(lexical_choice, 0.0)
            )
            gradient[lexical_choice] += (
                1.0 - self.neural_weight
            ) * lexical.get(lexical_choice, 0.0)
        result = (
            self.neural_weight * neural
            + (1.0 - self.neural_weight) * lexical_value,
            gradient,
        )
        self._relaxed_cache[cache_key] = (result[0], dict(result[1]))
        while len(self._relaxed_cache) > 131072:
            self._relaxed_cache.pop(next(iter(self._relaxed_cache)))
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-budget", type=int, default=640)
    parser.add_argument("--retrieval-budget", type=int, default=160)
    parser.add_argument(
        "--require-hyperedge-seed",
        action="store_true",
        help=(
            "Restrict retrieval expansion seeds to provenance-attested dependency "
            "hyperedges; useful for evaluating coherent-bundle retrieval."
        ),
    )
    parser.add_argument("--workload-window", type=int, default=16)
    parser.add_argument(
        "--forecast-min-margin",
        type=float,
        default=0.0,
        help="Minimum top-two bundle-score margin for a failed-event forecast label.",
    )
    parser.add_argument(
        "--forecast-min-support",
        type=int,
        default=2,
        help="Minimum observed outgoing transitions before forecast activation.",
    )
    parser.add_argument(
        "--forecast-min-confidence",
        type=float,
        default=0.75,
        help="Minimum empirical next-workflow probability for forecast activation.",
    )
    parser.add_argument(
        "--update-time-limit-ms",
        type=float,
        default=3000.0,
        help="Anytime maintenance deadline applied to matched retention methods.",
    )
    parser.add_argument(
        "--uncapped-maintenance",
        action="store_true",
        help="Use the uncapped E4 maintenance setting instead of the E3 deadline.",
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--online-model",
        type=Path,
        help="Frozen 256-feature logistic predictor fitted on verified development executions.",
    )
    parser.add_argument(
        "--composer-module",
        help="Import path of a fixed composer adapter exposing create_runner().",
    )
    parser.add_argument(
        "--service-catalog",
        type=Path,
        help="Public operation catalog covering every event service and version.",
    )
    parser.add_argument(
        "--model-dirs",
        type=Path,
        nargs="+",
        help=(
            "Development-only legacy comparison using candidate-stage HGT/BM25 "
            "scores; unavailable for reportable online runs."
        ),
    )
    parser.add_argument("--neural-weight", type=float, default=0.9)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--text-hash-dimension", type=int, default=64)
    parser.add_argument("--message-layers", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--lexical-features", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--source-executions",
        type=Path,
        help="Previously executed, independently checked exact event/bundle trajectories.",
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--max-events",
        type=int,
        help="Development-only prefix limit; omitted for a reportable run.",
    )
    return parser.parse_args()


def request_for(event: dict[str, Any]) -> Request:
    timestamp = parse_utc(event["event_time"], "event_time").timestamp()
    return Request(
        request_id=str(event["event_id"]),
        text=str(event["query"]),
        required_atoms=GENERIC_REQUIRED_ATOMS,
        operation_ids=frozenset(),
        timestamp=timestamp,
        allowed_scopes=frozenset((str(event["service_id"]),)),
        active_versions=frozenset((str(event["version_id"]),)),
    )


def active_archive_ids(
    memories: list[dict[str, Any]], timestamp: float
) -> frozenset[str]:
    active = []
    for row in memories:
        start = parse_utc(row["valid_from"], "valid_from").timestamp()
        end = (
            parse_utc(row["valid_until"], "valid_until").timestamp()
            if row.get("valid_until")
            else None
        )
        if start <= timestamp and (end is None or timestamp < end):
            active.append(str(row["memory_id"]))
    return frozenset(active)


def storage_cost(ids: frozenset[str], graph: EvidenceGraph) -> int:
    return sum(graph.records[memory_id].storage_cost for memory_id in ids)


def complete_groups(ids: frozenset[str], graph: EvidenceGraph) -> int:
    return sum(edge.members.issubset(ids) for edge in graph.hyperedges.values())


def _budgeted_ids(
    ranked: list[str], budget: int, graph: EvidenceGraph
) -> frozenset[str]:
    selected: set[str] = set()
    cost = 0
    for memory_id in ranked:
        item_cost = graph.records[memory_id].storage_cost
        if cost + item_cost <= budget:
            selected.add(memory_id)
            cost += item_cost
    return frozenset(selected)


def _bootstrap_groups(
    active: frozenset[str], budget: int, graph: EvidenceGraph
) -> frozenset[str]:
    """Label-blind cold start that admits complete grounded groups only."""

    edges = [
        edge
        for edge in graph.hyperedges.values()
        if edge.members.issubset(active)
    ]
    edges.sort(
        key=lambda edge: (
            -max(graph.records[item].timestamp for item in edge.members),
            edge.edge_id,
        )
    )
    bank: set[str] = set()
    cost = 0
    for edge in edges:
        additions = edge.members.difference(bank)
        added_cost = sum(graph.records[item].storage_cost for item in additions)
        if cost + added_cost <= budget:
            bank.update(additions)
            cost += added_cost
    return frozenset(bank)


@dataclass
class PolicyState:
    method: str
    bank: frozenset[str] = frozenset()
    workload: deque[Request] = field(default_factory=deque)
    access_clock: dict[str, int] = field(default_factory=dict)
    frequencies: Counter[str] = field(default_factory=Counter)
    tick: int = 0
    last_active: frozenset[str] = frozenset()
    last_signature: tuple[str, ...] | None = None
    transition_counts: dict[tuple[str, ...], Counter[tuple[str, ...]]] = field(
        default_factory=dict
    )
    signature_frequencies: Counter[tuple[str, ...]] = field(default_factory=Counter)


def prune_before_event(state: PolicyState, active: frozenset[str]) -> set[str]:
    """Remove expired records without using the current request to refill the bank."""

    evicted = set(state.bank.difference(active))
    state.bank = state.bank.intersection(active)
    return evicted


class OnlineController:
    def __init__(
        self,
        graph: EvidenceGraph,
        retriever: BundleBeamRetriever,
        storage_budget: int,
        workload_window: int,
        update_time_limit_ms: float | None = None,
        forecast_min_margin: float = 0.0,
        forecast_min_support: int = 2,
        forecast_min_confidence: float = 0.75,
    ) -> None:
        self.graph = graph
        self.retriever = retriever
        self.storage_budget = storage_budget
        self.workload_window = workload_window
        self.update_time_limit_ms = update_time_limit_ms
        self.forecast_min_margin = forecast_min_margin
        self.forecast_min_support = forecast_min_support
        self.forecast_min_confidence = forecast_min_confidence
        common_retention = {
            "storage_budget": storage_budget,
            "max_proposals": 64,
            "eviction_shortlist_size": 16,
            "max_eviction_group": 4,
            "proposal_dependency_hops": 2,
            "max_proposal_group": 8,
            "relaxation_steps": 8,
            "activation_cost_weight": 0.02,
            "removal_cost_weight": 0.02,
        }
        self.group_solvers = {
            "group_aware_online": GroupAwareRetention(
                retriever, RetentionConfig(**common_retention)
            ),
            "corem_retention": GroupAwareRetention(
                retriever,
                RetentionConfig(
                    **common_retention,
                    bank_value_objective="retriever",
                    proposal_priority="relaxed",
                ),
            ),
            "bundle_value_retention": GroupAwareRetention(
                retriever,
                RetentionConfig(
                    **common_retention,
                    bank_value_objective="bundle",
                    proposal_priority="relaxed",
                ),
            ),
            "item_value_retention": GroupAwareRetention(
                retriever,
                RetentionConfig(
                    **common_retention,
                    bank_value_objective="item",
                    proposal_priority="relaxed",
                ),
            ),
        }

    def _independent_bank(
        self, state: PolicyState, active: frozenset[str]
    ) -> frozenset[str]:
        scores = {memory_id: 0.0 for memory_id in active}
        workload = list(state.workload)
        for age, request in enumerate(reversed(workload)):
            compatible = [
                self._row(memory_id)
                for memory_id in active
                if self.graph.is_admissible(memory_id, request)
            ]
            lexical = bm25_scores(request.text, compatible)
            weight = 0.90**age
            for memory_id, value in lexical.items():
                scores[memory_id] += weight * value
        ranked = sorted(active, key=lambda item: (-scores[item], item))
        return _budgeted_ids(ranked, self.storage_budget, self.graph)

    def _group_candidate_pool(
        self, state: PolicyState, active: frozenset[str]
    ) -> frozenset[str]:
        """Bound the outer neighborhood using past-query lexical relevance."""

        scores = {memory_id: 0.0 for memory_id in active}
        for age, request in enumerate(reversed(state.workload)):
            compatible = [
                self._row(memory_id)
                for memory_id in active
                if self.graph.is_admissible(memory_id, request)
            ]
            for memory_id, value in bm25_scores(request.text, compatible).items():
                scores[memory_id] += (0.90**age) * value
        edges = [
            edge
            for edge in self.graph.hyperedges.values()
            if edge.members.issubset(active)
        ]
        edges.sort(
            key=lambda edge: (
                -sum(scores[item] for item in edge.members),
                edge.edge_id,
            )
        )
        candidates = set(state.bank.intersection(active))
        for edge in edges[:64]:
            candidates.update(edge.members)
        ranked_items = sorted(active, key=lambda item: (-scores[item], item))
        candidates.update(ranked_items[:64])
        return frozenset(candidates)

    def _group_bm25_bank(
        self,
        state: PolicyState,
        active: frozenset[str],
        *,
        fill_residual: bool = False,
    ) -> frozenset[str]:
        """Retain whole verified bundles using only past-query lexical scores."""
        if not state.workload and not fill_residual:
            return _bootstrap_groups(active, self.storage_budget, self.graph)
        scores = {memory_id: 0.0 for memory_id in active}
        for age, request in enumerate(reversed(state.workload)):
            compatible = [
                self._row(memory_id)
                for memory_id in active
                if self.graph.is_admissible(memory_id, request)
            ]
            for memory_id, value in bm25_scores(request.text, compatible).items():
                scores[memory_id] += (0.90**age) * value
        edges = [
            edge
            for edge in self.graph.hyperedges.values()
            if edge.members.issubset(active)
        ]
        edges.sort(
            key=lambda edge: (
                -sum(scores[item] for item in edge.members),
                -max(self.graph.records[item].timestamp for item in edge.members),
                edge.edge_id,
            )
        )
        bank: set[str] = set()
        cost = 0
        for edge in edges:
            additions = edge.members.difference(bank)
            added_cost = sum(self.graph.records[item].storage_cost for item in additions)
            if cost + added_cost <= self.storage_budget:
                bank.update(additions)
                cost += added_cost
        if fill_residual:
            remaining = sorted(
                active.difference(bank),
                key=lambda item: (
                    -scores[item],
                    -self.graph.records[item].timestamp,
                    item,
                ),
            )
            for item in remaining:
                item_cost = self.graph.records[item].storage_cost
                if cost + item_cost <= self.storage_budget:
                    bank.add(item)
                    cost += item_cost
        return frozenset(bank)

    def _row(self, memory_id: str) -> dict[str, Any]:
        record = self.graph.records[memory_id]
        return {"memory_id": memory_id, "text": record.text}

    def _edge_signature(self, members: frozenset[str]) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    operation
                    for memory_id in members
                    for operation in self.graph.records[memory_id].operation_ids
                }
            )
        )

    def _infer_request_signature(
        self, request: Request, active: frozenset[str]
    ) -> tuple[str, ...] | None:
        """Infer a latent workflow using only the observed request and archive."""
        edges = [
            edge
            for edge in self.graph.hyperedges.values()
            if edge.members.issubset(active)
            and self.graph.is_hyperedge_admissible(edge, request)
        ]
        if not edges:
            return None
        batched = getattr(self.retriever.model, "probabilities", None)
        probabilities = (
            batched(request, [edge.members for edge in edges], self.graph)
            if callable(batched)
            else tuple(
                self.retriever.model.probability(request, edge.members, self.graph)
                for edge in edges
            )
        )
        ranked = sorted(
            zip(edges, probabilities),
            key=lambda item: (item[1], item[0].edge_id),
            reverse=True,
        )
        best, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else float("-inf")
        if best_score - runner_up < self.forecast_min_margin:
            return None
        return self._edge_signature(best.members)

    def _forecast_bank(
        self, state: PolicyState, active: frozenset[str]
    ) -> frozenset[str] | None:
        """Fill the bank from an online first-order latent-workflow forecast."""
        edges_by_signature: dict[tuple[str, ...], list[Any]] = {}
        for edge in self.graph.hyperedges.values():
            if edge.members.issubset(active):
                edges_by_signature.setdefault(
                    self._edge_signature(edge.members), []
                ).append(edge)
        if state.last_signature is None or not edges_by_signature:
            return None

        first_outgoing = state.transition_counts.get(
            state.last_signature, Counter()
        )
        support = sum(first_outgoing.values())
        confidence = max(first_outgoing.values(), default=0) / max(1, support)
        if (
            support < self.forecast_min_support
            or confidence < self.forecast_min_confidence
        ):
            return None

        ranked_signatures: list[tuple[str, ...]] = []
        cursor = state.last_signature
        seen: set[tuple[str, ...]] = set()
        while cursor not in seen:
            seen.add(cursor)
            outgoing = state.transition_counts.get(cursor, Counter())
            eligible = [
                signature
                for signature in outgoing
                if signature in edges_by_signature
                and signature not in ranked_signatures
            ]
            if not eligible:
                break
            cursor = max(
                eligible,
                key=lambda signature: (
                    outgoing[signature],
                    state.signature_frequencies[signature],
                    signature,
                ),
            )
            ranked_signatures.append(cursor)

        remaining = sorted(
            edges_by_signature,
            key=lambda signature: (
                -state.signature_frequencies[signature],
                signature,
            ),
        )
        ranked_signatures.extend(
            signature
            for signature in remaining
            if signature not in ranked_signatures
        )
        bank: set[str] = set()
        cost = 0
        for signature in ranked_signatures:
            edge = max(
                edges_by_signature[signature],
                key=lambda item: (
                    max(self.graph.records[key].timestamp for key in item.members),
                    item.edge_id,
                ),
            )
            additions = edge.members.difference(bank)
            added_cost = storage_cost(additions, self.graph)
            if cost + added_cost <= self.storage_budget:
                bank.update(additions)
                cost += added_cost
        return frozenset(bank)

    def update(
        self, state: PolicyState, active: frozenset[str]
    ) -> tuple[set[str], set[str], int, int, bool]:
        before = state.bank
        current = before.intersection(active)
        evaluated = 0
        accepted = 0
        timed_out = False
        if state.method == "no_memory":
            after = frozenset()
        elif state.method in self.group_solvers:
            if not state.workload:
                after = _bootstrap_groups(active, self.storage_budget, self.graph)
            else:
                neighborhood = self._group_candidate_pool(state, active)
                current = current.intersection(neighborhood)
                result = self.group_solvers[state.method].optimize(
                    current,
                    neighborhood.difference(current),
                    list(state.workload),
                    self.graph,
                    time_limit_ms=self.update_time_limit_ms,
                    workload_weights=tuple(
                        0.90 ** age
                        for age in range(len(state.workload) - 1, -1, -1)
                    ),
                )
                after = result.retained_ids
                evaluated = result.evaluated_banks
                accepted = result.accepted_swaps
                timed_out = result.timed_out
        elif state.method == "independent_relevance_online":
            after = self._independent_bank(state, active)
        elif state.method == "group_bm25_online":
            after = self._group_bm25_bank(state, active)
        elif state.method == "group_bm25_residual_online":
            after = self._group_bm25_bank(state, active, fill_residual=True)
        elif state.method == "forecast_corem":
            forecast = self._forecast_bank(state, active)
            if forecast is not None:
                after = forecast
            elif not state.workload:
                after = _bootstrap_groups(active, self.storage_budget, self.graph)
            else:
                neighborhood = self._group_candidate_pool(state, active)
                current = current.intersection(neighborhood)
                result = self.group_solvers["bundle_value_retention"].optimize(
                    current,
                    neighborhood.difference(current),
                    list(state.workload),
                    self.graph,
                    time_limit_ms=self.update_time_limit_ms,
                    workload_weights=tuple(
                        0.90 ** age
                        for age in range(len(state.workload) - 1, -1, -1)
                    ),
                )
                after = result.retained_ids
                evaluated = result.evaluated_banks
                accepted = result.accepted_swaps
                timed_out = result.timed_out
        else:
            if state.method == "lru_online":
                ranked = sorted(
                    active,
                    key=lambda item: (
                        -state.access_clock.get(item, -1),
                        -self.graph.records[item].timestamp,
                        item,
                    ),
                )
            elif state.method == "lfu_online":
                ranked = sorted(
                    active,
                    key=lambda item: (
                        -state.frequencies[item],
                        -state.access_clock.get(item, -1),
                        -self.graph.records[item].timestamp,
                        item,
                    ),
                )
            else:  # pragma: no cover - argparse enforces this contract.
                raise ValueError(f"unknown online method: {state.method}")
            after = _budgeted_ids(ranked, self.storage_budget, self.graph)
        if storage_cost(after, self.graph) > self.storage_budget:
            raise RuntimeError(f"{state.method} exceeded the storage budget")
        state.bank = after
        state.last_active = active
        return (
            set(after.difference(before)),
            set(before.difference(after)),
            evaluated,
            accepted,
            timed_out,
        )

    def observe(
        self,
        state: PolicyState,
        request: Request,
        selected_ids: list[str],
        verified_success: bool = False,
        active: frozenset[str] | None = None,
    ) -> None:
        state.tick += 1
        for memory_id in selected_ids:
            state.access_clock[memory_id] = state.tick
            state.frequencies[memory_id] += 1
        selected_signature = (
            self._edge_signature(frozenset(selected_ids)) if selected_ids else None
        )
        signature = (
            selected_signature
            if verified_success and selected_signature
            else self._infer_request_signature(
                request, state.last_active if active is None else active
            )
        )
        if signature is not None:
            if state.last_signature is not None:
                outgoing = state.transition_counts.setdefault(
                    state.last_signature, Counter()
                )
                outgoing[signature] += 1
            state.signature_frequencies[signature] += 1
            state.last_signature = signature
        else:
            # Do not learn a spurious transition across an unobserved event.
            state.last_signature = None
        state.workload.append(request)
        while len(state.workload) > self.workload_window:
            state.workload.popleft()


def _failure_execution(stage: str, error: Exception) -> dict[str, Any]:
    return {
        "completed": False,
        "error_stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def execute_once(
    runner: Any,
    checker: Any,
    event: dict[str, Any],
    selected_memories: list[dict[str, Any]],
) -> dict[str, Any]:
    started = time.perf_counter()
    reset_verified = False
    snapshot_before = None
    snapshot_after = None
    execution_error = None
    try:
        reset = runner.reset(event)
        if not isinstance(reset, dict):
            raise RealTrackError("runner.reset() must return an object")
        reset_verified = reset.get("reset_verified") is True
        snapshot_before = reset.get("snapshot_id")
        if not reset_verified or not isinstance(snapshot_before, str):
            raise RealTrackError("runner reset was not independently verified")
        execution = runner.execute(event, selected_memories)
        snapshot_after = runner.snapshot(event)
        if not isinstance(execution, dict) or not isinstance(snapshot_after, str):
            raise RealTrackError("runner returned an invalid execution or snapshot")
    except Exception as error:  # noqa: BLE001 -- plugin boundary
        execution_error = f"{type(error).__name__}: {error}"
        execution = _failure_execution("runner", error)
    try:
        checked = checker.check(event, execution)
        if not isinstance(checked, dict):
            raise RealTrackError("checker.check() must return an object")
        checker_error = None
    except Exception as error:  # noqa: BLE001 -- plugin boundary
        checked = None
        checker_error = f"{type(error).__name__}: {error}"
    return {
        "execution_latency_ms": (time.perf_counter() - started) * 1000.0,
        "reset_verified": reset_verified,
        "snapshot_before": snapshot_before,
        "snapshot_after": snapshot_after,
        "execution_error": execution_error,
        "checker_error": checker_error,
        "execution": execution,
        "checker": checked,
    }


def load_exact_execution_cache(
    path: Path | None,
    event_ids: frozenset[str],
    memory_ids: frozenset[str],
    *,
    expected_composer: dict[str, Any] | None = None,
    expected_checker_contract: str | None = None,
    expected_dataset_sha256: str | None = None,
) -> dict[tuple[str, tuple[str, ...]], dict[str, Any]]:
    if path is None:
        return {}
    if not path.is_file():
        raise RealTrackError(f"source executions do not exist: {path}")
    cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for row in load_jsonl(path):
        event_id = str(row["event_id"])
        if event_id not in event_ids:
            continue
        selected = tuple(str(item) for item in row["selected_memory_ids"])
        if tuple(sorted(set(selected))) != selected or not set(selected).issubset(memory_ids):
            raise RealTrackError(f"invalid selected memory IDs in source: {event_id}")
        if (
            row.get("checker_independent") is not True
            or row.get("reset_verified") is not True
            or row.get("execution_error") is not None
            or row.get("checker_error") is not None
            or not isinstance(row.get("execution"), dict)
            or not isinstance(row.get("checker"), dict)
        ):
            raise RealTrackError(f"unverified source execution: {event_id}")
        if (
            expected_dataset_sha256 is not None
            and row.get("dataset_manifest_sha256") != expected_dataset_sha256
        ):
            raise RealTrackError(f"dataset manifest mismatch in source execution: {event_id}")
        if expected_composer is not None or expected_checker_contract is not None:
            validate_execution_contract(
                row, expected_composer, expected_checker_contract, event_id
            )
        record = {
            key: row[key]
            for key in (
                "execution_latency_ms",
                "reset_verified",
                "snapshot_before",
                "snapshot_after",
                "execution_error",
                "checker_error",
                "execution",
                "checker",
            )
        }
        key = (event_id, selected)
        previous = cache.get(key)
        if previous is not None and previous != record:
            raise RealTrackError(f"conflicting source execution: {event_id}")
        cache[key] = record
    if not cache:
        raise RealTrackError("source executions contain no matching event/bundle rows")
    return cache


def validate_execution_contract(
    record: dict[str, Any],
    expected_composer: dict[str, Any] | None,
    expected_checker_contract: str | None,
    event_id: str,
    expected_input_sha256: str | None = None,
) -> None:
    """Reject cached or fresh trajectories produced under a different contract."""
    execution = record.get("execution")
    checker = record.get("checker")
    if not isinstance(execution, dict) or not isinstance(checker, dict):
        raise RealTrackError(f"missing execution/checker evidence: {event_id}")
    if expected_composer is not None:
        for contract_key, execution_key in (
            ("model", "composer_model"),
            ("prompt_version", "composer_prompt_version"),
            ("config_sha256", "composer_config_sha256"),
            ("catalog_sha256", "composer_catalog_sha256"),
        ):
            if execution.get(execution_key) != expected_composer[contract_key]:
                raise RealTrackError(
                    f"composer {execution_key} mismatch in execution: {event_id}"
                )
        if not isinstance(execution.get("composer_input_sha256"), str) or len(
            execution["composer_input_sha256"]
        ) != 64:
            raise RealTrackError(f"missing composer input digest: {event_id}")
        if (
            expected_input_sha256 is not None
            and execution["composer_input_sha256"] != expected_input_sha256
        ):
            raise RealTrackError(f"composer input mismatch in execution: {event_id}")
    if expected_checker_contract is not None and (
        checker.get("contract_id") != expected_checker_contract
    ):
        raise RealTrackError(f"checker contract mismatch in execution: {event_id}")


def verify_online_model_provenance(
    model_path: Path,
    test_dataset: Path,
    test_event_count: int,
    model: OnlineLogisticPredictor,
) -> Path:
    """Require the frozen model's verified development-fit contract."""
    provenance_path = model_path.with_suffix(".provenance.json")
    if not provenance_path.is_file():
        raise ValueError("reportable online model has no training provenance")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        not isinstance(provenance, dict)
        or provenance.get("source_kind") != "verified_development_fit"
    ):
        raise ValueError("online model was not fitted on verified development executions")
    if (
        provenance.get("feature_version") != model.feature_version
        or provenance.get("dimension") != model.dimension
        or model.dimension != 256
    ):
        raise ValueError("online model feature contract does not match training provenance")
    if provenance.get("model_sha256") != hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest():
        raise ValueError("online model provenance checksum mismatch")
    if provenance.get("test_dataset_manifest_sha256") != hashlib.sha256(
        (test_dataset / "manifest.json").read_bytes()
    ).hexdigest():
        raise ValueError("online model was not frozen against this test split")
    if provenance.get("test_event_count") != test_event_count:
        raise ValueError("online model test event count does not match the frozen split")
    if type(provenance.get("verified_development_examples")) is not int or (
        provenance["verified_development_examples"] <= 0
    ):
        raise ValueError("online model has no verified development examples")
    for name in (
        "development_dataset_manifest_sha256",
        "development_executions_sha256",
    ):
        digest = provenance.get(name)
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"online model lacks a valid {name}")
    return provenance_path


def main() -> int:
    args = parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    if args.storage_budget <= 0 or args.retrieval_budget <= 0:
        raise ValueError("budgets must be positive")
    if args.workload_window <= 0:
        raise ValueError("workload window must be positive")
    if args.forecast_min_margin < 0.0:
        raise ValueError("forecast minimum margin must be non-negative")
    if args.forecast_min_support <= 0:
        raise ValueError("forecast minimum support must be positive")
    if not 0.0 <= args.forecast_min_confidence <= 1.0:
        raise ValueError("forecast minimum confidence must be in [0, 1]")
    if args.update_time_limit_ms is not None and args.update_time_limit_ms <= 0.0:
        raise ValueError("update time limit must be positive")
    if not 0.0 <= args.neural_weight <= 1.0:
        raise ValueError("neural weight must be between zero and one")
    if args.torch_threads <= 0:
        raise ValueError("torch threads must be positive")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("methods must be unique")
    python_hash_seed = os.environ.get("PYTHONHASHSEED")
    if args.model_dirs and python_hash_seed is None:
        raise ValueError(
            "PYTHONHASHSEED must be fixed for reportable frozen-model runs"
        )

    audit = audit_real_dataset(args.dataset)
    if not audit.eligible:
        raise RealTrackError(f"dataset audit failed: {audit.errors}")
    dataset_manifest_sha256 = hashlib.sha256(
        (args.dataset / "manifest.json").read_bytes()
    ).hexdigest()
    events = sorted(
        load_jsonl(args.dataset / "events.jsonl"),
        key=lambda row: (
            parse_utc(row["event_time"], "event_time"),
            str(row["event_id"]),
        ),
    )
    if args.max_events is not None:
        events = events[: args.max_events]
    memories = load_jsonl(args.dataset / "memories.jsonl")
    memories_by_id = {str(row["memory_id"]): row for row in memories}
    graph = make_graph(
        memories, load_jsonl(args.dataset / "dependency_hyperedges.jsonl")
    )
    model_metadata: dict[str, Any]
    if args.online_model and args.model_dirs:
        raise ValueError("online logistic and candidate-stage HGT models are distinct")
    if args.max_events is None and args.online_model is None:
        raise ValueError("reportable online runs require --online-model")
    if args.max_events is None and args.composer_module is None:
        raise ValueError("paper online runs require the fixed composer via --composer-module")
    if args.max_events is None and args.service_catalog is None:
        raise ValueError("paper online runs require --service-catalog")
    if args.online_model:
        scorer = OnlineLogisticPredictor.load(args.online_model)
        if scorer.dimension != 256:
            raise ValueError("paper online predictor requires 256 hash features")
        provenance_path = args.online_model.with_suffix(".provenance.json")
        if args.max_events is None:
            provenance_path = verify_online_model_provenance(
                args.online_model, args.dataset, len(events), scorer
            )
        model_metadata = {
            "kind": "fixed_online_logistic",
            "artifact": str(args.online_model.resolve()),
            "sha256": hashlib.sha256(args.online_model.read_bytes()).hexdigest(),
            "feature_version": scorer.feature_version,
            "dimension": scorer.dimension,
            "provenance": str(provenance_path.resolve()) if provenance_path.is_file() else None,
            "provenance_sha256": (
                hashlib.sha256(provenance_path.read_bytes()).hexdigest()
                if provenance_path.is_file() else None
            ),
        }
    elif args.model_dirs:
        import torch

        from experiments.evaluate_public_multiservice_hgt_ensemble import load_models

        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)
        frozen_models = load_models(
            args.model_dirs,
            graph,
            dimension=args.dimension,
            text_hash_dimension=args.text_hash_dimension,
            message_layers=args.message_layers,
            device=args.device,
            lexical_features=args.lexical_features,
        )
        scorer: Any = FrozenHGTBM25Ensemble(frozen_models, args.neural_weight)
        model_metadata = {
            "kind": "five_seed_hgt_bm25_ensemble",
            "model_seeds": [seed for seed, _, _ in frozen_models],
            "model_states": [
                {
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for _, _, path in frozen_models
            ],
            "neural_weight": args.neural_weight,
            "bm25_weight": 1.0 - args.neural_weight,
            "bm25_aggregation": "max",
            "bm25_normalization": "request_admissible_records_minmax",
            "dimension": args.dimension,
            "text_hash_dimension": args.text_hash_dimension,
            "message_layers": args.message_layers,
            "lexical_features": args.lexical_features,
            "device": args.device,
            "torch_threads": args.torch_threads,
            "python_hash_seed": python_hash_seed,
            "deterministic_algorithms": True,
        }
    else:
        scorer = SetUtilityModel()
        model_metadata = {
            "kind": "transparent_reference_set_utility",
            "warning": "Development-only scorer; paper online runs require a fixed logistic artifact.",
        }
    retriever = BundleBeamRetriever(
        scorer,
        RetrievalConfig(
            retrieval_budget=args.retrieval_budget,
            beam_width=8,
            max_depth=4,
            candidate_pool_size=64,
            dependency_hops=2,
            max_closure_size=8,
            enforce_dependency_closure=True,
            require_hyperedge_seed=args.require_hyperedge_seed,
            context_cost_weight=0.10,
            tie_seed=args.seed,
        ),
    )
    controller = OnlineController(
        graph,
        retriever,
        args.storage_budget,
        args.workload_window,
        None if args.uncapped_maintenance else args.update_time_limit_ms,
        args.forecast_min_margin,
        args.forecast_min_support,
        args.forecast_min_confidence,
    )
    states = {method: PolicyState(method) for method in args.methods}
    if args.composer_module:
        composer_module = importlib.import_module(args.composer_module)
        factory = getattr(composer_module, "create_runner", None)
        if not callable(factory):
            raise TypeError("composer module must expose create_runner()")
        runner = (
            factory(catalog_path=args.service_catalog)
            if args.service_catalog is not None
            else factory()
        )
    else:
        runner = create_runner()
    if any(not callable(getattr(runner, name, None)) for name in ("reset", "execute", "snapshot")):
        raise TypeError("composer runner must implement reset(), execute(), and snapshot()")
    composer_contract = None
    if args.max_events is None:
        contract_method = getattr(runner, "composer_contract", None)
        validate_events = getattr(runner, "validate_events", None)
        input_digest = getattr(runner, "input_sha256", None)
        if not all(callable(item) for item in (contract_method, validate_events, input_digest)):
            raise TypeError("paper composer must expose its contract, event and input checks")
        composer_contract = contract_method()
        if (
            not isinstance(composer_contract, dict)
            or composer_contract.get("model") != "gpt-4.1-2025-04-14"
            or not isinstance(composer_contract.get("prompt_version"), str)
            or not isinstance(composer_contract.get("config_sha256"), str)
            or not isinstance(composer_contract.get("catalog_sha256"), str)
            or len(composer_contract["config_sha256"]) != 64
            or len(composer_contract["catalog_sha256"]) != 64
        ):
            raise ValueError("paper composer contract is incomplete or has the wrong model")
        validate_events(events)
    checker = create_checker()
    source_cache = load_exact_execution_cache(
        args.source_executions,
        frozenset(str(row["event_id"]) for row in events),
        frozenset(memories_by_id),
        expected_composer=composer_contract,
        expected_checker_contract=(
            CHECKER_CONTRACT_ID if composer_contract is not None else None
        ),
        expected_dataset_sha256=(
            dataset_manifest_sha256 if composer_contract is not None else None
        ),
    )
    execution_cache = dict(source_cache)
    used_source_keys: set[tuple[str, tuple[str, ...]]] = set()
    rows: list[dict[str, Any]] = []
    banks: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    physical_executions = 0
    started_run = time.perf_counter()

    for event_index, event in enumerate(events, start=1):
        event_id = str(event["event_id"])
        request = request_for(event)
        active = active_archive_ids(memories, request.timestamp)
        for method in args.methods:
            state = states[method]
            pre_started = time.perf_counter()
            # At t, a public version change may invalidate old records. The
            # previous bank cannot activate replacements before event t has
            # been executed and independently checked.
            evicted_pre = prune_before_event(state, active)
            activated_pre: set[str] = set()
            evaluated_pre = accepted_pre = 0
            pre_update_ms = (time.perf_counter() - pre_started) * 1000.0
            bank_before = state.bank
            selection_started = time.perf_counter()
            if method == "no_memory":
                selected_ids: list[str] = []
                token_cost = 0
                retrieval_status = "fallback"
                evaluations = 0
            else:
                retrieval = retriever.retrieve(request, bank_before, graph)
                selected_ids = sorted(retrieval.selected_ids)
                token_cost = retrieval.token_cost
                retrieval_status = retrieval.status
                evaluations = retrieval.evaluations
            selection_ms = (time.perf_counter() - selection_started) * 1000.0
            if not set(selected_ids).issubset(bank_before):
                raise RuntimeError("retrieval escaped the active bank")

            cache_key = (event_id, tuple(selected_ids))
            fixed_refusal = method != "no_memory" and retrieval_status == "fallback"
            physical_reuse = not fixed_refusal and cache_key in execution_cache
            if fixed_refusal:
                execution_record = {
                    "execution_latency_ms": 0.0,
                    "reset_verified": False,
                    "snapshot_before": None,
                    "snapshot_after": None,
                    "execution_error": None,
                    "checker_error": None,
                    "execution": {"status": "fixed_refusal"},
                    "checker": None,
                }
                execution_provenance = "fixed_refusal"
            elif physical_reuse:
                execution_record = dict(execution_cache[cache_key])
                if composer_contract is not None:
                    validate_execution_contract(
                        execution_record, composer_contract, CHECKER_CONTRACT_ID,
                        event_id,
                        runner.input_sha256(
                            event, [memories_by_id[memory_id] for memory_id in selected_ids]
                        ),
                    )
                if cache_key in source_cache:
                    used_source_keys.add(cache_key)
                    execution_provenance = "prior_exact_execution"
                else:
                    execution_provenance = "within_run_reuse"
            else:
                execution_record = execute_once(
                    runner,
                    checker,
                    event,
                    [memories_by_id[memory_id] for memory_id in selected_ids],
                )
                if composer_contract is not None:
                    validate_execution_contract(
                        execution_record, composer_contract, CHECKER_CONTRACT_ID,
                        event_id,
                        runner.input_sha256(
                            event, [memories_by_id[memory_id] for memory_id in selected_ids]
                        ),
                    )
                execution_cache[cache_key] = dict(execution_record)
                physical_executions += 1
                execution_provenance = "executed_this_run"
            run_id = f"{method}:{args.seed}:{event_id}"
            row = {
                "run_id": run_id,
                "event_id": event_id,
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "method": method,
                "seed": args.seed,
                "selected_memory_ids": selected_ids,
                "token_cost": token_cost,
                "selection_latency_ms": selection_ms,
                "execution_latency_ms": execution_record["execution_latency_ms"],
                "latency_ms": selection_ms + execution_record["execution_latency_ms"],
                "checker_independent": not fixed_refusal,
                "physical_execution_reused": physical_reuse,
                "execution_provenance": execution_provenance,
                "physical_execution_cache_key": hashlib.sha256(
                    repr(cache_key).encode()
                ).hexdigest(),
                **{
                    key: value
                    for key, value in execution_record.items()
                    if key != "execution_latency_ms"
                },
            }
            if row["execution_error"] is not None or row["checker_error"] is not None:
                raise RealTrackError(f"infrastructure error in {run_id}")
            counts = (
                {
                    "verified_success": 0,
                    "goal_satisfied": 0,
                    "binding_passed": 0,
                    "binding_total": len(event["required_binding_atoms"]),
                    "evidence_covered": 0,
                    "evidence_total": len(event["required_evidence_atoms"]),
                    "evidence_complete": 0,
                    "communication_passed": 0,
                    "communication_total": len(event["communication_criteria"]),
                    "communication_success": 0,
                    "attempted_reuses": 0,
                    "stale_attempts": 0,
                    "stale_workflow": 0,
                }
                if fixed_refusal
                else score_execution_row(row, event)
            )
            scored_rows.append(
                {
                    "event_id": event_id,
                    "incident_id": event["incident_id"],
                    "provider_id": event["provider_id"],
                    "service_id": event["service_id"],
                    "version_id": event["version_id"],
                    "event_time": event["event_time"],
                    "event_index": event_index,
                    "service_event_index": 0,
                    "method": method,
                    "seed": args.seed,
                    "token_cost": token_cost,
                    "latency_ms": row["latency_ms"],
                    **counts,
                }
            )
            # The current request enters the workload only after independent checking.
            controller.observe(
                state,
                request,
                selected_ids,
                verified_success=counts["verified_success"] == 1,
                active=active,
            )
            post_started = time.perf_counter()
            (
                activated_post,
                evicted_post,
                evaluated_post,
                accepted_post,
                timed_out_post,
            ) = controller.update(state, active)
            post_update_ms = (time.perf_counter() - post_started) * 1000.0
            row.update(
                {
                    "active_bank_cost": storage_cost(bank_before, graph),
                    "retrieval_status": retrieval_status,
                    "evaluations": evaluations,
                    "pre_update_latency_ms": pre_update_ms,
                    "post_update_latency_ms": post_update_ms,
                    "bank_before_ids": sorted(bank_before),
                    "bank_after_ids": sorted(state.bank),
                    "activated_pre_ids": sorted(activated_pre),
                    "evicted_pre_ids": sorted(evicted_pre),
                    "activated_post_ids": sorted(activated_post),
                    "evicted_post_ids": sorted(evicted_post),
                    "retention_banks_evaluated": evaluated_pre + evaluated_post,
                    "retention_swaps_accepted": accepted_pre + accepted_post,
                    "retention_update_timed_out": timed_out_post,
                }
            )
            rows.append(row)
            banks.append(
                {
                    "event_id": event_id,
                    "event_time": event["event_time"],
                    "event_index": event_index,
                    "method": method,
                    "seed": args.seed,
                    "storage_budget": args.storage_budget,
                    "bank_before_cost": storage_cost(bank_before, graph),
                    "bank_after_cost": storage_cost(state.bank, graph),
                    "bank_before_records": len(bank_before),
                    "bank_after_records": len(state.bank),
                    "complete_groups_before": complete_groups(bank_before, graph),
                    "complete_groups_after": complete_groups(state.bank, graph),
                    "activated_records": len(activated_pre) + len(activated_post),
                    "evicted_records": len(evicted_pre) + len(evicted_post),
                    "update_latency_ms": pre_update_ms + post_update_ms,
                    "verified_success": counts["verified_success"],
                    "update_timed_out": timed_out_post,
                }
            )
        if event_index % 20 == 0 or event_index == len(events):
            elapsed = time.perf_counter() - started_run
            print(
                f"processed {event_index}/{len(events)} events; "
                f"physical executions={physical_executions}; elapsed={elapsed:.1f}s",
                flush=True,
            )

    service_counts: Counter[tuple[str, str]] = Counter()
    for row in scored_rows:
        key = (str(row["method"]), str(row["service_id"]))
        service_counts[key] += 1
        row["service_event_index"] = service_counts[key]

    summaries = []
    for method in args.methods:
        members = [row for row in scored_rows if row["method"] == method]
        execution_members = [row for row in rows if row["method"] == method]
        bank_members = [row for row in banks if row["method"] == method]
        summary = summarize_scored_rows(members)
        update_latencies = sorted(float(row["update_latency_ms"]) for row in bank_members)
        p95_index = max(0, int(0.95 * len(update_latencies) + 0.999999) - 1)
        summaries.append(
            {
                "method": method,
                **summary,
                "mean_update_latency_ms": statistics.mean(update_latencies),
                "p95_update_latency_ms": update_latencies[p95_index],
                "retrievable_complete_evidence_rate": sum(
                    row["retrieval_status"] == "ok" for row in execution_members
                )
                / len(execution_members),
                "peak_bank_cost": max(int(row["bank_after_cost"]) for row in bank_members),
                "total_activated_records": sum(
                    int(row["activated_records"]) for row in bank_members
                ),
                "total_evicted_records": sum(
                    int(row["evicted_records"]) for row in bank_members
                ),
                "total_churn_records": sum(
                    int(row["activated_records"]) + int(row["evicted_records"])
                    for row in bank_members
                ),
                "mean_churn_records_per_update": statistics.mean(
                    int(row["activated_records"]) + int(row["evicted_records"])
                    for row in bank_members
                ),
                "update_timeout_rate": sum(
                    bool(row["update_timed_out"]) for row in bank_members
                )
                / len(bank_members),
            }
        )

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "execution_results.jsonl", rows)
    write_jsonl(args.output / "scored_executions.jsonl", scored_rows)
    write_jsonl(args.output / "bank_history.jsonl", banks)
    write_json(
        args.output / "result.json",
        {
            "status": "complete_online_retention" if args.max_events is None else "development_prefix",
            "schema_version": "corem-online-retention-v2-strict-prequential",
            "dataset": str(args.dataset.resolve()),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "dataset_audit": audit.to_dict(),
            "events": len(events),
            "methods": args.methods,
            "seed": args.seed,
            "storage_budget": args.storage_budget,
            "retrieval_budget": args.retrieval_budget,
            "require_hyperedge_seed": args.require_hyperedge_seed,
            "workload_window": args.workload_window,
            "retrieval_beam_width": retriever.config.beam_width,
            "retrieval_beam_depth": retriever.config.max_depth,
            "retrieval_candidate_pool_size": retriever.config.candidate_pool_size,
            "retrieval_cost_weight": retriever.config.context_cost_weight,
            "retrieval_tie_seed": retriever.config.tie_seed,
            "retention_proposal_cap": controller.group_solvers["corem_retention"].config.max_proposals,
            "retention_eviction_shortlist": controller.group_solvers["corem_retention"].config.eviction_shortlist_size,
            "retention_transition_cost_weight": controller.group_solvers["corem_retention"].config.activation_cost_weight,
            "forecast_min_margin": args.forecast_min_margin,
            "forecast_min_support": args.forecast_min_support,
            "forecast_min_confidence": args.forecast_min_confidence,
            "update_time_limit_ms": None if args.uncapped_maintenance else args.update_time_limit_ms,
            "maintenance_uncapped": args.uncapped_maintenance,
            "retention_objective_contract": {
                "matched_methods": {
                    "item_value_retention": (
                        "sum of singleton predicted utilities after subtracting "
                        "the empty-set utility from each singleton"
                    ),
                    "bundle_value_retention": (
                        "best pre-enumerated complete bundle utility; deployed "
                        "retriever is not called for bank valuation"
                    ),
                    "corem_retention": (
                        "utility of the set returned by the deployed retriever"
                    ),
                },
                "common_components": (
                    "retriever, predictor, chronological requests, active archive, "
                    "cold-start bank, storage and retrieval budgets, workload window, "
                    "group proposals, eviction candidates, and proposal limits"
                ),
                "only_changed_factor": "bank-value objective",
                "proposal_and_eviction_priority": (
                    "relaxed bank-value sensitivity over the same past-only workload; "
                    "independent of the bank-value objective"
                ),
            },
            "forecast_policy_contract": {
                "method": "forecast_corem",
                "state": "latent workflow signatures and first-order transitions",
                "feedback_timing": (
                    "event t is retrieved, executed, and independently checked "
                    "before it updates the transition model or bank for t+1"
                ),
                "label_source": (
                    "verified selected bundle on success; otherwise the frozen "
                    "bundle scorer only when its top-two margin passes the gate"
                ),
                "activation_gate": {
                    "minimum_support": args.forecast_min_support,
                    "minimum_confidence": args.forecast_min_confidence,
                },
                "fallback": "matched Bundle-value retention",
            },
            "retrieval_model": model_metadata,
            "composer_module": args.composer_module or "controlled_plan_runner",
            "composer_contract": composer_contract,
            "checker_contract": CHECKER_CONTRACT_ID,
            "service_catalog": (
                str(args.service_catalog.resolve()) if args.service_catalog else None
            ),
            "physical_executions_this_run": physical_executions,
            "exact_prior_executions_used": len(used_source_keys),
            "unique_verified_executions": physical_executions + len(used_source_keys),
            "source_executions": (
                str(args.source_executions.resolve()) if args.source_executions else None
            ),
            "source_executions_sha256": (
                hashlib.sha256(args.source_executions.read_bytes()).hexdigest()
                if args.source_executions
                else None
            ),
            "logical_executions": len(rows),
            "summaries": summaries,
            "time_contract": (
                "At event t, remove version-invalid records from the bank frozen after "
                "t-1 without activating replacements; retrieve, execute, and independently "
                "check; only then add request t to the workload and update the bank."
            ),
            "archive_contract": (
                "Version-valid records remain in a read-only archive and may be reactivated "
                "only during an online update under the same storage budget."
            ),
            "execution_cache_contract": (
                "An exact event/selected-memory tuple may reuse an independently checked "
                "reset-environment trajectory only when dataset manifest, composer model, "
                "prompt, configuration, catalog, visible input, and checker contract match. "
                "Other distinct tuples are executed in this run; within-run duplicates "
                "share that trajectory."
            ),
            "claim_boundary": (
                "Controlled queries anchored to independently observed public releases; "
                "not natural production traffic."
            ),
            "elapsed_seconds": time.perf_counter() - started_run,
        },
    )
    write_checksums(
        args.output,
        (
            "bank_history.jsonl",
            "execution_results.jsonl",
            "result.json",
            "scored_executions.jsonl",
        ),
    )
    print(f"wrote online retention artifacts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
