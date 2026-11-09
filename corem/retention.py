"""Retrieval-consistent, dependency-group memory retention."""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from .retrieval import BundleBeamRetriever
from .types import EvidenceGraph, Request, RetentionResult


@dataclass(slots=True)
class RetentionConfig:
    storage_budget: int
    max_proposals: int = 16
    eviction_shortlist_size: int = 16
    max_eviction_group: int = 4
    proposal_dependency_hops: int = 2
    max_proposal_group: int = 8
    relaxation_steps: int = 8
    relaxation_step_size: float = 0.25
    sensitivity_epsilon: float = 0.02
    entropy_weight: float = 0.01
    coverage_penalty: float = 0.25
    activation_cost_weight: float = 0.0
    removal_cost_weight: float = 0.0
    # Backward-compatible symmetric update penalty. Prefer the two weights above.
    update_cost_weight: float = 0.0
    min_improvement: float = 0.0
    bank_value_objective: Literal["retriever", "bundle", "item"] = "retriever"
    proposal_priority: Literal["relaxed", "semantic"] = "relaxed"

    def __post_init__(self) -> None:
        if self.storage_budget <= 0:
            raise ValueError("storage_budget must be positive")
        limits = (
            self.max_proposals,
            self.eviction_shortlist_size,
            self.max_eviction_group,
            self.proposal_dependency_hops,
            self.max_proposal_group,
            self.relaxation_steps,
        )
        if any(value <= 0 for value in limits):
            raise ValueError("retention search limits must be positive")
        if not 0.0 < self.sensitivity_epsilon < 1.0:
            raise ValueError("sensitivity_epsilon must be in (0, 1)")
        if self.relaxation_step_size <= 0.0:
            raise ValueError("relaxation_step_size must be positive")
        weights = (
            self.entropy_weight,
            self.coverage_penalty,
            self.activation_cost_weight,
            self.removal_cost_weight,
            self.update_cost_weight,
            self.min_improvement,
        )
        if any(value < 0.0 for value in weights):
            raise ValueError("retention weights must be non-negative")
        if self.bank_value_objective not in {"retriever", "bundle", "item"}:
            raise ValueError("unknown bank-value objective")
        if self.proposal_priority not in {"relaxed", "semantic"}:
            raise ValueError("unknown proposal-priority rule")


class GroupAwareRetention:
    """Bounded group-swap solver for the nested retention objective."""

    def __init__(self, retriever: BundleBeamRetriever, config: RetentionConfig) -> None:
        self.retriever = retriever
        self.config = config

    def optimize(
        self,
        current_bank: Iterable[str],
        new_record_ids: Iterable[str],
        workload: Sequence[Request],
        graph: EvidenceGraph,
        time_limit_ms: float | None = None,
        workload_weights: Sequence[float] | None = None,
    ) -> RetentionResult:
        if not workload:
            raise ValueError("a non-empty past-only workload sample is required")
        if time_limit_ms is not None and time_limit_ms <= 0.0:
            raise ValueError("time_limit_ms must be positive")
        if workload_weights is None:
            weights = tuple(1.0 / len(workload) for _ in workload)
        else:
            if len(workload_weights) != len(workload):
                raise ValueError("workload weights must match workload length")
            if any(not math.isfinite(value) or value < 0 for value in workload_weights):
                raise ValueError("workload weights must be finite and non-negative")
            total = sum(workload_weights)
            if total <= 0:
                raise ValueError("workload weights must have positive total")
            weights = tuple(value / total for value in workload_weights)
        deadline = (
            time.perf_counter() + time_limit_ms / 1000.0
            if time_limit_ms is not None
            else None
        )
        original = frozenset(current_bank)
        new_ids = frozenset(new_record_ids)
        universe = original.union(new_ids)
        graph._require_records(universe)
        if self._storage_cost(original, graph) > self.config.storage_budget:
            raise ValueError("current bank already exceeds the storage budget")

        def deadline_result() -> RetentionResult:
            return RetentionResult(
                retained_ids=original,
                bank_value=self.retriever.config.fallback_value,
                storage_cost=self._storage_cost(original, graph),
                accepted_swaps=0,
                evaluated_banks=0,
                timed_out=True,
            )

        if deadline is not None and time.perf_counter() >= deadline:
            return deadline_result()
        sensitivity = self._proposal_priorities(
            universe, workload, graph, weights, deadline
        )
        if deadline is not None and time.perf_counter() >= deadline:
            return deadline_result()
        fixed_bundles = (
            self._fixed_complete_bundles(universe, workload, graph, deadline)
            if self.config.bank_value_objective == "bundle"
            else None
        )
        if deadline is not None and time.perf_counter() >= deadline:
            return deadline_result()
        proposals = self._bundle_proposals(new_ids, universe, graph, deadline)
        proposals.sort(
            key=lambda group: (
                -sum(sensitivity.get(key, 0.0) for key in sorted(group)),
                len(group),
                tuple(sorted(group)),
            )
        )
        bank = original
        if deadline is not None and time.perf_counter() >= deadline:
            return deadline_result()
        value = self._sample_bank_value(
            bank, workload, graph, original, fixed_bundles, deadline, weights
        )
        if value is None:
            return RetentionResult(
                retained_ids=bank,
                bank_value=self.retriever.config.fallback_value,
                storage_cost=self._storage_cost(bank, graph),
                accepted_swaps=0,
                evaluated_banks=0,
                timed_out=True,
            )
        evaluated = 1
        accepted = 0
        timed_out = deadline is not None and time.perf_counter() >= deadline

        for proposal in proposals[: self.config.max_proposals]:
            if deadline is not None and time.perf_counter() >= deadline:
                timed_out = True
                break
            if proposal.issubset(bank):
                continue
            augmented = bank.union(proposal)
            evictions = self._eviction_candidates(
                augmented, proposal, sensitivity, graph
            )
            for evicted in evictions:
                if deadline is not None and time.perf_counter() >= deadline:
                    timed_out = True
                    break
                candidate = augmented.difference(evicted)
                if self._storage_cost(candidate, graph) > self.config.storage_budget:
                    continue
                candidate_value = self._sample_bank_value(
                    candidate,
                    workload,
                    graph,
                    original,
                    fixed_bundles,
                    deadline,
                    weights,
                )
                if candidate_value is None:
                    timed_out = True
                    break
                evaluated += 1
                if deadline is not None and time.perf_counter() >= deadline:
                    timed_out = True
                if candidate_value > value + self.config.min_improvement:
                    bank = frozenset(candidate)
                    value = candidate_value
                    accepted += 1
                    break
            if timed_out:
                break

        return RetentionResult(
            retained_ids=bank,
            bank_value=value,
            storage_cost=self._storage_cost(bank, graph),
            accepted_swaps=accepted,
            evaluated_banks=evaluated,
            timed_out=timed_out,
        )

    def _sample_bank_value(
        self,
        bank: frozenset[str],
        workload: Sequence[Request],
        graph: EvidenceGraph,
        reference_bank: frozenset[str],
        fixed_bundles: tuple[tuple[frozenset[str], ...], ...] | None = None,
        deadline: float | None = None,
        workload_weights: Sequence[float] | None = None,
    ) -> float | None:
        if self.config.bank_value_objective == "retriever":
            values = self._retriever_values(bank, workload, graph, deadline)
        elif self.config.bank_value_objective == "bundle":
            if fixed_bundles is None or len(fixed_bundles) != len(workload):
                raise ValueError("fixed bundle candidates are required")
            values = self._bundle_values(
                bank, workload, graph, fixed_bundles, deadline
            )
        else:
            values = self._item_values(bank, workload, graph, deadline)
        if values is None:
            return None
        activated = self._storage_cost(bank.difference(reference_bank), graph)
        removed = self._storage_cost(reference_bank.difference(bank), graph)
        symmetric = activated + removed
        update_penalty = (
            self.config.activation_cost_weight * activated
            + self.config.removal_cost_weight * removed
            + self.config.update_cost_weight * symmetric
        ) / self.config.storage_budget
        weights = (
            workload_weights
            if workload_weights is not None
            else (1.0 / len(values) for _ in values)
        )
        return sum(weight * value for weight, value in zip(weights, values)) - update_penalty

    def _retriever_values(
        self,
        bank: frozenset[str],
        workload: Sequence[Request],
        graph: EvidenceGraph,
        deadline: float | None = None,
    ) -> list[float] | None:
        values = []
        for request in workload:
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            result = self.retriever.retrieve(request, bank, graph)
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            values.append(
                result.objective
                if result.status == "ok"
                else self.retriever.config.fallback_value
            )
        return values

    def _fixed_complete_bundles(
        self,
        universe: frozenset[str],
        workload: Sequence[Request],
        graph: EvidenceGraph,
        deadline: float | None = None,
    ) -> tuple[tuple[frozenset[str], ...], ...]:
        """Enumerate bank-independent complete candidates without retrieval.

        The candidate list is frozen once per update from the common universe.
        Candidate-bank comparisons therefore cannot change the list or invoke
        the deployed beam retriever.
        """
        result: list[tuple[frozenset[str], ...]] = []
        for request in workload:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            candidates: set[frozenset[str]] = set()
            for bundle in self.retriever.build_bundles(request, universe, graph):
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                members = bundle.members
                if (
                    self.retriever._token_cost(members, graph)
                    > self.retriever.config.retrieval_budget
                ):
                    continue
                if not self.retriever._is_feasible(request, members, graph):
                    continue
                candidates.add(members)
            result.append(
                tuple(sorted(candidates, key=lambda value: tuple(sorted(value))))
            )
        return tuple(result)

    def _bundle_values(
        self,
        bank: frozenset[str],
        workload: Sequence[Request],
        graph: EvidenceGraph,
        fixed_bundles: tuple[tuple[frozenset[str], ...], ...],
        deadline: float | None = None,
    ) -> list[float] | None:
        values: list[float] = []
        for request, candidates in zip(workload, fixed_bundles):
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            feasible = [
                candidate for candidate in candidates if candidate.issubset(bank)
            ]
            if not feasible:
                values.append(self.retriever.config.fallback_value)
                continue
            batched = getattr(self.retriever.model, "probabilities", None)
            probabilities = (
                batched(request, feasible, graph)
                if callable(batched)
                else tuple(
                    self.retriever.model.probability(request, candidate, graph)
                    for candidate in feasible
                )
            )
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            values.append(
                max(
                    self.retriever._objective(request, candidate, graph, probability)
                    for candidate, probability in zip(feasible, probabilities)
                )
            )
        return values

    def _item_values(
        self,
        bank: frozenset[str],
        workload: Sequence[Request],
        graph: EvidenceGraph,
        deadline: float | None = None,
    ) -> list[float] | None:
        """Sum independently scored singleton utilities for each request."""
        values: list[float] = []
        for request in workload:
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            singletons = [
                frozenset((key,))
                for key in sorted(bank)
                if graph.is_admissible(key, request)
            ]
            if not singletons:
                values.append(0.0)
                continue
            batched = getattr(self.retriever.model, "probabilities", None)
            probabilities = (
                batched(request, singletons, graph)
                if callable(batched)
                else tuple(
                    self.retriever.model.probability(request, singleton, graph)
                    for singleton in singletons
                )
            )
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            values.append(
                sum(
                    self.retriever._objective(
                        request, singleton, graph, probability
                    )
                    for singleton, probability in zip(singletons, probabilities)
                )
            )
        return values

    def _bundle_proposals(
        self,
        new_ids: frozenset[str],
        universe: frozenset[str],
        graph: EvidenceGraph,
        deadline: float | None = None,
    ) -> list[frozenset[str]]:
        proposals = {frozenset((key,)) for key in new_ids}
        if deadline is not None and time.perf_counter() >= deadline:
            return list(proposals)
        proposals.update(
            graph.dependency_groups(
                new_ids,
                universe,
                max_hops=self.config.proposal_dependency_hops,
                max_size=self.config.max_proposal_group,
                deadline=deadline,
            )
        )
        return list(proposals)

    def _relaxed_sensitivities(
        self,
        universe: frozenset[str],
        workload: Sequence[Request],
        graph: EvidenceGraph,
        workload_weights: Sequence[float] | None = None,
        deadline: float | None = None,
    ) -> dict[str, float]:
        """Finite-difference dV/dy through the unrolled projected optimizer.

        The retention gate ``y`` scales each relaxed inclusion.  These
        sensitivities rank proposals only; every accepted bank is still
        evaluated by the exact discrete retriever.
        """
        ordered_universe = tuple(sorted(universe))
        aggregate = {key: 0.0 for key in ordered_universe}
        weights = workload_weights or tuple(1.0 / len(workload) for _ in workload)
        for request, request_weight in zip(workload, weights):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            eligible = [
                key
                for key in ordered_universe
                if graph.is_admissible(key, request)
            ]
            if not eligible:
                continue
            availability = {key: 1.0 for key in eligible}
            base_value = self._relaxed_request_value(request, availability, graph)
            for key in eligible:
                if deadline is not None and time.perf_counter() >= deadline:
                    return aggregate
                perturbed = dict(availability)
                perturbed[key] -= self.config.sensitivity_epsilon
                value = self._relaxed_request_value(request, perturbed, graph)
                aggregate[key] += (
                    (base_value - value)
                    / self.config.sensitivity_epsilon
                    * request_weight
                )
        return aggregate

    def _proposal_priorities(
        self,
        universe: frozenset[str],
        workload: Sequence[Request],
        graph: EvidenceGraph,
        workload_weights: Sequence[float] | None = None,
        deadline: float | None = None,
    ) -> dict[str, float]:
        weights = workload_weights or tuple(1.0 / len(workload) for _ in workload)
        if self.config.proposal_priority == "relaxed":
            return self._relaxed_sensitivities(
                universe, workload, graph, weights, deadline
            )
        priorities: dict[str, float] = {}
        for key in sorted(universe):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            priorities[key] = sum(
                weight * self.retriever.model.semantic_similarity(request, key, graph)
                for request, weight in zip(workload, weights)
                if graph.is_admissible(key, request)
            )
        return priorities

    def _relaxed_request_value(
        self,
        request: Request,
        availability: dict[str, float],
        graph: EvidenceGraph,
    ) -> float:
        budget = self.retriever.config.retrieval_budget
        denominator = max(
            1.0,
            sum(
                graph.records[key].token_cost * availability[key]
                for key in availability
            ),
        )
        inclusion = {key: min(0.5, budget / denominator) for key in availability}
        self._project_budget_and_coverage(
            inclusion, availability, request, budget, graph
        )
        for _ in range(self.config.relaxation_steps):
            effective = {key: inclusion[key] * availability[key] for key in inclusion}
            _, probability_gradient = (
                self.retriever.model.relaxed_probability_and_gradient(
                    request, effective, graph
                )
            )
            coverage = self._fractional_coverage(request, effective, graph)
            updated: dict[str, float] = {}
            for key in inclusion:
                x = min(1.0 - 1e-6, max(1e-6, inclusion[key]))
                entropy_gradient = math.log((1.0 - x) / x)
                cost_gradient = (
                    self.retriever.config.context_cost_weight
                    * graph.records[key].token_cost
                    * availability[key]
                    / budget
                )
                missing_gain = sum(
                    availability[key]
                    for atom, amount in coverage.items()
                    if amount < 1.0 and atom in graph.records[key].evidence_atoms
                )
                gradient = (
                    probability_gradient[key] * availability[key]
                    - cost_gradient
                    + self.config.entropy_weight * entropy_gradient
                    + self.config.coverage_penalty * missing_gain
                )
                updated[key] = min(
                    1.0,
                    max(0.0, x + self.config.relaxation_step_size * gradient),
                )
            inclusion = updated
            self._project_budget_and_coverage(
                inclusion, availability, request, budget, graph
            )

        effective = {key: inclusion[key] * availability[key] for key in inclusion}
        probability, _ = self.retriever.model.relaxed_probability_and_gradient(
            request, effective, graph
        )
        normalized_cost = (
            sum(
                graph.records[key].token_cost * amount
                for key, amount in effective.items()
            )
            / budget
        )
        entropy = -sum(
            value * math.log(max(value, 1e-12))
            + (1.0 - value) * math.log(max(1.0 - value, 1e-12))
            for value in inclusion.values()
        )
        coverage = self._fractional_coverage(request, effective, graph)
        deficit = sum(max(0.0, 1.0 - value) for value in coverage.values())
        return (
            probability
            - self.retriever.config.context_cost_weight * normalized_cost
            + self.config.entropy_weight * entropy
            - self.config.coverage_penalty * deficit
        )

    @staticmethod
    def _fractional_coverage(
        request: Request, inclusion: dict[str, float], graph: EvidenceGraph
    ) -> dict[str, float]:
        coverage = {
            atom: (1.0 if atom in request.verified_atoms else 0.0)
            for atom in request.required_atoms
        }
        for key, amount in inclusion.items():
            for atom in graph.records[key].evidence_atoms.intersection(
                request.required_atoms
            ):
                coverage[atom] += amount
        return coverage

    @staticmethod
    def _project_budget_and_coverage(
        inclusion: dict[str, float],
        availability: dict[str, float],
        request: Request,
        budget: int,
        graph: EvidenceGraph,
    ) -> None:
        """Alternating projection onto box, budget, and evidence coverage."""
        for _ in range(8):
            for key in inclusion:
                inclusion[key] = min(1.0, max(0.0, inclusion[key]))
            cost = sum(
                graph.records[key].token_cost * inclusion[key] * availability[key]
                for key in inclusion
            )
            if cost > budget and cost > 0.0:
                scale = budget / cost
                for key in inclusion:
                    inclusion[key] *= scale
            effective = {key: inclusion[key] * availability[key] for key in inclusion}
            coverage = GroupAwareRetention._fractional_coverage(
                request, effective, graph
            )
            changed = False
            for atom, amount in coverage.items():
                deficit = max(0.0, 1.0 - amount)
                if deficit <= 1e-9:
                    continue
                providers = [
                    key
                    for key in inclusion
                    if atom in graph.records[key].evidence_atoms
                    and availability[key] * (1.0 - inclusion[key]) > 0.0
                ]
                for key in sorted(
                    providers,
                    key=lambda item: (graph.records[item].token_cost, item),
                ):
                    room = availability[key] * (1.0 - inclusion[key])
                    addition = min(deficit, room)
                    inclusion[key] += addition / availability[key]
                    deficit -= addition
                    changed = True
                    if deficit <= 1e-9:
                        break
            if not changed:
                break
        cost = sum(
            graph.records[key].token_cost * inclusion[key] * availability[key]
            for key in inclusion
        )
        if cost > budget and cost > 0.0:
            scale = budget / cost
            for key in inclusion:
                inclusion[key] *= scale

    def _eviction_candidates(
        self,
        augmented: frozenset[str],
        protected: frozenset[str],
        sensitivity: dict[str, float],
        graph: EvidenceGraph,
    ) -> list[frozenset[str]]:
        excess = self._storage_cost(augmented, graph) - self.config.storage_budget
        if excess <= 0:
            return [frozenset()]
        removable = sorted(
            augmented.difference(protected),
            key=lambda key: (sensitivity.get(key, 0.0), key),
        )
        candidates: set[frozenset[str]] = set()
        greedy: list[str] = []
        freed = 0
        for key in removable:
            greedy.append(key)
            freed += graph.records[key].storage_cost
            if freed >= excess:
                candidates.add(frozenset(greedy))
                break

        pool = removable[: min(12, len(removable))]
        for size in range(1, min(self.config.max_eviction_group, len(pool)) + 1):
            for combination in itertools.combinations(pool, size):
                if (
                    sum(graph.records[key].storage_cost for key in combination)
                    >= excess
                ):
                    candidates.add(frozenset(combination))
                    if len(candidates) >= self.config.eviction_shortlist_size * 4:
                        break
            if len(candidates) >= self.config.eviction_shortlist_size * 4:
                break
        return sorted(
            candidates,
            key=lambda group: (
                sum(sensitivity.get(key, 0.0) for key in sorted(group)),
                self._storage_cost(group, graph),
                sorted(group),
            ),
        )[: self.config.eviction_shortlist_size]

    @staticmethod
    def _storage_cost(record_ids: Iterable[str], graph: EvidenceGraph) -> int:
        return sum(graph.records[key].storage_cost for key in record_ids)
