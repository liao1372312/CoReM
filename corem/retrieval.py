"""Dependency-complete bundle beam search for CoReM."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from .model import SetUtilityModel
from .types import EvidenceGraph, Request, RetrievalResult


@dataclass(frozen=True, slots=True)
class Bundle:
    bundle_id: str
    members: frozenset[str]


@dataclass(slots=True)
class RetrievalConfig:
    retrieval_budget: int
    beam_width: int = 8
    max_depth: int = 4
    candidate_pool_size: int = 50
    dependency_hops: int = 2
    max_closure_size: int = 16
    max_witness_paths: int = 128
    enforce_dependency_closure: bool = True
    require_hyperedge_seed: bool = False
    context_cost_weight: float = 0.05
    uncovered_penalty: float = 0.5
    fallback_value: float = 0.0
    tie_seed: int = 2027

    def __post_init__(self) -> None:
        if self.retrieval_budget <= 0:
            raise ValueError("retrieval_budget must be positive")
        if (
            self.beam_width <= 0
            or self.max_depth <= 0
            or self.candidate_pool_size <= 0
            or self.dependency_hops <= 0
            or self.max_closure_size <= 0
            or self.max_witness_paths <= 0
        ):
            raise ValueError("search limits must be positive")
        if self.context_cost_weight < 0.0 or self.uncovered_penalty < 0.0:
            raise ValueError("retrieval penalty weights must be non-negative")
        if not 0.0 <= self.fallback_value <= 1.0:
            raise ValueError("fallback_value must be in [0, 1]")


class BundleBeamRetriever:
    """Bounded feasible-set search for Eq. (9), with an explicit fallback."""

    def __init__(self, model: SetUtilityModel, config: RetrievalConfig) -> None:
        self.model = model
        self.config = config

    def retrieve(
        self,
        request: Request,
        bank_ids: Iterable[str],
        graph: EvidenceGraph,
    ) -> RetrievalResult:
        bank = frozenset(bank_ids)
        graph._require_records(bank)
        bundles = self.build_bundles(request, bank, graph)
        probability_cache: dict[frozenset[str], float] = {}
        beam_score_cache: dict[frozenset[str], float] = {}

        def probabilities(states: list[frozenset[str]]) -> None:
            missing = [state for state in states if state not in probability_cache]
            if not missing:
                return
            batched = getattr(self.model, "probabilities", None)
            values = (
                batched(request, missing, graph)
                if callable(batched)
                else tuple(
                    self.model.probability(request, state, graph)
                    for state in missing
                )
            )
            if len(values) != len(missing):
                raise ValueError("batched probability scorer returned the wrong length")
            probability_cache.update(zip(missing, values))

        def beam_scores(states: list[frozenset[str]]) -> None:
            missing = [state for state in states if state not in beam_score_cache]
            probabilities(missing)
            required_count = max(1, len(request.required_atoms))
            for state in missing:
                objective = self._objective(
                    request, state, graph, probability_cache[state]
                )
                deficit = len(graph.uncovered_atoms(request, state)) / required_count
                beam_score_cache[state] = (
                    objective - self.config.uncovered_penalty * deficit
                )

        empty = frozenset()
        beam_scores([empty])
        beam: dict[frozenset[str], float] = {empty: beam_score_cache[empty]}
        best: tuple[float, frozenset[str], float] | None = None
        evaluations = 1

        if self._is_feasible(request, empty, graph):
            probability = probability_cache[empty]
            best = (
                self._objective(request, empty, graph, probability),
                empty,
                probability,
            )

        for _ in range(self.config.max_depth):
            candidates = dict(beam)
            expanded_states: list[frozenset[str]] = []
            seen_expansions: set[frozenset[str]] = set()
            parents: dict[frozenset[str], list[frozenset[str]]] = {}
            for selected in beam:
                for bundle in bundles:
                    expanded = selected.union(bundle.members)
                    if expanded == selected or not expanded.issubset(bank):
                        continue
                    if not self._within_budget(expanded, graph):
                        continue
                    if not graph.is_compatible(expanded, request):
                        continue
                    if any(not graph.is_admissible(key, request) for key in expanded):
                        continue
                    parents.setdefault(expanded, []).append(selected)
                    if expanded not in seen_expansions:
                        seen_expansions.add(expanded)
                        expanded_states.append(expanded)
            unseen = [state for state in expanded_states if state not in beam_score_cache]
            if not unseen:
                break
            beam_scores(unseen)
            evaluations += len(unseen)
            density = {
                expanded: max(
                    (
                        self._objective(request, expanded, graph, probability_cache[expanded])
                        - self._objective(request, selected, graph, probability_cache[selected])
                    )
                    / max(1, self._token_cost(expanded, graph) - self._token_cost(selected, graph))
                    for selected in parents[expanded]
                )
                for expanded in expanded_states
            }
            for expanded in sorted(expanded_states, key=lambda state: density[state], reverse=True):
                score = beam_score_cache[expanded]
                if score > candidates.get(expanded, float("-inf")):
                    candidates[expanded] = score
                if self._is_feasible(request, expanded, graph):
                    probability = probability_cache[expanded]
                    objective = self._objective(
                        request, expanded, graph, probability
                    )
                    if best is None or (
                        objective,
                        -self._token_cost(expanded, graph),
                        self._tie_rank(request, expanded),
                    ) > (
                        best[0],
                        -self._token_cost(best[1], graph),
                        self._tie_rank(request, best[1]),
                    ):
                        best = (objective, expanded, probability)
            beam = dict(
                sorted(
                    candidates.items(),
                    key=lambda item: (
                        item[1],
                        density.get(item[0], float("-inf")),
                        -self._token_cost(item[0], graph),
                        self._tie_rank(request, item[0]),
                    ),
                    reverse=True,
                )[: self.config.beam_width]
            )

        if best is None:
            return RetrievalResult(
                selected_ids=frozenset(),
                probability=self.config.fallback_value,
                objective=self.config.fallback_value,
                token_cost=0,
                status="fallback",
                uncovered_atoms=graph.uncovered_atoms(request, frozenset()),
                evaluations=evaluations,
            )
        objective, selected, probability = best
        return RetrievalResult(
            selected_ids=selected,
            probability=probability,
            objective=objective,
            token_cost=self._token_cost(selected, graph),
            status="ok",
            uncovered_atoms=frozenset(),
            evaluations=evaluations,
        )

    def build_bundles(
        self,
        request: Request,
        bank: frozenset[str],
        graph: EvidenceGraph,
    ) -> tuple[Bundle, ...]:
        """Semantic shortlist followed by grounded dependency expansion."""
        eligible = [key for key in bank if graph.is_admissible(key, request)]
        ranked = sorted(
            eligible,
            key=lambda key: (
                self.model.semantic_similarity(request, key, graph),
                self._tie_rank(request, frozenset((key,))),
            ),
            reverse=True,
        )
        shortlist = set(ranked[: self.config.candidate_pool_size])
        by_members: dict[frozenset[str], Bundle] = {}
        if not self.config.require_hyperedge_seed:
            for key in sorted(shortlist):
                for members in self._witness_paths(
                    request, frozenset((key,)), bank, graph
                ):
                    by_members[members] = Bundle(f"record:{key}", members)
        for edge in graph.hyperedges.values():
            if not graph.is_hyperedge_admissible(edge, request):
                continue
            if not edge.members.issubset(bank) or not edge.members.intersection(
                shortlist
            ):
                continue
            if edge.evidence_atoms and not edge.evidence_atoms.intersection(
                request.required_atoms
            ):
                continue
            for members in self._witness_paths(request, edge.members, bank, graph):
                by_members[members] = Bundle(f"dependency:{edge.edge_id}", members)
        return tuple(
            sorted(
                by_members.values(),
                key=lambda bundle: (len(bundle.members), bundle.bundle_id),
            )
        )

    def _witness_paths(
        self,
        request: Request,
        seed: frozenset[str],
        bank: frozenset[str],
        graph: EvidenceGraph,
    ) -> tuple[frozenset[str], ...]:
        """Enumerate bounded verified witness paths, keeping alternatives apart."""
        if (
            not seed.issubset(bank)
            or len(seed) > self.config.max_closure_size
            or not graph.is_compatible(seed, request)
            or any(not graph.is_admissible(key, request) for key in seed)
        ):
            return ()
        seen = {seed}
        frontier = [seed]
        if self.config.enforce_dependency_closure:
            edges = sorted(graph.hyperedges.values(), key=lambda edge: edge.edge_id)
            for _ in range(self.config.dependency_hops):
                next_frontier: list[frozenset[str]] = []
                for current in frontier:
                    for edge in edges:
                        if (
                            not edge.members.intersection(current)
                            or not edge.members.issubset(bank)
                            or not graph.is_hyperedge_admissible(edge, request)
                            or not edge.evidence_atoms.intersection(
                                request.required_atoms.difference(request.verified_atoms)
                            )
                        ):
                            continue
                        expanded = current.union(edge.members)
                        if (
                            expanded in seen
                            or len(expanded) > self.config.max_closure_size
                            or not graph.is_compatible(expanded, request)
                            or any(not graph.is_admissible(key, request) for key in expanded)
                        ):
                            continue
                        seen.add(expanded)
                        next_frontier.append(expanded)
                        if len(seen) >= self.config.max_witness_paths:
                            return tuple(sorted(seen, key=lambda path: (len(path), sorted(path))))
                frontier = next_frontier
                if not frontier:
                    break
        return tuple(sorted(seen, key=lambda path: (len(path), sorted(path))))

    def _beam_score(
        self, request: Request, selected: frozenset[str], graph: EvidenceGraph
    ) -> float:
        probability = self.model.probability(request, selected, graph)
        score = self._objective(request, selected, graph, probability)
        required_count = max(1, len(request.required_atoms))
        deficit = len(graph.uncovered_atoms(request, selected)) / required_count
        return score - self.config.uncovered_penalty * deficit

    def _objective(
        self,
        request: Request,
        selected: frozenset[str],
        graph: EvidenceGraph,
        probability: float | None = None,
    ) -> float:
        if probability is None:
            probability = self.model.probability(request, selected, graph)
        normalized_cost = (
            self._token_cost(selected, graph) / self.config.retrieval_budget
        )
        return probability - self.config.context_cost_weight * normalized_cost

    def _is_feasible(
        self, request: Request, selected: frozenset[str], graph: EvidenceGraph
    ) -> bool:
        return (
            self._within_budget(selected, graph)
            and graph.is_compatible(selected, request)
            and all(graph.is_admissible(key, request) for key in selected)
            and graph.is_evidence_complete(request, selected)
        )

    def _within_budget(self, selected: frozenset[str], graph: EvidenceGraph) -> bool:
        return self._token_cost(selected, graph) <= self.config.retrieval_budget

    @staticmethod
    def _token_cost(selected: Iterable[str], graph: EvidenceGraph) -> int:
        return sum(graph.records[key].token_cost for key in selected)

    def _tie_rank(self, request: Request, selected: frozenset[str]) -> int:
        payload = "\0".join(
            (str(self.config.tie_seed), request.request_id, *sorted(selected))
        )
        return int.from_bytes(
            hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
        )
