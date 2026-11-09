"""End-to-end neural set-utility model backed by :class:`TorchHGTEncoder`."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .encoding import Vector, cosine
from .model import FitResult, TrainingConfig
from .torch_encoding import TorchHGTEncoder
from .types import EvidenceGraph, OutcomeExample, Request


@dataclass(slots=True)
class TorchSetUtilityModel:
    """Paper-aligned non-additive model used directly by both optimizers."""

    encoder: TorchHGTEncoder = field(default_factory=TorchHGTEncoder)
    _cache_graph: EvidenceGraph | None = field(default=None, init=False, repr=False)
    _cache_revision: int = field(default=-1, init=False, repr=False)
    _graph_vectors: dict[str, Vector] = field(
        default_factory=dict, init=False, repr=False
    )

    def graph_vectors(self, graph: EvidenceGraph) -> dict[str, Vector]:
        if self._cache_graph is not graph or self._cache_revision != graph.revision:
            self._graph_vectors = self.encoder.encode_graph(graph)
            self._cache_graph = graph
            self._cache_revision = graph.revision
        return self._graph_vectors

    def semantic_similarity(
        self, request: Request, record_id: str, graph: EvidenceGraph
    ) -> float:
        query = self.encoder.encode_request(request)
        vectors = self.graph_vectors(graph)
        conditioned = self.encoder.condition_on_request(
            request, record_id, graph, vectors
        )
        return cosine(query, conditioned)

    def logit(
        self, request: Request, selected_ids: Iterable[str], graph: EvidenceGraph
    ) -> float:
        selected = frozenset(selected_ids)
        return float(self.encoder.auxiliary_logit(request, selected, graph))

    def probability(
        self, request: Request, selected_ids: Iterable[str], graph: EvidenceGraph
    ) -> float:
        selected = frozenset(selected_ids)
        return self.encoder.auxiliary_probability(request, selected, graph)

    def probabilities(
        self,
        request: Request,
        selected_sets: Sequence[frozenset[str]],
        graph: EvidenceGraph,
    ) -> tuple[float, ...]:
        """Score multiple search states with one device synchronization."""
        return self.encoder.auxiliary_probabilities(request, selected_sets, graph)

    def relaxed_probability_and_gradient(
        self,
        request: Request,
        inclusion: dict[str, float],
        graph: EvidenceGraph,
    ) -> tuple[float, dict[str, float]]:
        return self.encoder.relaxed_probability_and_gradient(request, inclusion, graph)

    def fit(
        self,
        examples: Sequence[OutcomeExample],
        graph: EvidenceGraph,
        config: TrainingConfig | None = None,
    ) -> FitResult:
        eligible = [
            example
            for example in examples
            if example.actually_executed
            and example.independently_verified
            and example.eligible_for_training
        ]
        if not eligible:
            raise ValueError("at least one verified training example is required")
        for example in eligible:
            graph._require_records(example.selected_ids)
            if any(
                not graph.is_admissible(record_id, example.request)
                for record_id in example.selected_ids
            ):
                raise ValueError(
                    "training examples must contain only admissible records"
                )
            if not graph.is_evidence_complete(example.request, example.selected_ids):
                raise ValueError("training examples must be evidence-complete")
        result = self.encoder.fit_supervised(
            eligible, graph, config or TrainingConfig()
        )
        self._cache_graph = None
        self._cache_revision = -1
        self._graph_vectors.clear()
        return FitResult(result.losses, False, result.epochs)
