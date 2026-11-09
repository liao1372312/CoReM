"""Non-additive, request-conditioned set-utility model."""

from __future__ import annotations

import itertools
import math
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .encoding import Encoder, StableHashEncoder, Vector, cosine
from .types import (
    DependencyHyperedge,
    EvidenceGraph,
    MemoryRecord,
    OutcomeExample,
    Request,
)


def _dot(weights: list[float], features: tuple[float, ...]) -> float:
    return sum(weight * feature for weight, feature in zip(weights, features))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _softplus(value: float) -> float:
    if value > 30.0:
        return value
    if value < -30.0:
        return math.exp(value)
    return math.log1p(math.exp(value))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left.union(right)
    return len(left.intersection(right)) / len(union) if union else 0.0


@dataclass(slots=True)
class TrainingConfig:
    learning_rate: float = 0.05
    epochs: int = 200
    calibration_weight: float = 0.1
    relation_weight: float = 0.1
    interaction_weight: float = 0.001
    l2_weight: float = 0.0001
    tolerance: float = 1e-6
    patience: int = 8
    gradient_clip: float = 10.0
    class_balance_power: float = 0.0
    ranking_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or self.epochs <= 0:
            raise ValueError("learning_rate and epochs must be positive")
        if self.patience <= 0 or self.gradient_clip <= 0.0:
            raise ValueError("patience and gradient_clip must be positive")
        if not 0.0 <= self.class_balance_power <= 1.0:
            raise ValueError("class_balance_power must be between 0 and 1")
        regularizers = (
            self.calibration_weight,
            self.relation_weight,
            self.interaction_weight,
            self.l2_weight,
            self.tolerance,
            self.ranking_weight,
        )
        if any(value < 0.0 for value in regularizers):
            raise ValueError("loss weights and tolerance must be non-negative")


@dataclass(frozen=True, slots=True)
class FitResult:
    losses: tuple[float, ...]
    converged: bool
    epochs: int


@dataclass(slots=True)
class SetUtilityModel:
    """Implements Eq. (5): unary + complement - redundancy utility."""

    encoder: Encoder = field(default_factory=StableHashEncoder)
    base_weights: list[float] = field(default_factory=lambda: [-1.25, -0.10])
    unary_weights: list[float] = field(
        default_factory=lambda: [-0.10, 1.25, 1.35, 0.45, 0.20, 0.10]
    )
    complement_weights: list[float] = field(
        default_factory=lambda: [-1.00, 0.50, 1.50, 0.25, 0.10]
    )
    redundancy_weights: list[float] = field(
        default_factory=lambda: [-1.00, 1.50, 1.00, 0.75]
    )
    relation_heads: dict[str, list[float]] = field(default_factory=dict)
    _cache_graph: EvidenceGraph | None = field(default=None, init=False, repr=False)
    _cache_revision: int = field(default=-1, init=False, repr=False)
    _graph_vectors: dict[str, Vector] = field(
        default_factory=dict, init=False, repr=False
    )
    _degree_graph: EvidenceGraph | None = field(default=None, init=False, repr=False)
    _degree_revision: int = field(default=-1, init=False, repr=False)
    _degrees: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _semantic_cache: OrderedDict[tuple[Request, str], float] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _semantic_cache_graph: EvidenceGraph | None = field(
        default=None, init=False, repr=False
    )
    _semantic_cache_signature: tuple[object, ...] | None = field(
        default=None, init=False, repr=False
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
        # Only the deterministic encoder is cached: neural/custom encoders may
        # update parameters without changing the graph revision.
        cacheable = type(self.encoder) is StableHashEncoder
        if cacheable:
            encoder = self.encoder
            signature = (
                graph.revision,
                id(encoder),
                encoder.dimension,
                encoder.message_layers,
                encoder.residual_weight,
                encoder.cross_attention_weight,
            )
            if (
                self._semantic_cache_graph is not graph
                or self._semantic_cache_signature != signature
            ):
                self._semantic_cache.clear()
                self._semantic_cache_graph = graph
                self._semantic_cache_signature = signature
            key = (request, record_id)
            if key in self._semantic_cache:
                self._semantic_cache.move_to_end(key)
                return self._semantic_cache[key]
        query = self.encoder.encode_request(request)
        vectors = self.graph_vectors(graph)
        condition = getattr(self.encoder, "condition_on_request", None)
        record = (
            condition(request, record_id, graph, vectors)
            if callable(condition)
            else vectors[record_id]
        )
        similarity = cosine(query, record)
        if cacheable:
            self._semantic_cache[key] = similarity
            if len(self._semantic_cache) > 8192:
                self._semantic_cache.popitem(last=False)
        return similarity

    def logit(
        self, request: Request, selected_ids: Iterable[str], graph: EvidenceGraph
    ) -> float:
        selected = frozenset(selected_ids)
        graph._require_records(selected)
        value = _dot(self.base_weights, self._base_features(request))
        for record_id in selected:
            value += _dot(
                self.unary_weights,
                self._unary_features(request, graph.records[record_id], graph),
            )
        for edge in self._eligible_hyperedges(request, graph):
            if edge.members.issubset(selected):
                raw = _dot(
                    self.complement_weights,
                    self._complement_features(request, edge, graph),
                )
                value += _softplus(raw)
        for left, right in graph.duplicate_pairs:
            if left in selected and right in selected:
                raw = _dot(
                    self.redundancy_weights,
                    self._redundancy_features(request, left, right, graph),
                )
                value -= _softplus(raw)
        return value

    def probability(
        self, request: Request, selected_ids: Iterable[str], graph: EvidenceGraph
    ) -> float:
        return _sigmoid(self.logit(request, selected_ids, graph))

    def relaxed_probability_and_gradient(
        self,
        request: Request,
        inclusion: dict[str, float],
        graph: EvidenceGraph,
    ) -> tuple[float, dict[str, float]]:
        """Multilinear extension and d probability / d inclusion."""
        graph._require_records(inclusion)
        x = {key: min(1.0, max(0.0, value)) for key, value in inclusion.items()}
        logit = _dot(self.base_weights, self._base_features(request))
        dlogit = {key: 0.0 for key in x}
        for key, inclusion_probability in x.items():
            unary = _dot(
                self.unary_weights,
                self._unary_features(request, graph.records[key], graph),
            )
            logit += inclusion_probability * unary
            dlogit[key] += unary
        for edge in self._eligible_hyperedges(request, graph):
            if not edge.members.issubset(x):
                continue
            coefficient = _softplus(
                _dot(
                    self.complement_weights,
                    self._complement_features(request, edge, graph),
                )
            )
            product = math.prod(x[key] for key in edge.members)
            logit += coefficient * product
            for key in edge.members:
                other_product = math.prod(
                    x[other] for other in edge.members if other != key
                )
                dlogit[key] += coefficient * other_product
        for left, right in graph.duplicate_pairs:
            if left not in x or right not in x:
                continue
            coefficient = _softplus(
                _dot(
                    self.redundancy_weights,
                    self._redundancy_features(request, left, right, graph),
                )
            )
            logit -= coefficient * x[left] * x[right]
            dlogit[left] -= coefficient * x[right]
            dlogit[right] -= coefficient * x[left]
        probability = _sigmoid(logit)
        scale = probability * (1.0 - probability)
        return probability, {key: scale * value for key, value in dlogit.items()}

    def fit(
        self,
        examples: Sequence[OutcomeExample],
        graph: EvidenceGraph,
        config: TrainingConfig | None = None,
    ) -> FitResult:
        """Fit observed set outcomes with full-batch gradient descent."""
        eligible_examples = [
            example
            for example in examples
            if example.actually_executed
            and example.independently_verified
            and example.eligible_for_training
        ]
        if not eligible_examples:
            raise ValueError("at least one training example is required")
        cfg = config or TrainingConfig()
        for example in eligible_examples:
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
        encoder_fit = getattr(self.encoder, "fit_supervised", None)
        if callable(encoder_fit):
            encoder_fit(eligible_examples, graph, cfg)
            self._cache_graph = None
            self._cache_revision = -1
            self._graph_vectors.clear()
        relation_observations = self._relation_observations(graph)
        self._ensure_relation_heads(
            {relation_type for _, _, relation_type in relation_observations}
        )
        losses: list[float] = []
        stable_epochs = 0
        converged = False
        for _ in range(cfg.epochs):
            gradients = {
                "base": [0.0] * len(self.base_weights),
                "unary": [0.0] * len(self.unary_weights),
                "complement": [0.0] * len(self.complement_weights),
                "redundancy": [0.0] * len(self.redundancy_weights),
            }
            loss = 0.0
            for example in eligible_examples:
                logit = self.logit(example.request, example.selected_ids, graph)
                probability = _sigmoid(logit)
                outcome = float(example.outcome)
                loss += (
                    max(logit, 0.0)
                    - outcome * logit
                    + math.log1p(math.exp(-abs(logit)))
                )
                loss += cfg.calibration_weight * (probability - outcome) ** 2
                chain = (probability - outcome) + (
                    2.0
                    * cfg.calibration_weight
                    * (probability - outcome)
                    * probability
                    * (1.0 - probability)
                )
                self._accumulate_logit_gradient(gradients, chain, example, graph)
            count = float(len(eligible_examples))
            loss /= count
            interaction_examples = [
                example
                for example in eligible_examples
                if example.eligible_for_interaction_training
            ]
            interaction_edges: dict[str, DependencyHyperedge] = {}
            interaction_support: dict[str, OutcomeExample] = {}
            for example in interaction_examples:
                for edge in self._eligible_hyperedges(example.request, graph):
                    interaction_edges.setdefault(edge.edge_id, edge)
                    interaction_support.setdefault(edge.edge_id, example)
            if not interaction_examples:
                # With no set-level interaction supervision, do not evaluate
                # request-scoped hyperedges against an unrelated fallback
                # request. Global L2 regularization still shrinks the shared
                # complement parameters toward zero.
                interaction_edges = {}
            for edge in interaction_edges.values():
                # The support map is constructed in the same example order as
                # the former linear search, but avoids rescanning every graph
                # hyperedge for every (edge, epoch) pair.
                supporting = interaction_support.get(
                    edge.edge_id, eligible_examples[0]
                )
                features = self._complement_features(supporting.request, edge, graph)
                raw = _dot(self.complement_weights, features)
                loss += cfg.interaction_weight * _softplus(raw)
                for index, feature in enumerate(features):
                    gradients["complement"][index] += (
                        count * cfg.interaction_weight * _sigmoid(raw) * feature
                    )
            relation_loss = self._update_relation_heads(
                relation_observations, cfg, graph
            )
            loss += cfg.relation_weight * relation_loss
            parameters = self._parameter_groups()
            for name, weights in parameters.items():
                for index, weight in enumerate(weights):
                    gradient = gradients[name][index] / count + cfg.l2_weight * weight
                    gradient = max(-cfg.gradient_clip, min(cfg.gradient_clip, gradient))
                    weights[index] -= cfg.learning_rate * gradient
                    loss += 0.5 * cfg.l2_weight * weight * weight
            losses.append(loss)
            if len(losses) > 1:
                denominator = max(1.0, abs(losses[-2]))
                relative_change = abs(losses[-1] - losses[-2]) / denominator
                stable_epochs = (
                    stable_epochs + 1 if relative_change < cfg.tolerance else 0
                )
                if stable_epochs >= cfg.patience:
                    converged = True
                    break
        return FitResult(tuple(losses), converged, len(losses))

    def _accumulate_logit_gradient(
        self,
        gradients: dict[str, list[float]],
        chain: float,
        example: OutcomeExample,
        graph: EvidenceGraph,
    ) -> None:
        def add(name: str, features: tuple[float, ...], scale: float) -> None:
            for index, feature in enumerate(features):
                gradients[name][index] += scale * feature

        add("base", self._base_features(example.request), chain)
        for record_id in example.selected_ids:
            add(
                "unary",
                self._unary_features(example.request, graph.records[record_id], graph),
                chain,
            )
        if example.eligible_for_interaction_training:
            for edge in self._eligible_hyperedges(example.request, graph):
                if edge.members.issubset(example.selected_ids):
                    features = self._complement_features(example.request, edge, graph)
                    raw = _dot(self.complement_weights, features)
                    add("complement", features, chain * _sigmoid(raw))
            for left, right in graph.duplicate_pairs:
                if left in example.selected_ids and right in example.selected_ids:
                    features = self._redundancy_features(
                        example.request, left, right, graph
                    )
                    raw = _dot(self.redundancy_weights, features)
                    add("redundancy", features, -chain * _sigmoid(raw))

    def _parameter_groups(self) -> dict[str, list[float]]:
        return {
            "base": self.base_weights,
            "unary": self.unary_weights,
            "complement": self.complement_weights,
            "redundancy": self.redundancy_weights,
        }

    @staticmethod
    def _base_features(request: Request) -> tuple[float, ...]:
        return (1.0, min(1.0, len(request.required_atoms) / 8.0))

    def _unary_features(
        self, request: Request, record: MemoryRecord, graph: EvidenceGraph
    ) -> tuple[float, ...]:
        similarity = self.semantic_similarity(request, record.record_id, graph)
        required = len(request.required_atoms)
        coverage = (
            len(record.evidence_atoms.intersection(request.required_atoms)) / required
            if required
            else 0.0
        )
        operation_overlap = _jaccard(record.operation_ids, request.operation_ids)
        age = max(0.0, request.timestamp - record.timestamp)
        recency = (
            math.exp(-age / 100.0) if request.timestamp and record.timestamp else 1.0
        )
        if self._degree_graph is not graph or self._degree_revision != graph.revision:
            degrees = {key: 0 for key in graph.records}
            for edge in graph.edges:
                if edge.source in degrees:
                    degrees[edge.source] += 1
                if edge.target in degrees:
                    degrees[edge.target] += 1
            self._degree_graph = graph
            self._degree_revision = graph.revision
            self._degrees = degrees
        degree = self._degrees.get(record.record_id, 0)
        return (
            1.0,
            similarity,
            coverage,
            operation_overlap,
            recency,
            min(1.0, degree / 10.0),
        )

    def _complement_features(
        self, request: Request, edge: DependencyHyperedge, graph: EvidenceGraph
    ) -> tuple[float, ...]:
        similarities = [
            self.semantic_similarity(request, key, graph) for key in edge.members
        ]
        covered: set[str] = set()
        for key in edge.members:
            covered.update(graph.records[key].evidence_atoms)
        required = len(request.required_atoms)
        coverage = (
            len(covered.intersection(request.required_atoms)) / required
            if required
            else 0.0
        )
        relation_hash = sum(ord(char) for char in edge.relation_type) % 101 / 100.0
        relation_support = self._hyperedge_relation_support(edge, graph)
        return (
            1.0,
            sum(similarities) / len(similarities),
            coverage,
            min(1.0, len(edge.members) / 4.0),
            0.5 * relation_hash + 0.5 * relation_support,
        )

    def _redundancy_features(
        self, request: Request, left: str, right: str, graph: EvidenceGraph
    ) -> tuple[float, ...]:
        records = graph.records
        vectors = self.graph_vectors(graph)
        condition = getattr(self.encoder, "condition_on_request", None)
        left_vector = (
            condition(request, left, graph, vectors)
            if callable(condition)
            else vectors[left]
        )
        right_vector = (
            condition(request, right, graph, vectors)
            if callable(condition)
            else vectors[right]
        )
        similarity = cosine(left_vector, right_vector)
        return (
            1.0,
            similarity,
            _jaccard(records[left].evidence_atoms, records[right].evidence_atoms),
            _jaccard(records[left].operation_ids, records[right].operation_ids),
        )

    def _relation_observations(
        self, graph: EvidenceGraph
    ) -> list[tuple[str, str, str]]:
        observations = [
            (edge.source, edge.target, edge.relation_type)
            for edge in graph.edges
            if edge.verified
        ]
        for edge in graph.hyperedges.values():
            observations.extend(
                (left, right, edge.relation_type)
                for left, right in itertools.combinations(sorted(edge.members), 2)
            )
        return observations

    def _ensure_relation_heads(self, relation_types: set[str]) -> None:
        for relation_type in sorted(relation_types):
            self.relation_heads.setdefault(relation_type, [0.0, 0.0, 0.0, 0.0])

    def _relation_features(
        self, source: str, target: str, graph: EvidenceGraph
    ) -> tuple[float, ...]:
        source_known = source in graph.records or source in graph.entities
        target_known = target in graph.records or target in graph.entities
        if not source_known or not target_known:
            return (1.0, 0.0, 0.0, 0.0)
        encoder_vectors = getattr(self.encoder, "_all_vectors", {})
        if source not in encoder_vectors or target not in encoder_vectors:
            self.graph_vectors(graph)
            encoder_vectors = getattr(self.encoder, "_all_vectors", {})
        similarity = (
            cosine(encoder_vectors[source], encoder_vectors[target])
            if source in encoder_vectors and target in encoder_vectors
            else 0.0
        )
        both_memories = float(source in graph.records and target in graph.records)
        evidence_overlap = 0.0
        if both_memories:
            evidence_overlap = _jaccard(
                graph.records[source].evidence_atoms,
                graph.records[target].evidence_atoms,
            )
        return (1.0, similarity, both_memories, evidence_overlap)

    def _relation_probabilities(
        self, source: str, target: str, graph: EvidenceGraph
    ) -> dict[str, float]:
        if not self.relation_heads:
            return {}
        features = self._relation_features(source, target, graph)
        logits = {
            key: _dot(weights, features) for key, weights in self.relation_heads.items()
        }
        maximum = max(logits.values())
        exponentials = {key: math.exp(value - maximum) for key, value in logits.items()}
        total = sum(exponentials.values())
        return {key: value / total for key, value in exponentials.items()}

    def _hyperedge_relation_support(
        self, edge: DependencyHyperedge, graph: EvidenceGraph
    ) -> float:
        pairs = list(itertools.combinations(sorted(edge.members), 2))
        if not pairs or edge.relation_type not in self.relation_heads:
            return 0.5
        return sum(
            self._relation_probabilities(left, right, graph).get(
                edge.relation_type, 0.0
            )
            for left, right in pairs
        ) / len(pairs)

    def _update_relation_heads(
        self,
        observations: Sequence[tuple[str, str, str]],
        config: TrainingConfig,
        graph: EvidenceGraph,
    ) -> float:
        if (
            not observations
            or len(self.relation_heads) < 2
            or config.relation_weight <= 0.0
        ):
            return 0.0
        gradients = {key: [0.0] * 4 for key in self.relation_heads}
        loss = 0.0
        for source, target, truth in observations:
            features = self._relation_features(source, target, graph)
            probabilities = self._relation_probabilities(source, target, graph)
            loss -= math.log(max(1e-12, probabilities[truth]))
            for relation_type, probability in probabilities.items():
                chain = probability - float(relation_type == truth)
                for index, feature in enumerate(features):
                    gradients[relation_type][index] += (
                        chain * feature / len(observations)
                    )
        for relation_type, weights in self.relation_heads.items():
            for index, weight in enumerate(weights):
                gradient = config.relation_weight * gradients[relation_type][index]
                gradient += config.l2_weight * weight
                weights[index] -= config.learning_rate * max(
                    -config.gradient_clip, min(config.gradient_clip, gradient)
                )
        return loss / len(observations)

    @staticmethod
    def _eligible_hyperedges(
        request: Request, graph: EvidenceGraph
    ) -> list[DependencyHyperedge]:
        return [
            edge
            for edge in graph.hyperedges.values()
            if graph.is_hyperedge_admissible(edge, request)
            and (
                not edge.evidence_atoms
                or edge.evidence_atoms.intersection(request.required_atoms)
            )
        ]
