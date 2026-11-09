"""Optional trainable HGT and request cross-attention backend.

This module is intentionally separate from :mod:`corem.encoding` so importing
the dependency-free reference implementation never requires PyTorch or PyG.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .encoding import Vector
from .types import (
    DependencyHyperedge,
    EvidenceEdge,
    EvidenceGraph,
    MemoryRecord,
    OutcomeExample,
    Request,
)

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
    from torch_geometric.nn import HGTConv
except ImportError as error:  # pragma: no cover - exercised only without the ML extra
    raise ImportError(
        "TorchHGTEncoder requires PyTorch and torch-geometric; "
        "create the environment with `conda env create -f environment.yml`."
    ) from error

_TOKEN_RE = re.compile(r"[\w./:-]+", re.UNICODE)
_WORD_RE = re.compile(r"\w+", re.UNICODE)
EdgeType = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class TorchEncoderFitResult:
    """Training trace for the optional neural encoder."""

    losses: tuple[float, ...]
    epochs: int


class TorchHGTEncoder(nn.Module):
    """Trainable HGT with request-conditioned neighborhood cross-attention.

    Text is initialized with stable feature hashing to keep this backend local
    and reproducible. HGT, attention, relation, and auxiliary non-additive set
    heads are learned from verified observations. The public encoding methods
    implement the same protocol as ``StableHashEncoder``.
    """

    def __init__(
        self,
        dimension: int = 64,
        text_hash_dimension: int | None = None,
        text_feature_mode: str = "tokens",
        message_layers: int = 2,
        attention_heads: int = 4,
        seed: int = 0,
        device: str = "cpu",
        enable_complements: bool = True,
        enable_redundancy: bool = True,
        enable_lexical_features: bool = False,
        enable_graph_context: bool = True,
    ) -> None:
        super().__init__()
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        if text_hash_dimension is not None and text_hash_dimension < 8:
            raise ValueError("text_hash_dimension must be at least 8")
        if text_feature_mode not in {"tokens", "subword"}:
            raise ValueError("text_feature_mode must be 'tokens' or 'subword'")
        if message_layers < 0:
            raise ValueError("message_layers must be non-negative")
        if attention_heads <= 0 or dimension % attention_heads:
            raise ValueError("attention_heads must divide dimension")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")

        self.dimension = dimension
        self.text_hash_dimension = text_hash_dimension or dimension
        self.text_feature_mode = text_feature_mode
        self.message_layers = message_layers
        self.attention_heads = attention_heads
        self.seed = seed
        self.device_name = device
        self.enable_complements = enable_complements
        self.enable_redundancy = enable_redundancy
        self.enable_lexical_features = enable_lexical_features
        self.enable_graph_context = enable_graph_context
        self._device = torch.device(device)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.request_projection = nn.Linear(self.text_hash_dimension, dimension)
            self.cross_attention = nn.MultiheadAttention(
                dimension, attention_heads, batch_first=True
            )
            self.cross_norm = nn.LayerNorm(dimension)
            self.base_head = nn.Linear(dimension + 1, 1)
            self.unary_head = nn.Linear(
                3 * dimension + 4 + int(enable_lexical_features), 1
            )
            self.complement_head = nn.Linear(
                3 * dimension + 3 + 2 * int(enable_lexical_features), 1
            )
            self.redundancy_head = nn.Linear(4 * dimension + 2, 1)

        self.input_projections = nn.ModuleList()
        self.hgt_layers = nn.ModuleList()
        self.hgt_norms = nn.ModuleList()
        self.relation_classifier: nn.Linear | None = None
        self._node_types: tuple[str, ...] = ()
        self._edge_types: tuple[EdgeType, ...] = ()
        self._relation_types: tuple[str, ...] = ()
        self._last_graph: EvidenceGraph | None = None
        self._last_revision = -1
        self._all_tensors: dict[str, Tensor] = {}
        self._all_vectors: dict[str, Vector] = {}
        self._index_graph: EvidenceGraph | None = None
        self._index_revision = -1
        self._adjacency: dict[str, tuple[tuple[str, EvidenceEdge], ...]] = {}
        self._hyperedges_by_record: dict[str, tuple[DependencyHyperedge, ...]] = {}
        self._text_cache: dict[str, Tensor] = {}
        self.last_fit_losses: tuple[float, ...] = ()
        self.to(self._device)

    def _hash_text(self, text: str) -> Tensor:
        cached = self._text_cache.get(text)
        if cached is not None:
            return cached
        values = [0.0] * self.text_hash_dimension
        normalized = text.casefold()
        features = list(_TOKEN_RE.findall(normalized))
        if self.text_feature_mode == "subword":
            words = _WORD_RE.findall(normalized)
            features.extend(f"word:{word}" for word in words)
            features.extend(
                f"bigram:{left}|{right}" for left, right in itertools.pairwise(words)
            )
            for word in words:
                if 4 <= len(word) <= 32:
                    bounded = f"^{word}$"
                    features.extend(
                        f"char3:{bounded[index : index + 3]}"
                        for index in range(len(bounded) - 2)
                    )
        for token in features:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self.text_hash_dimension
            values[index] += 1.0 if digest[8] & 1 else -1.0
        tensor = torch.tensor(values, dtype=torch.float32, device=self._device)
        tensor = F.normalize(tensor, dim=0) if tensor.norm() else tensor
        self._text_cache[text] = tensor
        return tensor

    @staticmethod
    def _record_text(record: MemoryRecord) -> str:
        return " ".join(
            (
                record.text,
                *sorted(record.evidence_atoms),
                *sorted(record.operation_ids),
                *sorted(record.version_scope),
            )
        )

    def _request_tensor(self, request: Request) -> Tensor:
        fields = (
            request.text,
            *sorted(request.required_atoms),
            *sorted(request.operation_ids),
            *sorted(request.active_versions),
        )
        return F.normalize(
            self.request_projection(self._hash_text(" ".join(fields))), dim=0
        )

    def _schema(
        self, graph: EvidenceGraph
    ) -> tuple[tuple[str, ...], tuple[EdgeType, ...]]:
        node_type = {record_id: "memory" for record_id in graph.records}
        node_type.update(
            {
                entity_id: entity.node_type
                for entity_id, entity in graph.entities.items()
            }
        )
        types = tuple(sorted(set(node_type.values())))
        edge_types: set[EdgeType] = set()
        for edge in graph.edges:
            edge_types.add(
                (node_type[edge.source], edge.relation_type, node_type[edge.target])
            )
            edge_types.add(
                (
                    node_type[edge.target],
                    f"{edge.relation_type}:rev",
                    node_type[edge.source],
                )
            )
        for edge in graph.hyperedges.values():
            edge_types.add(("memory", edge.relation_type, "memory"))
        return types, tuple(sorted(edge_types))

    def _ensure_schema(self, graph: EvidenceGraph) -> None:
        node_types, edge_types = self._schema(graph)
        if (node_types, edge_types) == (self._node_types, self._edge_types):
            return
        self._node_types = node_types
        self._edge_types = edge_types
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.input_projections = nn.ModuleList(
                nn.Linear(self.text_hash_dimension, self.dimension) for _ in node_types
            )
            self.hgt_layers = nn.ModuleList(
                HGTConv(
                    self.dimension,
                    self.dimension,
                    (list(node_types), list(edge_types)),
                    heads=self.attention_heads,
                )
                for _ in range(self.message_layers)
            )
            self.hgt_norms = nn.ModuleList(
                nn.ModuleList(nn.LayerNorm(self.dimension) for _ in node_types)
                for _ in range(self.message_layers)
            )
        self.relation_classifier = None
        self._relation_types = ()
        self.to(self._device)

    def _node_layout(
        self, graph: EvidenceGraph
    ) -> tuple[dict[str, list[str]], dict[str, tuple[str, int]]]:
        grouped = {node_type: [] for node_type in self._node_types}
        for record_id in sorted(graph.records):
            grouped["memory"].append(record_id)
        for entity_id, entity in sorted(graph.entities.items()):
            grouped[entity.node_type].append(entity_id)
        lookup = {
            node_id: (node_type, index)
            for node_type, node_ids in grouped.items()
            for index, node_id in enumerate(node_ids)
        }
        return grouped, lookup

    def _edge_index_dict(
        self, graph: EvidenceGraph, lookup: dict[str, tuple[str, int]]
    ) -> dict[EdgeType, Tensor]:
        if not self.enable_graph_context:
            return {}
        pairs: dict[EdgeType, list[tuple[int, int]]] = {
            edge_type: [] for edge_type in self._edge_types
        }

        def add(source: str, relation: str, target: str) -> None:
            source_type, source_index = lookup[source]
            target_type, target_index = lookup[target]
            pairs[(source_type, relation, target_type)].append(
                (source_index, target_index)
            )

        for edge in graph.edges:
            add(edge.source, edge.relation_type, edge.target)
            add(edge.target, f"{edge.relation_type}:rev", edge.source)
        for edge in graph.hyperedges.values():
            for left, right in itertools.combinations(sorted(edge.members), 2):
                add(left, edge.relation_type, right)
                add(right, edge.relation_type, left)
        return {
            edge_type: torch.tensor(values, dtype=torch.long, device=self._device)
            .t()
            .contiguous()
            for edge_type, values in pairs.items()
            if values
        }

    def _forward_graph(self, graph: EvidenceGraph) -> dict[str, Tensor]:
        self._ensure_schema(graph)
        self._ensure_graph_indices(graph)
        grouped, lookup = self._node_layout(graph)
        x_dict: dict[str, Tensor] = {}
        for type_index, node_type in enumerate(self._node_types):
            features = []
            for node_id in grouped[node_type]:
                if node_type == "memory":
                    text = self._record_text(graph.records[node_id])
                else:
                    entity = graph.entities[node_id]
                    text = f"type:{entity.node_type} {entity.text}"
                features.append(self._hash_text(text))
            stacked = torch.stack(features)
            x_dict[node_type] = torch.tanh(self.input_projections[type_index](stacked))

        edge_index = self._edge_index_dict(graph, lookup)
        if edge_index:
            for layer_index, convolution in enumerate(self.hgt_layers):
                messages = convolution(x_dict, edge_index)
                x_dict = {
                    node_type: self.hgt_norms[layer_index][type_index](
                        values
                        + (
                            messages.get(node_type)
                            if messages.get(node_type) is not None
                            else torch.zeros_like(values)
                        )
                    )
                    for type_index, (node_type, values) in enumerate(x_dict.items())
                }
        return {
            node_id: x_dict[node_type][index]
            for node_id, (node_type, index) in lookup.items()
        }

    def _ensure_graph_indices(self, graph: EvidenceGraph) -> None:
        if self._index_graph is graph and self._index_revision == graph.revision:
            return
        adjacency: dict[str, list[tuple[str, EvidenceEdge]]] = {
            key: [] for key in graph.records
        }
        for edge in graph.edges:
            if edge.source in adjacency:
                adjacency[edge.source].append((edge.target, edge))
            if edge.target in adjacency:
                adjacency[edge.target].append((edge.source, edge))
        memberships: dict[str, list[DependencyHyperedge]] = {
            key: [] for key in graph.records
        }
        for edge in graph.hyperedges.values():
            for member in edge.members:
                memberships[member].append(edge)
        self._index_graph = graph
        self._index_revision = graph.revision
        self._adjacency = {
            key: tuple(neighbors) for key, neighbors in adjacency.items()
        }
        self._hyperedges_by_record = {
            key: tuple(edges) for key, edges in memberships.items()
        }

    def encode_request(self, request: Request) -> Vector:
        self.eval()
        with torch.no_grad():
            return tuple(float(value) for value in self._request_tensor(request).cpu())

    def encode_graph(self, graph: EvidenceGraph) -> dict[str, Vector]:
        self.eval()
        with torch.no_grad():
            tensors = self._forward_graph(graph)
        self._last_graph = graph
        self._last_revision = graph.revision
        self._all_tensors = {key: value.detach() for key, value in tensors.items()}
        self._all_vectors = {
            key: tuple(float(value) for value in tensor.cpu())
            for key, tensor in self._all_tensors.items()
        }
        return {key: self._all_vectors[key] for key in graph.records}

    def _conditioned_tensor(
        self,
        request: Request,
        record_id: str,
        graph: EvidenceGraph,
        tensors: dict[str, Tensor],
    ) -> Tensor:
        self._ensure_graph_indices(graph)
        neighborhood = {record_id}
        if self.enable_graph_context:
            for neighbor, edge in self._adjacency.get(record_id, ()):
                if not graph.is_edge_admissible(edge, request):
                    continue
                neighborhood.add(neighbor)
            for edge in self._hyperedges_by_record.get(record_id, ()):
                if graph.is_hyperedge_admissible(edge, request):
                    neighborhood.update(edge.members)
        keys = [key for key in sorted(neighborhood) if key in tensors]
        query = self._request_tensor(request).view(1, 1, -1)
        context = torch.stack([tensors[key] for key in keys]).unsqueeze(0)
        attended, _ = self.cross_attention(query, context, context, need_weights=False)
        return self.cross_norm(tensors[record_id] + attended[0, 0] + query[0, 0])

    def condition_on_request(
        self,
        request: Request,
        record_id: str,
        graph: EvidenceGraph,
        record_vectors: dict[str, Vector],
    ) -> Vector:
        del record_vectors
        if self._last_graph is not graph or self._last_revision != graph.revision:
            self.encode_graph(graph)
        self.eval()
        with torch.no_grad():
            tensor = self._conditioned_tensor(
                request, record_id, graph, self._all_tensors
            )
        return tuple(float(value) for value in tensor.cpu())

    @staticmethod
    def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
        union = left.union(right)
        return len(left.intersection(right)) / len(union) if union else 0.0

    @staticmethod
    def _eligible_hyperedges(
        request: Request, graph: EvidenceGraph
    ) -> list[DependencyHyperedge]:
        eligible = []
        for edge in graph.hyperedges.values():
            if not graph.is_hyperedge_admissible(edge, request):
                continue
            relevant = edge.evidence_atoms.intersection(request.required_atoms)
            if edge.evidence_atoms and not relevant:
                continue
            if relevant and relevant.issubset(request.verified_atoms):
                continue
            eligible.append(edge)
        return eligible

    def _unary_value(
        self,
        request: Request,
        record_id: str,
        graph: EvidenceGraph,
        tensors: dict[str, Tensor],
        query: Tensor,
        conditioned: dict[str, Tensor] | None = None,
    ) -> Tensor:
        record = graph.records[record_id]
        representation = (
            conditioned[record_id]
            if conditioned is not None
            else self._conditioned_tensor(request, record_id, graph, tensors)
        )
        required = len(request.required_atoms)
        coverage = (
            len(record.evidence_atoms.intersection(request.required_atoms)) / required
            if required
            else 0.0
        )
        age = max(0.0, request.timestamp - record.timestamp)
        recency = (
            math.exp(-age / 100.0) if request.timestamp and record.timestamp else 1.0
        )
        self._ensure_graph_indices(graph)
        degree = (
            len(self._adjacency.get(record_id, ())) if self.enable_graph_context else 0
        )
        scalar_values = [
            coverage,
            self._jaccard(record.operation_ids, request.operation_ids),
            recency,
            min(1.0, degree / 10.0),
        ]
        if self.enable_lexical_features:
            scalar_values.append(self._lexical_similarity(request, record))
        scalars = torch.tensor(scalar_values, device=self._device)
        features = torch.cat((query, representation, query * representation, scalars))
        return self.unary_head(features).squeeze()

    def _complement_value(
        self,
        request: Request,
        edge: DependencyHyperedge,
        graph: EvidenceGraph,
        tensors: dict[str, Tensor],
        query: Tensor,
        conditioned: dict[str, Tensor] | None = None,
    ) -> Tensor:
        members = sorted(edge.members)
        pooled = torch.stack(
            [
                conditioned[key]
                if conditioned is not None
                else self._conditioned_tensor(request, key, graph, tensors)
                for key in members
            ]
        ).mean(dim=0)
        covered = set().union(*(graph.records[key].evidence_atoms for key in members))
        required = len(request.required_atoms)
        coverage = (
            len(covered.intersection(request.required_atoms)) / required
            if required
            else 0.0
        )
        relation_hash = (
            sum(ord(character) for character in edge.relation_type) % 101
        ) / 100.0
        scalar_values = [coverage, min(1.0, len(members) / 4.0), relation_hash]
        if self.enable_lexical_features:
            similarities = [
                self._lexical_similarity(request, graph.records[key]) for key in members
            ]
            scalar_values.extend(
                (sum(similarities) / len(similarities), max(similarities))
            )
        scalars = torch.tensor(scalar_values, device=self._device)
        features = torch.cat((query, pooled, query * pooled, scalars))
        return F.softplus(self.complement_head(features).squeeze())

    def _lexical_similarity(self, request: Request, record: MemoryRecord) -> float:
        """Stable raw-text cosine retained alongside learned graph representations."""
        query = self._hash_text(request.text)
        memory = self._hash_text(record.text)
        return float(torch.dot(query, memory).detach().cpu())

    def _redundancy_value(
        self,
        request: Request,
        left: str,
        right: str,
        graph: EvidenceGraph,
        tensors: dict[str, Tensor],
        query: Tensor,
        conditioned: dict[str, Tensor] | None = None,
    ) -> Tensor:
        left_vector = (
            conditioned[left]
            if conditioned is not None
            else self._conditioned_tensor(request, left, graph, tensors)
        )
        right_vector = (
            conditioned[right]
            if conditioned is not None
            else self._conditioned_tensor(request, right, graph, tensors)
        )
        scalars = torch.tensor(
            [
                self._jaccard(
                    graph.records[left].evidence_atoms,
                    graph.records[right].evidence_atoms,
                ),
                self._jaccard(
                    graph.records[left].operation_ids,
                    graph.records[right].operation_ids,
                ),
            ],
            device=self._device,
        )
        features = torch.cat(
            (query, left_vector, right_vector, left_vector * right_vector, scalars)
        )
        return F.softplus(self.redundancy_head(features).squeeze())

    def _set_logit(
        self, example: OutcomeExample, graph: EvidenceGraph, tensors: dict[str, Tensor]
    ) -> tuple[Tensor, list[Tensor]]:
        request = example.request
        query = self._request_tensor(request)
        required_count = torch.tensor(
            [min(1.0, len(request.required_atoms) / 8.0)], device=self._device
        )
        logit = self.base_head(torch.cat((query, required_count))).squeeze()
        for record_id in sorted(example.selected_ids):
            logit = logit + self._unary_value(request, record_id, graph, tensors, query)
        eligible_hyperedges = (
            [
                edge
                for edge in self._eligible_hyperedges(request, graph)
                if edge.members.issubset(example.selected_ids)
            ]
            if self.enable_complements
            else []
        )
        complements = [
            self._complement_value(request, edge, graph, tensors, query)
            for edge in eligible_hyperedges
        ]
        if example.eligible_for_interaction_training:
            for value in complements:
                logit = logit + value
            if self.enable_redundancy:
                for left, right in sorted(graph.duplicate_pairs):
                    if left in example.selected_ids and right in example.selected_ids:
                        logit = logit - self._redundancy_value(
                            request, left, right, graph, tensors, query
                        )
        return logit, complements

    @staticmethod
    def _relation_observations(graph: EvidenceGraph) -> list[tuple[str, str, str]]:
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

    def _ensure_relation_classifier(self, relation_types: tuple[str, ...]) -> None:
        if relation_types == self._relation_types:
            return
        self._relation_types = relation_types
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + 1)
            self.relation_classifier = nn.Linear(
                4 * self.dimension, len(relation_types)
            ).to(self._device)

    def _relation_loss(
        self,
        observations: list[tuple[str, str, str]],
        tensors: dict[str, Tensor],
    ) -> Tensor:
        if not observations or self.relation_classifier is None:
            return torch.zeros((), device=self._device)
        labels = {
            relation: index for index, relation in enumerate(self._relation_types)
        }
        features = []
        targets = []
        for source, target, relation in observations:
            source_vector = tensors[source]
            target_vector = tensors[target]
            features.append(
                torch.cat(
                    (
                        source_vector,
                        target_vector,
                        torch.abs(source_vector - target_vector),
                        source_vector * target_vector,
                    )
                )
            )
            targets.append(labels[relation])
        logits = self.relation_classifier(torch.stack(features))
        target = torch.tensor(targets, dtype=torch.long, device=self._device)
        return F.cross_entropy(logits, target)

    def fit_supervised(
        self,
        examples: Sequence[OutcomeExample],
        graph: EvidenceGraph,
        config: object,
    ) -> TorchEncoderFitResult:
        """Train HGT and attention on verified set outcomes plus relation types."""
        eligible = [
            example
            for example in examples
            if example.actually_executed
            and example.independently_verified
            and example.eligible_for_training
        ]
        if not eligible:
            raise ValueError("at least one verified training example is required")
        learning_rate = float(getattr(config, "learning_rate", 0.01))
        epochs = int(getattr(config, "epochs", 50))
        calibration_weight = float(getattr(config, "calibration_weight", 0.1))
        relation_weight = float(getattr(config, "relation_weight", 0.1))
        interaction_weight = float(getattr(config, "interaction_weight", 0.001))
        gradient_clip = float(getattr(config, "gradient_clip", 10.0))
        class_balance_power = float(getattr(config, "class_balance_power", 0.0))
        ranking_weight = float(getattr(config, "ranking_weight", 0.0))
        self._ensure_schema(graph)
        observations = self._relation_observations(graph) if relation_weight else []
        if relation_weight:
            relation_types = tuple(sorted({relation for _, _, relation in observations}))
            self._ensure_relation_classifier(relation_types)
        positive_count = sum(example.outcome for example in eligible)
        negative_count = len(eligible) - positive_count
        if class_balance_power and (positive_count == 0 or negative_count == 0):
            raise ValueError("class-balanced training requires both outcome classes")
        class_weights = {
            1: (len(eligible) / (2.0 * positive_count)) ** class_balance_power
            if positive_count
            else 1.0,
            0: (len(eligible) / (2.0 * negative_count)) ** class_balance_power
            if negative_count
            else 1.0,
        }
        optimizer = torch.optim.Adam(self.parameters(), lr=min(learning_rate, 0.01))
        losses: list[float] = []
        for _ in range(epochs):
            self.train()
            optimizer.zero_grad()
            tensors = self._forward_graph(graph)
            outcome_losses = []
            brier_losses = []
            interaction_penalties = []
            ranking_groups: dict[str, list[tuple[Tensor, int]]] = defaultdict(list)
            for example in eligible:
                logit, complements = self._set_logit(example, graph, tensors)
                target = torch.tensor(float(example.outcome), device=self._device)
                outcome_losses.append(
                    class_weights[example.outcome]
                    * F.binary_cross_entropy_with_logits(logit, target)
                )
                brier_losses.append((torch.sigmoid(logit) - target).square())
                interaction_penalties.extend(complements)
                ranking_groups[example.request.request_id].append(
                    (logit, example.outcome)
                )
            loss = torch.stack(outcome_losses).mean()
            loss = loss + calibration_weight * torch.stack(brier_losses).mean()
            if relation_weight:
                loss = loss + relation_weight * self._relation_loss(observations, tensors)
            if ranking_weight:
                ranking_losses = []
                for group in ranking_groups.values():
                    positive = [logit for logit, outcome in group if outcome]
                    negative = [logit for logit, outcome in group if not outcome]
                    if not positive or not negative:
                        continue
                    differences = torch.stack(
                        [
                            negative_logit - positive_logit
                            for positive_logit in positive
                            for negative_logit in negative
                        ]
                    )
                    ranking_losses.append(F.softplus(differences).mean())
                if ranking_losses:
                    loss = loss + ranking_weight * torch.stack(ranking_losses).mean()
            if interaction_penalties:
                loss = (
                    loss
                    + interaction_weight * torch.stack(interaction_penalties).mean()
                )
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        self.last_fit_losses = tuple(losses)
        self.encode_graph(graph)
        return TorchEncoderFitResult(self.last_fit_losses, len(losses))

    def auxiliary_probability(
        self,
        request: Request,
        selected_ids: set[str] | frozenset[str],
        graph: EvidenceGraph,
    ) -> float:
        """Return the neural pretraining head's diagnostic set probability."""
        return float(torch.sigmoid(self.auxiliary_logit(request, selected_ids, graph)))

    def auxiliary_probabilities(
        self,
        request: Request,
        selected_sets: Sequence[frozenset[str]],
        graph: EvidenceGraph,
    ) -> tuple[float, ...]:
        """Score search states while reusing request-conditioned components."""
        if not selected_sets:
            return ()
        for selected_ids in selected_sets:
            graph._require_records(selected_ids)
        tensors = self._cached_graph_tensors(graph)
        selected_union = frozenset().union(*selected_sets)
        self.eval()
        with torch.no_grad():
            query = self._request_tensor(request)
            conditioned = {
                record_id: self._conditioned_tensor(
                    request, record_id, graph, tensors
                )
                for record_id in sorted(selected_union)
            }
            required_count = torch.tensor(
                [min(1.0, len(request.required_atoms) / 8.0)],
                device=self._device,
            )
            base = self.base_head(torch.cat((query, required_count))).squeeze()
            unary = {
                record_id: self._unary_value(
                    request,
                    record_id,
                    graph,
                    tensors,
                    query,
                    conditioned,
                )
                for record_id in sorted(selected_union)
            }
            eligible_edges = (
                [
                    edge
                    for edge in self._eligible_hyperedges(request, graph)
                    if edge.members.issubset(selected_union)
                ]
                if self.enable_complements
                else []
            )
            complements = {
                edge.edge_id: self._complement_value(
                    request, edge, graph, tensors, query, conditioned
                )
                for edge in eligible_edges
            }
            redundancies = (
                {
                    (left, right): self._redundancy_value(
                        request,
                        left,
                        right,
                        graph,
                        tensors,
                        query,
                        conditioned,
                    )
                    for left, right in sorted(graph.duplicate_pairs)
                    if left in selected_union and right in selected_union
                }
                if self.enable_redundancy
                else {}
            )
            logits = []
            for selected_ids in selected_sets:
                logit = base
                for record_id in sorted(selected_ids):
                    logit = logit + unary[record_id]
                for edge in eligible_edges:
                    if edge.members.issubset(selected_ids):
                        logit = logit + complements[edge.edge_id]
                for (left, right), value in redundancies.items():
                    if left in selected_ids and right in selected_ids:
                        logit = logit - value
                logits.append(logit)
            values = torch.sigmoid(torch.stack(logits)).detach().cpu().tolist()
        return tuple(float(value) for value in values)

    def _cached_graph_tensors(self, graph: EvidenceGraph) -> dict[str, Tensor]:
        if self._last_graph is not graph or self._last_revision != graph.revision:
            self.encode_graph(graph)
        return self._all_tensors

    def auxiliary_logit(
        self,
        request: Request,
        selected_ids: set[str] | frozenset[str],
        graph: EvidenceGraph,
    ) -> Tensor:
        """Return the fitted non-additive neural set logit."""
        graph._require_records(selected_ids)
        example = OutcomeExample(request, frozenset(selected_ids), outcome=0)
        tensors = self._cached_graph_tensors(graph)
        self.eval()
        with torch.no_grad():
            logit, _ = self._set_logit(example, graph, tensors)
        return logit.cpu()

    def relaxed_probability_and_gradient(
        self,
        request: Request,
        inclusion: dict[str, float],
        graph: EvidenceGraph,
    ) -> tuple[float, dict[str, float]]:
        """Differentiate the multilinear neural set utility with respect to inclusion."""
        graph._require_records(inclusion)
        tensors = self._cached_graph_tensors(graph)
        self.eval()
        with torch.no_grad():
            query = self._request_tensor(request)
            required_count = torch.tensor(
                [min(1.0, len(request.required_atoms) / 8.0)],
                device=self._device,
            )
            base = self.base_head(torch.cat((query, required_count))).squeeze()
            unary = {
                key: self._unary_value(request, key, graph, tensors, query)
                for key in sorted(inclusion)
            }
            complements = (
                [
                    (
                        edge,
                        self._complement_value(request, edge, graph, tensors, query),
                    )
                    for edge in self._eligible_hyperedges(request, graph)
                    if edge.members.issubset(inclusion)
                ]
                if self.enable_complements
                else []
            )
            redundancies = (
                [
                    (
                        left,
                        right,
                        self._redundancy_value(
                            request, left, right, graph, tensors, query
                        ),
                    )
                    for left, right in sorted(graph.duplicate_pairs)
                    if left in inclusion and right in inclusion
                ]
                if self.enable_redundancy
                else []
            )
        variables = {
            key: torch.tensor(
                min(1.0, max(0.0, value)),
                dtype=torch.float32,
                device=self._device,
                requires_grad=True,
            )
            for key, value in sorted(inclusion.items())
        }
        logit = base
        for key, value in variables.items():
            logit = logit + value * unary[key]
        for edge, coefficient in complements:
            logit = logit + coefficient * torch.prod(
                torch.stack([variables[key] for key in sorted(edge.members)])
            )
        for left, right, coefficient in redundancies:
            logit = logit - coefficient * variables[left] * variables[right]
        probability = torch.sigmoid(logit)
        if not variables:
            return float(probability.cpu()), {}
        derivatives = torch.autograd.grad(probability, tuple(variables.values()))
        return float(probability.detach().cpu()), {
            key: float(gradient.detach().cpu())
            for key, gradient in zip(variables, derivatives)
        }
