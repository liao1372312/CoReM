"""Deterministic text and relation-aware graph encoders.

The paper intentionally treats the HGT as an interchangeable component.  This
module supplies a dependency-free message-passing encoder with the same
request-conditioned interface, so the optimization method can run anywhere.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

from .types import EvidenceEdge, EvidenceGraph, EvidenceNode, MemoryRecord, Request

Vector = tuple[float, ...]
_TOKEN_RE = re.compile(r"[\w./:-]+", re.UNICODE)


@lru_cache(maxsize=4096)
def _encode_text_cached(text: str, dimension: int) -> Vector:
    """Hash immutable text features once, including the encoder dimension."""
    values = [0.0] * dimension
    for token in _TOKEN_RE.findall(text.casefold()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        index = int.from_bytes(digest[:8], "big") % dimension
        sign = 1.0 if digest[8] & 1 else -1.0
        values[index] += sign
    return _normalize(values)


def _normalize(values: list[float]) -> Vector:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return tuple(values)
    return tuple(value / norm for value in values)


def cosine(left: Vector, right: Vector) -> float:
    return sum(a * b for a, b in zip(left, right))


class Encoder(Protocol):
    dimension: int

    def encode_request(self, request: Request) -> Vector: ...

    def encode_graph(self, graph: EvidenceGraph) -> dict[str, Vector]: ...

    def condition_on_request(
        self,
        request: Request,
        record_id: str,
        graph: EvidenceGraph,
        record_vectors: dict[str, Vector],
    ) -> Vector: ...


@dataclass(slots=True)
class StableHashEncoder:
    """Stable feature hashing plus typed relation message passing."""

    dimension: int = 64
    message_layers: int = 2
    residual_weight: float = 0.65
    cross_attention_weight: float = 0.35
    _last_graph: EvidenceGraph | None = field(default=None, init=False, repr=False)
    _last_revision: int = field(default=-1, init=False, repr=False)
    _all_vectors: dict[str, Vector] = field(
        default_factory=dict, init=False, repr=False
    )
    _adjacency: dict[str, tuple[tuple[str, EvidenceEdge], ...]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.dimension < 8:
            raise ValueError("dimension must be at least 8")
        if self.message_layers < 0:
            raise ValueError("message_layers must be non-negative")
        if not 0.0 <= self.cross_attention_weight <= 1.0:
            raise ValueError("cross_attention_weight must be in [0, 1]")

    def encode_text(self, text: str) -> Vector:
        return _encode_text_cached(text, self.dimension)

    def encode_request(self, request: Request) -> Vector:
        fields = [
            request.text,
            *sorted(request.required_atoms),
            *sorted(request.operation_ids),
            *sorted(request.active_versions),
        ]
        return self.encode_text(" ".join(fields))

    def encode_record(self, record: MemoryRecord) -> Vector:
        fields = [
            record.text,
            *sorted(record.evidence_atoms),
            *sorted(record.operation_ids),
            *sorted(record.version_scope),
        ]
        return self.encode_text(" ".join(fields))

    def encode_entity(self, entity: EvidenceNode) -> Vector:
        return self.encode_text(f"type:{entity.node_type} {entity.text}")

    def encode_graph(self, graph: EvidenceGraph) -> dict[str, Vector]:
        vectors = {
            key: self.encode_record(record) for key, record in graph.records.items()
        }
        vectors.update(
            {key: self.encode_entity(entity) for key, entity in graph.entities.items()}
        )
        for _ in range(self.message_layers):
            messages = {key: [0.0] * self.dimension for key in vectors}
            counts = {key: 0 for key in vectors}
            for edge in graph.edges:
                self._add_typed_message(
                    messages[edge.target], vectors[edge.source], edge.relation_type
                )
                counts[edge.target] += 1
                self._add_typed_message(
                    messages[edge.source],
                    vectors[edge.target],
                    edge.relation_type + ":rev",
                )
                counts[edge.source] += 1
            updated: dict[str, Vector] = {}
            for key, vector in vectors.items():
                if counts[key] == 0:
                    updated[key] = vector
                    continue
                mixed = [
                    self.residual_weight * value
                    + (1.0 - self.residual_weight) * messages[key][i] / counts[key]
                    for i, value in enumerate(vector)
                ]
                updated[key] = _normalize(mixed)
            vectors = updated
        self._last_graph = graph
        self._last_revision = graph.revision
        self._all_vectors = vectors
        adjacency: dict[str, list[tuple[str, EvidenceEdge]]] = {
            key: [] for key in vectors
        }
        for edge in graph.edges:
            adjacency[edge.source].append((edge.target, edge))
            adjacency[edge.target].append((edge.source, edge))
        self._adjacency = {
            key: tuple(neighbors) for key, neighbors in adjacency.items()
        }
        return {key: vectors[key] for key in graph.records}

    def condition_on_request(
        self,
        request: Request,
        record_id: str,
        graph: EvidenceGraph,
        record_vectors: dict[str, Vector],
    ) -> Vector:
        """Deterministic cross-attention over a record's typed one-hop neighborhood."""
        if (
            self._last_graph is not graph
            or self._last_revision != graph.revision
            or record_id not in self._all_vectors
        ):
            record_vectors = self.encode_graph(graph)
        neighborhood = {record_id}
        for neighbor, edge in self._adjacency.get(record_id, ()):
            if not graph.is_edge_admissible(edge, request):
                continue
            neighborhood.add(neighbor)
        available = [key for key in sorted(neighborhood) if key in self._all_vectors]
        if not available:
            return record_vectors[record_id]
        if len(available) == 1:
            return record_vectors[record_id]
        query = self.encode_request(request)
        logits = [cosine(query, self._all_vectors[key]) for key in available]
        maximum = max(logits)
        weights = [math.exp(value - maximum) for value in logits]
        normalizer = sum(weights)
        pooled = [0.0] * self.dimension
        for key, weight in zip(available, weights):
            for index, value in enumerate(self._all_vectors[key]):
                pooled[index] += weight / normalizer * value
        base = record_vectors[record_id]
        mixed = [
            (1.0 - self.cross_attention_weight) * base[index]
            + self.cross_attention_weight * pooled[index]
            for index in range(self.dimension)
        ]
        return _normalize(mixed)

    def _add_typed_message(
        self, target: list[float], source: Vector, relation_type: str
    ) -> None:
        digest = hashlib.blake2b(relation_type.encode("utf-8"), digest_size=16).digest()
        shift = int.from_bytes(digest[:4], "big") % self.dimension
        sign = 1.0 if digest[4] & 1 else -1.0
        for index, value in enumerate(source):
            target[(index + shift) % self.dimension] += sign * value
