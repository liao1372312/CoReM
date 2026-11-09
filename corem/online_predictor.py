"""Fixed, regularized logistic set predictor for the online E3/E4 stage.

The paper specifies feature families and a 256-dimensional hash, but does not
publish the exact feature template or fitted coefficients. This implementation
keeps that choice explicit in a versioned artifact instead of silently using
the candidate-freezing HGT/BM25 score as an online success probability.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .types import EvidenceGraph, OutcomeExample, Request


def _tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\w+", text.lower()))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass(slots=True)
class OnlineLogisticPredictor:
    """A deterministic set-level probability model frozen after development fit."""

    dimension: int = 256
    weights: list[float] = field(default_factory=lambda: [0.0] * 256)
    intercept: float = 0.0
    feature_version: str = "corem-online-hash-v1"

    def __post_init__(self) -> None:
        if self.dimension <= 0 or len(self.weights) != self.dimension:
            raise ValueError("weights must match a positive hash dimension")
        if not math.isfinite(self.intercept) or any(
            not math.isfinite(weight) for weight in self.weights
        ):
            raise ValueError("online predictor coefficients must be finite")

    @classmethod
    def seeded_random(
        cls, *, seed: int = 2027, dimension: int = 256, scale: float = 0.01
    ) -> OnlineLogisticPredictor:
        """Create a reproducible exploratory model without a success claim."""
        if dimension <= 0 or not math.isfinite(scale) or scale <= 0:
            raise ValueError("dimension and random scale must be positive")
        generator = random.Random(seed)
        return cls(
            dimension=dimension,
            weights=[generator.gauss(0.0, scale) for _ in range(dimension)],
        )

    def _index(self, name: str) -> int:
        return int.from_bytes(
            hashlib.sha256(name.encode("utf-8")).digest()[:8], "big"
        ) % self.dimension

    def _terms(
        self, request: Request, inclusion: dict[str, float], graph: EvidenceGraph
    ) -> tuple[list[float], dict[str, list[float]]]:
        features = [0.0] * self.dimension
        gradients = {key: [0.0] * self.dimension for key in inclusion}

        def add(name: str, value: float, derivatives: dict[str, float] | None = None) -> None:
            index = self._index(name)
            features[index] += value
            if derivatives:
                for key, derivative in derivatives.items():
                    gradients[key][index] += derivative

        request_words = _tokens(request.text)
        for word in sorted(request_words):
            add(f"request:{word}", 1.0)
        for atom in sorted(request.required_atoms):
            add(f"required:{atom}", 1.0)
        for version in sorted(request.active_versions):
            add(f"active-version:{version}", 1.0)
        for key in sorted(inclusion):
            amount = inclusion[key]
            record = graph.records[key]
            derivative = {key: 1.0}
            add("record-count", amount, derivative)
            add("token-cost", amount * record.token_cost / 160.0,
                {key: record.token_cost / 160.0})
            for word in sorted(_tokens(record.text)):
                add(f"memory:{word}", amount, derivative)
                if word in request_words:
                    add(f"request-memory:{word}", amount, derivative)
            for atom in sorted(record.evidence_atoms):
                add(f"evidence:{atom}", amount, derivative)
                if atom in request.required_atoms:
                    add(f"required-evidence:{atom}", amount, derivative)
            for version in sorted(record.version_scope):
                add(f"record-version:{version}", amount, derivative)
                if version in request.active_versions:
                    add(f"matching-version:{version}", amount, derivative)
            for operation in sorted(record.operation_ids):
                add(f"operation:{operation}", amount, derivative)
        for edge in sorted(graph.hyperedges.values(), key=lambda item: item.edge_id):
            if not graph.is_hyperedge_admissible(edge, request):
                continue
            members = edge.members
            if not members.issubset(inclusion):
                continue
            product = math.prod(inclusion[key] for key in members)
            derivatives = {
                key: math.prod(inclusion[other] for other in members if other != key)
                for key in sorted(members)
            }
            add(f"dependency:{edge.relation_type}", product, derivatives)
            for atom in sorted(edge.evidence_atoms.intersection(request.required_atoms)):
                add(f"dependency-evidence:{atom}", product, derivatives)
        ordered = sorted(inclusion)
        for position, left in enumerate(ordered):
            for right in ordered[position + 1:]:
                left_amount, right_amount = inclusion[left], inclusion[right]
                product = left_amount * right_amount
                derivatives = {left: right_amount, right: left_amount}
                shared = graph.records[left].evidence_atoms.intersection(
                    graph.records[right].evidence_atoms
                )
                add("cross-record", product, derivatives)
                for atom in sorted(shared):
                    add(f"shared-evidence:{atom}", product, derivatives)
                if tuple(sorted((left, right))) in graph.duplicate_pairs:
                    add("duplicate-pair", product, derivatives)
        return features, gradients

    def probability(
        self, request: Request, selected_ids: Iterable[str], graph: EvidenceGraph
    ) -> float:
        inclusion = {key: 1.0 for key in frozenset(selected_ids)}
        features, _ = self._terms(request, inclusion, graph)
        return _sigmoid(self.intercept + sum(
            weight * value for weight, value in zip(self.weights, features)
        ))

    def probabilities(
        self, request: Request, selections: Sequence[frozenset[str]], graph: EvidenceGraph
    ) -> tuple[float, ...]:
        return tuple(self.probability(request, selected, graph) for selected in selections)

    def semantic_similarity(
        self, request: Request, record_id: str, graph: EvidenceGraph
    ) -> float:
        request_words = _tokens(request.text)
        memory_words = _tokens(graph.records[record_id].text)
        union = request_words | memory_words
        return len(request_words & memory_words) / len(union) if union else 0.0

    def relaxed_probability_and_gradient(
        self, request: Request, inclusion: dict[str, float], graph: EvidenceGraph
    ) -> tuple[float, dict[str, float]]:
        features, derivatives = self._terms(request, inclusion, graph)
        probability = _sigmoid(self.intercept + sum(
            weight * value for weight, value in zip(self.weights, features)
        ))
        factor = probability * (1.0 - probability)
        return probability, {
            key: factor * sum(weight * value for weight, value in zip(self.weights, row))
            for key, row in derivatives.items()
        }

    def fit_development(
        self,
        examples: Sequence[OutcomeExample],
        graph: EvidenceGraph,
        *,
        l2: float = 0.01,
        learning_rate: float = 0.1,
        epochs: int = 200,
    ) -> None:
        if l2 < 0 or learning_rate <= 0 or epochs <= 0:
            raise ValueError("invalid regularized logistic training parameters")
        verified = [
            example for example in examples
            if example.actually_executed
            and example.independently_verified
            and example.eligible_for_training
        ]
        if not verified:
            raise ValueError("no independently verified development executions")
        rows = [
            (self._terms(example.request,
                         {key: 1.0 for key in example.selected_ids}, graph)[0],
             example.outcome)
            for example in verified
        ]
        for _ in range(epochs):
            gradient = [0.0] * self.dimension
            intercept_gradient = 0.0
            for features, outcome in rows:
                probability = _sigmoid(self.intercept + sum(
                    weight * value for weight, value in zip(self.weights, features)
                ))
                error = probability - outcome
                intercept_gradient += error
                for index, value in enumerate(features):
                    gradient[index] += error * value
            count = len(rows)
            self.intercept -= learning_rate * intercept_gradient / count
            for index in range(self.dimension):
                self.weights[index] -= learning_rate * (
                    gradient[index] / count + l2 * self.weights[index]
                )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "kind": "fixed_online_logistic",
            "feature_version": self.feature_version,
            "dimension": self.dimension,
            "intercept": self.intercept,
            "weights": self.weights,
        }, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> OnlineLogisticPredictor:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("kind") != "fixed_online_logistic" or data.get("feature_version") != "corem-online-hash-v1":
            raise ValueError("unsupported online predictor artifact")
        return cls(
            dimension=int(data["dimension"]),
            weights=[float(value) for value in data["weights"]],
            intercept=float(data["intercept"]),
        )
