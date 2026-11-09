"""Core data structures for the CoReM reference implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Iterable


def _nonempty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """One grounded experience record in the active memory bank."""

    record_id: str
    text: str
    evidence_atoms: frozenset[str]
    operation_ids: frozenset[str] = frozenset()
    token_cost: int = 1
    storage_cost: int = 1
    timestamp: float = 0.0
    provenance: str = "verified"
    service_scope: frozenset[str] = frozenset()
    version_scope: frozenset[str] = frozenset()
    valid_from: float = 0.0
    valid_until: float | None = None
    execution_outcome: int | None = None
    provenance_verified: bool = True
    admissible: bool = True

    def __post_init__(self) -> None:
        _nonempty(self.record_id, "record_id")
        _nonempty(self.text, "text")
        _nonempty(self.provenance, "provenance")
        if self.token_cost <= 0 or self.storage_cost <= 0:
            raise ValueError("record costs must be positive")
        if self.valid_until is not None and self.valid_until < self.valid_from:
            raise ValueError("valid_until cannot precede valid_from")
        if self.execution_outcome not in (None, 0, 1):
            raise ValueError("execution_outcome must be 0, 1, or None")


@dataclass(frozen=True, slots=True)
class Request:
    """A composition request and the evidence it requires."""

    request_id: str
    text: str
    required_atoms: frozenset[str]
    verified_atoms: frozenset[str] = frozenset()
    operation_ids: frozenset[str] = frozenset()
    timestamp: float = 0.0
    allowed_scopes: frozenset[str] = frozenset()
    active_versions: frozenset[str] = frozenset()
    conflict_pairs: frozenset[frozenset[str]] = frozenset()
    forbidden_subsets: frozenset[frozenset[str]] = frozenset()

    def __post_init__(self) -> None:
        _nonempty(self.request_id, "request_id")
        _nonempty(self.text, "text")
        if any(len(pair) != 2 for pair in self.conflict_pairs):
            raise ValueError("each request conflict must contain two records")
        if any(len(group) < 2 for group in self.forbidden_subsets):
            raise ValueError("each forbidden subset must contain at least two records")


@dataclass(frozen=True, slots=True)
class EvidenceEdge:
    """A verified, typed relation between heterogeneous evidence nodes."""

    source: str
    target: str
    relation_type: str
    provenance: str
    verified: bool = True
    signature: str = ""
    version_scope: frozenset[str] = frozenset()
    valid_from: float = 0.0
    valid_until: float | None = None

    def __post_init__(self) -> None:
        _nonempty(self.relation_type, "relation_type")
        _nonempty(self.provenance, "provenance")
        if self.valid_until is not None and self.valid_until < self.valid_from:
            raise ValueError("valid_until cannot precede valid_from")


@dataclass(frozen=True, slots=True)
class EvidenceNode:
    """A verified non-memory node in the heterogeneous evidence graph."""

    node_id: str
    node_type: str
    text: str
    provenance: str
    verified: bool = True

    def __post_init__(self) -> None:
        _nonempty(self.node_id, "node_id")
        _nonempty(self.node_type, "node_type")
        _nonempty(self.text, "text")
        _nonempty(self.provenance, "provenance")


@dataclass(frozen=True, slots=True)
class DependencyHyperedge:
    """A provenance-backed group whose members are jointly useful."""

    edge_id: str
    members: frozenset[str]
    evidence_atoms: frozenset[str]
    relation_type: str = "dependency"
    provenance: str = "verified"
    verified: bool = True
    signature: str = ""
    version_scope: frozenset[str] = frozenset()
    valid_from: float = 0.0
    valid_until: float | None = None

    def __post_init__(self) -> None:
        _nonempty(self.edge_id, "edge_id")
        _nonempty(self.relation_type, "relation_type")
        _nonempty(self.provenance, "provenance")
        if len(self.members) < 2:
            raise ValueError("a dependency hyperedge needs at least two members")
        if self.valid_until is not None and self.valid_until < self.valid_from:
            raise ValueError("valid_until cannot precede valid_from")


@dataclass(slots=True)
class EvidenceGraph:
    """Mutable evidence graph with a revision number for encoder caches."""

    records: dict[str, MemoryRecord] = field(default_factory=dict)
    entities: dict[str, EvidenceNode] = field(default_factory=dict)
    edges: list[EvidenceEdge] = field(default_factory=list)
    hyperedges: dict[str, DependencyHyperedge] = field(default_factory=dict)
    duplicate_pairs: set[tuple[str, str]] = field(default_factory=set)
    conflict_pairs: set[frozenset[str]] = field(default_factory=set)
    forbidden_subsets: set[frozenset[str]] = field(default_factory=set)
    revision: int = 0

    def add_record(self, record: MemoryRecord) -> None:
        if record.record_id in self.entities:
            raise ValueError("a memory record id cannot shadow an evidence entity")
        self.records[record.record_id] = record
        self.revision += 1

    def add_records(self, records: Iterable[MemoryRecord]) -> None:
        for record in records:
            self.add_record(record)

    def add_entity(self, entity: EvidenceNode) -> None:
        if not entity.verified:
            raise ValueError("unverified entities cannot enter the evidence graph")
        if entity.node_id in self.records:
            raise ValueError("an entity id cannot shadow a memory record")
        self.entities[entity.node_id] = entity
        self.revision += 1

    def add_entities(self, entities: Iterable[EvidenceNode]) -> None:
        for entity in entities:
            self.add_entity(entity)

    def add_edge(self, edge: EvidenceEdge) -> None:
        self._require_nodes((edge.source, edge.target))
        if not edge.verified:
            raise ValueError("unverified relations cannot enter the evidence graph")
        self.edges.append(edge)
        self.revision += 1

    def add_hyperedge(self, edge: DependencyHyperedge) -> None:
        self._require_records(edge.members)
        if not edge.verified:
            raise ValueError(
                "unverified dependency relations cannot enter the evidence graph"
            )
        self.hyperedges[edge.edge_id] = edge
        self.revision += 1

    def add_duplicate(self, left: str, right: str) -> None:
        if left == right:
            raise ValueError("a record cannot duplicate itself")
        self._require_records((left, right))
        self.duplicate_pairs.add(tuple(sorted((left, right))))
        self.revision += 1

    def add_conflict(self, left: str, right: str) -> None:
        """Record a verified incompatibility, distinct from soft redundancy."""
        if left == right:
            raise ValueError("a record cannot conflict with itself")
        self._require_records((left, right))
        self.conflict_pairs.add(frozenset((left, right)))
        self.revision += 1

    def add_forbidden_subset(self, record_ids: Iterable[str]) -> None:
        members = frozenset(record_ids)
        if len(members) < 2:
            raise ValueError("a forbidden subset needs at least two records")
        self._require_records(members)
        self.forbidden_subsets.add(members)
        self.revision += 1

    def is_compatible(self, selected: Iterable[str], request: Request) -> bool:
        members = frozenset(selected)
        forbidden = (
            self.conflict_pairs
            | self.forbidden_subsets
            | request.conflict_pairs
            | request.forbidden_subsets
        )
        return all(not group.issubset(members) for group in forbidden)

    def _require_records(self, record_ids: Iterable[str]) -> None:
        missing = set(record_ids).difference(self.records)
        if missing:
            raise KeyError(f"unknown record ids: {sorted(missing)}")

    def _require_nodes(self, node_ids: Iterable[str]) -> None:
        known = set(self.records).union(self.entities)
        missing = set(node_ids).difference(known)
        if missing:
            raise KeyError(f"unknown evidence node ids: {sorted(missing)}")

    def is_admissible(self, record_id: str, request: Request) -> bool:
        record = self.records[record_id]
        if not record.admissible or not record.provenance_verified:
            return False
        if request.allowed_scopes and record.service_scope:
            if not request.allowed_scopes.intersection(record.service_scope):
                return False
        if record.version_scope:
            if not request.active_versions:
                return False
            if not request.active_versions.intersection(record.version_scope):
                return False
        if request.timestamp:
            if record.valid_from and request.timestamp < record.valid_from:
                return False
            if (
                record.valid_until is not None
                and request.timestamp > record.valid_until
            ):
                return False
        return True

    @staticmethod
    def _scope_is_admissible(
        version_scope: frozenset[str],
        valid_from: float,
        valid_until: float | None,
        request: Request,
    ) -> bool:
        if version_scope and (
            not request.active_versions
            or not request.active_versions.intersection(version_scope)
        ):
            return False
        if request.timestamp:
            if valid_from and request.timestamp < valid_from:
                return False
            if valid_until is not None and request.timestamp > valid_until:
                return False
        return True

    def is_edge_admissible(self, edge: EvidenceEdge, request: Request) -> bool:
        return edge.verified and self._scope_is_admissible(
            edge.version_scope, edge.valid_from, edge.valid_until, request
        )

    def is_hyperedge_admissible(
        self, edge: DependencyHyperedge, request: Request
    ) -> bool:
        return edge.verified and self._scope_is_admissible(
            edge.version_scope, edge.valid_from, edge.valid_until, request
        )

    def dependency_closure(
        self,
        request: Request,
        seed_ids: Iterable[str],
        bank_ids: Iterable[str],
        *,
        max_hops: int,
        max_size: int | None = None,
    ) -> frozenset[str]:
        """Close a selected set over request-relevant grounded dependency paths."""
        if max_hops < 0:
            raise ValueError("max_hops must be non-negative")
        bank = frozenset(bank_ids)
        selected = set(seed_ids)
        self._require_records(bank)
        if not selected.issubset(bank):
            raise ValueError("dependency-closure seeds must belong to the bank")
        frontier = set(selected)
        for _ in range(max_hops):
            additions: set[str] = set()
            for edge in self.hyperedges.values():
                if not self.is_hyperedge_admissible(edge, request):
                    continue
                if not edge.members.issubset(bank) or not edge.members.intersection(
                    frontier
                ):
                    continue
                required_by_edge = edge.evidence_atoms.intersection(
                    request.required_atoms
                )
                if edge.evidence_atoms and not required_by_edge:
                    continue
                if required_by_edge and required_by_edge.issubset(
                    request.verified_atoms
                ):
                    continue
                additions.update(edge.members.difference(selected))
            if not additions:
                break
            if max_size is not None and len(selected.union(additions)) > max_size:
                break
            selected.update(additions)
            frontier = additions
        return frozenset(selected)

    def dependency_groups(
        self,
        seed_ids: Iterable[str],
        universe_ids: Iterable[str],
        *,
        max_hops: int,
        max_size: int,
        deadline: float | None = None,
    ) -> set[frozenset[str]]:
        """Return bounded hypergraph-connected groups for retention proposals."""
        universe = frozenset(universe_ids)
        seeds = frozenset(seed_ids)
        self._require_records(universe)
        groups: set[frozenset[str]] = set()
        for seed in sorted(seeds):
            if deadline is not None and time.perf_counter() >= deadline:
                return groups
            group = {seed}
            frontier = {seed}
            for _ in range(max_hops):
                additions: set[str] = set()
                for edge in sorted(self.hyperedges.values(), key=lambda item: item.edge_id):
                    if deadline is not None and time.perf_counter() >= deadline:
                        return groups
                    if edge.members.issubset(universe) and edge.members.intersection(
                        frontier
                    ):
                        additions.update(edge.members.difference(group))
                if not additions or len(group.union(additions)) > max_size:
                    break
                group.update(additions)
                frontier = additions
            if len(group) > 1:
                groups.add(frozenset(group))
        for edge in sorted(self.hyperedges.values(), key=lambda item: item.edge_id):
            if deadline is not None and time.perf_counter() >= deadline:
                return groups
            if (
                edge.members.issubset(universe)
                and edge.members.intersection(seeds)
                and len(edge.members) <= max_size
            ):
                groups.add(edge.members)
        return groups

    def uncovered_atoms(
        self, request: Request, selected: Iterable[str]
    ) -> frozenset[str]:
        covered = set(request.verified_atoms)
        for record_id in selected:
            covered.update(self.records[record_id].evidence_atoms)
        return frozenset(request.required_atoms.difference(covered))

    def is_evidence_complete(self, request: Request, selected: Iterable[str]) -> bool:
        return not self.uncovered_atoms(request, selected)


@dataclass(frozen=True, slots=True)
class OutcomeExample:
    """An actually executed set-level training observation."""

    request: Request
    selected_ids: frozenset[str]
    outcome: int
    actually_executed: bool = True
    independently_verified: bool = True
    eligible_for_training: bool = True
    eligible_for_interaction_training: bool = True

    def __post_init__(self) -> None:
        if self.outcome not in (0, 1):
            raise ValueError("outcome must be 0 or 1")


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    selected_ids: frozenset[str]
    probability: float
    objective: float
    token_cost: int
    status: str
    uncovered_atoms: frozenset[str] = frozenset()
    evaluations: int = 0


@dataclass(frozen=True, slots=True)
class RetentionResult:
    retained_ids: frozenset[str]
    bank_value: float
    storage_cost: int
    accepted_swaps: int
    evaluated_banks: int
    timed_out: bool = False
