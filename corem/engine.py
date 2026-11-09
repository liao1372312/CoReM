"""Online orchestration for CoReM retrieval, execution feedback, and retention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .model import FitResult, SetUtilityModel, TrainingConfig
from .retention import GroupAwareRetention
from .retrieval import BundleBeamRetriever
from .types import (
    DependencyHyperedge,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    MemoryRecord,
    OutcomeExample,
    Request,
    RetentionResult,
    RetrievalResult,
)


@dataclass(frozen=True, slots=True)
class ExecutionFeedback:
    """Output from an independent composer/verifier supplied by the caller."""

    outcome: int
    actually_executed: bool = True
    independently_verified: bool = True
    service_version_validated: bool = True
    eligible_for_training: bool = True
    eligible_for_interaction_training: bool = True
    new_records: tuple[MemoryRecord, ...] = ()
    new_entities: tuple[EvidenceNode, ...] = ()
    new_edges: tuple[EvidenceEdge, ...] = ()
    new_hyperedges: tuple[DependencyHyperedge, ...] = ()
    duplicate_pairs: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.outcome not in (0, 1):
            raise ValueError("outcome must be 0 or 1")


Executor = Callable[[Request, tuple[MemoryRecord, ...], str], ExecutionFeedback]


@dataclass(frozen=True, slots=True)
class StepResult:
    retrieval: RetrievalResult
    outcome: int
    retention: RetentionResult
    fit: FitResult | None


class CoremEngine:
    """Implements the online sequence in Algorithm 1 of the paper."""

    def __init__(
        self,
        graph: EvidenceGraph,
        active_bank: frozenset[str],
        model: SetUtilityModel,
        retriever: BundleBeamRetriever,
        retainer: GroupAwareRetention,
    ) -> None:
        graph._require_records(active_bank)
        self.graph = graph
        self.active_bank = active_bank
        self.model = model
        self.retriever = retriever
        self.retainer = retainer
        self.observations: list[OutcomeExample] = []

    def step(
        self,
        request: Request,
        executor: Executor,
        past_only_workload: Sequence[Request],
        training_config: TrainingConfig | None = None,
        workload_weights: Sequence[float] | None = None,
    ) -> StepResult:
        retrieval = self.retriever.retrieve(request, self.active_bank, self.graph)
        selected_records = tuple(
            self.graph.records[key] for key in sorted(retrieval.selected_ids)
        )
        feedback = (
            executor(request, selected_records, retrieval.status)
            if retrieval.status == "ok"
            else ExecutionFeedback(
                outcome=0,
                actually_executed=False,
                independently_verified=False,
                eligible_for_training=False,
                eligible_for_interaction_training=False,
            )
        )
        training_eligible = (
            feedback.eligible_for_training
            and feedback.service_version_validated
            and retrieval.status == "ok"
        )
        observation = OutcomeExample(
            request,
            retrieval.selected_ids,
            feedback.outcome,
            actually_executed=feedback.actually_executed,
            independently_verified=feedback.independently_verified,
            eligible_for_training=training_eligible,
            eligible_for_interaction_training=(
                feedback.eligible_for_interaction_training and training_eligible
            ),
        )
        self.observations.append(observation)

        evidence_admissible = (
            feedback.actually_executed
            and feedback.independently_verified
            and feedback.service_version_validated
        )
        admitted_records = tuple(
            record
            for record in feedback.new_records
            if evidence_admissible and record.provenance_verified
        )
        if evidence_admissible:
            self.graph.add_records(admitted_records)
            self.graph.add_entities(
                entity for entity in feedback.new_entities if entity.verified
            )
            known_nodes = set(self.graph.records).union(self.graph.entities)
            for edge in feedback.new_edges:
                if edge.verified and {edge.source, edge.target}.issubset(known_nodes):
                    self.graph.add_edge(edge)
            for edge in feedback.new_hyperedges:
                if edge.verified and edge.members.issubset(self.graph.records):
                    self.graph.add_hyperedge(edge)
            for left, right in feedback.duplicate_pairs:
                if {left, right}.issubset(self.graph.records):
                    self.graph.add_duplicate(left, right)

        fit_result = None
        if training_config is not None and any(
            item.actually_executed
            and item.independently_verified
            and item.eligible_for_training
            for item in self.observations
        ):
            fit_result = self.model.fit(self.observations, self.graph, training_config)

        new_ids = frozenset(record.record_id for record in admitted_records)
        workload = tuple(past_only_workload) or (request,)
        retention = self.retainer.optimize(
            self.active_bank,
            new_ids,
            workload,
            self.graph,
            workload_weights=workload_weights,
        )
        self.active_bank = retention.retained_ids
        return StepResult(retrieval, feedback.outcome, retention, fit_result)
