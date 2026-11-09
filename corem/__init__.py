"""CoReM: coupled relational memory retrieval and retention."""

from .attestation import attest_workflow_hyperedge, partition_attested_hyperedges
from .encoding import StableHashEncoder
from .dataset import PublicCoremDataset
from .engine import CoremEngine, ExecutionFeedback, StepResult
from .model import FitResult, SetUtilityModel, TrainingConfig
from .online_predictor import OnlineLogisticPredictor
from .retention import GroupAwareRetention, RetentionConfig
from .retrieval import Bundle, BundleBeamRetriever, RetrievalConfig
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

__all__ = [
    "Bundle",
    "BundleBeamRetriever",
    "CoremEngine",
    "DependencyHyperedge",
    "EvidenceEdge",
    "EvidenceGraph",
    "EvidenceNode",
    "ExecutionFeedback",
    "FitResult",
    "GroupAwareRetention",
    "MemoryRecord",
    "OutcomeExample",
    "OnlineLogisticPredictor",
    "PublicCoremDataset",
    "Request",
    "RetentionConfig",
    "RetentionResult",
    "RetrievalConfig",
    "RetrievalResult",
    "SetUtilityModel",
    "StableHashEncoder",
    "StepResult",
    "TrainingConfig",
    "attest_workflow_hyperedge",
    "partition_attested_hyperedges",
]
