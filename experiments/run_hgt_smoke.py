"""Train the optional HGT backend on a small, fully verified fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch_geometric

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem import (
    DependencyHyperedge,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    MemoryRecord,
    OutcomeExample,
    Request,
    TrainingConfig,
)
from corem.torch_encoding import TorchHGTEncoder
from corem.torch_model import TorchSetUtilityModel


def build_fixture() -> tuple[EvidenceGraph, tuple[OutcomeExample, ...], Request]:
    def record(record_id: str, atoms: set[str]) -> MemoryRecord:
        return MemoryRecord(
            record_id,
            f"verified memory {record_id}",
            frozenset(atoms),
            provenance="controlled-smoke-trace",
        )

    graph = EvidenceGraph()
    graph.add_records(
        (record("binding", {"x"}), record("guard", {"y"}), record("duplicate", {"x"}))
    )
    graph.add_entities(
        (
            EvidenceNode("operation", "operation", "execute x", "trace"),
            EvidenceNode("schema", "constraint", "requires y", "trace"),
        )
    )
    graph.add_edge(EvidenceEdge("binding", "operation", "implements", "trace"))
    graph.add_edge(EvidenceEdge("guard", "schema", "constrained_by", "trace"))
    graph.add_hyperedge(
        DependencyHyperedge(
            "binding-chain",
            frozenset({"binding", "guard"}),
            frozenset({"x", "y"}),
            "dependency",
            "trace",
        )
    )
    graph.add_duplicate("binding", "duplicate")
    compose = Request("compose", "compose x then y", frozenset({"x", "y"}))
    single = Request("single", "execute x", frozenset({"x"}))
    empty = Request("empty", "no evidence", frozenset())
    examples = (
        OutcomeExample(compose, frozenset({"binding", "guard"}), 1),
        OutcomeExample(compose, frozenset({"binding", "guard", "duplicate"}), 0),
        OutcomeExample(single, frozenset({"binding"}), 1),
        OutcomeExample(single, frozenset({"duplicate"}), 0),
        OutcomeExample(empty, frozenset(), 0),
    )
    return graph, examples * 3, compose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    graph, examples, request = build_fixture()
    encoder = TorchHGTEncoder(
        dimension=32, message_layers=2, attention_heads=4, seed=2027
    )
    model = TorchSetUtilityModel(encoder=encoder)
    fit = model.fit(
        examples,
        graph,
        TrainingConfig(
            epochs=args.epochs,
            learning_rate=0.01,
            relation_weight=0.2,
        ),
    )
    print(
        json.dumps(
            {
                "device": encoder.device_name,
                "torch": torch.__version__,
                "torch_geometric": torch_geometric.__version__,
                "epochs": args.epochs,
                "encoder_loss_first": encoder.last_fit_losses[0],
                "encoder_loss_last": encoder.last_fit_losses[-1],
                "joint_loss_first": fit.losses[0],
                "joint_loss_last": fit.losses[-1],
                "auxiliary_positive_probability": encoder.auxiliary_probability(
                    request, {"binding", "guard"}, graph
                ),
                "auxiliary_duplicate_probability": encoder.auxiliary_probability(
                    request, {"binding", "guard", "duplicate"}, graph
                ),
                "interpretation": "gradient smoke check; not an effectiveness result",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
