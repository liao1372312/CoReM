from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from corem.dataset import PrivateCoremDataset, PublicCoremDataset


class PrivateDatasetTests(unittest.TestCase):
    def test_streaming_loader_preserves_observed_label_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "PRIVATE").write_text("private\n", encoding="utf-8")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "name": "fixture",
                        "schema_version": "1.0.0",
                        "private": True,
                        "files": {},
                    }
                ),
                encoding="utf-8",
            )
            memory = {
                "memory_id": "m1",
                "source": "fixture",
                "text": "verified operation memory",
                "operation_ids": ["op1"],
                "evidence_atoms": ["operation:op1"],
                "token_cost": 3,
                "storage_cost": 4,
                "event_time": None,
                "provenance": "event-1",
                "split": "train",
                "admissible": True,
                "eligible_for_outcome_training": True,
            }
            outcome = {
                "task_id": "q1",
                "query": "run op1",
                "required_atoms": ["operation:op1"],
                "selected_memory_ids": ["m1"],
                "outcome": 1,
                "split": "train",
                "event_time": None,
                "scope": "singleton_endpoint_only",
            }
            (root / "memory_records.jsonl").write_text(
                json.dumps(memory) + "\n", encoding="utf-8"
            )
            (root / "set_outcomes.jsonl").write_text(
                json.dumps(outcome) + "\n", encoding="utf-8"
            )
            (root / "dependency_hyperedges.jsonl").write_text("", encoding="utf-8")
            (root / "duplicate_groups.jsonl").write_text("", encoding="utf-8")
            operation = {
                "operation_id": "op1",
                "service_id": "service1",
                "method": "POST",
                "endpoint": "/run",
                "name": "run operation",
                "description": "execute the fixture",
                "required_input_atoms": ["token"],
                "output_atoms": ["status"],
                "provenance": {"source_path": "fixture.json"},
            }
            (root / "operations.jsonl").write_text(
                json.dumps(operation) + "\n", encoding="utf-8"
            )

            dataset = PrivateCoremDataset(root)
            example = next(dataset.iter_outcome_examples())
            graph = dataset.build_graph(memory_ids=frozenset({"m1"}))

            self.assertTrue(dataset.summary.private)
            self.assertEqual(example.outcome, 1)
            self.assertEqual(example.selected_ids, frozenset({"m1"}))
            self.assertTrue(example.actually_executed)
            self.assertTrue(example.independently_verified)
            self.assertFalse(example.eligible_for_interaction_training)
            self.assertIn("m1", graph.records)

            heterogeneous = dataset.build_graph(
                memory_ids=frozenset({"m1"}),
                include_operation_entities=True,
                include_schema_entities=True,
            )
            self.assertEqual(
                {entity.node_type for entity in heterogeneous.entities.values()},
                {"operation", "service", "input_constraint", "output_schema"},
            )
            self.assertEqual(
                {edge.relation_type for edge in heterogeneous.edges},
                {
                    "documents_operation",
                    "belongs_to_service",
                    "requires_input",
                    "produces_output",
                },
            )

    def test_public_loader_accepts_verified_multi_operation_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "PUBLIC").write_text("public\n", encoding="utf-8")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "name": "public-fixture",
                        "schema_version": "2.0.0",
                        "private": False,
                        "label_scope": "multi_operation_state_criteria",
                        "files": {},
                    }
                ),
                encoding="utf-8",
            )
            records = [
                {
                    "memory_id": memory_id,
                    "source": "fixture",
                    "text": f"execute {memory_id}",
                    "operation_ids": [f"op{index}"],
                    "evidence_atoms": ["goal:q1"],
                    "token_cost": 2,
                    "storage_cost": 3,
                    "event_time": None,
                    "provenance": f"replay-{index}",
                    "provenance_verified": True,
                    "split": "train",
                    "admissible": True,
                }
                for index, memory_id in enumerate(("m1", "m2"), start=1)
            ]
            (root / "memory_records.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            outcome = {
                "task_id": "q1",
                "query": "execute both operations",
                "required_atoms": ["goal:q1"],
                "verified_atoms": [],
                "gold_operation_ids": ["op1", "op2"],
                "selected_memory_ids": ["m1", "m2"],
                "outcome": 1,
                "split": "train",
                "event_time": None,
                "active_versions": ["commit"],
                "scope": "multi_operation_state_criteria",
                "actually_executed": True,
                "independently_verified": True,
                "eligible_for_training": True,
                "eligible_for_interaction_training": True,
            }
            (root / "set_outcomes.jsonl").write_text(
                json.dumps(outcome) + "\n", encoding="utf-8"
            )
            retrieval_task = {
                **outcome,
                "gold_memory_ids": ["m1", "m2"],
            }
            (root / "retrieval_tasks.jsonl").write_text(
                json.dumps(retrieval_task) + "\n", encoding="utf-8"
            )
            edge = {
                "edge_id": "edge1",
                "task_id": "q1",
                "members": ["m1", "m2"],
                "evidence_atoms": ["goal:q1"],
                "relation_type": "execution_irreducible_dependency",
                "provenance": "verified-replay",
                "verified": True,
                "admissible_default": True,
                "confidence": 1.0,
                "version_scope": ["commit"],
            }
            (root / "dependency_hyperedges.jsonl").write_text(
                json.dumps(edge) + "\n", encoding="utf-8"
            )
            (root / "duplicate_groups.jsonl").write_text("", encoding="utf-8")

            dataset = PublicCoremDataset(root)
            example = next(dataset.iter_outcome_examples())
            graph = dataset.build_graph(memory_ids=frozenset({"m1", "m2"}))

            self.assertFalse(dataset.summary.private)
            self.assertEqual(example.selected_ids, frozenset({"m1", "m2"}))
            self.assertFalse(example.request.operation_ids)
            self.assertFalse(dataset.get_task("q1").get("operation_ids"))
            request, _, _ = dataset.load_task_graph("q1")
            self.assertFalse(request.operation_ids)
            self.assertTrue(example.eligible_for_interaction_training)
            self.assertEqual(set(graph.hyperedges), {"edge1"})
            self.assertTrue(graph.is_evidence_complete(example.request, example.selected_ids))
            additive_graph = dataset.build_graph(
                memory_ids=frozenset({"m1", "m2"}), include_hyperedges=False
            )
            self.assertFalse(additive_graph.hyperedges)


if __name__ == "__main__":
    unittest.main()
