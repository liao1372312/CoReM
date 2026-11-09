from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from corem.online_predictor import OnlineLogisticPredictor
from corem.real_eval import RealTrackError, write_jsonl
from corem.types import Request
from experiments.run_online_retention import (
    FrozenHGTBM25Ensemble,
    OnlineController,
    PolicyState,
    load_exact_execution_cache,
    prune_before_event,
    validate_execution_contract,
    verify_online_model_provenance,
)


class OnlineModelProvenanceTests(unittest.TestCase):
    def test_reportable_model_requires_verified_fit_and_matching_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "online.json"
            dataset = root / "test"
            dataset.mkdir()
            manifest = dataset / "manifest.json"
            manifest.write_text('{"events": 3}\n', encoding="utf-8")
            model = OnlineLogisticPredictor()
            model.save(model_path)
            provenance_path = model_path.with_suffix(".provenance.json")
            provenance = {
                "source_kind": "verified_development_fit",
                "feature_version": model.feature_version,
                "dimension": model.dimension,
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "test_dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "test_event_count": 3,
                "verified_development_examples": 12,
                "development_dataset_manifest_sha256": "a" * 64,
                "development_executions_sha256": "b" * 64,
            }
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
            self.assertEqual(
                verify_online_model_provenance(model_path, dataset, 3, model),
                provenance_path,
            )
            with self.assertRaisesRegex(ValueError, "event count"):
                verify_online_model_provenance(model_path, dataset, 4, model)
            provenance["source_kind"] = "unknown"
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "verified development"):
                verify_online_model_provenance(model_path, dataset, 3, model)


class _FixedModel:
    def semantic_similarity(self, request, record_id, graph):
        return {"a": 0.8, "b": 0.2}[record_id]

    def probability(self, request, selected_ids, graph):
        return 0.5

    def probabilities(self, request, selected_sets, graph):
        return tuple(0.5 for _ in selected_sets)

    def relaxed_probability_and_gradient(self, request, inclusion, graph):
        return 0.5, {memory_id: 0.2 for memory_id in inclusion}


class OnlineTimeBoundaryTests(unittest.TestCase):
    def test_frozen_ensemble_uses_same_bm25_residual_for_discrete_and_relaxed(self) -> None:
        records = {
            "a": SimpleNamespace(text="alpha operation"),
            "b": SimpleNamespace(text="beta operation"),
        }
        graph = SimpleNamespace(
            records=records,
            revision=1,
            is_admissible=lambda memory_id, request: True,
        )
        request = Request("request-1", "alpha", frozenset())
        model = FrozenHGTBM25Ensemble(
            [(17, _FixedModel(), Path("seed17.pt"))], neural_weight=0.9
        )

        probabilities = model.probabilities(
            request, [frozenset({"a"}), frozenset({"b"})], graph
        )
        self.assertAlmostEqual(probabilities[0], 0.55)
        self.assertAlmostEqual(probabilities[1], 0.45)
        relaxed, gradient = model.relaxed_probability_and_gradient(
            request, {"a": 1.0, "b": 1.0}, graph
        )
        self.assertAlmostEqual(relaxed, 0.55)
        self.assertAlmostEqual(gradient["a"], 0.28)
        self.assertAlmostEqual(gradient["b"], 0.18)

    def test_group_bm25_retains_whole_past_relevant_bundle(self) -> None:
        records = {
            "a1": SimpleNamespace(text="create issue", timestamp=1, storage_cost=1),
            "a2": SimpleNamespace(text="issue created", timestamp=1, storage_cost=1),
            "b1": SimpleNamespace(text="delete index", timestamp=2, storage_cost=1),
            "b2": SimpleNamespace(text="index deleted", timestamp=2, storage_cost=1),
        }
        graph = SimpleNamespace(
            records=records,
            hyperedges={
                "a": SimpleNamespace(edge_id="a", members=frozenset(("a1", "a2"))),
                "b": SimpleNamespace(edge_id="b", members=frozenset(("b1", "b2"))),
            },
            is_admissible=lambda memory_id, request: True,
        )
        controller = OnlineController.__new__(OnlineController)
        controller.graph = graph
        controller.storage_budget = 2
        state = PolicyState("group_bm25_online")
        active = frozenset(records)
        self.assertEqual(controller._group_bm25_bank(state, active), {"b1", "b2"})
        state.workload.append(SimpleNamespace(text="create issue"))
        self.assertEqual(controller._group_bm25_bank(state, active), {"a1", "a2"})
        controller.storage_budget = 3
        residual_bank = controller._group_bm25_bank(
            state, active, fill_residual=True
        )
        self.assertTrue({"a1", "a2"}.issubset(residual_bank))
        self.assertEqual(len(residual_bank), 3)

    def test_release_invalidates_without_activating_new_version(self) -> None:
        state = PolicyState("group_aware_online", bank=frozenset({"old", "shared"}))
        evicted = prune_before_event(state, frozenset({"shared", "new"}))
        self.assertEqual(evicted, {"old"})
        self.assertEqual(state.bank, frozenset({"shared"}))
        self.assertNotIn("new", state.bank)

    def test_forecast_bank_follows_only_observed_past_transitions(self) -> None:
        records = {
            "a1": SimpleNamespace(
                operation_ids=frozenset({"a"}), timestamp=1, storage_cost=1
            ),
            "a2": SimpleNamespace(
                operation_ids=frozenset({"a"}), timestamp=1, storage_cost=1
            ),
            "b1": SimpleNamespace(
                operation_ids=frozenset({"b"}), timestamp=1, storage_cost=1
            ),
            "b2": SimpleNamespace(
                operation_ids=frozenset({"b"}), timestamp=1, storage_cost=1
            ),
            "c1": SimpleNamespace(
                operation_ids=frozenset({"c"}), timestamp=1, storage_cost=1
            ),
            "c2": SimpleNamespace(
                operation_ids=frozenset({"c"}), timestamp=1, storage_cost=1
            ),
        }
        graph = SimpleNamespace(
            records=records,
            hyperedges={
                "a": SimpleNamespace(edge_id="a", members=frozenset({"a1", "a2"})),
                "b": SimpleNamespace(edge_id="b", members=frozenset({"b1", "b2"})),
                "c": SimpleNamespace(edge_id="c", members=frozenset({"c1", "c2"})),
            },
        )
        controller = OnlineController.__new__(OnlineController)
        controller.graph = graph
        controller.storage_budget = 4
        controller.forecast_min_support = 2
        controller.forecast_min_confidence = 0.75
        state = PolicyState("forecast_corem", last_signature=("a",))
        state.transition_counts = {
            ("a",): Counter({("b",): 3}),
            ("b",): Counter({("c",): 2}),
        }
        state.signature_frequencies.update({("a",): 3, ("b",): 3, ("c",): 2})
        self.assertEqual(
            controller._forecast_bank(state, frozenset(records)),
            frozenset({"b1", "b2", "c1", "c2"}),
        )
        state.transition_counts[("a",)] = Counter({("b",): 1, ("c",): 1})
        self.assertIsNone(controller._forecast_bank(state, frozenset(records)))

    def test_exact_cache_rejects_unverified_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "executions.jsonl"
            write_jsonl(
                source,
                [
                    {
                        "event_id": "event-1",
                        "selected_memory_ids": ["memory-1"],
                        "checker_independent": False,
                        "reset_verified": True,
                        "execution_error": None,
                        "checker_error": None,
                        "execution": {},
                        "checker": {},
                        "execution_latency_ms": 1.0,
                        "snapshot_before": "a",
                        "snapshot_after": "b",
                    }
                ],
            )
            with self.assertRaisesRegex(RealTrackError, "unverified source"):
                load_exact_execution_cache(
                    source, frozenset({"event-1"}), frozenset({"memory-1"})
                )

    def test_cached_execution_requires_matching_composer_and_input(self) -> None:
        contract = {
            "model": "gpt-4.1-2025-04-14",
            "prompt_version": "corem-composer-v2",
            "config_sha256": "a" * 64,
            "catalog_sha256": "b" * 64,
        }
        record = {
            "execution": {
                "composer_model": contract["model"],
                "composer_prompt_version": contract["prompt_version"],
                "composer_config_sha256": contract["config_sha256"],
                "composer_catalog_sha256": contract["catalog_sha256"],
                "composer_input_sha256": "c" * 64,
            },
            "checker": {
                "contract_id": "public-multiservice-challenge-checker-v1"
            },
        }
        validate_execution_contract(
            record, contract, "public-multiservice-challenge-checker-v1",
            "event-1", "c" * 64,
        )
        with self.assertRaisesRegex(RealTrackError, "input mismatch"):
            validate_execution_contract(
                record, contract, "public-multiservice-challenge-checker-v1",
                "event-1", "d" * 64,
            )
        record["execution"]["composer_catalog_sha256"] = "e" * 64
        with self.assertRaisesRegex(RealTrackError, "catalog"):
            validate_execution_contract(
                record, contract, "public-multiservice-challenge-checker-v1",
                "event-1", "c" * 64,
            )

    def test_source_cache_rejects_another_dataset_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "executions.jsonl"
            write_jsonl(source, [{
                "event_id": "event-1", "selected_memory_ids": ["memory-1"],
                "checker_independent": True, "reset_verified": True,
                "execution_error": None, "checker_error": None,
                "execution": {}, "checker": {},
                "execution_latency_ms": 1.0,
                "snapshot_before": "a", "snapshot_after": "b",
                "dataset_manifest_sha256": "a" * 64,
            }])
            with self.assertRaisesRegex(RealTrackError, "dataset manifest mismatch"):
                load_exact_execution_cache(
                    source, frozenset({"event-1"}), frozenset({"memory-1"}),
                    expected_dataset_sha256="b" * 64,
                )


if __name__ == "__main__":
    unittest.main()
