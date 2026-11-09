from __future__ import annotations

import unittest

from corem import (
    CoremEngine,
    DependencyHyperedge,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    ExecutionFeedback,
    GroupAwareRetention,
    MemoryRecord,
    OutcomeExample,
    Request,
    RetentionConfig,
    RetrievalConfig,
    SetUtilityModel,
    TrainingConfig,
)
from corem.retrieval import BundleBeamRetriever
from corem.encoding import StableHashEncoder


def record(
    record_id: str,
    atoms: set[str],
    *,
    token_cost: int = 1,
    storage_cost: int = 1,
    admissible: bool = True,
) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        text=f"memory {record_id} {' '.join(sorted(atoms))}",
        evidence_atoms=frozenset(atoms),
        token_cost=token_cost,
        storage_cost=storage_cost,
        provenance=f"trace-{record_id}",
        admissible=admissible,
    )


class SetUtilityTests(unittest.TestCase):
    def test_text_encoding_cache_respects_dimension_and_is_immutable(self) -> None:
        text = "weather forecast city"
        first = StableHashEncoder(dimension=64)
        second = StableHashEncoder(dimension=32)
        vector = first.encode_text(text)
        self.assertIs(vector, first.encode_text(text))
        self.assertEqual(len(vector), 64)
        self.assertEqual(len(second.encode_text(text)), 32)
        first.dimension = 32
        self.assertEqual(first.encode_text(text), second.encode_text(text))

    def test_heterogeneous_graph_supports_request_conditioned_neighbors(self) -> None:
        graph = EvidenceGraph()
        graph.add_record(record("memory", {"weather"}))
        graph.add_entity(
            EvidenceNode(
                "operation",
                "service_operation",
                "forecast endpoint with city parameter",
                "openapi-trace",
            )
        )
        graph.add_edge(
            EvidenceEdge("memory", "operation", "documents", "openapi-trace")
        )
        request = Request("q", "forecast for a city", frozenset({"weather"}))
        model = SetUtilityModel()

        vectors = model.graph_vectors(graph)
        conditioned = model.encoder.condition_on_request(
            request, "memory", graph, vectors
        )
        self.assertEqual(set(vectors), {"memory"})
        self.assertIn("operation", model.encoder._all_vectors)
        self.assertNotEqual(conditioned, vectors["memory"])

    def test_only_independently_verified_outcomes_are_trainable(self) -> None:
        graph = EvidenceGraph()
        graph.add_record(record("a", {"x"}))
        example = OutcomeExample(
            Request("q", "x", frozenset({"x"})),
            frozenset({"a"}),
            1,
            independently_verified=False,
        )
        with self.assertRaises(ValueError):
            SetUtilityModel().fit((example,), graph)

    def test_relation_auxiliary_heads_train_on_typed_edges(self) -> None:
        graph = EvidenceGraph()
        graph.add_records((record("a", {"x"}), record("b", {"y"})))
        graph.add_entities(
            (
                EvidenceNode("op", "operation", "execute x", "trace"),
                EvidenceNode("schema", "constraint", "requires y", "trace"),
            )
        )
        graph.add_edge(EvidenceEdge("a", "op", "implements", "trace"))
        graph.add_edge(EvidenceEdge("b", "schema", "constrained_by", "trace"))
        example = OutcomeExample(
            Request("q", "x", frozenset({"x"})),
            frozenset({"a"}),
            1,
            eligible_for_interaction_training=False,
        )
        model = SetUtilityModel()
        model.fit(
            (example,),
            graph,
            TrainingConfig(epochs=3, relation_weight=0.5),
        )
        self.assertEqual(set(model.relation_heads), {"implements", "constrained_by"})
        self.assertTrue(
            any(
                value != 0.0
                for weights in model.relation_heads.values()
                for value in weights
            )
        )

    def test_graph_vector_cache_tracks_graph_identity(self) -> None:
        first = EvidenceGraph()
        first.add_record(record("first", {"x"}))
        second = EvidenceGraph()
        second.add_record(record("second", {"y"}))
        model = SetUtilityModel()

        self.assertIn("first", model.graph_vectors(first))
        self.assertIn("second", model.graph_vectors(second))
        self.assertIn("first", model.graph_vectors(first))

    def test_semantic_cache_tracks_request_and_graph_revision(self) -> None:
        graph = EvidenceGraph()
        graph.add_record(record("memory", {"weather"}))
        model = SetUtilityModel()
        first = Request("first", "weather", frozenset({"weather"}))
        second = Request("second", "forecast", frozenset({"weather"}))

        self.assertEqual(
            model.semantic_similarity(first, "memory", graph),
            SetUtilityModel().semantic_similarity(first, "memory", graph),
        )
        model.semantic_similarity(second, "memory", graph)
        self.assertEqual(len(model._semantic_cache), 2)
        graph.add_entity(EvidenceNode("op", "operation", "forecast", "trace"))
        graph.add_edge(EvidenceEdge("memory", "op", "documents", "trace"))
        self.assertEqual(
            model.semantic_similarity(first, "memory", graph),
            SetUtilityModel().semantic_similarity(first, "memory", graph),
        )
        self.assertEqual(len(model._semantic_cache), 1)

    def test_complement_and_redundancy_are_non_additive(self) -> None:
        graph = EvidenceGraph()
        graph.add_records((record("a", {"x"}), record("b", {"y"})))
        graph.add_hyperedge(
            DependencyHyperedge(
                "ab",
                frozenset({"a", "b"}),
                frozenset({"x", "y"}),
                provenance="trace-ab",
            )
        )
        request = Request("q", "x y", frozenset({"x", "y"}))
        model = SetUtilityModel(
            base_weights=[0.0, 0.0],
            unary_weights=[0.0] * 6,
            complement_weights=[2.0, 0.0, 0.0, 0.0, 0.0],
            redundancy_weights=[2.0, 0.0, 0.0, 0.0],
        )
        single = model.probability(request, {"a"}, graph)
        together = model.probability(request, {"a", "b"}, graph)
        self.assertGreater(together, single)

        graph.add_duplicate("a", "b")
        penalized = model.probability(request, {"a", "b"}, graph)
        self.assertAlmostEqual(penalized, single, places=7)

    def test_observed_outcome_training_reduces_loss(self) -> None:
        graph = EvidenceGraph()
        graph.add_record(record("a", {"x"}))
        positive = Request("positive", "x", frozenset({"x"}))
        negative = Request("negative", "nothing", frozenset())
        examples = tuple(
            [OutcomeExample(positive, frozenset({"a"}), 1)] * 4
            + [OutcomeExample(negative, frozenset(), 0)] * 4
        )
        model = SetUtilityModel(
            base_weights=[0.0, 0.0],
            unary_weights=[0.0] * 6,
            complement_weights=[0.0] * 5,
            redundancy_weights=[0.0] * 4,
        )
        fit = model.fit(
            examples,
            graph,
            TrainingConfig(learning_rate=0.15, epochs=80, calibration_weight=0.1),
        )
        self.assertLess(fit.losses[-1], fit.losses[0])
        self.assertGreater(model.probability(positive, {"a"}, graph), 0.5)
        self.assertLess(model.probability(negative, set(), graph), 0.5)


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = EvidenceGraph()
        self.graph.add_records(
            (
                record("binding", {"binding"}),
                record("precondition", {"precondition"}),
                record("inadmissible", {"binding", "precondition"}, admissible=False),
            )
        )
        self.graph.add_hyperedge(
            DependencyHyperedge(
                "chain",
                frozenset({"binding", "precondition"}),
                frozenset({"binding", "precondition"}),
                provenance="trace-chain",
            )
        )
        self.request = Request(
            "compose", "binding precondition", frozenset({"binding", "precondition"})
        )
        model = SetUtilityModel(
            base_weights=[0.0, 0.0],
            unary_weights=[0.0] * 6,
            complement_weights=[2.0, 0.0, 0.0, 0.0, 0.0],
            redundancy_weights=[0.0] * 4,
        )
        self.model = model

    def test_beam_returns_evidence_complete_dependency_bundle(self) -> None:
        retriever = BundleBeamRetriever(
            self.model,
            RetrievalConfig(retrieval_budget=2, beam_width=4, max_depth=2),
        )
        result = retriever.retrieve(
            self.request, frozenset(self.graph.records), self.graph
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.selected_ids, frozenset({"binding", "precondition"}))
        self.assertEqual(result.token_cost, 2)

    def test_infeasible_budget_triggers_explicit_fallback(self) -> None:
        retriever = BundleBeamRetriever(
            self.model,
            RetrievalConfig(
                retrieval_budget=1, beam_width=4, max_depth=2, fallback_value=0.1
            ),
        )
        result = retriever.retrieve(
            self.request, {"binding", "precondition"}, self.graph
        )
        self.assertEqual(result.status, "fallback")
        self.assertEqual(result.selected_ids, frozenset())
        self.assertEqual(result.probability, 0.1)

    def test_attested_bundle_mode_rejects_ungrouped_singletons(self) -> None:
        graph = EvidenceGraph()
        graph.add_records((record("a", {"x"}), record("b", {"y"})))
        request = Request("q", "x y", frozenset({"x", "y"}))
        retriever = BundleBeamRetriever(
            self.model,
            RetrievalConfig(
                retrieval_budget=2,
                beam_width=4,
                max_depth=2,
                require_hyperedge_seed=True,
            ),
        )
        result = retriever.retrieve(request, graph.records, graph)
        self.assertEqual(result.status, "fallback")

        graph.add_hyperedge(
            DependencyHyperedge(
                "attested",
                frozenset({"a", "b"}),
                frozenset({"x", "y"}),
                provenance="verified-contract",
            )
        )
        result = retriever.retrieve(request, graph.records, graph)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.selected_ids, frozenset({"a", "b"}))

    def test_multi_hop_dependency_path_is_added_atomically(self) -> None:
        graph = EvidenceGraph()
        graph.add_records(
            (
                record("a", {"x"}),
                record("bridge", set()),
                record("c", {"y"}),
                record("irrelevant", {"z"}),
            )
        )
        graph.add_hyperedge(
            DependencyHyperedge("ab", frozenset({"a", "bridge"}), frozenset({"x"}))
        )
        graph.add_hyperedge(
            DependencyHyperedge("bc", frozenset({"bridge", "c"}), frozenset({"y"}))
        )
        graph.add_hyperedge(
            DependencyHyperedge(
                "irrelevant-path",
                frozenset({"a", "irrelevant"}),
                frozenset({"z"}),
            )
        )
        request = Request("path", "x then y", frozenset({"x", "y"}))
        retriever = BundleBeamRetriever(
            self.model,
            RetrievalConfig(retrieval_budget=3, dependency_hops=2, max_depth=1),
        )

        result = retriever.retrieve(request, graph.records, graph)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.selected_ids, frozenset({"a", "bridge", "c"}))

    def test_alternative_witnesses_do_not_get_union_closure(self) -> None:
        graph = EvidenceGraph()
        graph.add_records((record("a", {"x"}), record("b", {"y"}), record("c", {"y"})))
        for edge_id, other in (("ab", "b"), ("ac", "c")):
            graph.add_hyperedge(
                DependencyHyperedge(edge_id, frozenset({"a", other}), frozenset({"x", "y"}))
            )
        request = Request("alternatives", "x y", frozenset({"x", "y"}))
        retriever = BundleBeamRetriever(
            self.model, RetrievalConfig(retrieval_budget=2, max_depth=1)
        )
        result = retriever.retrieve(request, graph.records, graph)
        self.assertEqual(result.status, "ok")
        self.assertIn(result.selected_ids, (frozenset({"a", "b"}), frozenset({"a", "c"})))

    def test_pair_and_higher_order_conflicts_are_hard_constraints(self) -> None:
        graph = EvidenceGraph()
        graph.add_records((record("a", {"x"}), record("b", {"y"}), record("c", {"y"})))
        graph.add_conflict("a", "b")
        request = Request("conflict", "x y", frozenset({"x", "y"}))
        retriever = BundleBeamRetriever(
            self.model, RetrievalConfig(retrieval_budget=2, max_depth=2)
        )
        self.assertEqual(
            retriever.retrieve(request, graph.records, graph).selected_ids,
            frozenset({"a", "c"}),
        )
        graph.add_forbidden_subset({"a", "c"})
        self.assertEqual(retriever.retrieve(request, graph.records, graph).status, "fallback")

    def test_equal_utility_uses_configured_tie_seed(self) -> None:
        graph = EvidenceGraph()
        graph.add_records((record("a", {"x"}), record("b", {"x"})))
        request = Request("tie", "x", frozenset({"x"}))
        model = SetUtilityModel(
            base_weights=[0.0, 0.0],
            unary_weights=[0.0] * 6,
            complement_weights=[0.0] * 5,
            redundancy_weights=[0.0] * 4,
        )
        retriever = BundleBeamRetriever(
            model, RetrievalConfig(retrieval_budget=1, tie_seed=2027)
        )
        expected = max(
            (frozenset({"a"}), frozenset({"b"})),
            key=lambda selected: retriever._tie_rank(request, selected),
        )
        self.assertEqual(retriever.retrieve(request, graph.records, graph).selected_ids, expected)

    def test_version_and_temporal_scope_are_hard_constraints(self) -> None:
        graph = EvidenceGraph()
        graph.add_record(
            MemoryRecord(
                "versioned",
                "v2 endpoint",
                frozenset({"x"}),
                version_scope=frozenset({"v2"}),
                valid_from=10.0,
                valid_until=20.0,
            )
        )
        self.assertTrue(
            graph.is_admissible(
                "versioned",
                Request(
                    "valid",
                    "x",
                    frozenset({"x"}),
                    timestamp=15.0,
                    active_versions=frozenset({"v2"}),
                ),
            )
        )
        self.assertFalse(
            graph.is_admissible(
                "versioned",
                Request(
                    "expired",
                    "x",
                    frozenset({"x"}),
                    timestamp=25.0,
                    active_versions=frozenset({"v2"}),
                ),
            )
        )
        self.assertFalse(
            graph.is_admissible(
                "versioned",
                Request(
                    "wrong",
                    "x",
                    frozenset({"x"}),
                    timestamp=15.0,
                    active_versions=frozenset({"v1"}),
                ),
            )
        )


class RetentionTests(unittest.TestCase):
    def test_anytime_deadline_returns_current_feasible_bank(self) -> None:
        graph = EvidenceGraph()
        graph.add_records((record("old", {"x"}), record("new", {"x"})))
        request = Request("q", "x", frozenset({"x"}))
        retriever = BundleBeamRetriever(
            SetUtilityModel(), RetrievalConfig(retrieval_budget=1)
        )
        retainer = GroupAwareRetention(
            retriever,
            RetentionConfig(
                storage_budget=1,
                proposal_priority="semantic",
            ),
        )
        result = retainer.optimize(
            {"old"}, {"new"}, (request,), graph, time_limit_ms=1e-9
        )
        self.assertEqual(result.retained_ids, frozenset({"old"}))
        self.assertTrue(result.timed_out)
        self.assertEqual(result.evaluated_banks, 0)

    def test_expired_deadline_stops_relaxed_proposal_guidance(self) -> None:
        graph = EvidenceGraph()
        graph.add_records((record("old", {"x"}), record("new", {"x"})))
        request = Request("q", "x", frozenset({"x"}))
        retriever = BundleBeamRetriever(
            SetUtilityModel(), RetrievalConfig(retrieval_budget=1)
        )
        retainer = GroupAwareRetention(
            retriever, RetentionConfig(storage_budget=1, proposal_priority="relaxed")
        )
        priorities = retainer._proposal_priorities(
            frozenset({"old", "new"}), (request,), graph, deadline=0.0
        )
        self.assertEqual(priorities, {"new": 0.0, "old": 0.0})

    def test_bundle_bank_value_uses_fixed_candidates_without_retrieval(self) -> None:
        graph = EvidenceGraph()
        graph.add_records((record("a", {"x"}), record("b", {"y"})))
        graph.add_hyperedge(
            DependencyHyperedge(
                "complete",
                frozenset({"a", "b"}),
                frozenset({"x", "y"}),
            )
        )
        request = Request("q", "x y", frozenset({"x", "y"}))
        model = SetUtilityModel(
            base_weights=[0.0, 0.0],
            unary_weights=[0.0] * 6,
            complement_weights=[2.0, 0.0, 0.0, 0.0, 0.0],
            redundancy_weights=[0.0] * 4,
        )
        retriever = BundleBeamRetriever(
            model,
            RetrievalConfig(retrieval_budget=2, context_cost_weight=0.0),
        )
        retainer = GroupAwareRetention(
            retriever,
            RetentionConfig(storage_budget=2, bank_value_objective="bundle"),
        )
        candidates = retainer._fixed_complete_bundles(
            frozenset({"a", "b"}), (request,), graph
        )

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("bundle valuation called the deployed retriever")

        retriever.retrieve = fail_if_called  # type: ignore[method-assign]
        complete = retainer._sample_bank_value(
            frozenset({"a", "b"}),
            (request,),
            graph,
            frozenset({"a", "b"}),
            candidates,
        )
        incomplete = retainer._sample_bank_value(
            frozenset({"a"}),
            (request,),
            graph,
            frozenset({"a"}),
            candidates,
        )
        self.assertGreater(complete, incomplete)

    def test_item_bank_value_is_sum_of_singleton_utilities(self) -> None:
        graph = EvidenceGraph()
        graph.add_records((record("a", {"x"}), record("b", {"y"})))
        request = Request("q", "x y", frozenset({"x", "y"}))
        model = SetUtilityModel(
            base_weights=[0.0, 0.0],
            unary_weights=[0.0] * 6,
            complement_weights=[0.0] * 5,
            redundancy_weights=[0.0] * 4,
        )
        retriever = BundleBeamRetriever(
            model,
            RetrievalConfig(retrieval_budget=2, context_cost_weight=0.0),
        )
        retainer = GroupAwareRetention(
            retriever,
            RetentionConfig(storage_budget=2, bank_value_objective="item"),
        )
        both = retainer._sample_bank_value(
            frozenset({"a", "b"}),
            (request,),
            graph,
            frozenset({"a", "b"}),
        )
        a = retainer._sample_bank_value(
            frozenset({"a"}), (request,), graph, frozenset({"a"})
        )
        b = retainer._sample_bank_value(
            frozenset({"b"}), (request,), graph, frozenset({"b"})
        )
        self.assertAlmostEqual(both, a + b)
        self.assertAlmostEqual(
            a, retriever._objective(request, frozenset({"a"}), graph)
        )
        self.assertAlmostEqual(
            b, retriever._objective(request, frozenset({"b"}), graph)
        )

    def test_bank_value_uses_normalized_workload_weights(self) -> None:
        graph = EvidenceGraph()
        graph.add_record(record("a", {"x"}))
        requests = (
            Request("x", "x", frozenset({"x"})),
            Request("y", "y", frozenset({"y"})),
        )
        retriever = BundleBeamRetriever(
            SetUtilityModel(), RetrievalConfig(retrieval_budget=1)
        )
        retainer = GroupAwareRetention(retriever, RetentionConfig(storage_budget=1))
        bank = frozenset({"a"})
        values = retainer._retriever_values(bank, requests, graph)
        weighted = retainer._sample_bank_value(
            bank, requests, graph, bank, workload_weights=(0.75, 0.25)
        )
        self.assertIsNotNone(values)
        self.assertAlmostEqual(weighted, 0.75 * values[0] + 0.25 * values[1])
        with self.assertRaises(ValueError):
            retainer.optimize(bank, (), requests, graph, workload_weights=(1.0,))

    def test_joint_bundle_can_replace_individually_weak_memory(self) -> None:
        graph = EvidenceGraph()
        graph.add_records(
            (
                record("old", {"x", "y"}, token_cost=2, storage_cost=2),
                record("new-x", {"x"}),
                record("new-y", {"y"}),
            )
        )
        graph.add_hyperedge(
            DependencyHyperedge(
                "new-pair",
                frozenset({"new-x", "new-y"}),
                frozenset({"x", "y"}),
                provenance="trace-new-pair",
            )
        )
        request = Request("q", "x y", frozenset({"x", "y"}))
        model = SetUtilityModel(
            base_weights=[0.0, 0.0],
            unary_weights=[0.0] * 6,
            complement_weights=[3.0, 0.0, 0.0, 0.0, 0.0],
            redundancy_weights=[0.0] * 4,
        )
        retriever = BundleBeamRetriever(
            model,
            RetrievalConfig(
                retrieval_budget=2,
                beam_width=8,
                max_depth=3,
                context_cost_weight=0.0,
            ),
        )
        retainer = GroupAwareRetention(
            retriever,
            RetentionConfig(
                storage_budget=2, max_proposals=8, eviction_shortlist_size=8
            ),
        )
        result = retainer.optimize({"old"}, {"new-x", "new-y"}, (request,), graph)
        self.assertEqual(result.retained_ids, frozenset({"new-x", "new-y"}))
        self.assertEqual(result.storage_cost, 2)
        self.assertGreaterEqual(result.accepted_swaps, 1)

    def test_path_connected_group_can_be_inserted_together(self) -> None:
        graph = EvidenceGraph()
        graph.add_records(
            (
                record("old", {"x", "y", "z"}, token_cost=3, storage_cost=3),
                record("new-x", {"x"}),
                record("bridge", {"y"}),
                record("new-z", {"z"}),
            )
        )
        graph.add_hyperedge(
            DependencyHyperedge(
                "left", frozenset({"new-x", "bridge"}), frozenset({"x", "y"})
            )
        )
        graph.add_hyperedge(
            DependencyHyperedge(
                "right", frozenset({"bridge", "new-z"}), frozenset({"y", "z"})
            )
        )
        request = Request("q", "x y z", frozenset({"x", "y", "z"}))
        model = SetUtilityModel(
            base_weights=[0.0, 0.0],
            unary_weights=[0.0] * 6,
            complement_weights=[3.0, 0.0, 0.0, 0.0, 0.0],
            redundancy_weights=[0.0] * 4,
        )
        retriever = BundleBeamRetriever(
            model,
            RetrievalConfig(retrieval_budget=3, max_depth=2, context_cost_weight=0.0),
        )
        retainer = GroupAwareRetention(
            retriever,
            RetentionConfig(storage_budget=3, proposal_dependency_hops=2),
        )
        result = retainer.optimize(
            {"old"}, {"new-x", "bridge", "new-z"}, (request,), graph
        )
        self.assertEqual(result.retained_ids, frozenset({"new-x", "bridge", "new-z"}))


class EngineTests(unittest.TestCase):
    def test_step_connects_feedback_to_retention(self) -> None:
        graph = EvidenceGraph()
        graph.add_record(record("old", {"irrelevant"}))
        request = Request("q", "required x", frozenset({"x"}))
        model = SetUtilityModel(
            base_weights=[0.0, 0.0],
            unary_weights=[0.0] * 6,
            complement_weights=[0.0] * 5,
            redundancy_weights=[0.0] * 4,
        )
        retriever = BundleBeamRetriever(
            model,
            RetrievalConfig(retrieval_budget=1, fallback_value=0.0),
        )
        retainer = GroupAwareRetention(retriever, RetentionConfig(storage_budget=1))
        engine = CoremEngine(graph, frozenset({"old"}), model, retriever, retainer)

        def executor(
            _request: Request, _selected: tuple[MemoryRecord, ...], status: str
        ) -> ExecutionFeedback:
            self.fail("fallback must not execute the workflow")

        result = engine.step(request, executor, (request,))
        self.assertEqual(result.outcome, 0)
        self.assertEqual(engine.active_bank, frozenset({"old"}))
        self.assertEqual(result.retention.retained_ids, frozenset({"old"}))

    def test_unverified_feedback_cannot_train_or_enter_active_memory(self) -> None:
        graph = EvidenceGraph()
        graph.add_record(record("old", {"x"}))
        request = Request("q", "x", frozenset({"x"}))
        model = SetUtilityModel()
        retriever = BundleBeamRetriever(model, RetrievalConfig(retrieval_budget=1))
        retainer = GroupAwareRetention(retriever, RetentionConfig(storage_budget=1))
        engine = CoremEngine(graph, frozenset({"old"}), model, retriever, retainer)

        def executor(
            _request: Request, _selected: tuple[MemoryRecord, ...], _status: str
        ) -> ExecutionFeedback:
            return ExecutionFeedback(
                outcome=1,
                independently_verified=False,
                new_records=(record("unverified-new", {"x"}),),
            )

        result = engine.step(
            request,
            executor,
            (request,),
            TrainingConfig(epochs=2),
        )
        self.assertIsNone(result.fit)
        self.assertNotIn("unverified-new", graph.records)
        self.assertEqual(engine.active_bank, frozenset({"old"}))


if __name__ == "__main__":
    unittest.main()
