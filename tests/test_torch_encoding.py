from __future__ import annotations

import math
import unittest

from corem import (
    DependencyHyperedge,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    MemoryRecord,
    OutcomeExample,
    Request,
    RetrievalConfig,
    SetUtilityModel,
    TrainingConfig,
)
from corem.retrieval import BundleBeamRetriever

try:
    import torch

    from corem.torch_encoding import TorchHGTEncoder
    from corem.torch_model import TorchSetUtilityModel
except ImportError:
    torch = None
    TorchHGTEncoder = None
    TorchSetUtilityModel = None


@unittest.skipIf(torch is None, "PyTorch/PyG optional dependencies are unavailable")
class TorchHGTEncoderTests(unittest.TestCase):
    @staticmethod
    def graph_and_examples() -> tuple[EvidenceGraph, tuple[OutcomeExample, ...]]:
        def record(record_id: str, atoms: set[str]) -> MemoryRecord:
            return MemoryRecord(
                record_id,
                f"memory {record_id}",
                frozenset(atoms),
                provenance="trace",
            )

        graph = EvidenceGraph()
        graph.add_records(
            (
                record("binding", {"x"}),
                record("precondition", {"y"}),
                record("duplicate", {"x"}),
            )
        )
        graph.add_entities(
            (
                EvidenceNode("operation", "operation", "execute x", "trace"),
                EvidenceNode("schema", "constraint", "requires y", "trace"),
            )
        )
        graph.add_edge(EvidenceEdge("binding", "operation", "implements", "trace"))
        graph.add_edge(
            EvidenceEdge("precondition", "schema", "constrained_by", "trace")
        )
        graph.add_hyperedge(
            DependencyHyperedge(
                "chain",
                frozenset({"binding", "precondition"}),
                frozenset({"x", "y"}),
                "dependency",
                "trace",
            )
        )
        graph.add_duplicate("binding", "duplicate")
        compose = Request("compose", "compose x then y", frozenset({"x", "y"}))
        single = Request("single", "execute x", frozenset({"x"}))
        empty = Request("empty", "no evidence needed", frozenset())
        examples = (
            OutcomeExample(compose, frozenset({"binding", "precondition"}), 1),
            OutcomeExample(
                compose,
                frozenset({"binding", "precondition", "duplicate"}),
                0,
            ),
            OutcomeExample(single, frozenset({"binding"}), 1),
            OutcomeExample(single, frozenset({"duplicate"}), 0),
            OutcomeExample(empty, frozenset(), 0),
        )
        return graph, examples * 2

    def test_hgt_and_cross_attention_receive_supervised_gradients(self) -> None:
        graph, examples = self.graph_and_examples()
        encoder = TorchHGTEncoder(
            dimension=16, message_layers=1, attention_heads=2, seed=7
        )
        encoder.encode_graph(graph)
        before = {
            name: parameter.detach().clone()
            for name, parameter in encoder.named_parameters()
        }
        model = SetUtilityModel(encoder=encoder)

        result = model.fit(
            examples,
            graph,
            TrainingConfig(epochs=4, learning_rate=0.02, relation_weight=0.2),
        )

        self.assertLess(encoder.last_fit_losses[-1], encoder.last_fit_losses[0])
        self.assertLess(result.losses[-1], result.losses[0])
        self.assertEqual(set(model.graph_vectors(graph)), set(graph.records))
        changed = sum(
            float((parameter.detach() - before[name]).abs().sum())
            for name, parameter in encoder.named_parameters()
            if name in before
        )
        self.assertGreater(changed, 0.0)

    def test_text_hash_width_can_differ_from_hidden_width(self) -> None:
        graph, _ = self.graph_and_examples()
        encoder = TorchHGTEncoder(
            dimension=16,
            text_hash_dimension=64,
            message_layers=1,
            attention_heads=2,
            seed=5,
        )

        vectors = encoder.encode_graph(graph)

        self.assertEqual(encoder.request_projection.in_features, 64)
        self.assertEqual(encoder.input_projections[0].in_features, 64)
        self.assertTrue(all(len(vector) == 16 for vector in vectors.values()))

    def test_subword_text_features_are_supported_and_cached(self) -> None:
        encoder = TorchHGTEncoder(
            dimension=16,
            text_hash_dimension=32,
            text_feature_mode="subword",
            message_layers=0,
            attention_heads=2,
        )

        first = encoder._hash_text("GET /user-profile/{user_id}")
        second = encoder._hash_text("GET /user-profile/{user_id}")

        self.assertIs(first, second)
        self.assertEqual(first.shape, (32,))

    def test_optional_lexical_features_extend_neural_heads(self) -> None:
        baseline = TorchHGTEncoder(
            dimension=16,
            text_hash_dimension=32,
            message_layers=0,
            attention_heads=2,
        )
        lexical = TorchHGTEncoder(
            dimension=16,
            text_hash_dimension=32,
            message_layers=0,
            attention_heads=2,
            enable_lexical_features=True,
        )

        self.assertEqual(
            lexical.unary_head.in_features, baseline.unary_head.in_features + 1
        )
        self.assertEqual(
            lexical.complement_head.in_features,
            baseline.complement_head.in_features + 2,
        )

    def test_text_only_factorial_cell_keeps_complements_without_graph_context(
        self,
    ) -> None:
        graph, examples = self.graph_and_examples()
        from experiments.train_public_multiservice_hgt import build_model

        model = build_model(
            "text_complement",
            seed=7,
            dimension=16,
            text_hash_dimension=32,
            message_layers=2,
            device="cpu",
            lexical_features=False,
        )
        encoder = model.encoder
        encoder._ensure_schema(graph)
        _, lookup = encoder._node_layout(graph)
        self.assertEqual(len(encoder.hgt_layers), 0)
        self.assertEqual(encoder._edge_index_dict(graph, lookup), {})
        self.assertTrue(encoder.enable_complements)
        self.assertFalse(encoder.enable_graph_context)
        tensors = encoder._forward_graph(graph)
        _, complements = encoder._set_logit(examples[0], graph, tensors)
        self.assertEqual(len(complements), 1)

        additive = build_model(
            "text_additive",
            seed=7,
            dimension=16,
            text_hash_dimension=32,
            message_layers=2,
            device="cpu",
            lexical_features=False,
        )
        additive_tensors = additive.encoder._forward_graph(graph)
        _, no_complements = additive.encoder._set_logit(
            examples[0], graph, additive_tensors
        )
        self.assertEqual(no_complements, [])

    def test_text_hash_width_rejects_too_small_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "text_hash_dimension"):
            TorchHGTEncoder(dimension=16, text_hash_dimension=4)

        with self.assertRaisesRegex(ValueError, "text_feature_mode"):
            TorchHGTEncoder(dimension=16, text_feature_mode="unknown")

    def test_end_to_end_neural_utility_exposes_relaxed_gradients(self) -> None:
        graph, examples = self.graph_and_examples()
        model = TorchSetUtilityModel(
            TorchHGTEncoder(dimension=16, message_layers=1, attention_heads=2, seed=11)
        )
        fit = model.fit(
            examples,
            graph,
            TrainingConfig(epochs=12, learning_rate=0.02, relation_weight=0.2),
        )
        request = examples[0].request
        positive = model.probability(request, {"binding", "precondition"}, graph)
        duplicate = model.probability(
            request, {"binding", "precondition", "duplicate"}, graph
        )
        probability, gradient = model.relaxed_probability_and_gradient(
            request,
            {"binding": 0.5, "precondition": 0.5, "duplicate": 0.5},
            graph,
        )

        self.assertLess(fit.losses[-1], fit.losses[0])
        self.assertGreater(positive, duplicate)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
        self.assertEqual(set(gradient), set(graph.records))

        retrieval = BundleBeamRetriever(
            model,
            RetrievalConfig(
                retrieval_budget=2,
                beam_width=4,
                max_depth=2,
                candidate_pool_size=3,
            ),
        ).retrieve(request, graph.records, graph)
        self.assertEqual(retrieval.status, "ok")
        self.assertEqual(retrieval.selected_ids, frozenset({"binding", "precondition"}))

    def test_class_balanced_neural_training_is_supported(self) -> None:
        graph, examples = self.graph_and_examples()
        model = TorchSetUtilityModel(
            TorchHGTEncoder(dimension=16, message_layers=1, attention_heads=2, seed=19)
        )

        result = model.fit(
            examples,
            graph,
            TrainingConfig(
                epochs=4,
                learning_rate=0.01,
                class_balance_power=1.0,
            ),
        )

        self.assertEqual(result.epochs, 4)
        self.assertTrue(all(math.isfinite(loss) for loss in result.losses))

    def test_batched_probabilities_match_scalar_inference(self) -> None:
        graph, examples = self.graph_and_examples()
        model = TorchSetUtilityModel(
            TorchHGTEncoder(dimension=16, message_layers=1, attention_heads=2, seed=23)
        )
        request = examples[0].request
        selected_sets = (
            frozenset(),
            frozenset({"binding"}),
            frozenset({"binding", "precondition"}),
            frozenset({"binding", "precondition", "duplicate"}),
        )

        scalar = tuple(
            model.probability(request, selected, graph)
            for selected in selected_sets
        )
        batched = model.probabilities(request, selected_sets, graph)

        self.assertEqual(len(batched), len(scalar))
        for observed, expected in zip(batched, scalar):
            self.assertAlmostEqual(observed, expected, places=6)

    def test_event_level_ranking_loss_is_supported(self) -> None:
        graph, examples = self.graph_and_examples()
        model = TorchSetUtilityModel(
            TorchHGTEncoder(dimension=16, message_layers=1, attention_heads=2, seed=29)
        )

        result = model.fit(
            examples,
            graph,
            TrainingConfig(
                epochs=4,
                learning_rate=0.01,
                ranking_weight=1.0,
            ),
        )

        self.assertEqual(result.epochs, 4)
        self.assertTrue(all(math.isfinite(loss) for loss in result.losses))

    def test_set_logit_scores_only_selected_hyperedges(self) -> None:
        graph, examples = self.graph_and_examples()
        graph.add_hyperedge(
            DependencyHyperedge(
                "unselected-chain",
                frozenset({"binding", "duplicate"}),
                frozenset({"x", "y"}),
                "dependency",
                "trace",
            )
        )
        encoder = TorchHGTEncoder(
            dimension=16, message_layers=1, attention_heads=2, seed=31
        )

        tensors = encoder._forward_graph(graph)
        _, complement_terms = encoder._set_logit(examples[0], graph, tensors)

        self.assertEqual(len(complement_terms), 1)

    def test_encoder_matched_additive_gate_keeps_graph_but_disables_set_terms(
        self,
    ) -> None:
        graph, examples = self.graph_and_examples()
        encoder = TorchHGTEncoder(
            dimension=16,
            message_layers=1,
            attention_heads=2,
            seed=23,
            enable_complements=False,
            enable_redundancy=False,
        )

        tensors = encoder._forward_graph(graph)
        _, complement_terms = encoder._set_logit(examples[0], graph, tensors)
        probability, gradient = encoder.relaxed_probability_and_gradient(
            examples[0].request,
            {"binding": 0.5, "precondition": 0.5, "duplicate": 0.5},
            graph,
        )

        self.assertIn("chain", graph.hyperedges)
        self.assertFalse(complement_terms)
        self.assertTrue(0.0 <= probability <= 1.0)
        self.assertEqual(set(gradient), set(graph.records))


if __name__ == "__main__":
    unittest.main()
