"""Checks for the fixed online logistic scorer."""

import tempfile
import unittest
from pathlib import Path

from corem.online_predictor import OnlineLogisticPredictor
from corem.types import EvidenceGraph, MemoryRecord, OutcomeExample, Request


class OnlinePredictorTests(unittest.TestCase):
    def test_nonfinite_coefficients_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            OnlineLogisticPredictor(weights=[float("nan")] + [0.0] * 255)

    def test_verified_development_fit_and_artifact_round_trip(self) -> None:
        graph = EvidenceGraph()
        graph.add_record(MemoryRecord("useful", "matching evidence", frozenset({"x"})))
        graph.add_record(MemoryRecord("other", "unrelated", frozenset({"z"})))
        request = Request("q", "matching evidence", frozenset({"x"}))
        examples = (
            OutcomeExample(request, frozenset({"useful"}), 1),
            OutcomeExample(request, frozenset({"other"}), 0),
            OutcomeExample(
                request, frozenset({"other"}), 1,
                independently_verified=False,
            ),
        )
        model = OnlineLogisticPredictor()
        model.fit_development(examples, graph, epochs=100)
        self.assertGreater(
            model.probability(request, {"useful"}, graph),
            model.probability(request, {"other"}, graph),
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "online.json"
            model.save(artifact)
            restored = OnlineLogisticPredictor.load(artifact)
            self.assertAlmostEqual(
                restored.probability(request, {"useful"}, graph),
                model.probability(request, {"useful"}, graph),
            )

    def test_relaxed_gradient_matches_finite_difference(self) -> None:
        graph = EvidenceGraph()
        graph.add_record(MemoryRecord("a", "matching evidence", frozenset({"x"})))
        request = Request("q", "matching", frozenset({"x"}))
        model = OnlineLogisticPredictor()
        model.weights[model._index("required-evidence:x")] = 1.0
        probability, gradient = model.relaxed_probability_and_gradient(
            request, {"a": 0.4}, graph
        )
        offset, _ = model.relaxed_probability_and_gradient(
            request, {"a": 0.40001}, graph
        )
        self.assertAlmostEqual(gradient["a"], (offset - probability) / 0.00001, places=4)

    def test_feature_values_do_not_depend_on_inclusion_order(self) -> None:
        graph = EvidenceGraph()
        graph.add_record(MemoryRecord("a", "alpha beta", frozenset({"x", "y"})))
        graph.add_record(MemoryRecord("b", "beta gamma", frozenset({"y", "z"})))
        request = Request("q", "alpha gamma", frozenset({"x", "z"}))
        model = OnlineLogisticPredictor()
        forward = model._terms(request, {"a": 0.4, "b": 0.7}, graph)
        reversed_order = model._terms(request, {"b": 0.7, "a": 0.4}, graph)
        self.assertEqual(forward, reversed_order)


if __name__ == "__main__":
    unittest.main()
