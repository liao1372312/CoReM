"""Tests for the private-benchmark probe's label-blind selection helpers."""

from __future__ import annotations

import unittest

from experiments.appworld_train_retrieval_probe import (
    _choose_memories,
    _paired_generator_interval,
    _rank_memory_apis,
)


class AppWorldTrainRetrievalProbeTests(unittest.TestCase):
    def test_diversity_can_select_nonredundant_memory(self) -> None:
        scores = {"one": 1.0, "duplicate": 0.9, "different": 0.8}
        api_sets = {
            "one": {"a", "b"},
            "duplicate": {"a", "b"},
            "different": {"c", "d"},
        }
        self.assertEqual(
            _choose_memories(scores, api_sets, 2, None), ["one", "duplicate"]
        )
        self.assertEqual(
            _choose_memories(scores, api_sets, 2, 0.7), ["one", "different"]
        )

    def test_api_budget_is_enforced(self) -> None:
        actual = _rank_memory_apis(
            ["one", "two"],
            {"one": 2.0, "two": 1.0},
            {"one": {"a", "b"}, "two": {"c", "d"}},
            2,
        )
        self.assertEqual(actual, {"a", "b"})

    def test_paired_interval_preserves_zero_effect(self) -> None:
        rows = [{"recall": 0.5}, {"recall": 1.0}, {"recall": 0.0}]
        result = _paired_generator_interval(rows, rows, ["g1", "g1", "g2"], 100)
        self.assertEqual(result["mean"], 0.0)
        self.assertEqual(result["cluster_bootstrap_95_interval"], [0.0, 0.0])
        self.assertEqual(result["ties"], 3)


if __name__ == "__main__":
    unittest.main()
