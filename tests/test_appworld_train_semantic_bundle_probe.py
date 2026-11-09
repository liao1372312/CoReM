"""Unit checks for the train-only dependency-conditioned selector."""

from __future__ import annotations

import unittest

from experiments.appworld_train_semantic_bundle_probe import (
    _pad_apis,
    _select_bundles,
)


class AppWorldSemanticBundleProbeTests(unittest.TestCase):
    def test_overlap_penalty_changes_second_bundle(self) -> None:
        candidates = ["base", "duplicate", "new"]
        similarity = {"base": 0.9, "duplicate": 0.8, "new": 0.7}
        api_sets = {
            "base": {"a", "b"},
            "duplicate": {"a", "b"},
            "new": {"c", "d"},
        }
        evidence = {api: 0.2 for api in "abcd"}
        self.assertEqual(
            _select_bundles(candidates, similarity, api_sets, evidence, 2, 0, 0),
            ["base", "duplicate"],
        )
        self.assertEqual(
            _select_bundles(candidates, similarity, api_sets, evidence, 2, 0, 0.25),
            ["base", "new"],
        )

    def test_padding_obeys_exact_budget(self) -> None:
        predicted = _pad_apis(
            ["one"],
            {"one": 0.8},
            {"one": {"a", "b"}},
            ["c", "d", "a", "b"],
            3,
        )
        self.assertEqual(predicted, {"a", "b", "c"})


if __name__ == "__main__":
    unittest.main()
