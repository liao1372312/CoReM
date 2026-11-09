"""Disjoint train controls preserve source context without copied edges."""

from __future__ import annotations

import unittest

from experiments.build_appworld_train_disjoint_edge_control import disjoint_hint
from experiments.build_appworld_train_mirrored_edge_control import EDGE_MARKER


class DisjointEdgeControlTests(unittest.TestCase):
    def test_same_source_count_forward_edges_and_repeatability(self) -> None:
        order = "Static AppWorld API-call occurrences:\n1. a.one\n2. a.two\n3. a.three\n4. a.four"
        original = order + EDGE_MARKER + "1->2, 1->4"
        first, count = disjoint_hint(order, original, 2027)
        second, _ = disjoint_hint(order, original, 2027)
        self.assertEqual(first, second)
        self.assertEqual(count, 2)
        self.assertTrue(first.startswith(order + EDGE_MARKER))
        edges = {
            tuple(map(int, token.split("->")))
            for token in first.split(EDGE_MARKER)[1].split(", ")
        }
        self.assertEqual(len(edges), 2)
        self.assertFalse(edges & {(1, 2), (1, 4)})
        self.assertTrue(all(1 <= left < right <= 4 for left, right in edges))

    def test_rejects_impossible_or_unpaired_graph(self) -> None:
        order = "1. a.one\n2. a.two"
        with self.assertRaises(ValueError):
            disjoint_hint(order, order + EDGE_MARKER + "1->2", 2027)
        with self.assertRaises(ValueError):
            disjoint_hint(order, "different" + EDGE_MARKER + "1->2", 2027)


if __name__ == "__main__":
    unittest.main()
