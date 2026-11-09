"""Mirrored edge controls preserve source text and edge-list format."""

from __future__ import annotations

import unittest

from experiments.build_appworld_train_mirrored_edge_control import (
    EDGE_MARKER,
    mirror_edges,
    mirrored_hint,
)


class MirroredEdgeControlTests(unittest.TestCase):
    def test_mirror_keeps_forward_edges_and_count(self) -> None:
        self.assertEqual(mirror_edges(((1, 2), (1, 4)), 4), ((1, 4), (3, 4)))

    def test_hint_preserves_order_prefix(self) -> None:
        order = "Example task:\nDo work\nStatic AppWorld API-call occurrences:\n1. a.one\n2. a.two\n3. a.three"
        changed, count, overlap = mirrored_hint(order, order + EDGE_MARKER + "1->2")
        self.assertEqual(changed, order + EDGE_MARKER + "2->3")
        self.assertEqual((count, overlap), (1, 0))

    def test_rejects_mismatched_source(self) -> None:
        with self.assertRaises(ValueError):
            mirrored_hint("1. a.one", "other" + EDGE_MARKER + "None detected in this source.")

    def test_rejects_invalid_edges(self) -> None:
        with self.assertRaises(ValueError):
            mirror_edges(((2, 1),), 2)


if __name__ == "__main__":
    unittest.main()
