"""Order audit distinguishes call-name compatibility from binding evidence."""

from __future__ import annotations

import unittest

from experiments.audit_appworld_static_runtime_order import (
    edge_order_compatible,
    lcs_length,
)


class StaticRuntimeOrderTests(unittest.TestCase):
    def test_lcs_handles_repeats_and_branch_omission(self) -> None:
        self.assertEqual(lcs_length(("a", "b", "a"), ["a", "x", "b", "a"]), 3)
        self.assertEqual(lcs_length(("a", "b", "c"), ["a", "c"]), 2)

    def test_edge_order_compatibility_is_only_name_level(self) -> None:
        names = ("a", "b", "a")
        self.assertEqual(edge_order_compatible(names, ((1, 2), (2, 3)), ["b", "a"]), (1, 2))
        self.assertEqual(edge_order_compatible(("a", "a"), ((1, 2),), ["a"]), (0, 1))

    def test_invalid_static_edge_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            edge_order_compatible(("a", "b"), ((2, 1),), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
