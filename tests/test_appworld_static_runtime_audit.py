"""Name-set coverage is narrower than runtime dependency validation."""

from __future__ import annotations

import unittest

from experiments.audit_appworld_static_dependency_runtime import name_set_difference


class NameCoverageTests(unittest.TestCase):
    def test_occurrence_repeats_do_not_change_name_set(self) -> None:
        self.assertEqual(name_set_difference(("a.x", "a.x", "b.y"), ["b.y", "a.x"]), (0, 0))

    def test_missing_names_are_counted_both_ways(self) -> None:
        self.assertEqual(name_set_difference(("a.x", "b.y"), ["a.x", "c.z"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
