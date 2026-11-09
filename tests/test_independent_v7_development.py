"""Guard development-family labels and complete negative construction."""

from __future__ import annotations

import unittest

from experiments.build_independent_v7_development import _spec


class DevelopmentFamilySpecTest(unittest.TestCase):
    def test_each_family_has_one_distinct_executable_negative(self) -> None:
        families = ("gitea_label", "gitea_milestone", "meili_document", "meili_search")
        for family in families:
            with self.subTest(family=family):
                a = _spec(family, "bundle_a")
                b = _spec(family, "bundle_b")
                self.assertNotEqual(a["positive"], b["positive"])
                self.assertEqual(len(a["plans"]), 3)
                self.assertEqual(len(b["plans"]), 3)
                self.assertEqual(a["query"], b["query"])
                self.assertEqual(a["contract"], b["contract"])
                self.assertNotEqual(a["plans"], b["plans"])
                self.assertNotEqual(a["text"], b["text"])
                for bundle in (a, b):
                    first, second, third = bundle["plans"]
                    self.assertIn("capture", first)
                    self.assertIn("capture", second)
                    self.assertIn("{" + first["capture"]["name"] + "}", second["path"])
                    self.assertIn("{" + second["capture"]["name"] + "}", third["path"])


if __name__ == "__main__":
    unittest.main()
