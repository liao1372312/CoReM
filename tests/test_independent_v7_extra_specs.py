"""Offline invariants for the six source-derived v7 candidate workflows."""

from __future__ import annotations

import unittest

from experiments.independent_v7_checker_v5 import _bodies
from experiments.independent_v7_checker_v6 import _facet_goal, _multi_goal
from experiments.independent_v7_extra_specs import (
    GITEA_EXTRA,
    MEILI_EXTRA,
    extra_spec,
)


class IndependentV7ExtraSpecsTest(unittest.TestCase):
    def test_each_incident_has_two_complete_distinct_bundles(self) -> None:
        for family in GITEA_EXTRA + MEILI_EXTRA:
            for incident in range(1, 9):
                with self.subTest(family=family, incident=incident):
                    positive = extra_spec(family, incident, True)
                    negative = extra_spec(family, incident, False)
                    self.assertEqual(positive["query"], negative["query"])
                    self.assertEqual(positive["contract"], negative["contract"])
                    self.assertNotEqual(positive["plans"], negative["plans"])
                    for spec in (positive, negative):
                        self.assertGreaterEqual(len(spec["plans"]), 3)
                        self.assertEqual(
                            len(spec["descriptions"]), len(spec["plans"]) + 1
                        )
                        self.assertEqual(len(spec["docs"]), len(spec["descriptions"]))
                        self.assertGreaterEqual(len(spec["contract"]["handoffs"]), 2)
                        self.assertEqual(
                            len(spec["contract"]["statuses"]), len(spec["plans"])
                        )

    def test_checker_helpers_fail_closed_on_malformed_response(self) -> None:
        self.assertIsNone(_bodies([{"response": {"status": None, "body": {}}}]))
        self.assertFalse(_multi_goal({}, [], {}))
        self.assertFalse(_facet_goal({}, [], {}))


if __name__ == "__main__":
    unittest.main()
