"""The tau3 paper table preserves the metric and adaptation boundaries."""

from __future__ import annotations

import hashlib
import json
import unittest

from experiments.generate_tau3_paper_table import SOURCE, render_table


class Tau3PaperTableTests(unittest.TestCase):
    def test_frozen_table_and_metric_boundary(self) -> None:
        raw = SOURCE.read_bytes()
        summary = json.loads(raw)
        rendered = render_table(summary, hashlib.sha256(raw).hexdigest())
        self.assertNotIn(r"\notmeasured", rendered)
        self.assertIn(r"\notapplicable", rendered)
        self.assertIn(r"\textbf{\corm}", rendered)
        summary["summaries"][0]["ecr"] = None
        with self.assertRaises(ValueError):
            render_table(summary, "dummy")


if __name__ == "__main__":
    unittest.main()
