"""The v5 challenge paper table must reject inconsistent frozen metrics."""

from __future__ import annotations

import hashlib
import json
import unittest

from experiments.generate_v5_challenge_paper_table import SOURCE, render_table


class V5ChallengePaperTableTests(unittest.TestCase):
    def test_frozen_summary_and_invalid_interval(self) -> None:
        raw = SOURCE.read_bytes()
        summary = json.loads(raw)
        rendered = render_table(summary, hashlib.sha256(raw).hexdigest())
        self.assertIn(r"Version BM25 + closure & \textbf{60.9}", rendered)
        self.assertIn(r"Recency BM25 & 12.5 & 22.9 & 12.5 & 7.8 & $[4.7,20.3]$", rendered)
        summary["method_metrics"][2]["vsr_ci_high"] = 0
        with self.assertRaises(ValueError):
            render_table(summary, "dummy")


if __name__ == "__main__":
    unittest.main()
