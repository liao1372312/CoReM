"""E5 paper values must come from the frozen attested test artifact."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments/results_expanded_e5_corem_v2_attested/result.json"
PAPER = ROOT / "CoReM_WWW2027_Main_Expanded_Experiments.tex"


class E5AttestationPaperTests(unittest.TestCase):
    def test_frozen_result_and_paper_rows(self) -> None:
        result = json.loads(RESULT.read_text())
        self.assertEqual(result["schema_version"], "corem-expanded-e5-v2-attested")
        self.assertEqual(result["split"], "test")
        summaries = {row["condition"]: row for row in result["summaries"]}
        self.assertEqual(summaries["clean"]["mean_opt"], 0.5)
        self.assertEqual(summaries["spurious_30"]["mean_opt"], 0.5)
        self.assertEqual(
            summaries["spurious_30"]["mean_corrupt_edge_rejection"], 1.0
        )
        self.assertIsNone(summaries["binding_rewrite"]["mean_opt"])
        self.assertEqual(summaries["binding_rewrite"]["mean_safe_rejection"], 1.0)

        paper = PAPER.read_text()
        self.assertIn(
            r"Spurious edges: 30\% & 50.0 & 100.0 & 0.0",
            paper,
        )
        self.assertNotIn("Binding-edge rewrite & 0.0", paper)
        self.assertIn("does not report SUR", paper)


if __name__ == "__main__":
    unittest.main()
