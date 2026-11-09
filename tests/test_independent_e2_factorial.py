import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results_independent_e2_factorial_v2"


class IndependentE2FactorialTests(unittest.TestCase):
    def test_frozen_factorial_is_complete_and_matches_paper(self) -> None:
        result = json.loads((RESULTS / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(
            result["schema_version"], "corem-independent-e2-factorial-v2"
        )
        self.assertEqual(result["status"], "complete_post_hoc_factorial")
        self.assertEqual(result["events"], 128)
        self.assertEqual(result["candidate_sets"], 256)
        rows = result["rows"]
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            {(row["graph"], row["set_term"], row["lexical"]) for row in rows},
            {
                (graph, set_term, lexical)
                for graph in ("On", "Off")
                for set_term in ("On", "Off")
                for lexical in ("On", "Off")
            },
        )
        self.assertTrue(all(row["events"] == 128 for row in rows))
        self.assertTrue(all(row["candidate_sets"] == 256 for row in rows))
        self.assertTrue(all(row["vsr"] == row["top1"] for row in rows))
        self.assertAlmostEqual(rows[0]["vsr"], 102 / 128)
        self.assertAlmostEqual(rows[-1]["vsr"], 57 / 128)
        self.assertAlmostEqual(
            result["main_effects_vsr_pp"]["lexical_on_minus_off"], 41.015625
        )
        paired = result["paired_comparison"]
        self.assertEqual(
            (paired["event_wins"], paired["event_losses"], paired["event_ties"]),
            (7, 2, 119),
        )
        self.assertAlmostEqual(paired["event_exact_sign_p"], 0.1796875)
        self.assertAlmostEqual(paired["family_bootstrap_ci_low_pp"], -4.6875)
        self.assertAlmostEqual(paired["family_bootstrap_ci_high_pp"], 16.40625)

        self.assertIn("post-hoc", result["claim_boundary"])

    def test_checksums(self) -> None:
        for line in (RESULTS / "checksums.sha256").read_text(
            encoding="utf-8"
        ).splitlines():
            digest, name = line.split("  ", 1)
            observed = hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
            self.assertEqual(observed, digest)


if __name__ == "__main__":
    unittest.main()
