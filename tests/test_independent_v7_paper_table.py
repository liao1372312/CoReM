"""Paper table rows are generated from validated method aggregates."""

from __future__ import annotations

import unittest

from experiments.generate_independent_v7_paper_table import METHODS, render_rows


class IndependentV7PaperTableTests(unittest.TestCase):
    def test_renders_reconciled_method_rows(self) -> None:
        methods = {}
        for key, _ in METHODS:
            methods[key] = {
                "successes": 100,
                "events": 128,
                "mean_tokens": 170,
                "p95_selection_latency_ms": 10.0,
                "by_provider": {
                    "gitea": {"events": 64, "successes": 60},
                    "meilisearch": {"events": 64, "successes": 40},
                },
            }
        result = {
            "status": "complete_independent_v7_confirmatory_analysis",
            "events": 128,
            "families": 16,
            "methods": methods,
        }
        rendered = render_rows(result, "a" * 64)
        self.assertEqual(rendered.count(" & "), 36)
        self.assertIn(r"\begin{tabular}{lrrrr}", rendered)
        self.assertIn(r"\end{tabular}", rendered)
        self.assertIn(r"\corm & \textbf{100}", rendered)
        methods["graph_complement"]["by_provider"]["gitea"]["successes"] = 59
        with self.assertRaises(ValueError):
            render_rows(result, "a" * 64)


if __name__ == "__main__":
    unittest.main()
