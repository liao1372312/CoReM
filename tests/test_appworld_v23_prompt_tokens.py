"""Aggregate token-cost summaries do not depend on private task text."""

from __future__ import annotations

import unittest

from experiments.audit_appworld_v23_prompt_tokens import summarize_counts


class V23PromptTokenTests(unittest.TestCase):
    def test_summary(self) -> None:
        self.assertEqual(
            summarize_counts([1, 2, 3]),
            {"count": 3, "min": 1, "max": 3, "total": 6, "mean": 2.0},
        )

    def test_empty_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarize_counts([])


if __name__ == "__main__":
    unittest.main()
