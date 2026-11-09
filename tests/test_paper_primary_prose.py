"""Headline prose stays linked to frozen experimental summaries."""

from __future__ import annotations

import unittest

from experiments.check_paper_primary_prose import assert_text, main


class PaperPrimaryProseTests(unittest.TestCase):
    def test_frozen_headline_claims(self) -> None:
        self.assertEqual(main(), 0)

    def test_mismatched_claim_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_text("one verified success", "two verified successes", "test")


if __name__ == "__main__":
    unittest.main()
