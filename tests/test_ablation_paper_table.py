"""The panelled ablation table must retain its setting-specific baselines."""

from __future__ import annotations

import hashlib
import json
import unittest

from experiments.generate_ablation_paper_table import SOURCES, render_table


class AblationPaperTableTests(unittest.TestCase):
    def test_frozen_table_and_coverage_rejection(self) -> None:
        raw = {key: path.read_bytes() for key, path in SOURCES.items()}
        sources = {key: json.loads(value) for key, value in raw.items()}
        hashes = {key: hashlib.sha256(value).hexdigest() for key, value in raw.items()}
        rendered = render_table(sources, hashes)
        self.assertIn(
            r"\quad w/o non-additive set utility & Controlled challenge (64) & 23.4 & $-26.6$",
            rendered,
        )
        self.assertIn(
            r"\quad Uniform workload forecast & Past-only storage sweep & 69.4 & $+1.6$",
            rendered,
        )
        sources["v6"]["method_metrics"][0]["events"] = 64
        with self.assertRaises(ValueError):
            render_table(sources, hashes)


if __name__ == "__main__":
    unittest.main()
