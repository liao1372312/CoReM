"""Fresh-query paper rows reconcile the summary and method analyses."""

from __future__ import annotations

import hashlib
import json
import unittest

from experiments.generate_fresh_query_paper_table import ANALYSES, SUMMARY, render_table


class FreshQueryPaperTableTests(unittest.TestCase):
    def test_frozen_inputs_and_mismatch_rejection(self) -> None:
        paths = {"summary": SUMMARY, **ANALYSES}
        payloads = {key: json.loads(path.read_bytes()) for key, path in paths.items()}
        hashes = {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in paths.items()
        }
        analyses = {key: payloads[key] for key in ANALYSES}
        rendered = render_table(payloads["summary"], analyses, hashes)
        self.assertIn(r"\corm & \textbf{79.7}", rendered)
        self.assertIn("Version BM25 + closure & 78.1", rendered)
        self.assertIn(r"\end{tabular}", rendered)
        analyses["corem_query_only"]["methods"][0]["vsr"] = 0
        with self.assertRaises(ValueError):
            render_table(payloads["summary"], analyses, hashes)


if __name__ == "__main__":
    unittest.main()
