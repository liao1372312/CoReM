"""Portable manifest resolution must preserve strict frozen-file integrity."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from experiments.summarize_independent_v7_confirmation import (
    relocated_path,
    verify_frozen_sources,
)


class IndependentV7SummaryTest(unittest.TestCase):
    def test_relocated_file_is_verified_by_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data" / "events.jsonl"
            source.parent.mkdir()
            source.write_bytes(b"frozen events\n")
            frozen = {
                "dataset": {
                    "events": {
                        "path": "/old/repository/data/events.jsonl",
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                }
            }
            self.assertEqual(
                verify_frozen_sources(frozen, Path("/old/repository"), root), 1
            )
            source.write_bytes(b"changed events\n")
            with self.assertRaisesRegex(ValueError, "frozen source changed"):
                verify_frozen_sources(frozen, Path("/old/repository"), root)

    def test_path_outside_original_repository_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "escapes repository"):
            relocated_path(
                "/other/private-data.jsonl", Path("/old/repository"), Path("/tmp/new")
            )


if __name__ == "__main__":
    unittest.main()
