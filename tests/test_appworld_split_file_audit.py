"""Split fingerprints must not disclose IDs or accept overlap."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.audit_appworld_split_files import SPLITS, audit_splits


class AppWorldSplitAuditTests(unittest.TestCase):
    def _root(self, directory: str) -> Path:
        root = Path(directory)
        split_root = root / "data" / "datasets"
        split_root.mkdir(parents=True)
        for index, split in enumerate(SPLITS):
            (split_root / f"{split}.txt").write_text(f"task_{index}\n")
        return root

    def test_only_counts_and_hashes_are_exported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = audit_splits(self._root(directory))
            self.assertEqual(result["total_unique_task_ids"], 4)
            self.assertFalse(result["task_ids_exported"])
            self.assertNotIn("task_0", str(result))

    def test_cross_split_overlap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            (root / "data" / "datasets" / "dev.txt").write_text("task_0\n")
            with self.assertRaises(ValueError):
                audit_splits(root)


if __name__ == "__main__":
    unittest.main()
