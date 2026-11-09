"""Synthetic checks for aggregate-only AppWorld report summaries."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.summarize_appworld_state_reports import summarize_reports


class StateReportSummaryTests(unittest.TestCase):
    def test_counts_requirements_without_task_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, passed, failed in (("one", 2, 1), ("two", 3, 0)):
                path = root / "tasks" / name / "evaluation" / "report.md"
                path.parent.mkdir(parents=True)
                path.write_text(f"Passed Tests : {passed}\nFailed Tests : {failed}\n")
            self.assertEqual(
                summarize_reports(root, 2),
                {
                    "reports": 2,
                    "passed_requirements": 5,
                    "failed_requirements": 1,
                    "success_reports": 1,
                },
            )

    def test_wrong_report_count_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                summarize_reports(Path(directory), 1)


if __name__ == "__main__":
    unittest.main()
