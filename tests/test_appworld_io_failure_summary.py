"""Aggregate-only interaction diagnostics for the private AppWorld pilot."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.analyze_appworld_io_failures import summarize_io


class InteractionSummaryTests(unittest.TestCase):
    def test_counts_clipped_docs_and_exception_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "tasks" / "sample" / "logs" / "environment_io.md"
            path.parent.mkdir(parents=True)
            path.write_text(
                "\n### Environment Interaction 1\n---\n```python\n"
                "print(apis.api_docs.show_app_descriptions())\n```\n\n```\n"
                + "x" * 2001
                + "\n```\n\n"
                "\n### Environment Interaction 2\n---\n```python\n"
                "print(apis.foo())\n```\n\n```\n"
                "Execution failed. Traceback:\nValueError: secret\n```\n\n"
            )
            self.assertEqual(
                summarize_io(root, 1),
                {
                    "task_logs": 1,
                    "interactions": 2,
                    "outputs_over_2000_chars": 1,
                    "message_only_json_outputs": 0,
                    "api_doc_interactions": 1,
                    "api_doc_outputs_over_2000_chars": 1,
                    "execution_failures": 1,
                    "syntax_failures": 0,
                    "safety_failures": 0,
                    "runtime_or_other_failures": 1,
                    "failure_outputs_over_2000_chars": 0,
                    "failures_by_exception_class": {"ValueError": 1},
                    "protected_text_exported": False,
                },
            )

    def test_incomplete_log_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "tasks" / "sample" / "logs" / "environment_io.md"
            path.parent.mkdir(parents=True)
            path.write_text("private malformed content")
            with self.assertRaises(ValueError):
                summarize_io(root, 1)

    def test_aggregate_reveals_missing_final_syntax_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "tasks" / "sample" / "logs" / "environment_io.md"
            path.parent.mkdir(parents=True)
            path.write_text(
                "\n### Environment Interaction 1\n---\n```python\n"
                "print(1)\n```\n\n```\n1\n```\n\n"
            )
            aggregate_path = root / "aggregate.json"
            aggregate_path.write_text(
                json.dumps(
                    {
                        "num_tasks": 1,
                        "private_task_outcomes": [
                            {
                                "task_sha256": hashlib.sha256(b"sample").hexdigest(),
                                "steps": 2,
                                "syntax_failure_messages": 1,
                            }
                        ],
                        "execution_failure_messages": 1,
                        "syntax_failure_messages": 1,
                    }
                )
            )
            result = summarize_io(root, 1, aggregate_path)
            self.assertEqual(result["steps_missing_from_logs"], 1)
            self.assertEqual(result["failures_missing_from_logs"], 1)
            self.assertEqual(result["aggregate_recorded_syntax_failures"], 1)
            self.assertEqual(result["tasks_with_missing_log_steps"], 1)
            self.assertTrue(result["missing_steps_accounted_for_by_syntax"])

    def test_message_only_json_is_counted_without_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "tasks" / "sample" / "logs" / "environment_io.md"
            path.parent.mkdir(parents=True)
            path.write_text(
                "\n### Environment Interaction 1\n---\n```python\n"
                "print(apis.foo())\n```\n\n```\n"
                '{"message":"private value"}\n```\n\n'
            )
            result = summarize_io(root, 1)
            self.assertEqual(result["message_only_json_outputs"], 1)
            self.assertNotIn("private value", str(result))


if __name__ == "__main__":
    unittest.main()
