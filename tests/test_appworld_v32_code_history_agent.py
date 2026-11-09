"""Tests for the v32 bounded-observation, complete-code context policy."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments import run_appworld_v32_code_history_agent as agent


class V32CodeHistoryAgentTests(unittest.TestCase):
    def test_context_preserves_every_code_action(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ]
        for index in range(6):
            messages.extend((
                {"role": "assistant", "content": f"code-{index}"},
                {"role": "user", "content": f"observation-{index}"},
            ))
        selected = agent.context_messages(messages, "preserve_task")
        self.assertEqual(
            [item["content"] for item in selected if item["role"] == "assistant"],
            [f"code-{index}" for index in range(6)],
        )
        self.assertEqual(selected[0:2], messages[0:2])
        self.assertEqual(selected[-1], messages[-1])
        self.assertEqual(
            sum("Earlier environment output omitted" in item["content"] for item in selected),
            2,
        )
        self.assertEqual(messages[3]["content"], "observation-0")

    def test_context_keeps_latest_long_doc_and_bounds_prior_one(self) -> None:
        long_prior = "p" * 3000
        long_latest = "l" * 3000
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "doc-1"},
            {"role": "user", "content": long_prior},
            {"role": "assistant", "content": "doc-2"},
            {"role": "user", "content": long_latest},
        ]
        selected = agent.context_messages(messages, "preserve_task")
        self.assertTrue(selected[3]["content"].startswith("p" * 2000))
        self.assertIn("Older observation shortened", selected[3]["content"])
        self.assertEqual(selected[-1]["content"], long_latest)
        with self.assertRaises(ValueError):
            agent.context_messages(messages, "preserve_code")

    def test_task_row_records_context_provenance(self) -> None:
        with patch.object(agent, "_original_run_task", return_value={"success": False}):
            row = agent.run_task()
        self.assertEqual(row["context_policy"], agent.CONTEXT_POLICY)
        self.assertEqual(len(row["context_script_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
