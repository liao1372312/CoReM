"""Tests for the explicit AppWorld authentication contract."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments import run_appworld_v33_auth_contract_agent as agent


class V33AuthContractAgentTests(unittest.TestCase):
    def test_prompt_requires_explicit_per_app_token(self) -> None:
        supervisor = SimpleNamespace(
            first_name="First", last_name="Last", email="x@example.test",
            phone_number="000",
        )
        messages = agent.initial_messages(
            "do task", supervisor, "checklist", None, None
        )
        system = messages[0]["content"]
        self.assertIn("access_token=<saved token>", system)
        self.assertIn("a successful login does not implicitly authenticate", system)
        self.assertIn("HTTP 401", system)
        self.assertIn("HTTP 422", system)
        self.assertIn("Task:\ndo task", messages[1]["content"])

    def test_initial_messages_does_not_mutate_parent_result(self) -> None:
        parent = [
            {"role": "system", "content": "base"},
            {"role": "user", "content": "task"},
        ]
        with patch.object(agent, "_original_initial_messages", return_value=parent):
            result = agent.initial_messages("x", object(), "checklist", None)
        self.assertEqual(parent[0]["content"], "base")
        self.assertIn(agent.AUTH_CONTRACT, result[0]["content"])

    def test_task_row_records_auth_prompt_provenance(self) -> None:
        with patch.object(agent, "_original_run_task", return_value={"success": False}):
            row = agent.run_task()
        self.assertEqual(row["auth_prompt_policy"], agent.PROMPT_POLICY)
        self.assertEqual(len(row["auth_prompt_script_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
