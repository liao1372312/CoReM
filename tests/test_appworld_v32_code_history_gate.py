"""The v32 launcher must isolate the code-history intervention."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments import run_appworld_v31_fresh_train_gate as v31
from experiments import run_appworld_v32_code_history_gate as gate


class V32CodeHistoryGateTests(unittest.TestCase):
    def test_runner_changes_only_agent_and_private_output(self) -> None:
        original = v31.runner_argv()
        command = gate.runner_argv()
        for flag in (
            "--model", "--endpoint", "--num-tasks", "--max-steps",
            "--context-tokens", "--predict-tokens", "--prompt-variant",
            "--history-strategy", "--seed",
        ):
            self.assertEqual(
                original[original.index(flag) + 1],
                command[command.index(flag) + 1],
            )
        self.assertEqual(command[1], str(gate.AGENT))
        self.assertNotIn("--memory-view", command)
        self.assertNotIn("--doc-view", command)
        self.assertEqual(command[command.index("--output") + 1], str(gate.RESULT))

    def test_validator_requires_context_provenance(self) -> None:
        payload = {"private_task_outcomes": [{
            "context_policy": gate.agent.CONTEXT_POLICY,
            "context_script_sha256": gate.AGENT_SHA256,
        }]}
        with patch.object(v31, "validate_result", return_value=(5, 2, 3)):
            self.assertEqual(gate.validate_result(payload), (5, 2, 3))
            payload["private_task_outcomes"][0]["context_policy"] = "wrong"
            with self.assertRaises(ValueError):
                gate.validate_result(payload)


if __name__ == "__main__":
    unittest.main()
