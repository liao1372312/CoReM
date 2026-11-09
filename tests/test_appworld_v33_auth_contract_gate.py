"""The v33 launcher must isolate its authentication prompt."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments import run_appworld_v32_code_history_gate as v32
from experiments import run_appworld_v33_auth_contract_gate as gate


class V33AuthContractGateTests(unittest.TestCase):
    def test_runner_changes_only_agent_and_private_output(self) -> None:
        original = v32.runner_argv()
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

    def test_validator_requires_auth_prompt_provenance(self) -> None:
        payload = {"private_task_outcomes": [{
            "auth_prompt_policy": gate.agent.PROMPT_POLICY,
            "auth_prompt_script_sha256": gate.AGENT_SHA256,
        }]}
        with patch.object(v32, "validate_result", return_value=(7, 3, 4)):
            self.assertEqual(gate.validate_result(payload), (7, 3, 4))
            payload["private_task_outcomes"][0]["auth_prompt_policy"] = "wrong"
            with self.assertRaises(ValueError):
                gate.validate_result(payload)


if __name__ == "__main__":
    unittest.main()
