"""The v34 launcher must isolate the token-schema guard."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments import run_appworld_v33_auth_contract_gate as v33
from experiments import run_appworld_v34_token_schema_guard_gate as gate


class V34TokenSchemaGuardGateTests(unittest.TestCase):
    def test_runner_changes_only_agent_and_private_output(self) -> None:
        original = v33.runner_argv()
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

    def test_validator_requires_token_guard_provenance(self) -> None:
        payload = {"private_task_outcomes": [{
            "token_schema_guard_interventions": 2,
            "token_schema_guard_policy": gate.agent.TOKEN_GUARD_POLICY,
            "token_schema_guard_script_sha256": gate.AGENT_SHA256,
        }]}
        with patch.object(v33, "validate_result", return_value=(7, 3, 4)):
            self.assertEqual(gate.validate_result(payload), (7, 3, 4, 2))
            payload["private_task_outcomes"][0]["token_schema_guard_interventions"] = -1
            with self.assertRaises(ValueError):
                gate.validate_result(payload)


if __name__ == "__main__":
    unittest.main()
