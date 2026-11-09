"""The context-fidelity gate changes only the agent and private outputs."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments import run_appworld_v29_qwen32_guard_gate as v29
from experiments import run_appworld_v30_qwen32_full_docs_gate as gate


class V30FullDocsGateTests(unittest.TestCase):
    def test_runner_preserves_model_budget_and_guarded_no_memory_scope(self) -> None:
        original = v29.runner_argv()
        command = gate.runner_argv()
        self.assertEqual(command[1], str(gate.AGENT))
        for flag in ("--model", "--endpoint", "--num-tasks", "--max-steps",
                     "--context-tokens", "--predict-tokens", "--prompt-variant",
                     "--history-strategy", "--seed"):
            self.assertEqual(
                original[original.index(flag) + 1],
                command[command.index(flag) + 1],
            )
        self.assertNotIn("--memory-view", command)
        self.assertNotEqual(original[original.index("--output") + 1],
                            command[command.index("--output") + 1])

    def test_validator_requires_full_doc_provenance(self) -> None:
        payload = {"private_task_outcomes": [{
            "observation_policy": "latest_complete_api_doc_older_replies_2000_prefix",
            "observation_script_sha256": gate.AGENT_SHA256,
            "full_doc_observations": 2,
            "max_doc_observation_chars": 9093,
            "steps": 3,
        }]}
        with patch.object(v29, "validate_result", return_value=(3, 1)):
            self.assertEqual(gate.validate_result(payload), (3, 1, 2))
            payload["private_task_outcomes"][0]["max_doc_observation_chars"] = 12_001
            with self.assertRaises(ValueError):
                gate.validate_result(payload)


if __name__ == "__main__":
    unittest.main()
