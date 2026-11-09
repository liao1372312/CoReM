"""The post-v28 repair changes only the agent wrapper and private outputs."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments import run_appworld_v28_qwen32_gate as v28
from experiments import run_appworld_v29_qwen32_guard_gate as gate


class V29GuardGateTests(unittest.TestCase):
    def test_runner_reuses_model_and_settings_with_fresh_outputs(self) -> None:
        original = v28.runner_argv()
        guarded = gate.runner_argv()
        self.assertEqual(original[0], guarded[0])
        self.assertEqual(guarded[1], str(gate.GUARD))
        self.assertNotEqual(original[1], guarded[1])
        for flag in ("--model", "--endpoint", "--num-tasks", "--max-steps",
                     "--context-tokens", "--predict-tokens", "--prompt-variant",
                     "--history-strategy", "--seed"):
            self.assertEqual(
                original[original.index(flag) + 1],
                guarded[guarded.index(flag) + 1],
            )
        self.assertNotIn("--memory-view", guarded)
        self.assertNotEqual(original[original.index("--output") + 1],
                            guarded[guarded.index("--output") + 1])

    def test_validator_rejects_missing_guard_provenance(self) -> None:
        payload = {"private_task_outcomes": [{
            "api_guard_interventions": 2,
            "api_guard_policy": "read_only_documentation_for_invalid_nested_api_paths",
            "api_guard_script_sha256": gate.GUARD_SHA256,
        }]}
        with patch.object(v28, "validate_result", return_value=3):
            self.assertEqual(gate.validate_result(payload), (3, 2))
            payload["private_task_outcomes"][0]["api_guard_script_sha256"] = "bad"
            with self.assertRaises(ValueError):
                gate.validate_result(payload)


if __name__ == "__main__":
    unittest.main()
