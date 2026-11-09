"""The stronger-composer gate remains train-only and fails closed."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments import run_appworld_v28_qwen32_gate as gate
from experiments.serve_local_qwen3_32b_awq import sufficient_vram


def valid_payload() -> dict[str, object]:
    return {
        "status": "train_only_state_based_agent_pilot_not_corem_result",
        "benchmark": "AppWorld 0.1.3.post1", "split": "train",
        "task_sampling": "seeded_distinct_generators_first_variant",
        "model": gate.MODEL_NAME, "model_endpoint": gate.ENDPOINT,
        "model_digest": gate.MODEL_DIGEST, "seed": 2027,
        "appworld_random_seed": 100, "max_steps": 20,
        "context_tokens": 8192, "predict_tokens": 512,
        "prompt_variant": "checklist", "history_strategy": "preserve_task",
        "raise_on_failure": True, "num_tasks": 9,
        "memory_condition": "none", "memory_view_sha256": None,
        "doc_condition": "none", "doc_view_sha256": None,
        "script_sha256": gate.RUNNER_SHA256,
        "train_split_sha256": gate.TRAIN_SPLIT_SHA256,
        "protected_text_exported": False, "successes": 1,
        "model_errors": 0, "execution_errors": 0, "evaluation_errors": 0,
        "private_task_outcomes": [
            {"task_sha256": f"synthetic-{index}", "success": index == 0,
             "steps": 2, "model_errors": 0, "execution_errors": 0,
             "evaluation_error": False}
            for index in range(9)
        ],
    }


class V28GateTests(unittest.TestCase):
    def test_runner_is_no_memory_train_gate(self) -> None:
        argv = gate.runner_argv()
        self.assertIn("v28_qwen32_none_seed2027_n9", argv)
        self.assertEqual(argv[argv.index("--num-tasks") + 1], "9")
        self.assertEqual(argv[argv.index("--predict-tokens") + 1], "512")
        self.assertEqual(argv[argv.index("--history-strategy") + 1], "preserve_task")
        self.assertNotIn("--memory-view", argv)
        self.assertNotIn("--doc-view", argv)

    def test_result_validator_rejects_wrong_target_or_technical_error(self) -> None:
        payload = valid_payload()
        targets = {f"synthetic-{index}" for index in range(9)}
        with patch.object(gate, "expected_target_hashes", return_value=targets):
            self.assertEqual(gate.validate_result(payload), 18)
            payload["model_errors"] = 1
            with self.assertRaises(ValueError):
                gate.validate_result(payload)
            payload["model_errors"] = 0
            payload["private_task_outcomes"][0]["task_sha256"] = "unselected"
            with self.assertRaises(ValueError):
                gate.validate_result(payload)

    def test_server_vram_reserve_is_inclusive(self) -> None:
        self.assertTrue(sufficient_vram(25 * 1024**3, 25))
        self.assertFalse(sufficient_vram(25 * 1024**3 - 1, 25))


if __name__ == "__main__":
    unittest.main()
