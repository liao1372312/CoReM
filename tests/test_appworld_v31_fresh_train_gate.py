"""The fresh-train gate must preserve the frozen v30 executor."""

from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

from experiments import run_appworld_v30_qwen32_full_docs_gate as v30
from experiments import run_appworld_v31_fresh_train_gate as gate


class V31FreshTrainGateTests(unittest.TestCase):
    def test_runner_changes_only_seed_and_private_output(self) -> None:
        original = v30.runner_argv()
        command = gate.runner_argv()
        for flag in ("--model", "--endpoint", "--num-tasks", "--max-steps",
                     "--context-tokens", "--predict-tokens", "--prompt-variant",
                     "--history-strategy"):
            self.assertEqual(original[original.index(flag) + 1],
                             command[command.index(flag) + 1])
        self.assertEqual(command[1], str(v30.AGENT))
        self.assertEqual(command[command.index("--seed") + 1], "2033")
        self.assertNotIn("--memory-view", command)
        self.assertNotIn("--doc-view", command)
        self.assertEqual(command[command.index("--output") + 1], str(gate.RESULT))

    @unittest.skipUnless(importlib.util.find_spec("appworld"),
                         "AppWorld is installed in the dedicated runner environment")
    def test_fresh_targets_are_distinct_from_prior_targets(self) -> None:
        from appworld.task import task_id_to_generator_id
        from experiments.run_appworld_local_agent_pilot import _sample_task_ids

        self.assertEqual(len(gate.fresh_target_hashes()), 9)
        task_ids = (gate.v28.APPWORLD_ROOT / "data" / "datasets" / "train.txt").read_text().splitlines()
        prior_generators = {
            task_id_to_generator_id(task_id)
            for task_id in _sample_task_ids(task_ids, 2027, 9)
        }
        fresh_generators = {
            task_id_to_generator_id(task_id)
            for task_id in _sample_task_ids(task_ids, 2033, 9)
        }
        self.assertEqual(len(prior_generators), 9)
        self.assertEqual(len(fresh_generators), 9)
        self.assertFalse(prior_generators & fresh_generators)

    def test_validator_rejects_wrong_targets(self) -> None:
        with patch.object(gate, "fresh_target_hashes", return_value={"expected"}):
            with self.assertRaisesRegex(ValueError, "frozen v30 settings"):
                gate.validate_result({"seed": 2027})
            with self.assertRaisesRegex(ValueError, "incomplete task rows"):
                gate.validate_result({
                    "status": "train_only_state_based_agent_pilot_not_corem_result",
                    "benchmark": "AppWorld 0.1.3.post1",
                    "split": "train",
                    "task_sampling": "seeded_distinct_generators_first_variant",
                    "model": gate.v28.MODEL_NAME,
                    "model_endpoint": gate.v28.ENDPOINT,
                    "model_digest": gate.v28.MODEL_DIGEST,
                    "seed": 2033,
                    "appworld_random_seed": 100,
                    "max_steps": 20,
                    "context_tokens": 8192,
                    "predict_tokens": 512,
                    "prompt_variant": "checklist",
                    "history_strategy": "preserve_task",
                    "raise_on_failure": True,
                    "num_tasks": 9,
                    "memory_condition": "none",
                    "memory_view_sha256": None,
                    "doc_condition": "none",
                    "doc_view_sha256": None,
                    "script_sha256": gate.RUNNER_SHA256,
                    "train_split_sha256": gate.TRAIN_SPLIT_SHA256,
                    "protected_text_exported": False,
                    "model_errors": 0,
                    "execution_errors": 0,
                    "evaluation_errors": 0,
                    "private_task_outcomes": [],
                })


if __name__ == "__main__":
    unittest.main()
