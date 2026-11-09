"""The v23 launcher preserves paired settings and never aliases outputs."""

from __future__ import annotations

import unittest
from pathlib import Path

from experiments.run_appworld_v23_frozen_pair import (
    ARM_ORDER,
    result_path,
    runner_argv,
    token_cost_delta,
)


class V23FrozenPairTests(unittest.TestCase):
    def test_runner_arms_only_change_output_name_and_memory_view(self) -> None:
        python = Path("/test/python")
        commands = {arm: runner_argv(arm, python) for arm in ARM_ORDER}
        self.assertEqual(len({result_path(arm) for arm in ARM_ORDER}), 4)
        fixed = ["--num-tasks", "9", "--max-steps", "20", "--context-tokens", "6144",
                 "--predict-tokens", "512", "--prompt-variant", "checklist",
                 "--history-strategy", "preserve_code", "--raise-on-failure",
                 "--seed", "2027"]
        for command in commands.values():
            self.assertEqual(command[command.index("--num-tasks"):command.index("--seed") + 2], fixed)
            for index, token in enumerate(fixed):
                if token.startswith("--"):
                    self.assertIn(token, command)
                    if index + 1 < len(fixed) and not fixed[index + 1].startswith("--"):
                        self.assertEqual(command[command.index(token) + 1], fixed[index + 1])
        self.assertNotIn("--memory-view", commands["none"])
        for arm in ARM_ORDER[1:]:
            self.assertIn("--memory-view", commands[arm])
        self.assertTrue(
            commands["disjoint"][commands["disjoint"].index("--memory-view") + 1]
            .endswith("appworld_train_static_disjoint_edges_seed2027_n9.json")
        )

    def test_unknown_arm_rejected(self) -> None:
        with self.assertRaises(ValueError):
            runner_argv("dev", Path("/test/python"))

    def test_token_cost_is_paired_to_clean_model_requests(self) -> None:
        before = {
            "requests": 3, "accepted_requests": 3,
            "truncated_requests": 0, "rejected_oversize_requests": 0,
            "prompt_tokens_sum": 450, "completion_tokens_sum": 80,
            "max_prompt_tokens": 200,
        }
        after = {
            "requests": 5, "accepted_requests": 5,
            "truncated_requests": 0, "rejected_oversize_requests": 0,
            "prompt_tokens_sum": 750, "completion_tokens_sum": 130,
            "max_prompt_tokens": 210,
        }
        self.assertEqual(
            token_cost_delta(before, after, 2),
            {"requests": 2, "prompt_tokens": 300, "completion_tokens": 50,
             "server_cumulative_max_prompt_tokens": 210},
        )
        with self.assertRaises(ValueError):
            token_cost_delta(before, after, 1)
        after["rejected_oversize_requests"] = 1
        with self.assertRaises(ValueError):
            token_cost_delta(before, after, 2)


if __name__ == "__main__":
    unittest.main()
