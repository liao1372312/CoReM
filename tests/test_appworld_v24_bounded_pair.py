"""The post-failure train pilot has fresh artifacts and matched settings."""

from __future__ import annotations

import unittest
from pathlib import Path

from experiments.run_appworld_v24_bounded_pair import (
    ARM_ORDER, experiment_dir, result_path, runner_argv, stats_path,
)


class V24BoundedPairTests(unittest.TestCase):
    def test_new_paths_do_not_alias_v23_or_other_arms(self) -> None:
        self.assertEqual(len({result_path(arm) for arm in ARM_ORDER}), 4)
        self.assertEqual(len({stats_path(arm) for arm in ARM_ORDER}), 4)
        self.assertEqual(len({experiment_dir(arm) for arm in ARM_ORDER}), 4)
        for arm in ARM_ORDER:
            self.assertIn("v24", str(result_path(arm)))
            self.assertIn("v24", str(stats_path(arm)))
            self.assertIn("v24", str(experiment_dir(arm)))

    def test_all_arms_use_identical_bounded_composer_settings(self) -> None:
        commands = {arm: runner_argv(arm, Path("/test/python")) for arm in ARM_ORDER}
        expected = {
            "--num-tasks": "9", "--max-steps": "20",
            "--context-tokens": "8192", "--predict-tokens": "512",
            "--prompt-variant": "checklist", "--history-strategy": "preserve_task",
            "--seed": "2027",
        }
        for command in commands.values():
            for flag, value in expected.items():
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertIn("--raise-on-failure", command)
        self.assertNotIn("--memory-view", commands["none"])
        for arm in ARM_ORDER[1:]:
            self.assertIn("--memory-view", commands[arm])

    def test_unknown_arm_refused(self) -> None:
        with self.assertRaises(ValueError):
            runner_argv("test", Path("/test/python"))


if __name__ == "__main__":
    unittest.main()
