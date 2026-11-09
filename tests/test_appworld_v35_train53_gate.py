"""Tests for the frozen, disjoint v35 53-task train stress test."""

from __future__ import annotations

import importlib.util
import unittest

from experiments import run_appworld_v33_auth_contract_gate as v33
from experiments import run_appworld_v35_train53_agent as agent
from experiments import run_appworld_v35_train53_gate as gate


class V35Train53GateTests(unittest.TestCase):
    def test_runner_preserves_executor_settings_and_changes_shard(self) -> None:
        original = v33.runner_argv()
        for shard, size in enumerate(agent.SHARD_SIZES):
            command = gate.runner_argv(shard)
            for flag in (
                "--model", "--endpoint", "--max-steps", "--context-tokens",
                "--predict-tokens", "--prompt-variant", "--history-strategy",
            ):
                self.assertEqual(
                    original[original.index(flag) + 1],
                    command[command.index(flag) + 1],
                )
            self.assertEqual(command[1], str(gate.AGENT))
            self.assertEqual(command[command.index("--num-tasks") + 1], str(size))
            self.assertEqual(command[command.index("--seed") + 1], "2035")
            self.assertEqual(
                command[command.index("--output") + 1], str(gate.RESULTS[shard])
            )
            self.assertNotIn("--memory-view", command)
            self.assertNotIn("--doc-view", command)

    def test_sampler_rejects_undeclared_invocations(self) -> None:
        with self.assertRaises(ValueError):
            agent.sample_shard([], 2034, 30)
        with self.assertRaises(ValueError):
            agent.sample_shard([], 2035, 29)

    @unittest.skipUnless(
        importlib.util.find_spec("appworld"),
        "AppWorld is installed in the dedicated runner environment",
    )
    def test_selection_is_unique_disjoint_and_committed(self) -> None:
        from appworld.task import task_id_to_generator_id

        path = gate.v28.APPWORLD_ROOT / "data/datasets/train.txt"
        task_ids = path.read_text().splitlines()
        selected = agent.selected_task_ids(task_ids)
        excluded = set(agent._original_sampler(task_ids, 2033, 9))
        first = agent.sample_shard(task_ids, 2035, 30)
        second = agent.sample_shard(task_ids, 2035, 23)
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(len(selected), 53)
        self.assertEqual(len({task_id_to_generator_id(x) for x in selected}), 30)
        self.assertFalse(set(selected) & excluded)
        self.assertFalse(set(first) & set(second))
        self.assertEqual(first + second, selected)
        self.assertEqual(
            agent.selection_digest(task_ids),
            "671f959f27d01c2b7da7f7863d1a9b227fec3cd2678d0dbbe99b6decf63513cf",
        )


if __name__ == "__main__":
    unittest.main()
