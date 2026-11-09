"""Thinking responses execute only the final answer and fail closed."""

from __future__ import annotations

import unittest
from pathlib import Path

from experiments.run_appworld_v27_thinking_agent import strip_thinking
from experiments.run_appworld_v27_thinking_gate import runner_argv


class ThinkingAgentTests(unittest.TestCase):
    def test_discards_private_reasoning_before_execution(self) -> None:
        self.assertEqual(strip_thinking("<think>reason</think>\nprint(1)"), "print(1)")

    def test_discards_fenced_final_code(self) -> None:
        self.assertEqual(
            strip_thinking("<think>reason</think>\n```python\nprint(1)\n```"),
            "print(1)",
        )

    def test_unfinished_reasoning_refused(self) -> None:
        with self.assertRaises(ValueError):
            strip_thinking("<think>reason without final answer")

    def test_gate_uses_fresh_thinking_settings(self) -> None:
        command = runner_argv(Path("/test/python"))
        self.assertIn("run_appworld_v27_thinking_agent.py", command[1])
        self.assertEqual(command[command.index("--predict-tokens") + 1], "1024")
        self.assertEqual(command[command.index("--context-tokens") + 1], "8192")
        self.assertNotIn("--memory-view", command)


if __name__ == "__main__":
    unittest.main()
