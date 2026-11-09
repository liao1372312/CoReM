"""Small deterministic checks for the local AppWorld agent harness."""

from __future__ import annotations

import unittest
import sys
from unittest.mock import patch
from urllib.error import HTTPError
from types import SimpleNamespace

from experiments.build_appworld_train_exemplar_hints import select_source
from experiments.run_appworld_local_agent_pilot import (
    ModelContextOverflowError,
    ModelServerResourceError,
    _context_messages,
    _execute_and_flush,
    _initial_messages,
    _next_code,
    _require_appworld_version,
    _run_task,
    _strip_code_fences,
)


class LocalAgentPilotTests(unittest.TestCase):
    def test_plain_code_is_unchanged(self) -> None:
        self.assertEqual(_strip_code_fences("print(1)\n"), "print(1)")

    def test_appworld_version_must_match_frozen_environment(self) -> None:
        _require_appworld_version("0.1.3.post1")
        with self.assertRaises(ValueError):
            _require_appworld_version("0.1.4")

    def test_python_fence_is_removed(self) -> None:
        self.assertEqual(_strip_code_fences("```python\nprint(1)\n```"), "print(1)")

    def test_embedded_fence_is_extracted(self) -> None:
        self.assertEqual(
            _strip_code_fences("Here is the code:\n```python\nprint(2)\n```\nDone."),
            "print(2)",
        )

    def test_exemplar_selection_excludes_target_generator(self) -> None:
        target = {
            "task_id": "target",
            "generator_id": "same",
            "instruction": "book a blue car",
        }
        candidates = [
            target,
            {
                "task_id": "same_2",
                "generator_id": "same",
                "instruction": "book a blue car",
            },
            {
                "task_id": "other",
                "generator_id": "different",
                "instruction": "book a car",
            },
        ]
        self.assertEqual(select_source(target, candidates)["task_id"], "other")

    def test_checklist_prompt_uses_only_supplied_identity(self) -> None:
        supervisor = SimpleNamespace(
            first_name="A", last_name="B", email="a@example.test", phone_number="123"
        )
        messages = _initial_messages("Do the task", supervisor, "checklist", None)
        self.assertIn("a@example.test", messages[1]["content"])
        self.assertIn("Do the task", messages[1]["content"])
        self.assertIn("complete_task", messages[0]["content"])

    def test_unknown_prompt_variant_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            _initial_messages("task", object(), "other", None)

    def test_worked_example_is_marked_dummy_and_preserves_real_task(self) -> None:
        supervisor = SimpleNamespace(
            first_name="A", last_name="B", email="a@example.test", phone_number="123"
        )
        messages = _initial_messages(
            "Actual task", supervisor, "worked_example", None
        )
        self.assertIn("Demonstration only", messages[0]["content"])
        self.assertIn("Do not copy", messages[0]["content"])
        self.assertIn("Actual task", messages[1]["content"])

    def test_doc_prefetch_is_added_without_memory(self) -> None:
        messages = _initial_messages(
            "task", object(), "generic", None, "prefetched API specification"
        )
        self.assertIn("prefetched API specification", messages[1]["content"])
        self.assertNotIn("prefetched API specification", messages[0]["content"])

    def test_history_preserves_task_after_many_turns(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "original task"},
            *[
                {"role": "assistant", "content": f"step {number}"}
                for number in range(20)
            ],
        ]
        self.assertNotIn(messages[1], _context_messages(messages, "legacy_tail11"))
        selected = _context_messages(messages, "preserve_task")
        self.assertEqual(selected[1], messages[1])
        self.assertEqual(selected[-8:], messages[-8:])

    def test_gpu_resource_refusal_aborts_instead_of_becoming_zero_score(self) -> None:
        refusal = HTTPError("http://127.0.0.1:11435/api/chat", 503, "busy", None, None)
        with patch(
            "experiments.run_appworld_local_agent_pilot.urllib.request.urlopen",
            side_effect=refusal,
        ):
            with self.assertRaises(ModelServerResourceError):
                _next_code(
                    "http://127.0.0.1:11435/api/chat",
                    "Qwen3.5-9B",
                    [{"role": "user", "content": "synthetic task"}],
                    2027,
                )

    def test_context_refusal_aborts_instead_of_dropping_task(self) -> None:
        refusal = HTTPError("http://127.0.0.1:11435/api/chat", 413, "long", None, None)
        with patch(
            "experiments.run_appworld_local_agent_pilot.urllib.request.urlopen",
            side_effect=refusal,
        ):
            with self.assertRaises(ModelContextOverflowError):
                _next_code(
                    "http://127.0.0.1:11435/api/chat",
                    "Qwen3.5-9B",
                    [{"role": "user", "content": "synthetic task"}],
                    2027,
                )

    def test_code_history_keeps_assignments_and_compresses_old_outputs(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "original task"},
        ]
        for index in range(8):
            messages.append({"role": "assistant", "content": f"value_{index} = {index}"})
            messages.append({"role": "user", "content": f"observation {index}"})
        selected = _context_messages(messages, "preserve_code")
        self.assertEqual(selected[1]["content"], "original task")
        self.assertIn("value_0 = 0", [row["content"] for row in selected])
        self.assertIn("value_7 = 7", [row["content"] for row in selected])
        self.assertIn("omitted", selected[3]["content"])
        self.assertEqual(selected[-1]["content"], "observation 7")
        self.assertEqual(messages[3]["content"], "observation 0")

    def test_execution_flushes_early_return_log(self) -> None:
        class FakeWorld:
            def __init__(self) -> None:
                self.flushed = False

            def execute(self, code: str) -> str:
                self.code = code
                return "Execution failed. Traceback:\nSyntax error in line:"

            def save_logs(self) -> None:
                self.flushed = True

        world = FakeWorld()
        self.assertIn("Syntax error", _execute_and_flush(world, "invalid code"))
        self.assertTrue(world.flushed)
        self.assertEqual(world.code, "invalid code")

    def test_explicit_api_failure_feedback_is_passed_to_world(self) -> None:
        seen: dict[str, object] = {}

        class FakeWorld:
            def __init__(self, **kwargs: object) -> None:
                seen.update(kwargs)
                self.task = SimpleNamespace(instruction="synthetic task", supervisor=object())

            def __enter__(self) -> "FakeWorld":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def execute(self, code: str) -> str:
                return "ok"

            def save_logs(self) -> None:
                return None

            def task_completed(self) -> bool:
                return True

            def evaluate(self) -> object:
                return SimpleNamespace(success=False)

        with patch.dict(sys.modules, {"appworld": SimpleNamespace(AppWorld=FakeWorld)}):
            with patch(
                "experiments.run_appworld_local_agent_pilot._next_code",
                return_value="print(1)",
            ):
                result = _run_task(
                    "synthetic_train_task",
                    "synthetic_experiment",
                    "http://127.0.0.1:11435/api/chat",
                    "Qwen3.5-9B-4bit",
                    1,
                    2027,
                    raise_on_failure=True,
                )
        self.assertIs(seen["raise_on_failure"], True)
        self.assertEqual(seen["random_seed"], 100)
        self.assertEqual(result["steps"], 1)


if __name__ == "__main__":
    unittest.main()
