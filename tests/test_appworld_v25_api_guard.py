"""Invalid AppWorld API calls become read-only documentation interactions."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.run_appworld_v25_guard_gate import runner_argv
from experiments.run_appworld_v25_api_guard import (
    guard_code, guarded_execute_and_flush, safe_api_names,
)


NAMES = {
    "api_docs": {"show_app_descriptions", "show_api_descriptions", "show_api_doc"},
    "spotify": {"login", "search_songs", "create_playlist"},
}


class ApiGuardTests(unittest.TestCase):
    def test_gate_keeps_v24_train_composer_settings_without_memory(self) -> None:
        command = runner_argv(Path("/test/python"))
        for option, value in {
            "--num-tasks": "9", "--max-steps": "20",
            "--context-tokens": "8192", "--predict-tokens": "512",
            "--prompt-variant": "checklist", "--history-strategy": "preserve_task",
            "--seed": "2027",
        }.items():
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertNotIn("--memory-view", command)
        self.assertIn("--raise-on-failure", command)

    def test_reads_only_task_visible_api_names(self) -> None:
        world = SimpleNamespace(apis={"api_docs": {"show_app_descriptions": object()},
                                      "spotify": {"login": object()}})
        self.assertEqual(safe_api_names(world), {
            "api_docs": {"show_app_descriptions"}, "spotify": {"login"},
        })

    def test_valid_call_is_unchanged(self) -> None:
        code = "print(apis.spotify.search_songs(query='music'))"
        self.assertEqual(guard_code(code, NAMES), (code, None))

    def test_invalid_method_uses_read_only_docs(self) -> None:
        code, reason = guard_code("apis.spotify.search_tracks(query='x')", NAMES)
        self.assertEqual(reason, "invalid_method")
        self.assertEqual(
            code,
            "print(apis.api_docs.show_api_descriptions(app_name='spotify'))",
        )

    def test_invalid_app_lists_apps(self) -> None:
        code, reason = guard_code("apis.unknown.search(query='x')", NAMES)
        self.assertEqual(reason, "invalid_app")
        self.assertEqual(code, "print(apis.api_docs.show_app_descriptions())")

    def test_earlier_valid_state_action_is_not_executed_with_invalid_code(self) -> None:
        generated = "apis.spotify.create_playlist(name='x')\napis.spotify.fake()"
        code, reason = guard_code(generated, NAMES)
        self.assertEqual(reason, "invalid_method")
        self.assertNotIn("create_playlist", code)

    def test_syntax_error_is_left_for_environment_feedback(self) -> None:
        code = "apis.spotify.search_songs("
        self.assertEqual(guard_code(code, NAMES), (code, None))

    def test_guard_never_executes_invalid_generated_code(self) -> None:
        world = SimpleNamespace(apis={key: {name: object() for name in values}
                                      for key, values in NAMES.items()})
        with patch(
            "experiments.run_appworld_v25_api_guard._original_execute_and_flush",
            return_value="documented methods",
        ) as execute:
            observation = guarded_execute_and_flush(world, "apis.spotify.fake()")
        self.assertIn("not executed", observation)
        self.assertIn("documented methods", observation)
        execute.assert_called_once_with(
            world, "print(apis.api_docs.show_api_descriptions(app_name='spotify'))"
        )


if __name__ == "__main__":
    unittest.main()
