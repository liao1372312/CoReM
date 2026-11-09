"""Preserve complete current API documentation without mutating task history."""

from __future__ import annotations

import unittest

from experiments.run_appworld_v30_full_docs_agent import (
    MAX_COMPLETE_DOC_CHARS, context_messages, history_observation,
)


class FullDocsAgentTests(unittest.TestCase):
    def test_documentation_is_kept_whole(self) -> None:
        document = "API guard:\n" + "method_specification " * 400
        selected, is_doc = history_observation("apis.fake.method()", document)
        self.assertTrue(is_doc)
        self.assertEqual(selected, document)

    def test_non_doc_observation_retains_tail(self) -> None:
        observation = "a" * 3000 + "final status"
        selected, is_doc = history_observation("print(1)", observation)
        self.assertFalse(is_doc)
        self.assertEqual(selected, observation[-2000:])

    def test_oversize_documentation_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            history_observation("print(apis.api_docs.show_app_descriptions())",
                                "d" * (MAX_COMPLETE_DOC_CHARS + 1))

    def test_context_keeps_task_latest_doc_and_compacts_older(self) -> None:
        old_doc = "old " * 2000
        latest_doc = "new " * 2000
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "show docs"},
            {"role": "user", "content": old_doc},
            {"role": "assistant", "content": "show newer docs"},
            {"role": "user", "content": latest_doc},
        ]
        selected = context_messages(messages, "preserve_task")
        self.assertEqual(selected[1]["content"], "task")
        self.assertLess(len(selected[3]["content"]), len(old_doc))
        self.assertEqual(selected[-1]["content"], latest_doc)
        self.assertEqual(messages[3]["content"], old_doc)
        with self.assertRaises(ValueError):
            context_messages(messages, "legacy_tail11")


if __name__ == "__main__":
    unittest.main()
