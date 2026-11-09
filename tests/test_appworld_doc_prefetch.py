"""Synthetic checks for label-free AppWorld API-doc prefetch."""

from __future__ import annotations

import unittest

from experiments.build_appworld_train_doc_prefetch import select_docs


class DocPrefetchTests(unittest.TestCase):
    def test_selects_complete_docs_in_lexical_order(self) -> None:
        docs = [
            {
                "api": "mail.send",
                "index_text": "mail send message",
                "specification": "SEND",
            },
            {
                "api": "calendar.find",
                "index_text": "calendar search event",
                "specification": "FIND",
            },
        ]
        selected, names = select_docs("find calendar event", docs, 2, 100)
        self.assertEqual(names[0], "calendar.find")
        self.assertEqual(selected[0], "FIND")

    def test_budget_never_truncates_a_specification(self) -> None:
        docs = [
            {"api": "a.first", "index_text": "first", "specification": "abcdef"},
            {"api": "b.second", "index_text": "second", "specification": "uvwxyz"},
        ]
        selected, names = select_docs("first", docs, 2, 10)
        self.assertEqual(selected, ["abcdef"])
        self.assertEqual(names, ["a.first"])

    def test_too_small_budget_fails_closed(self) -> None:
        docs = [{"api": "a.first", "index_text": "first", "specification": "abcdef"}]
        with self.assertRaises(ValueError):
            select_docs("first", docs, 1, 2)


if __name__ == "__main__":
    unittest.main()
