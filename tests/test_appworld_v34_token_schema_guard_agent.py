"""Tests for the read-only token-schema guard."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experiments import run_appworld_v34_token_schema_guard_agent as agent


class V34TokenSchemaGuardAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        show_doc = Mock(return_value={
            "parameters": [
                {"name": "access_token"},
                {"name": "query"},
            ]
        })
        self.world = SimpleNamespace(
            apis=SimpleNamespace(api_docs=SimpleNamespace(show_api_doc=show_doc))
        )
        self.names = {"simple_note": {"search_notes"}, "api_docs": {"show_api_doc"}}

    def test_guard_replaces_missing_token_with_read_only_docs(self) -> None:
        code, target = agent.token_schema_guard(
            "print(apis.simple_note.search_notes(query='x'))",
            self.world,
            self.names,
        )
        self.assertEqual(target, ("simple_note", "search_notes"))
        self.assertEqual(
            code,
            "print(apis.api_docs.show_api_doc("
            "app_name='simple_note', api_name='search_notes'))",
        )

    def test_guard_allows_explicit_token_and_double_star_kwargs(self) -> None:
        explicit = (
            "print(apis.simple_note.search_notes("
            "access_token=token, query='x'))"
        )
        self.assertEqual(
            agent.token_schema_guard(explicit, self.world, self.names),
            (explicit, None),
        )
        kwargs = "print(apis.simple_note.search_notes(**params))"
        self.assertEqual(
            agent.token_schema_guard(kwargs, self.world, self.names),
            (kwargs, None),
        )

    def test_guard_ignores_api_without_token_parameter(self) -> None:
        self.world.apis.api_docs.show_api_doc.return_value = {
            "parameters": [{"name": "username"}, {"name": "password"}]
        }
        code = "print(apis.simple_note.search_notes(query='x'))"
        self.assertEqual(
            agent.token_schema_guard(code, self.world, self.names),
            (code, None),
        )

    def test_task_row_records_guard_provenance(self) -> None:
        with patch.object(agent, "_original_run_task", return_value={"success": False}):
            row = agent.run_task()
        self.assertEqual(row["token_schema_guard_interventions"], 0)
        self.assertEqual(row["token_schema_guard_policy"], agent.TOKEN_GUARD_POLICY)
        self.assertEqual(len(row["token_schema_guard_script_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
