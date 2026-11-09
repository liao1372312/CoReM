"""Nested invalid AppWorld API paths are redirected to documented names."""

from __future__ import annotations

import unittest

from experiments.run_appworld_v26_nested_guard import api_attribute_chain, guard_code

NAMES = {
    "api_docs": {"show_app_descriptions", "show_api_descriptions", "show_api_doc"},
    "spotify": {"login", "search_songs", "create_playlist"},
}


class NestedApiGuardTests(unittest.TestCase):
    def test_nested_invalid_method_is_guarded(self) -> None:
        code = "print(apis.spotify.api_docs.show_app_descriptions())"
        self.assertEqual(
            guard_code(code, NAMES),
            (
                "print(apis.api_docs.show_api_descriptions(app_name='spotify'))",
                "invalid_method",
            ),
        )

    def test_valid_documentation_call_is_unchanged(self) -> None:
        code = "print(apis.api_docs.show_app_descriptions())"
        self.assertEqual(guard_code(code, NAMES), (code, None))

    def test_invalid_app_is_guarded_even_when_nested(self) -> None:
        code = "apis.fake.api_docs.show_app_descriptions()"
        self.assertEqual(
            guard_code(code, NAMES),
            ("print(apis.api_docs.show_app_descriptions())", "invalid_app"),
        )

    def test_api_chain_ignores_non_api_objects(self) -> None:
        import ast

        node = ast.parse("object.spotify.api_docs").body[0].value
        self.assertIsNone(api_attribute_chain(node))


if __name__ == "__main__":
    unittest.main()
