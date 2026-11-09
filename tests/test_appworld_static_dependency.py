"""Static dependency sketches must be occurrence based and value free."""

from __future__ import annotations

import unittest

from experiments.appworld_static_dependency import extract_static_dependency_sketch


class StaticDependencyTests(unittest.TestCase):
    def test_direct_and_transitive_assignments(self) -> None:
        source = """
def solution():
    response = apis.spotify.show_user(access_token="secret")
    user_id = response["id"]
    apis.spotify.show_playlist(user_id=user_id)
"""
        result = extract_static_dependency_sketch(source)
        self.assertEqual(
            result.api_calls,
            ("spotify.show_user", "spotify.show_playlist"),
        )
        self.assertEqual(result.edges, ((1, 2),))
        self.assertNotIn("secret", str(result))

    def test_branch_producers_are_potential_not_certified(self) -> None:
        source = """
def solution(flag):
    if flag:
        value = apis.foo.first()
    else:
        value = apis.foo.second()
    apis.foo.last(value=value)
"""
        result = extract_static_dependency_sketch(source)
        self.assertEqual(result.edges, ((1, 3), (2, 3)))

    def test_nested_call_and_reassignment(self) -> None:
        source = """
x = apis.foo.first()
y = apis.foo.second(value=x)
x = 1
apis.foo.third(value=x, other=y)
"""
        result = extract_static_dependency_sketch(source)
        self.assertEqual(result.edges, ((1, 2), (2, 3)))


if __name__ == "__main__":
    unittest.main()
