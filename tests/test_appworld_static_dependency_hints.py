"""Paired static dependency hints must stay source-disjoint and value free."""

from __future__ import annotations

import unittest

from experiments.appworld_static_dependency import StaticDependencySketch
from experiments.build_appworld_train_static_dependency_hints import (
    render_edges,
    render_order,
    strict_source_rows,
)


class StaticHintTests(unittest.TestCase):
    def test_strict_source_excludes_every_target_generator(self) -> None:
        rows = [
            {"task_id": "a", "generator_id": "target_a", "instruction": "private"},
            {"task_id": "b", "generator_id": "target_b", "instruction": "private"},
            {"task_id": "c", "generator_id": "source", "instruction": "public"},
        ]
        eligible = strict_source_rows(rows, {"target_a", "target_b"})
        self.assertEqual([row["task_id"] for row in eligible], ["c"])

    def test_renderer_contains_only_api_names_and_anonymous_edges(self) -> None:
        sketch = StaticDependencySketch(
            api_calls=("foo.first", "foo.second"), edges=((1, 2),)
        )
        self.assertEqual(render_order(sketch), "1. foo.first\n2. foo.second")
        self.assertEqual(render_edges(sketch), "1->2")


if __name__ == "__main__":
    unittest.main()
