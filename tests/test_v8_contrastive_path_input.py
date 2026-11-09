"""Development-only contrastive path serialization checks."""

from __future__ import annotations

import unittest

from experiments.develop_v8_contrastive_path_input import difference_focused_paths


class ContrastivePathInputTests(unittest.TestCase):
    def test_changed_step_retains_role_and_omits_shared_setup(self) -> None:
        memories = {
            "a1": {"memory_id": "a1", "role": "workflow_step", "step_index": 1,
                   "plan": {"method": "PUT", "path": "/collections/example"}},
            "a2": {"memory_id": "a2", "role": "workflow_step", "step_index": 2,
                   "plan": {"method": "POST", "path": "/collections/example/points/delete",
                            "json": {"points": [1]}}},
            "b1": {"memory_id": "b1", "role": "workflow_step", "step_index": 1,
                   "plan": {"method": "PUT", "path": "/collections/example"}},
            "b2": {"memory_id": "b2", "role": "workflow_step", "step_index": 2,
                   "plan": {"method": "POST", "path": "/collections/example/points/delete",
                            "json": {"points": [2]}}},
        }
        first, second, count = difference_focused_paths(
            {"selected_memory_ids": ["a2", "a1"]},
            {"selected_memory_ids": ["b1", "b2"]},
            memories,
        )
        self.assertEqual(count, 1)
        self.assertIn("Call 2:", first)
        self.assertIn('"points":[1]', first)
        self.assertIn('"points":[2]', second)
        self.assertNotIn("Call 1:", first)

    def test_identical_candidate_plans_are_rejected(self) -> None:
        memories = {
            "a": {"memory_id": "a", "role": "workflow_step", "step_index": 1,
                  "plan": {"method": "GET", "path": "/health"}},
            "b": {"memory_id": "b", "role": "workflow_step", "step_index": 1,
                  "plan": {"method": "GET", "path": "/health"}},
        }
        with self.assertRaisesRegex(ValueError, "no executable difference"):
            difference_focused_paths(
                {"selected_memory_ids": ["a"]},
                {"selected_memory_ids": ["b"]},
                memories,
            )


if __name__ == "__main__":
    unittest.main()
