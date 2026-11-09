"""The train sample audit exports breadth statistics, not task text."""

from __future__ import annotations

import unittest

from experiments.audit_appworld_train_sample import summarize_sample


class TrainSampleAuditTests(unittest.TestCase):
    def test_private_text_is_not_returned(self) -> None:
        tasks = [
            {"task_id": "a", "required_apis": ["one", "two"], "instruction": "secret"},
            {"task_id": "b", "required_apis": ["one"], "instruction": "private"},
        ]
        result = summarize_sample(tasks, {"a"})
        self.assertEqual(result["train_mean_required_apis"], 1.5)
        self.assertEqual(result["pilot_mean_required_apis"], 2)
        self.assertNotIn("secret", str(result))
        self.assertNotIn("private", str(result))

    def test_unknown_selection_fails(self) -> None:
        with self.assertRaises(ValueError):
            summarize_sample([{"task_id": "a", "required_apis": ["one"]}], {"b"})


if __name__ == "__main__":
    unittest.main()
