"""Synthetic feedback classifier must not surface private response text."""

from __future__ import annotations

import unittest

from experiments.diagnose_appworld_api_failure_feedback import classify_feedback


class FeedbackClassifierTests(unittest.TestCase):
    def test_exception_feedback(self) -> None:
        result = classify_feedback(
            "Execution failed. Traceback:\nException: Response status code is 401:\nsecret"
        )
        self.assertEqual(result["http_status_in_traceback"], 401)
        self.assertTrue(result["execution_failed"])
        self.assertNotIn("secret", str(result))

    def test_json_feedback(self) -> None:
        result = classify_feedback('{"message": "private"}')
        self.assertTrue(result["json_object_returned"])
        self.assertFalse(result["execution_failed"])
        self.assertNotIn("private", str(result))


if __name__ == "__main__":
    unittest.main()
