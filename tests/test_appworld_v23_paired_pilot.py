"""The v23 summary must pair identical tasks and reject technical failures."""

from __future__ import annotations

import unittest

from experiments.summarize_appworld_v23_paired_pilot import (
    METHODS,
    MODEL_DIGEST,
    RUNNER_SHA256,
    VIEW_SHA256,
    paired_counts,
    validated_outcomes,
)


class V23PairedPilotTests(unittest.TestCase):
    def test_paired_counts(self) -> None:
        self.assertEqual(
            paired_counts({"a": True, "b": False, "c": True}, {"a": False, "b": True, "c": True}),
            {"actual_wins": 1, "baseline_wins": 1, "ties": 1},
        )
        with self.assertRaises(ValueError):
            paired_counts({"a": True}, {"b": True})

    def test_validated_outcomes_rejects_service_error(self) -> None:
        rows = [
            {
                "task_sha256": f"{index:064x}",
                "success": False,
                "evaluation_error": False,
                "model_errors": 0,
                "execution_errors": 0,
            }
            for index in range(9)
        ]
        payload = {
            "status": "train_only_state_based_agent_pilot_not_corem_result",
            "benchmark": "AppWorld 0.1.3.post1",
            "split": "train",
            "model": "Qwen3.5-9B-4bit",
            "model_endpoint": "http://127.0.0.1:11435/api/chat",
            "model_digest": MODEL_DIGEST,
            "task_sampling": "seeded_distinct_generators_first_variant",
            "seed": 2027,
            "appworld_random_seed": 100,
            "max_steps": 20,
            "context_tokens": 6144,
            "predict_tokens": 512,
            "prompt_variant": "checklist",
            "history_strategy": "preserve_code",
            "raise_on_failure": True,
            "num_tasks": 9,
            "script_sha256": RUNNER_SHA256,
            "memory_condition": METHODS["none"],
            "memory_view_sha256": VIEW_SHA256["none"],
            "doc_condition": "none",
            "doc_view_sha256": None,
            "train_split_sha256": "93d9fe71e7a2e3b7529803d4a20b604f4ebf5ae806f321081140238068189d37",
            "protected_text_exported": False,
            "evaluation_errors": 0,
            "model_errors": 0,
            "execution_errors": 0,
            "successes": 0,
            "private_task_outcomes": rows,
        }
        self.assertEqual(len(validated_outcomes(payload, "none")), 9)
        with self.assertRaises(ValueError):
            validated_outcomes(payload, "none", {"different"})
        self.assertEqual(
            len(validated_outcomes(payload, "none", {row["task_sha256"] for row in rows})),
            9,
        )
        payload["model_endpoint"] = "http://127.0.0.1:11436/api/chat"
        with self.assertRaises(ValueError):
            validated_outcomes(payload, "none")
        payload["model_endpoint"] = "http://127.0.0.1:11435/api/chat"
        payload["memory_condition"] = METHODS["disjoint"]
        payload["memory_view_sha256"] = VIEW_SHA256["disjoint"]
        self.assertEqual(len(validated_outcomes(payload, "disjoint")), 9)
        payload["memory_condition"] = METHODS["none"]
        payload["memory_view_sha256"] = VIEW_SHA256["none"]
        payload["raise_on_failure"] = False
        with self.assertRaises(ValueError):
            validated_outcomes(payload, "none")
        payload["raise_on_failure"] = True
        payload["model_errors"] = 1
        with self.assertRaises(ValueError):
            validated_outcomes(payload, "none")


if __name__ == "__main__":
    unittest.main()
