"""Structural gate regression checks; the development data must fail closed."""

from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path

from corem.real_eval import load_jsonl
from experiments.audit_independent_v7_confirmation_gate import audit

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "public_independent_v7_development_v1"
VIEW = ROOT / "experiments" / "results_independent_v7_development_v1_selections"


class IndependentV7ConfirmationGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.events = load_jsonl(DATASET / "events.jsonl")
        cls.memories = load_jsonl(DATASET / "memories.jsonl")
        cls.candidates = load_jsonl(VIEW / "candidate_selections.jsonl")
        cls.query_view = load_jsonl(VIEW / "query_only_events.jsonl")

    def test_inspected_four_family_data_cannot_pass_sealed_gate(self) -> None:
        excluded = {
            (str(row["provider_id"]), str(row["task_key"])) for row in self.events
        }
        result = audit(
            self.events, self.memories, self.candidates, self.query_view, excluded
        )
        self.assertEqual(result["status"], "not_ready")
        self.assertIn("requires >=128 incidents; found 4", result["issues"])
        self.assertTrue(
            any(
                "overlaps inspected pilot/development" in issue
                for issue in result["issues"]
            )
        )

    def test_label_or_checker_fields_cannot_enter_selector_inputs(self) -> None:
        candidates = deepcopy(self.candidates)
        candidates[0]["goal_satisfied"] = True
        candidates[0]["candidate_edge_id"] = "oracle_answer"
        candidates[1]["candidate_edge_id"] = "example:bundle_a"
        query_view = deepcopy(self.query_view)
        query_view[0]["workflow_contract"] = self.events[0]["workflow_contract"]
        result = audit(self.events, self.memories, candidates, query_view, set())
        self.assertTrue(
            any(
                "candidate input contains non-selection fields" in issue
                for issue in result["issues"]
            )
        )
        self.assertTrue(
            any(
                "answer vocabulary in candidate identifiers" in issue
                for issue in result["issues"]
            )
        )
        self.assertTrue(
            any(
                "candidate identifiers expose bundle side" in issue
                for issue in result["issues"]
            )
        )
        self.assertTrue(
            any(
                "query view contains missing/extra" in issue
                for issue in result["issues"]
            )
        )

    def test_terminal_404_is_only_valid_for_index_deletion_absence(self) -> None:
        events = deepcopy(self.events)
        target = events[0]
        target["operation_count"] = 7
        target["workflow_contract"]["kind"] = "v7_meili_index_delete"
        target["workflow_contract"]["statuses"] = [202, 200, 202, 200, 202, 200, 404]
        result = audit(events, self.memories, self.candidates, self.query_view, set())
        self.assertFalse(
            any(
                str(target["event_id"]) in issue
                and "status contract does not match" in issue
                for issue in result["issues"]
            )
        )
        target["workflow_contract"]["statuses"][4] = 404
        result = audit(events, self.memories, self.candidates, self.query_view, set())
        self.assertTrue(
            any(
                str(target["event_id"]) in issue
                and "status contract does not match" in issue
                for issue in result["issues"]
            )
        )


if __name__ == "__main__":
    unittest.main()
