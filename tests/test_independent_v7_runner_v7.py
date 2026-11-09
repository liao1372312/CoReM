"""Opt-in real-service tests for distinct Meilisearch update workflows."""

from __future__ import annotations

import os
import unittest
from copy import deepcopy

from experiments.independent_v7_checker_v7 import IndependentV7CheckerV7
from experiments.independent_v7_diverse_specs import diverse_meili_spec
from experiments.independent_v7_runner import IndependentV7Runner


@unittest.skipUnless(
    os.environ.get("COREM_RUN_V7_INTEGRATION") == "1",
    "requires pinned isolated Meilisearch binary and Docker",
)
class IndependentV7RunnerV7IntegrationTest(unittest.TestCase):
    def test_document_partial_update_preserves_other_fields(self) -> None:
        checker = IndependentV7CheckerV7()
        for positive in (True, False):
            spec = diverse_meili_spec("meili_document_update", 1, positive)
            event = {
                "event_id": f"v7-smoke-meili-update-{positive}",
                "event_time": "2026-10-18T00:00:00Z",
                "service_id": "meilisearch",
                "version_id": "1.55.0",
                "operation_count": len(spec["plans"]),
                "required_binding_atoms": [
                    "binding:step_order",
                    "binding:runtime_handoffs",
                    "binding:status_codes",
                ],
                "required_evidence_atoms": [],
                "communication_criteria": [
                    "communication:all_statuses",
                    "communication:workflow_summary",
                ],
                "workflow_contract": spec["contract"],
            }
            memories = [
                {
                    "memory_id": f"v7-v7-smoke-step-{index}",
                    "role": "workflow_step",
                    "step_index": index,
                    "plan": plan,
                }
                for index, plan in enumerate(spec["plans"], 1)
            ]
            runner = IndependentV7Runner()
            try:
                self.assertTrue(runner.reset(event)["reset_verified"])
                execution = runner.execute(event, memories)
                self.assertTrue(execution["completed"], execution)
                report = checker.check(event, execution)
                self.assertEqual(
                    report["goal_satisfied"],
                    positive,
                    {
                        "report": report,
                        "steps": [
                            {
                                "request": step.get("request"),
                                "response": step.get("response"),
                            }
                            for step in execution.get("steps", [])
                        ],
                    },
                )
                if positive:
                    corrupted = deepcopy(execution)
                    corrupted["steps"][-1]["response"]["body"]["owner"] = "wrong"
                    self.assertFalse(checker.check(event, corrupted)["goal_satisfied"])
                    misbound = deepcopy(execution)
                    misbound["steps"][-1]["request"]["path"] = "/wrong-resource"
                    self.assertFalse(checker.check(event, misbound)["goal_satisfied"])
            finally:
                runner._cleanup()

    def test_index_swap_moves_candidate_document_into_live_uid(self) -> None:
        checker = IndependentV7CheckerV7()
        for positive in (True, False):
            spec = diverse_meili_spec("meili_index_swap", 1, positive)
            event = {
                "event_id": f"v7-smoke-meili-swap-{positive}",
                "event_time": "2026-10-18T00:00:00Z",
                "service_id": "meilisearch",
                "version_id": "1.55.0",
                "operation_count": len(spec["plans"]),
                "required_binding_atoms": [
                    "binding:step_order",
                    "binding:runtime_handoffs",
                    "binding:status_codes",
                ],
                "required_evidence_atoms": [],
                "communication_criteria": [
                    "communication:all_statuses",
                    "communication:workflow_summary",
                ],
                "workflow_contract": spec["contract"],
            }
            memories = [
                {
                    "memory_id": f"v7-v7-swap-step-{index}",
                    "role": "workflow_step",
                    "step_index": index,
                    "plan": plan,
                }
                for index, plan in enumerate(spec["plans"], 1)
            ]
            runner = IndependentV7Runner()
            try:
                self.assertTrue(runner.reset(event)["reset_verified"])
                execution = runner.execute(event, memories)
                self.assertTrue(execution["completed"], execution)
                report = checker.check(event, execution)
                self.assertEqual(
                    report["goal_satisfied"],
                    positive,
                    {
                        "report": report,
                        "steps": [
                            {
                                "request": row.get("request"),
                                "response": row.get("response"),
                            }
                            for row in execution.get("steps", [])
                        ],
                    },
                )
                if positive:
                    corrupted = deepcopy(execution)
                    corrupted["steps"][-1]["response"]["body"]["title"] = "wrong"
                    self.assertFalse(checker.check(event, corrupted)["goal_satisfied"])
                    misbound = deepcopy(execution)
                    misbound["steps"][4]["request"]["json"] = []
                    self.assertFalse(checker.check(event, misbound)["goal_satisfied"])
            finally:
                runner._cleanup()

    def test_index_delete_removes_only_target_uid(self) -> None:
        checker = IndependentV7CheckerV7()
        for positive in (True, False):
            spec = diverse_meili_spec("meili_index_delete", 1, positive)
            event = {
                "event_id": f"v7-smoke-meili-delete-index-{positive}",
                "event_time": "2026-10-18T00:00:00Z",
                "service_id": "meilisearch",
                "version_id": "1.55.0",
                "operation_count": len(spec["plans"]),
                "required_binding_atoms": [
                    "binding:step_order",
                    "binding:runtime_handoffs",
                    "binding:status_codes",
                ],
                "required_evidence_atoms": [],
                "communication_criteria": [
                    "communication:all_statuses",
                    "communication:workflow_summary",
                ],
                "workflow_contract": spec["contract"],
            }
            memories = [
                {
                    "memory_id": f"v7-v7-index-delete-step-{index}",
                    "role": "workflow_step",
                    "step_index": index,
                    "plan": plan,
                }
                for index, plan in enumerate(spec["plans"], 1)
            ]
            runner = IndependentV7Runner()
            try:
                self.assertTrue(runner.reset(event)["reset_verified"])
                execution = runner.execute(event, memories)
                self.assertTrue(execution["completed"], execution)
                report = checker.check(event, execution)
                self.assertEqual(
                    report["goal_satisfied"],
                    positive,
                    {
                        "report": report,
                        "steps": [
                            {
                                "request": row.get("request"),
                                "response": row.get("response"),
                            }
                            for row in execution.get("steps", [])
                        ],
                    },
                )
                if positive:
                    corrupted = deepcopy(execution)
                    corrupted["steps"][-1]["response"]["status"] = 200
                    self.assertFalse(checker.check(event, corrupted)["goal_satisfied"])
                    misbound = deepcopy(execution)
                    misbound["steps"][4]["request"]["path"] = "/wrong-resource"
                    self.assertFalse(checker.check(event, misbound)["goal_satisfied"])
            finally:
                runner._cleanup()


if __name__ == "__main__":
    unittest.main()
