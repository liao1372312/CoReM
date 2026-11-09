"""Opt-in independent real-service tests for diverse Meilisearch workflows."""

from __future__ import annotations

import os
import unittest
from copy import deepcopy

from experiments.independent_v7_checker_v6 import IndependentV7CheckerV6
from experiments.independent_v7_runner import IndependentV7Runner


def _capture(name: str, path: list[str]) -> dict:
    return {"name": name, "source": "body_path", "path": path}


def _event(
    kind: str, uid: str, handoffs: list[tuple[str, int]], **contract: object
) -> dict:
    return {
        "event_id": f"v7-smoke-{kind}",
        "event_time": "2026-10-18T00:00:00Z",
        "service_id": "meilisearch",
        "version_id": "1.55.0",
        "operation_count": 5,
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
        "workflow_contract": {
            "kind": kind,
            "index_uid": uid,
            "statuses": [202, 200, 202, 200, 200],
            "handoffs": [
                {"name": name, "target_step": step} for name, step in handoffs
            ],
            **contract,
        },
    }


@unittest.skipUnless(
    os.environ.get("COREM_RUN_V7_INTEGRATION") == "1",
    "requires pinned isolated Meilisearch binary and Docker",
)
class IndependentV7RunnerV6IntegrationTest(unittest.TestCase):
    def _run(self, event: dict, plans: list[dict], corrupt_field: str) -> None:
        runner = IndependentV7Runner()
        try:
            self.assertTrue(runner.reset(event)["reset_verified"])
            memories = [
                {
                    "memory_id": f"v7-v6-smoke-step-{index}",
                    "role": "workflow_step",
                    "step_index": index,
                    "plan": plan,
                }
                for index, plan in enumerate(plans, 1)
            ]
            execution = runner.execute(event, memories)
            self.assertTrue(
                execution["completed"],
                {key: value for key, value in execution.items() if key != "steps"},
            )
            checker = IndependentV7CheckerV6()
            report = checker.check(event, execution)
            self.assertTrue(report["goal_satisfied"], report)
            self.assertTrue(
                all(row["passed"] for row in report["communication_checks"])
            )
            corrupted = deepcopy(execution)
            corrupted["steps"][-1]["response"]["body"][corrupt_field] = "wrong"
            self.assertFalse(checker.check(event, corrupted)["goal_satisfied"])
            misbound = deepcopy(execution)
            misbound["steps"][-1]["request"]["path"] = "/wrong-resource"
            self.assertFalse(checker.check(event, misbound)["goal_satisfied"])
        finally:
            runner._cleanup()

    def test_document_deletion_preserves_expected_document(self) -> None:
        uid = "v7_smoke_deletion"
        event = _event(
            "v7_meili_deletion",
            uid,
            [
                ("task_add", 2),
                ("index_uid", 3),
                ("task_delete", 4),
                ("index_uid_confirm", 5),
            ],
            keep_id=11,
            delete_id=22,
            keep_title="Release handbook",
        )
        plans = [
            {
                "method": "POST",
                "path": f"/indexes/{uid}/documents?primaryKey=id",
                "json": [
                    {"id": 11, "title": "Release handbook"},
                    {"id": 22, "title": "Draft notice"},
                ],
                "capture": _capture("task_add", ["taskUid"]),
            },
            {
                "method": "GET",
                "path": "/tasks/{task_add}",
                "capture": _capture("index_uid", ["indexUid"]),
            },
            {
                "method": "DELETE",
                "path": "/indexes/{index_uid}/documents/22",
                "capture": _capture("task_delete", ["taskUid"]),
            },
            {
                "method": "GET",
                "path": "/tasks/{task_delete}",
                "capture": _capture("index_uid_confirm", ["indexUid"]),
            },
            {"method": "GET", "path": "/indexes/{index_uid_confirm}/documents/11"},
        ]
        self._run(event, plans, "title")

    def test_multi_index_search_uses_both_returned_index_ids(self) -> None:
        uid_a, uid_b = "v7_smoke_multi_a", "v7_smoke_multi_b"
        event = _event(
            "v7_meili_multi_search",
            uid_a,
            [("task_a", 2), ("index_a", 5), ("task_b", 4), ("index_b", 5)],
            index_a=uid_a,
            index_b=uid_b,
            term_a="handbook",
            term_b="invoice",
            document_a=11,
            document_b=22,
        )
        plans = [
            {
                "method": "POST",
                "path": f"/indexes/{uid_a}/documents?primaryKey=id",
                "json": [{"id": 11, "title": "Release handbook"}],
                "capture": _capture("task_a", ["taskUid"]),
            },
            {
                "method": "GET",
                "path": "/tasks/{task_a}",
                "capture": _capture("index_a", ["indexUid"]),
            },
            {
                "method": "POST",
                "path": f"/indexes/{uid_b}/documents?primaryKey=id",
                "json": [{"id": 22, "title": "Payment invoice"}],
                "capture": _capture("task_b", ["taskUid"]),
            },
            {
                "method": "GET",
                "path": "/tasks/{task_b}",
                "capture": _capture("index_b", ["indexUid"]),
            },
            {
                "method": "POST",
                "path": "/multi-search",
                "json": {
                    "queries": [
                        {"indexUid": "{index_a}", "q": "handbook"},
                        {"indexUid": "{index_b}", "q": "invoice"},
                    ]
                },
            },
        ]
        self._run(event, plans, "results")

    def test_facet_distribution_after_settings_task(self) -> None:
        uid = "v7_smoke_facet"
        event = _event(
            "v7_meili_facet",
            uid,
            [
                ("task_add", 2),
                ("index_uid", 3),
                ("task_settings", 4),
                ("index_uid_confirm", 5),
            ],
            category="urgent",
            search_term="report",
            document_id=11,
        )
        plans = [
            {
                "method": "POST",
                "path": f"/indexes/{uid}/documents?primaryKey=id",
                "json": [{"id": 11, "title": "Release report", "category": "urgent"}],
                "capture": _capture("task_add", ["taskUid"]),
            },
            {
                "method": "GET",
                "path": "/tasks/{task_add}",
                "capture": _capture("index_uid", ["indexUid"]),
            },
            {
                "method": "PUT",
                "path": "/indexes/{index_uid}/settings/filterable-attributes",
                "json": ["category"],
                "capture": _capture("task_settings", ["taskUid"]),
            },
            {
                "method": "GET",
                "path": "/tasks/{task_settings}",
                "capture": _capture("index_uid_confirm", ["indexUid"]),
            },
            {
                "method": "POST",
                "path": "/indexes/{index_uid_confirm}/search",
                "json": {"q": "report", "facets": ["category"]},
            },
        ]
        self._run(event, plans, "facetDistribution")


if __name__ == "__main__":
    unittest.main()
