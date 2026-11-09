"""Opt-in real-service smoke checks for the prospective v7 runner."""

from __future__ import annotations

import os
import unittest
from copy import deepcopy

from experiments.independent_v7_checker import IndependentV7Checker
from experiments.independent_v7_checker_v2 import IndependentV7CheckerV2
from experiments.independent_v7_checker_v3 import IndependentV7CheckerV3
from experiments.independent_v7_checker_v4 import IndependentV7CheckerV4
from experiments.independent_v7_checker_v5 import IndependentV7CheckerV5
from experiments.independent_v7_runner import IndependentV7Runner


def _memory(index: int, plan: dict[str, object]) -> dict[str, object]:
    return {
        "memory_id": f"v7-smoke-step-{index}",
        "role": "workflow_step",
        "step_index": index,
        "plan": plan,
    }


@unittest.skipUnless(
    os.environ.get("COREM_RUN_V7_INTEGRATION") == "1",
    "requires pinned Gitea/Meilisearch binaries and Docker",
)
class IndependentV7RunnerIntegrationTest(unittest.TestCase):
    def _run_new_family(
        self,
        event: dict[str, object],
        plans: list[dict[str, object]],
        corrupt_step: int,
        corrupt_field: str,
        checker_type: type[IndependentV7CheckerV2] = IndependentV7CheckerV2,
    ) -> None:
        runner = IndependentV7Runner()
        try:
            self.assertTrue(runner.reset(event)["reset_verified"])
            execution = runner.execute(
                event, [_memory(index, plan) for index, plan in enumerate(plans, 1)]
            )
            self.assertTrue(
                execution["completed"],
                (
                    {key: value for key, value in execution.items() if key != "steps"},
                    [
                        (step["response"]["status"], step["request"]["path"])
                        for step in execution.get("steps", [])
                    ],
                ),
            )
            checker = checker_type()
            report = checker.check(event, execution)
            self.assertTrue(
                report["goal_satisfied"],
                (
                    report,
                    [
                        (step["response"]["status"], step["request"]["path"])
                        for step in execution["steps"]
                    ],
                ),
            )
            self.assertTrue(
                all(row["passed"] for row in report["communication_checks"])
            )
            corrupted = deepcopy(execution)
            if corrupt_field == "__list__":
                corrupted["steps"][corrupt_step]["response"]["body"] = ["wrong"]
            else:
                corrupted["steps"][corrupt_step]["response"]["body"][corrupt_field] = (
                    "wrong"
                )
            self.assertFalse(checker.check(event, corrupted)["goal_satisfied"])
            misbound = deepcopy(execution)
            misbound["steps"][-1]["request"]["path"] = "/wrong-resource"
            self.assertFalse(checker.check(event, misbound)["goal_satisfied"])
            kind = event["workflow_contract"]["kind"]
            if kind in {
                "v7_meili_synonyms",
                "v7_meili_stopwords",
                "v7_meili_filterable",
                "v7_meili_sortable",
            }:
                wrong_payload = deepcopy(execution)
                wrong_payload["steps"][2]["request"]["json"] = ["wrong"]
                self.assertFalse(checker.check(event, wrong_payload)["goal_satisfied"])
            runner.snapshot(event)
        finally:
            runner._cleanup()

    def test_gitea_label_family(self) -> None:
        event = {
            "event_id": "v7-smoke-gitea-label",
            "event_time": "2026-10-18T00:00:00Z",
            "service_id": "gitea",
            "version_id": "28.1.0",
            "operation_count": 3,
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
                "kind": "v7_gitea_label",
                "repo_name": "v7-smoke-label",
                "label_name": "Urgent triage",
                "statuses": [201, 201, 200],
                "handoffs": [
                    {"name": "repo", "target_step": 2},
                    {"name": "label_id", "target_step": 3},
                ],
            },
        }
        plans = [
            {
                "method": "POST",
                "path": "/api/v1/user/repos",
                "json": {"name": "v7-smoke-label", "private": True, "auto_init": True},
                "capture": {"name": "repo", "source": "body_path", "path": ["name"]},
            },
            {
                "method": "POST",
                "path": "/api/v1/repos/corem_eval/{repo}/labels",
                "json": {"name": "Urgent triage", "color": "#ff0000"},
                "capture": {"name": "label_id", "source": "body_path", "path": ["id"]},
            },
            {
                "method": "GET",
                "path": "/api/v1/repos/corem_eval/{repo}/labels/{label_id}",
            },
        ]
        self._run_new_family(event, plans, 2, "name")

    def test_meilisearch_document_family(self) -> None:
        event = {
            "event_id": "v7-smoke-meili-document",
            "event_time": "2026-10-18T00:00:00Z",
            "service_id": "meilisearch",
            "version_id": "1.55.0",
            "operation_count": 3,
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
                "kind": "v7_meili_document",
                "index_uid": "v7_smoke_docs",
                "document_id": 1,
                "document_title": "Release audit",
                "statuses": [202, 200, 200],
                "handoffs": [
                    {"name": "task_uid", "target_step": 2},
                    {"name": "index_uid", "target_step": 3},
                ],
            },
        }
        plans = [
            {
                "method": "POST",
                "path": "/indexes/v7_smoke_docs/documents?primaryKey=id",
                "json": [{"id": 1, "title": "Release audit"}],
                "capture": {
                    "name": "task_uid",
                    "source": "body_path",
                    "path": ["taskUid"],
                },
            },
            {
                "method": "GET",
                "path": "/tasks/{task_uid}",
                "capture": {
                    "name": "index_uid",
                    "source": "body_path",
                    "path": ["indexUid"],
                },
            },
            {
                "method": "GET",
                "path": "/indexes/{index_uid}/documents/1",
            },
        ]
        self._run_new_family(event, plans, 2, "title")

    def test_gitea_milestone_family(self) -> None:
        event = {
            "event_id": "v7-smoke-gitea-milestone",
            "event_time": "2026-10-18T00:00:00Z",
            "service_id": "gitea",
            "version_id": "28.1.0",
            "operation_count": 3,
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
                "kind": "v7_gitea_milestone",
                "repo_name": "v7-smoke-milestone",
                "milestone_title": "Sprint gate",
                "milestone_state": "open",
                "statuses": [201, 201, 200],
                "handoffs": [
                    {"name": "repo", "target_step": 2},
                    {"name": "milestone_id", "target_step": 3},
                ],
            },
        }
        plans = [
            {
                "method": "POST",
                "path": "/api/v1/user/repos",
                "json": {
                    "name": "v7-smoke-milestone",
                    "private": True,
                    "auto_init": True,
                },
                "capture": {"name": "repo", "source": "body_path", "path": ["name"]},
            },
            {
                "method": "POST",
                "path": "/api/v1/repos/corem_eval/{repo}/milestones",
                "json": {"title": "Sprint gate", "state": "open"},
                "capture": {
                    "name": "milestone_id",
                    "source": "body_path",
                    "path": ["id"],
                },
            },
            {
                "method": "GET",
                "path": "/api/v1/repos/corem_eval/{repo}/milestones/{milestone_id}",
            },
        ]
        self._run_new_family(event, plans, 2, "state")

    def test_meilisearch_search_family(self) -> None:
        event = {
            "event_id": "v7-smoke-meili-search",
            "event_time": "2026-10-18T00:00:00Z",
            "service_id": "meilisearch",
            "version_id": "1.55.0",
            "operation_count": 3,
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
                "kind": "v7_meili_search",
                "index_uid": "v7_smoke_search",
                "document_id": 11,
                "document_title": "Release audit handbook",
                "search_query": "handbook",
                "statuses": [202, 200, 200],
                "handoffs": [
                    {"name": "task_uid", "target_step": 2},
                    {"name": "index_uid", "target_step": 3},
                ],
            },
        }
        plans = [
            {
                "method": "POST",
                "path": "/indexes/v7_smoke_search/documents?primaryKey=id",
                "json": [{"id": 11, "title": "Release audit handbook"}],
                "capture": {
                    "name": "task_uid",
                    "source": "body_path",
                    "path": ["taskUid"],
                },
            },
            {
                "method": "GET",
                "path": "/tasks/{task_uid}",
                "capture": {
                    "name": "index_uid",
                    "source": "body_path",
                    "path": ["indexUid"],
                },
            },
            {
                "method": "POST",
                "path": "/indexes/{index_uid}/search",
                "json": {"q": "handbook"},
            },
        ]
        self._run_new_family(event, plans, 2, "hits")

    def test_gitea_branch_family(self) -> None:
        event = {
            "event_id": "v7-smoke-gitea-branch",
            "event_time": "2026-10-18T00:00:00Z",
            "service_id": "gitea",
            "version_id": "28.1.0",
            "operation_count": 3,
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
                "kind": "v7_gitea_branch",
                "repo_name": "v7-smoke-branch",
                "branch_name": "release-readiness",
                "statuses": [201, 201, 200],
                "handoffs": [
                    {"name": "repo", "target_step": 2},
                    {"name": "branch_name", "target_step": 3},
                ],
            },
        }
        plans = [
            {
                "method": "POST",
                "path": "/api/v1/user/repos",
                "json": {"name": "v7-smoke-branch", "private": True, "auto_init": True},
                "capture": {"name": "repo", "source": "body_path", "path": ["name"]},
            },
            {
                "method": "POST",
                "path": "/api/v1/repos/corem_eval/{repo}/branches",
                "json": {
                    "new_branch_name": "release-readiness",
                    "old_branch_name": "main",
                },
                "capture": {
                    "name": "branch_name",
                    "source": "body_path",
                    "path": ["name"],
                },
            },
            {
                "method": "GET",
                "path": "/api/v1/repos/corem_eval/{repo}/branches/{branch_name}",
            },
        ]
        self._run_new_family(event, plans, 2, "name", IndependentV7CheckerV3)

    def test_meilisearch_index_stats_family(self) -> None:
        event = {
            "event_id": "v7-smoke-meili-stats",
            "event_time": "2026-10-18T00:00:00Z",
            "service_id": "meilisearch",
            "version_id": "1.55.0",
            "operation_count": 3,
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
                "kind": "v7_meili_stats",
                "index_uid": "v7_smoke_stats",
                "document_count": 1,
                "statuses": [202, 200, 200],
                "handoffs": [
                    {"name": "task_uid", "target_step": 2},
                    {"name": "index_uid", "target_step": 3},
                ],
            },
        }
        plans = [
            {
                "method": "POST",
                "path": "/indexes/v7_smoke_stats/documents?primaryKey=id",
                "json": [{"id": 71, "title": "Release audit"}],
                "capture": {
                    "name": "task_uid",
                    "source": "body_path",
                    "path": ["taskUid"],
                },
            },
            {
                "method": "GET",
                "path": "/tasks/{task_uid}",
                "capture": {
                    "name": "index_uid",
                    "source": "body_path",
                    "path": ["indexUid"],
                },
            },
            {"method": "GET", "path": "/indexes/{index_uid}/stats"},
        ]
        self._run_new_family(
            event, plans, 2, "numberOfDocuments", IndependentV7CheckerV3
        )

    def test_gitea_release_family(self) -> None:
        event = {
            "event_id": "v7-smoke-gitea-release",
            "event_time": "2026-10-18T00:00:00Z",
            "service_id": "gitea",
            "version_id": "28.1.0",
            "operation_count": 3,
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
                "kind": "v7_gitea_release",
                "repo_name": "v7-smoke-release",
                "release_name": "Ready release",
                "tag_name": "v1.0.0",
                "statuses": [201, 201, 200],
                "handoffs": [
                    {"name": "repo", "target_step": 2},
                    {"name": "release_id", "target_step": 3},
                ],
            },
        }
        plans = [
            {
                "method": "POST",
                "path": "/api/v1/user/repos",
                "json": {
                    "name": "v7-smoke-release",
                    "private": True,
                    "auto_init": True,
                },
                "capture": {"name": "repo", "source": "body_path", "path": ["name"]},
            },
            {
                "method": "POST",
                "path": "/api/v1/repos/corem_eval/{repo}/releases",
                "json": {
                    "tag_name": "v1.0.0",
                    "name": "Ready release",
                    "draft": False,
                    "prerelease": False,
                    "target_commitish": "main",
                },
                "capture": {
                    "name": "release_id",
                    "source": "body_path",
                    "path": ["id"],
                },
            },
            {
                "method": "GET",
                "path": "/api/v1/repos/corem_eval/{repo}/releases/{release_id}",
            },
        ]
        self._run_new_family(event, plans, 2, "name", IndependentV7CheckerV3)

    def test_meilisearch_synonyms_family(self) -> None:
        event = {
            "event_id": "v7-smoke-meili-synonyms",
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
                "kind": "v7_meili_synonyms",
                "index_uid": "v7_smoke_synonyms",
                "term": "report",
                "synonyms": ["briefing"],
                "statuses": [202, 200, 202, 200, 200],
                "handoffs": [
                    {"name": "task_create", "target_step": 2},
                    {"name": "index_uid", "target_step": 3},
                    {"name": "task_settings", "target_step": 4},
                    {"name": "index_uid_confirm", "target_step": 5},
                ],
            },
        }
        plans = [
            {
                "method": "POST",
                "path": "/indexes",
                "json": {"uid": "v7_smoke_synonyms", "primaryKey": "id"},
                "capture": {
                    "name": "task_create",
                    "source": "body_path",
                    "path": ["taskUid"],
                },
            },
            {
                "method": "GET",
                "path": "/tasks/{task_create}",
                "capture": {
                    "name": "index_uid",
                    "source": "body_path",
                    "path": ["indexUid"],
                },
            },
            {
                "method": "PUT",
                "path": "/indexes/{index_uid}/settings/synonyms",
                "json": {"report": ["briefing"]},
                "capture": {
                    "name": "task_settings",
                    "source": "body_path",
                    "path": ["taskUid"],
                },
            },
            {
                "method": "GET",
                "path": "/tasks/{task_settings}",
                "capture": {
                    "name": "index_uid_confirm",
                    "source": "body_path",
                    "path": ["indexUid"],
                },
            },
            {
                "method": "GET",
                "path": "/indexes/{index_uid_confirm}/settings/synonyms",
            },
        ]
        self._run_new_family(event, plans, 4, "report", IndependentV7CheckerV3)

    def test_meilisearch_list_settings_families(self) -> None:
        families = (
            ("stopwords", "stop-words", ["a", "the"]),
            ("filterable", "filterable-attributes", ["category", "rating"]),
            ("sortable", "sortable-attributes", ["price", "rating"]),
        )
        for kind_suffix, route, values in families:
            with self.subTest(kind=kind_suffix):
                uid = f"v7_smoke_{kind_suffix}"
                event = {
                    "event_id": f"v7-smoke-meili-{kind_suffix}",
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
                        "kind": f"v7_meili_{kind_suffix}",
                        "index_uid": uid,
                        "setting_values": values,
                        "statuses": [202, 200, 202, 200, 200],
                        "handoffs": [
                            {"name": "task_create", "target_step": 2},
                            {"name": "index_uid", "target_step": 3},
                            {"name": "task_settings", "target_step": 4},
                            {"name": "index_uid_confirm", "target_step": 5},
                        ],
                    },
                }
                plans = [
                    {
                        "method": "POST",
                        "path": "/indexes",
                        "json": {"uid": uid, "primaryKey": "id"},
                        "capture": {
                            "name": "task_create",
                            "source": "body_path",
                            "path": ["taskUid"],
                        },
                    },
                    {
                        "method": "GET",
                        "path": "/tasks/{task_create}",
                        "capture": {
                            "name": "index_uid",
                            "source": "body_path",
                            "path": ["indexUid"],
                        },
                    },
                    {
                        "method": "PUT",
                        "path": f"/indexes/{{index_uid}}/settings/{route}",
                        "json": values,
                        "capture": {
                            "name": "task_settings",
                            "source": "body_path",
                            "path": ["taskUid"],
                        },
                    },
                    {
                        "method": "GET",
                        "path": "/tasks/{task_settings}",
                        "capture": {
                            "name": "index_uid_confirm",
                            "source": "body_path",
                            "path": ["indexUid"],
                        },
                    },
                    {
                        "method": "GET",
                        "path": f"/indexes/{{index_uid_confirm}}/settings/{route}",
                    },
                ]
                self._run_new_family(
                    event, plans, 4, "__list__", IndependentV7CheckerV3
                )

    def test_gitea_issue_comment_family(self) -> None:
        event = {
            "event_id": "v7-smoke-gitea-issue-comment",
            "event_time": "2026-10-18T00:00:00Z",
            "service_id": "gitea",
            "version_id": "28.1.0",
            "operation_count": 4,
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
                "kind": "v7_gitea_issue_comment",
                "repo_name": "v7-smoke-comment",
                "issue_title": "Release gate",
                "comment_body": "Review before deployment",
                "statuses": [201, 201, 201, 200],
                "handoffs": [
                    {"name": "repo", "target_step": 2},
                    {"name": "issue_index", "target_step": 3},
                    {"name": "comment_id", "target_step": 4},
                ],
            },
        }
        plans = [
            {
                "method": "POST",
                "path": "/api/v1/user/repos",
                "json": {
                    "name": "v7-smoke-comment",
                    "private": True,
                    "auto_init": True,
                },
                "capture": {"name": "repo", "source": "body_path", "path": ["name"]},
            },
            {
                "method": "POST",
                "path": "/api/v1/repos/corem_eval/{repo}/issues",
                "json": {"title": "Release gate"},
                "capture": {
                    "name": "issue_index",
                    "source": "body_path",
                    "path": ["number"],
                },
            },
            {
                "method": "POST",
                "path": "/api/v1/repos/corem_eval/{repo}/issues/{issue_index}/comments",
                "json": {"body": "Review before deployment"},
                "capture": {
                    "name": "comment_id",
                    "source": "body_path",
                    "path": ["id"],
                },
            },
            {
                "method": "GET",
                "path": "/api/v1/repos/corem_eval/{repo}/issues/comments/{comment_id}",
            },
        ]
        self._run_new_family(event, plans, 3, "body", IndependentV7CheckerV4)

    def test_gitea_wiki_family(self) -> None:
        content = "UmVsZWFzZSBndWlkZQ=="
        event = {
            "event_id": "v7-smoke-gitea-wiki",
            "event_time": "2026-10-18T00:00:00Z",
            "service_id": "gitea",
            "version_id": "28.1.0",
            "operation_count": 3,
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
                "kind": "v7_gitea_wiki",
                "repo_name": "v7-smoke-wiki",
                "wiki_title": "Release-Guide",
                "content_base64": content,
                "statuses": [201, 201, 200],
                "handoffs": [
                    {"name": "repo", "target_step": 2},
                    {"name": "wiki_sub_url", "target_step": 3},
                ],
            },
        }
        plans = [
            {
                "method": "POST",
                "path": "/api/v1/user/repos",
                "json": {"name": "v7-smoke-wiki", "private": True, "auto_init": True},
                "capture": {"name": "repo", "source": "body_path", "path": ["name"]},
            },
            {
                "method": "POST",
                "path": "/api/v1/repos/corem_eval/{repo}/wiki/new",
                "json": {"title": "Release-Guide", "content_base64": content},
                "capture": {
                    "name": "wiki_sub_url",
                    "source": "body_path",
                    "path": ["sub_url"],
                },
            },
            {
                "method": "GET",
                "path": "/api/v1/repos/corem_eval/{repo}/wiki/page/{wiki_sub_url}",
            },
        ]
        self._run_new_family(event, plans, 2, "content_base64", IndependentV7CheckerV4)

    def test_gitea_project_family(self) -> None:
        event = {
            "event_id": "v7-smoke-gitea-project",
            "event_time": "2026-10-18T00:00:00Z",
            "service_id": "gitea",
            "version_id": "28.1.0",
            "operation_count": 3,
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
                "kind": "v7_gitea_project",
                "repo_name": "v7-smoke-project",
                "project_title": "Release board",
                "statuses": [201, 201, 200],
                "handoffs": [
                    {"name": "repo", "target_step": 2},
                    {"name": "project_id", "target_step": 3},
                ],
            },
        }
        plans = [
            {
                "method": "POST",
                "path": "/api/v1/user/repos",
                "json": {
                    "name": "v7-smoke-project",
                    "private": True,
                    "auto_init": True,
                },
                "capture": {"name": "repo", "source": "body_path", "path": ["name"]},
            },
            {
                "method": "POST",
                "path": "/api/v1/repos/corem_eval/{repo}/projects",
                "json": {"title": "Release board", "template_type": "none"},
                "capture": {
                    "name": "project_id",
                    "source": "body_path",
                    "path": ["id"],
                },
            },
            {
                "method": "GET",
                "path": "/api/v1/repos/corem_eval/{repo}/projects/{project_id}",
            },
        ]
        self._run_new_family(event, plans, 2, "title", IndependentV7CheckerV4)

    def test_gitea_file_family(self) -> None:
        content = "UmVsZWFzZSBmaWxlIG9r"
        event = {
            "event_id": "v7-smoke-gitea-file",
            "event_time": "2026-10-18T00:00:00Z",
            "service_id": "gitea",
            "version_id": "28.1.0",
            "operation_count": 3,
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
                "kind": "v7_gitea_file",
                "repo_name": "v7-smoke-file",
                "file_path": "release.txt",
                "file_text": "Release file ok",
                "statuses": [201, 201, 200],
                "handoffs": [
                    {"name": "repo", "target_step": 2},
                    {"name": "file_path", "target_step": 3},
                ],
            },
        }
        plans = [
            {
                "method": "POST",
                "path": "/api/v1/user/repos",
                "json": {"name": "v7-smoke-file", "private": True, "auto_init": True},
                "capture": {"name": "repo", "source": "body_path", "path": ["name"]},
            },
            {
                "method": "POST",
                "path": "/api/v1/repos/corem_eval/{repo}/contents/release.txt",
                "json": {"content": content, "message": "Add release file"},
                "capture": {
                    "name": "file_path",
                    "source": "body_path",
                    "path": ["content", "path"],
                },
            },
            {
                "method": "GET",
                "path": "/api/v1/repos/corem_eval/{repo}/contents/{file_path}",
            },
        ]
        self._run_new_family(event, plans, 2, "content", IndependentV7CheckerV5)

    def test_gitea_inactive_hook_family(self) -> None:
        event = {
            "event_id": "v7-smoke-gitea-hook",
            "event_time": "2026-10-18T00:00:00Z",
            "service_id": "gitea",
            "version_id": "28.1.0",
            "operation_count": 3,
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
                "kind": "v7_gitea_hook",
                "repo_name": "v7-smoke-hook",
                "hook_event": "push",
                "statuses": [201, 201, 200],
                "handoffs": [
                    {"name": "repo", "target_step": 2},
                    {"name": "hook_id", "target_step": 3},
                ],
            },
        }
        plans = [
            {
                "method": "POST",
                "path": "/api/v1/user/repos",
                "json": {"name": "v7-smoke-hook", "private": True, "auto_init": True},
                "capture": {"name": "repo", "source": "body_path", "path": ["name"]},
            },
            {
                "method": "POST",
                "path": "/api/v1/repos/corem_eval/{repo}/hooks",
                "json": {
                    "type": "gitea",
                    "active": False,
                    "events": ["push"],
                    "config": {
                        "url": "https://example.invalid/corem-eval-hook",
                        "content_type": "json",
                    },
                },
                "capture": {"name": "hook_id", "source": "body_path", "path": ["id"]},
            },
            {
                "method": "GET",
                "path": "/api/v1/repos/corem_eval/{repo}/hooks/{hook_id}",
            },
        ]
        self._run_new_family(event, plans, 2, "events", IndependentV7CheckerV5)

    def test_gitea_topic_lifecycle_family(self) -> None:
        event = {
            "event_id": "v7-smoke-gitea-topic",
            "event_time": "2026-10-18T00:00:00Z",
            "service_id": "gitea",
            "version_id": "28.1.0",
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
                "kind": "v7_gitea_topic",
                "repo_name": "v7-smoke-topic",
                "topic": "release-review",
                "statuses": [201, 204, 200, 204, 200],
                "handoffs": [
                    {"name": "repo", "target_step": 2},
                    {"name": "topic", "target_step": 4},
                ],
            },
        }
        plans = [
            {
                "method": "POST",
                "path": "/api/v1/user/repos",
                "json": {"name": "v7-smoke-topic", "private": True, "auto_init": True},
                "capture": {"name": "repo", "source": "body_path", "path": ["name"]},
            },
            {
                "method": "PUT",
                "path": "/api/v1/repos/corem_eval/{repo}/topics",
                "json": {"topics": ["release-review"]},
            },
            {
                "method": "GET",
                "path": "/api/v1/repos/corem_eval/{repo}/topics",
                "capture": {
                    "name": "topic",
                    "source": "body_path",
                    "path": ["topics", 0],
                },
            },
            {
                "method": "DELETE",
                "path": "/api/v1/repos/corem_eval/{repo}/topics/{topic}",
            },
            {"method": "GET", "path": "/api/v1/repos/corem_eval/{repo}/topics"},
        ]
        self._run_new_family(event, plans, 4, "topics", IndependentV7CheckerV5)

    def _run_workflow(
        self,
        service: str,
        version: str,
        plans: list[dict[str, object]],
    ) -> dict[str, object]:
        event = {
            "event_id": f"v7-smoke-{service}",
            "event_time": "2026-10-18T00:00:00+00:00",
            "service_id": service,
            "version_id": version,
            "operation_count": 3,
            "required_binding_atoms": [
                "binding:step_order",
                "binding:runtime_handoffs",
                "binding:status_codes",
            ],
            "required_evidence_atoms": [],
            "communication_criteria": [],
            "workflow_contract": (
                {
                    "kind": "v7_gitea_repo_issue",
                    "repo_name": "v7-smoke",
                    "issue_title": "Verify response handoff",
                    "statuses": [201, 201, 200],
                    "handoffs": [
                        {"name": "repo", "target_step": 2},
                        {"name": "issue_index", "target_step": 3},
                    ],
                }
                if service == "gitea"
                else {
                    "kind": "v7_meili_index",
                    "index_uid": "v7_smoke",
                    "statuses": [202, 200, 200],
                    "handoffs": [
                        {"name": "task_uid", "target_step": 2},
                        {"name": "index_uid", "target_step": 3},
                    ],
                }
            ),
        }
        runner = IndependentV7Runner()
        try:
            reset = runner.reset(event)
            self.assertTrue(reset["reset_verified"])
            execution = runner.execute(
                event,
                [_memory(index, plan) for index, plan in enumerate(plans, 1)],
            )
            self.assertTrue(execution["completed"])
            self.assertEqual(len(execution["steps"]), 3)
            self.assertEqual(
                [step["response"]["status"] for step in execution["steps"]],
                [
                    201 if service == "gitea" else 202,
                    201 if service == "gitea" else 200,
                    200,
                ],
            )
            checker = IndependentV7Checker()
            self.assertTrue(checker.check(event, execution)["goal_satisfied"])
            corrupted = deepcopy(execution)
            if service == "gitea":
                corrupted["steps"][2]["response"]["body"]["title"] = "Wrong title"
            else:
                corrupted["steps"][1]["response"]["body"]["status"] = "failed"
            self.assertFalse(checker.check(event, corrupted)["goal_satisfied"])
            runner.snapshot(event)
            return execution
        finally:
            runner._cleanup()

    def test_gitea_two_runtime_handoffs(self) -> None:
        execution = self._run_workflow(
            "gitea",
            "28.1.0",
            [
                {
                    "method": "POST",
                    "path": "/api/v1/user/repos",
                    "json": {"name": "v7-smoke", "private": True, "auto_init": True},
                    "capture": {
                        "name": "repo",
                        "source": "body_path",
                        "path": ["name"],
                    },
                },
                {
                    "method": "POST",
                    "path": "/api/v1/repos/corem_eval/{repo}/issues",
                    "json": {"title": "Verify response handoff"},
                    "capture": {
                        "name": "issue_index",
                        "source": "body_path",
                        "path": ["number"],
                    },
                },
                {
                    "method": "GET",
                    "path": "/api/v1/repos/corem_eval/{repo}/issues/{issue_index}",
                },
            ],
        )
        self.assertEqual(execution["bindings"]["repo"], "v7-smoke")
        self.assertEqual(execution["bindings"]["issue_index"], 1)
        self.assertEqual(
            execution["steps"][2]["response"]["body"]["title"],
            "Verify response handoff",
        )

    def test_meilisearch_two_runtime_handoffs(self) -> None:
        execution = self._run_workflow(
            "meilisearch",
            "1.55.0",
            [
                {
                    "method": "POST",
                    "path": "/indexes",
                    "json": {"uid": "v7_smoke", "primaryKey": "id"},
                    "capture": {
                        "name": "task_uid",
                        "source": "body_path",
                        "path": ["taskUid"],
                    },
                },
                {
                    "method": "GET",
                    "path": "/tasks/{task_uid}",
                    "capture": {
                        "name": "index_uid",
                        "source": "body_path",
                        "path": ["indexUid"],
                    },
                },
                {"method": "GET", "path": "/indexes/{index_uid}"},
            ],
        )
        self.assertEqual(
            execution["steps"][1]["response"]["body"]["status"], "succeeded"
        )
        self.assertEqual(execution["bindings"]["index_uid"], "v7_smoke")
        self.assertEqual(execution["steps"][2]["response"]["body"]["uid"], "v7_smoke")


if __name__ == "__main__":
    unittest.main()
