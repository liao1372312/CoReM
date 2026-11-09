from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from corem.model import SetUtilityModel
from corem.retrieval import BundleBeamRetriever, RetrievalConfig
from corem.types import DependencyHyperedge, EvidenceGraph, MemoryRecord, Request
from experiments.generate_public_challenge_retention_selections import (
    _group_bank,
    _item_bank,
)
from experiments.generate_public_multiservice_bundle_candidates import (
    candidate_bundles,
    lexical_rolewise_candidates,
)
from experiments.generate_public_multiservice_challenge_ablations import (
    AdditiveUtilityModel,
    ItemOnlyRetriever,
)
from experiments.generate_public_multiservice_matched_baseline import select_bundle
from experiments.public_challenge_budget_runner import PublicChallengeBudgetRunner
from experiments.public_challenge_checker import PublicChallengeChecker
from experiments.public_challenge_checker_v2 import _create_fetch_query_goal
from experiments.public_challenge_runner import PublicChallengeRunner
from experiments.public_multiservice_challenge_runner import (
    PublicMultiserviceChallengeRunner,
    _render_json,
)
from experiments.public_workflows_checker import PublicWorkflowsChecker
from experiments.public_workflows_runner import PublicWorkflowsRunner, _capture_value


def _event() -> dict[str, object]:
    return {
        "event_id": "workflow-test",
        "service_id": "json-server",
        "version_id": "1.0.0-beta.3",
        "event_time": "2024-09-25T00:00:00Z",
        "required_binding_atoms": [
            "binding:step_order",
            "binding:runtime_handoff",
            "binding:status_codes",
        ],
        "required_evidence_atoms": [
            "evidence:step1",
            "evidence:step2",
            "evidence:verification",
        ],
        "communication_criteria": [
            "communication:both_statuses",
            "communication:workflow_summary",
        ],
        "workflow_contract": {
            "kind": "json_relationship_fetch",
            "statuses": [200, 200],
            "handoff_name": "post_id",
            "id_type": "string",
        },
    }


def _execution() -> dict[str, object]:
    return {
        "completed": True,
        "steps": [
            {
                "step_index": 1,
                "request": {"path": "/comments/1?_embed=post"},
                "response": {"status": 200, "body": {"post": {"id": "1"}}},
            },
            {
                "step_index": 2,
                "request": {"path": "/posts/1"},
                "response": {
                    "status": 200,
                    "body": {"id": "1", "title": "alpha"},
                },
            },
        ],
        "bindings": {"post_id": "1"},
        "selected_memory_observations": [
            {
                "memory_id": "memory-valid",
                "service_id": "json-server",
                "version_scope": ["1.0.0-beta.3"],
                "valid_from": "2024-09-24T00:00:00Z",
                "valid_until": None,
                "evidence_atoms": [
                    "evidence:step1",
                    "evidence:step2",
                    "evidence:verification",
                ],
                "provenance_verified": True,
            }
        ],
        "communication": {
            "explicit_statuses": True,
            "outcome_summary": "Executed two-step workflow successfully.",
        },
    }


class PublicWorkflowRunnerTests(unittest.TestCase):
    def test_capture_and_url_safe_runtime_substitution(self) -> None:
        response = {"status": 200, "body": {"post": {"id": "A/B C"}}}
        value = _capture_value(
            response, {"source": "body_path", "path": ["post", "id"]}
        )
        self.assertEqual(value, "A/B C")
        self.assertEqual(
            PublicWorkflowsRunner._render_path(
                "/anything/{post_id}", {"post_id": value}
            ),
            "/anything/A%2FB%20C",
        )

    def test_missing_runtime_binding_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unavailable binding"):
            PublicWorkflowsRunner._render_path("/posts/{post_id}", {})

    def test_json_renderer_distinguishes_literals_from_named_bindings(self) -> None:
        payload = {
            "id": "{expectation_id}",
            "body": '{"proof":"alpha"}',
        }
        self.assertEqual(
            _render_json(payload, {"expectation_id": "runtime-123"}),
            {"id": "runtime-123", "body": '{"proof":"alpha"}'},
        )
        with self.assertRaisesRegex(ValueError, "unavailable binding"):
            _render_json({"id": "{expectation_id}"}, {})

    def test_json_renderer_preserves_numeric_full_binding(self) -> None:
        self.assertEqual(
            _render_json(
                {"options": {"cas": "{version}"}, "description": "v{version}"},
                {"version": 2},
            ),
            {"options": {"cas": 2}, "description": "v2"},
        )


class PublicWorkflowCheckerTests(unittest.TestCase):
    def test_complete_two_step_execution_passes(self) -> None:
        checked = PublicWorkflowsChecker().check(_event(), _execution())
        self.assertTrue(checked["goal_satisfied"])
        self.assertTrue(all(row["passed"] for row in checked["binding_checks"]))
        self.assertTrue(all(row["covered"] for row in checked["evidence_checks"]))
        self.assertFalse(checked["reuse_attempts"][0]["invalid_for_event"])

    def test_unobserved_handoff_fails_independent_check(self) -> None:
        execution = copy.deepcopy(_execution())
        execution["steps"][1]["request"]["path"] = "/posts/2"
        checked = PublicWorkflowsChecker().check(_event(), execution)
        self.assertFalse(checked["goal_satisfied"])
        handoff = {row["atom"]: row["passed"] for row in checked["binding_checks"]}
        self.assertFalse(handoff["binding:runtime_handoff"])


def _challenge_event() -> dict[str, object]:
    return {
        "event_id": "challenge-test",
        "service_id": "go-httpbin",
        "version_id": "v2.15.0",
        "event_time": "2024-09-25T00:00:00Z",
        "required_binding_atoms": [
            "binding:step_order",
            "binding:runtime_handoffs",
            "binding:status_codes",
        ],
        "required_evidence_atoms": [
            "step1:go-httpbin:uuid_status_chain",
            "step2:go-httpbin:uuid_status_chain",
            "step3:go-httpbin:uuid_status_chain",
            "verify:go-httpbin:uuid_status_chain",
        ],
        "communication_criteria": [
            "communication:all_statuses",
            "communication:workflow_summary",
        ],
        "workflow_contract": {
            "kind": "httpbin_uuid_status_chain",
            "statuses": [200, 200, 200],
            "handoffs": [
                {"name": "uuid", "target_step": 2},
                {"name": "reflected_status", "target_step": 3},
            ],
        },
    }


def _challenge_execution() -> dict[str, object]:
    uuid = "12345678-1234-1234-1234-123456789abc"
    evidence = [
        "step1:go-httpbin:uuid_status_chain",
        "step2:go-httpbin:uuid_status_chain",
        "step3:go-httpbin:uuid_status_chain",
        "verify:go-httpbin:uuid_status_chain",
    ]
    return {
        "completed": True,
        "steps": [
            {
                "step_index": 1,
                "request": {"path": "/uuid"},
                "response": {"status": 200, "body": {"uuid": uuid}},
            },
            {
                "step_index": 2,
                "request": {"path": f"/anything/{uuid}"},
                "response": {
                    "status": 200,
                    "body": {"url": f"http://example.test/anything/{uuid}"},
                },
            },
            {
                "step_index": 3,
                "request": {"path": "/anything/status-200"},
                "response": {
                    "status": 200,
                    "body": {"url": "http://example.test/anything/status-200"},
                },
            },
        ],
        "bindings": {"uuid": uuid, "reflected_status": 200},
        "selected_memory_observations": [
            {
                "memory_id": "challenge-valid",
                "service_id": "go-httpbin",
                "version_scope": ["v2.15.0"],
                "valid_from": "2024-09-24T00:00:00Z",
                "valid_until": None,
                "evidence_atoms": evidence,
                "provenance_verified": True,
            }
        ],
        "communication": {
            "explicit_statuses": True,
            "outcome_summary": "Executed three-step workflow successfully.",
        },
    }


class PublicChallengeTests(unittest.TestCase):
    def test_v5_bundle_candidates_are_complete_and_budgeted(self) -> None:
        event = {
            "event_id": "event",
            "event_time": "2025-01-02T00:00:00Z",
            "query": "alpha workflow",
            "service_id": "service",
            "version_id": "v1",
        }
        memories = [
            {
                "memory_id": f"m{index}",
                "text": f"alpha {role}",
                "evidence_atoms": [f"role:{role}"],
                "operation_ids": [],
                "token_cost": 40,
                "storage_cost": 40,
                "provenance_uri": "test",
                "provenance_verified": True,
                "service_id": "service",
                "version_scope": ["v1"],
                "valid_from": "2025-01-01T00:00:00Z",
                "valid_until": None,
            }
            for index, role in enumerate(("step1", "step2", "step3", "verify"), start=1)
        ]
        hyperedges = [
            {
                "edge_id": "dependency:test",
                "members": [row["memory_id"] for row in memories],
                "evidence_atoms": [
                    "role:step1",
                    "role:step2",
                    "role:step3",
                    "role:verify",
                ],
                "relation_type": "runtime-handoff",
                "provenance": "test",
                "verified": True,
                "version_scope": ["v1"],
                "valid_from": "2025-01-01T00:00:00Z",
                "valid_until": None,
            }
        ]
        candidates = candidate_bundles(event, memories, hyperedges, 160)
        self.assertEqual(
            candidates, [("dependency:test", ["m1", "m2", "m3", "m4"], 160)]
        )
        self.assertEqual(candidate_bundles(event, memories, hyperedges, 159), [])
        lexical = lexical_rolewise_candidates(event, memories, 160)
        self.assertEqual(len(lexical), 1)
        self.assertEqual(lexical[0][1], ["m1", "m2", "m3", "m4"])

    def test_v5_runner_normalizes_selected_plan_binding_failure(self) -> None:
        memories = [
            {
                "memory_id": f"step-{index}",
                "role": "workflow_step",
                "step_index": index,
                "plan": {
                    "method": "GET",
                    "path": "/step/{missing}" if index == 1 else f"/step-{index}",
                },
                "service_id": "go-httpbin",
                "version_scope": ["v2.15.0"],
                "valid_from": "2024-09-24T00:00:00Z",
                "valid_until": None,
                "evidence_atoms": [],
                "provenance_verified": True,
            }
            for index in (1, 2, 3)
        ]
        event = _challenge_event()
        event["operation_count"] = 3
        execution = PublicMultiserviceChallengeRunner().execute(event, memories)
        self.assertFalse(execution["completed"])
        self.assertIn("runtime bindings", execution["failure"])

    def test_v5_runner_does_not_hide_unrelated_value_error(self) -> None:
        runner = PublicMultiserviceChallengeRunner()
        with (
            patch.object(
                runner,
                "_execute_selected",
                side_effect=ValueError("unrelated infrastructure failure"),
            ),
            self.assertRaisesRegex(ValueError, "infrastructure"),
        ):
            runner.execute(_challenge_event(), [])

    def test_v5_item_only_ablation_excludes_dependency_bundles(self) -> None:
        graph = EvidenceGraph()
        for index, role in enumerate(("step1", "step2", "step3", "verify"), 1):
            graph.add_record(
                MemoryRecord(
                    record_id=f"m{index}",
                    text=f"workflow {role}",
                    evidence_atoms=frozenset((f"role:{role}",)),
                    token_cost=40,
                    storage_cost=40,
                    provenance_verified=True,
                )
            )
        graph.add_hyperedge(
            DependencyHyperedge(
                edge_id="dependency:test",
                members=frozenset(("m1", "m2", "m3", "m4")),
                evidence_atoms=frozenset(
                    ("role:step1", "role:step2", "role:step3", "role:verify")
                ),
                relation_type="three-step-runtime-handoff",
                provenance="test",
            )
        )
        request = Request(
            request_id="q",
            text="workflow",
            required_atoms=frozenset(
                ("role:step1", "role:step2", "role:step3", "role:verify")
            ),
        )
        config = RetrievalConfig(
            retrieval_budget=160,
            beam_width=16,
            max_depth=4,
            candidate_pool_size=10,
            enforce_dependency_closure=False,
        )
        full = BundleBeamRetriever(SetUtilityModel(), config)
        item_only = ItemOnlyRetriever(SetUtilityModel(), config)
        bank = frozenset(graph.records)
        self.assertTrue(
            any(
                len(bundle.members) == 4
                for bundle in full.build_bundles(request, bank, graph)
            )
        )
        self.assertTrue(
            all(
                len(bundle.members) == 1
                for bundle in item_only.build_bundles(request, bank, graph)
            )
        )

    def test_v5_additive_ablation_removes_complement_gain(self) -> None:
        graph = EvidenceGraph()
        graph.add_record(
            MemoryRecord(
                record_id="m1",
                text="workflow",
                evidence_atoms=frozenset(("role:step1",)),
                token_cost=40,
                storage_cost=40,
                provenance_verified=True,
            )
        )
        graph.add_record(
            MemoryRecord(
                record_id="m2",
                text="workflow verification",
                evidence_atoms=frozenset(("role:verify",)),
                token_cost=40,
                storage_cost=40,
                provenance_verified=True,
            )
        )
        graph.add_hyperedge(
            DependencyHyperedge(
                edge_id="dependency:test",
                members=frozenset(("m1", "m2")),
                evidence_atoms=frozenset(("role:step1", "role:verify")),
                relation_type="runtime-handoff",
                provenance="test",
            )
        )
        request = Request(
            request_id="q",
            text="workflow",
            required_atoms=frozenset(("role:step1",)),
        )
        selected = frozenset(("m1", "m2"))
        self.assertGreater(
            SetUtilityModel().logit(request, selected, graph),
            AdditiveUtilityModel().logit(request, selected, graph),
        )

    def test_v5_matched_bm25_selects_an_atomic_dependency_bundle(self) -> None:
        graph = EvidenceGraph()
        rows = []
        for index, role in enumerate(("step1", "step2", "step3", "verify"), 1):
            record = MemoryRecord(
                record_id=f"m{index}",
                text=f"alpha workflow {role}",
                evidence_atoms=frozenset((f"role:{role}",)),
                token_cost=40,
                storage_cost=40,
                provenance_verified=True,
            )
            graph.add_record(record)
            rows.append(
                {
                    "memory_id": record.record_id,
                    "text": record.text,
                    "valid_from": "2025-01-01T00:00:00Z",
                }
            )
        graph.add_hyperedge(
            DependencyHyperedge(
                edge_id="dependency:test",
                members=frozenset(graph.records),
                evidence_atoms=frozenset(
                    ("role:step1", "role:step2", "role:step3", "role:verify")
                ),
                relation_type="three-step-runtime-handoff",
                provenance="test",
            )
        )
        request = Request(
            request_id="q",
            text="alpha workflow",
            required_atoms=frozenset(
                ("role:step1", "role:step2", "role:step3", "role:verify")
            ),
        )
        selected, cost, _ = select_bundle(
            request,
            rows,
            graph,
            seed=2027,
            event_id="q",
            token_budget=160,
        )
        self.assertEqual(selected, ["m1", "m2", "m3", "m4"])
        self.assertEqual(cost, 160)

    def test_challenge_runner_requires_all_three_steps(self) -> None:
        memories = [
            {
                "memory_id": f"step-{index}",
                "role": "workflow_step",
                "step_index": index,
                "plan": {"method": "GET", "path": f"/step-{index}"},
            }
            for index in (1, 2, 3)
        ]
        plans = PublicChallengeRunner._challenge_plans(memories, 3)
        self.assertEqual([row[0] for row in plans], [1, 2, 3])

    def test_challenge_checker_validates_both_runtime_handoffs(self) -> None:
        checked = PublicChallengeChecker().check(
            _challenge_event(), _challenge_execution()
        )
        self.assertTrue(checked["goal_satisfied"])
        self.assertTrue(all(row["passed"] for row in checked["binding_checks"]))
        self.assertTrue(all(row["covered"] for row in checked["evidence_checks"]))

        broken = copy.deepcopy(_challenge_execution())
        broken["steps"][2]["request"]["path"] = "/anything/status-201"
        checked = PublicChallengeChecker().check(_challenge_event(), broken)
        self.assertFalse(checked["goal_satisfied"])
        handoffs = {row["atom"]: row["passed"] for row in checked["binding_checks"]}
        self.assertFalse(handoffs["binding:runtime_handoffs"])

    def test_repaired_create_fetch_query_goal(self) -> None:
        contract = {
            "statuses": [201, 200, 200],
            "id_type": "string",
        }
        steps = [
            {
                "response": {
                    "status": 201,
                    "body": {"id": "a1", "title": "Echo", "views": 7},
                }
            },
            {
                "response": {
                    "status": 200,
                    "body": {"id": "a1", "title": "Echo", "views": 7},
                }
            },
            {
                "response": {
                    "status": 200,
                    "body": [{"id": "a1", "title": "Echo", "views": 7}],
                }
            },
        ]
        self.assertTrue(
            _create_fetch_query_goal(
                contract, steps, {"created_id": "a1", "post_title": "Echo"}
            )
        )

    def test_budget_runner_normalizes_selected_plan_binding_failure(self) -> None:
        runner = PublicChallengeBudgetRunner()
        with patch.object(
            runner,
            "base_execute",
            side_effect=ValueError("workflow step requires an unavailable binding"),
        ):
            execution = runner.execute(_challenge_event(), [])
        self.assertFalse(execution["completed"])
        self.assertIn("runtime bindings", execution["failure"])

    def test_budget_runner_does_not_hide_unrelated_value_error(self) -> None:
        runner = PublicChallengeBudgetRunner()
        with (
            patch.object(
                runner,
                "base_execute",
                side_effect=ValueError("unrelated infrastructure failure"),
            ),
            self.assertRaisesRegex(ValueError, "infrastructure"),
        ):
            runner.execute(_challenge_event(), [])


class RetentionPolicyTests(unittest.TestCase):
    def test_group_bank_never_keeps_a_partial_dependency_group(self) -> None:
        memories = {
            memory_id: {"storage_cost": 50} for memory_id in ("a1", "a2", "b1", "b2")
        }
        hyperedges = [
            {"edge_id": "a", "members": ["a1", "a2"]},
            {"edge_id": "b", "members": ["b1", "b2"]},
        ]
        selected, cost, groups = _group_bank(
            hyperedges,
            memories,
            {"a1": 3.0, "a2": 3.0, "b1": 1.0, "b2": 1.0},
            seed=2027,
            budget=150,
        )
        self.assertEqual(selected, ["a1", "a2"])
        self.assertEqual(cost, 100)
        self.assertEqual(groups, 1)

    def test_item_bank_obeys_budget_without_group_closure(self) -> None:
        memories = [
            {
                "memory_id": memory_id,
                "storage_cost": 50,
                "valid_from": "2024-01-01T00:00:00Z",
            }
            for memory_id in ("a1", "a2", "b1")
        ]
        selected, cost, groups = _item_bank(
            memories,
            {"a1": 3.0, "a2": 2.0, "b1": 1.0},
            seed=2027,
            budget=75,
            recency=False,
        )
        self.assertEqual(selected, ["a1"])
        self.assertEqual(cost, 50)
        self.assertEqual(groups, 0)


if __name__ == "__main__":
    unittest.main()
