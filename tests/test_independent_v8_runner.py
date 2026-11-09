"""Opt-in real-service smoke tests for the prospective v8 runner."""

from __future__ import annotations

import copy
import os
import unittest
from pathlib import Path

from corem.real_eval import load_jsonl
from experiments.independent_v8_checker import IndependentV8Checker, _assertions_hold
from experiments.independent_v8_runner import IndependentV8Runner


class V8CheckerAssertionTests(unittest.TestCase):
    def test_nonempty_requires_present_nonempty_collection(self) -> None:
        assertion = [{"path": ["result", "snapshots"], "nonempty": True}]
        self.assertTrue(_assertions_hold({"result": {"snapshots": ["one"]}}, assertion))
        self.assertFalse(_assertions_hold({"result": {"snapshots": []}}, assertion))
        self.assertFalse(_assertions_hold({"result": {}}, assertion))


@unittest.skipUnless(
    os.environ.get("COREM_RUN_V8_INTEGRATION") == "1",
    "set COREM_RUN_V8_INTEGRATION=1 for local Qdrant/OpenBao integration",
)
class IndependentV8RunnerTests(unittest.TestCase):
    def test_alias_routing_development_oracle_and_negative(self) -> None:
        checker = IndependentV8Checker()
        for dataset_name in (
            "public_independent_v8_alias_development_pilot_v1",
            "public_independent_v8_alias_role_swap_development_v1",
        ):
            dataset = Path("data") / dataset_name
            event = load_jsonl(dataset / "events.jsonl")[0]
            memories = {
                str(row["memory_id"]): row for row in load_jsonl(dataset / "memories.jsonl")
            }
            for name, expected in (
                ("pilot_positive_selections.jsonl", True),
                ("pilot_negative_selections.jsonl", False),
            ):
                with self.subTest(dataset=dataset_name, selection=name):
                    selection = load_jsonl(dataset / name)[0]
                    runner = IndependentV8Runner()
                    runner.reset(event)
                    try:
                        execution = runner.execute(
                            event,
                            [memories[memory_id] for memory_id in selection["selected_memory_ids"]],
                        )
                        result = checker.check(event, execution, runner.checker_context())
                        self.assertEqual(result["goal_satisfied"], expected)
                    finally:
                        runner.snapshot(event)

    def _run(self, service: str, version: str, plans: list[dict]) -> tuple[dict, dict]:
        runner = IndependentV8Runner()
        event = {
            "event_id": f"v8-development-{service}",
            "service_id": service,
            "version_id": version,
            "operation_count": len(plans),
            "workflow_contract": (
                {
                    "kind": "pilot_qdrant_point",
                    "collection": "v8pilot",
                    "point_id": 1,
                    "tag": "alpha",
                }
                if service == "qdrant"
                else {
                    "kind": "pilot_openbao_kv",
                    "secret_key": "v8pilot",
                    "marker": "good",
                }
            ),
        }
        memories = [
            {
                "memory_id": f"{service}:step-{index}",
                "role": "workflow_step",
                "step_index": index,
                "plan": plan,
            }
            for index, plan in enumerate(plans, 1)
        ]
        reset = runner.reset(event)
        try:
            execution = runner.execute(event, memories)
            context = runner.checker_context()
            checker = IndependentV8Checker()
            self.assertTrue(checker.check(event, execution, context)["goal_satisfied"])
            assertion_event = dict(event)
            if service == "qdrant":
                state_path = "/collections/v8pilot/points/1?with_payload=true"
                state_assertions = [
                    {"path": ["result", "id"], "equals": 1},
                    {"path": ["result", "payload", "tag"], "equals": "alpha"},
                ]
                trace_path = ["steps", 2, "response", "body", "result", "points", 0, "id"]
                trace_value = 1
            else:
                state_path = "/v1/secret/data/v8pilot"
                state_assertions = [
                    {"path": ["data", "data", "marker"], "equals": "good"},
                    {"path": ["data", "metadata", "version"], "equals": 1},
                ]
                trace_path = ["steps", 1, "response", "body", "data", "data", "marker"]
                trace_value = "good"
            assertion_event["workflow_contract"] = {
                "kind": "assertions_v1",
                "trace_statuses": [200, 200, 200],
                "trace_assertions": [{"path": trace_path, "equals": trace_value}],
                "state_checks": [
                    {"path": state_path, "status": 200, "assertions": state_assertions}
                ],
            }
            self.assertTrue(
                checker.check(assertion_event, execution, context)["goal_satisfied"]
            )
            corrupted = copy.deepcopy(execution)
            if service == "qdrant":
                corrupted["steps"][2]["response"]["body"]["result"]["points"][0]["id"] = 2
            else:
                corrupted["steps"][1]["response"]["body"]["data"]["data"]["marker"] = "wrong"
            self.assertFalse(checker.check(event, corrupted, context)["goal_satisfied"])
            self.assertFalse(
                checker.check(assertion_event, corrupted, context)["goal_satisfied"]
            )
            if service == "qdrant":
                check = runner._request(
                    {"method": "GET"},
                    "/collections/v8pilot/points/1?with_payload=true",
                )
            else:
                check = runner._request(
                    {"method": "GET"}, "/v1/secret/data/v8pilot"
                )
            self.assertEqual(check["status"], 200)
            self.assertTrue(reset["reset_verified"])
            self.assertIn("127.0.0.1", reset["healthcheck"])
            return execution, check
        finally:
            runner.snapshot(event)
            self.assertIsNone(runner.process)

    def test_qdrant_collection_point_query(self) -> None:
        execution, check = self._run(
            "qdrant",
            "1.15.5",
            [
                {
                    "method": "PUT",
                    "path": "/collections/v8pilot",
                    "json": {"vectors": {"size": 2, "distance": "Cosine"}},
                },
                {
                    "method": "PUT",
                    "path": "/collections/v8pilot/points?wait=true",
                    "json": {
                        "points": [
                            {"id": 1, "vector": [0.9, 0.1], "payload": {"tag": "alpha"}},
                            {"id": 2, "vector": [0.1, 0.9], "payload": {"tag": "beta"}},
                        ]
                    },
                },
                {
                    "method": "POST",
                    "path": "/collections/v8pilot/points/query",
                    "json": {"query": [1.0, 0.0], "limit": 1, "with_payload": True},
                },
            ],
        )
        self.assertTrue(execution["completed"])
        self.assertEqual([step["response"]["status"] for step in execution["steps"]], [200, 200, 200])
        self.assertEqual(execution["steps"][2]["response"]["body"]["result"]["points"][0]["id"], 1)
        self.assertEqual(check["body"]["result"]["payload"]["tag"], "alpha")

    def test_openbao_kv_versioned_readback(self) -> None:
        execution, check = self._run(
            "openbao",
            "2.7.0",
            [
                {
                    "method": "POST",
                    "path": "/v1/secret/data/v8pilot",
                    "json": {"data": {"marker": "good"}},
                },
                {"method": "GET", "path": "/v1/secret/data/v8pilot"},
                {"method": "GET", "path": "/v1/secret/metadata/v8pilot"},
            ],
        )
        self.assertTrue(execution["completed"])
        self.assertEqual([step["response"]["status"] for step in execution["steps"]], [200, 200, 200])
        self.assertEqual(check["body"]["data"]["data"]["marker"], "good")
        self.assertEqual(execution["steps"][2]["response"]["body"]["data"]["current_version"], 1)


if __name__ == "__main__":
    unittest.main()
