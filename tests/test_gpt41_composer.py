"""Offline contract checks for the fixed composer adapter."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.gpt41_composer import (
    MODEL,
    GPT41ComposerRunner,
    _materialize_steps,
    load_service_catalog,
)
from experiments.public_multiservice_challenge_runner import (
    PublicMultiserviceChallengeRunner,
)


class _Responses:
    def __init__(self, output: dict) -> None:
        self.output = output
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(id="response-test", output_text=json.dumps(self.output))


class ComposerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.event = {
            "query": "Follow the record through three operations",
            "service_id": "example",
            "version_id": "v1",
            "operation_count": 3,
            "workflow_contract": {"hidden_answer": "never send this"},
            "task_key": "hidden-task-key",
        }
        self.memories = [
            {
                "memory_id": f"m{index}",
                "text": f"step {index}",
                "role": "workflow_step",
                "step_index": index,
                "plan": {"method": "GET", "path": f"/step{index}"},
            }
            for index in range(1, 4)
        ]

    def test_composer_uses_only_exposed_fields_and_selected_memories(self) -> None:
        output = {"steps": [
            {
                "step_index": index,
                "memory_id": f"m{index}",
                "plan_json": json.dumps({"method": "GET", "path": f"/step{index}"}),
            }
            for index in range(1, 4)
        ]}
        responses = _Responses(output)
        runner = GPT41ComposerRunner(SimpleNamespace(responses=responses))
        with patch.object(PublicMultiserviceChallengeRunner, "execute", return_value={
            "completed": True, "steps": [], "bindings": {}, "communication": {}
        }) as execute:
            result = runner.execute(self.event, self.memories)
        self.assertTrue(result["completed"])
        self.assertEqual(result["composer_model"], MODEL)
        self.assertEqual(responses.request["model"], MODEL)
        self.assertTrue(responses.request["text"]["format"]["strict"])
        self.assertEqual(responses.request["temperature"], 0)
        prompt = responses.request["input"][1]["content"]
        self.assertNotIn("hidden_answer", prompt)
        self.assertNotIn("hidden-task-key", prompt)
        visible = json.loads(prompt)
        self.assertEqual(visible["operation_skeleton"], [1, 2, 3])
        self.assertEqual(visible["service_catalog"]["version_id"], "v1")
        self.assertEqual(len(result["composer_config_sha256"]), 64)
        self.assertEqual(len(result["composer_input_sha256"]), 64)
        self.assertEqual(len(execute.call_args.args[1]), 3)

    def test_unselected_or_misaligned_step_is_rejected(self) -> None:
        rows = [
            {"step_index": index, "memory_id": f"m{index}",
             "plan_json": json.dumps({"method": "GET", "path": f"/step{index}"})}
            for index in range(1, 4)
        ]
        rows[1]["memory_id"] = "outside-bank"
        with self.assertRaisesRegex(ValueError, "unavailable"):
            _materialize_steps({"steps": rows}, self.event, self.memories)
        rows[1]["memory_id"] = "m2"
        rows[1]["plan_json"] = '{"method":"GET","path":"//other-host"}'
        with self.assertRaisesRegex(ValueError, "unsafe"):
            _materialize_steps({"steps": rows}, self.event, self.memories)

    def test_rewritten_plan_is_rejected_without_execution(self) -> None:
        output = {"steps": [
            {
                "step_index": index,
                "memory_id": f"m{index}",
                "plan_json": json.dumps({
                    "method": "GET",
                    "path": "/invented" if index == 2 else f"/step{index}",
                }),
            }
            for index in range(1, 4)
        ]}
        responses = _Responses(output)
        runner = GPT41ComposerRunner(SimpleNamespace(responses=responses))
        with patch.object(PublicMultiserviceChallengeRunner, "execute") as execute:
            result = runner.execute(self.event, self.memories)
        self.assertFalse(result["completed"])
        self.assertIn("differs from the cited memory", result["failure"])
        execute.assert_not_called()

    def test_catalog_whitelists_public_operation_fields(self) -> None:
        event = dict(self.event)
        event["service_catalog"] = {"operations": [{
            "operation_id": "create",
            "method": "POST",
            "path": "/items",
            "description": "Create an item",
            "hidden_checker_result": "secret",
        }]}
        output = {"steps": [
            {
                "step_index": index,
                "memory_id": f"m{index}",
                "plan_json": json.dumps({"method": "GET", "path": f"/step{index}"}),
            }
            for index in range(1, 4)
        ]}
        responses = _Responses(output)
        runner = GPT41ComposerRunner(SimpleNamespace(responses=responses))
        with patch.object(PublicMultiserviceChallengeRunner, "execute", return_value={
            "completed": True, "steps": [], "bindings": {}, "communication": {}
        }):
            runner.execute(event, self.memories)
        prompt = responses.request["input"][1]["content"]
        self.assertIn("Create an item", prompt)
        self.assertNotIn("hidden_checker_result", prompt)
        self.assertNotIn("secret", prompt)

    def test_incomplete_model_response_does_not_execute(self) -> None:
        responses = _Responses({"steps": []})
        responses.create = lambda **kwargs: SimpleNamespace(
            id="incomplete-response", status="incomplete", output_text='{"steps": []}'
        )
        runner = GPT41ComposerRunner(SimpleNamespace(responses=responses))
        with patch.object(PublicMultiserviceChallengeRunner, "execute") as execute:
            result = runner.execute(self.event, self.memories)
        self.assertFalse(result["completed"])
        self.assertIn("did not complete", result["failure"])
        execute.assert_not_called()

    def test_catalog_file_covers_event_and_binds_visible_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "catalog.json"
            path.write_text(json.dumps({
                "schema_version": "corem-service-catalog-v1",
                "entries": [{
                    "service_id": "example", "version_id": "v1",
                    "operations": [{
                        "operation_id": "read", "method": "GET",
                        "path": "/items", "description": "Read items",
                    }],
                }],
            }), encoding="utf-8")
            runner = GPT41ComposerRunner(
                SimpleNamespace(responses=_Responses({"steps": []})),
                catalog_path=path,
            )
            runner.validate_events([self.event])
            digest = runner.input_sha256(self.event, self.memories)
            self.assertEqual(len(digest), 64)
            self.assertEqual(runner.composer_contract()["catalog_sha256"],
                             load_service_catalog(path)[1])
            with self.assertRaisesRegex(ValueError, "lacks event scopes"):
                runner.validate_events([{**self.event, "version_id": "v2"}])
            path.write_text(path.read_text(encoding="utf-8").replace(
                '"/items"', '"//other-host/items"'
            ), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "local"):
                load_service_catalog(path)


if __name__ == "__main__":
    unittest.main()
