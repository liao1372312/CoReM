"""Independent checker rules for Gitea file, webhook, and topic workflows."""

from __future__ import annotations

import base64
import urllib.parse
from collections.abc import Mapping
from typing import Any

from experiments.independent_v7_checker import _request_path, _response
from experiments.independent_v7_checker_v4 import IndependentV7CheckerV4


def _requests(steps: list[Any]) -> list[dict[str, Any]] | None:
    rows = [step.get("request") if isinstance(step, dict) else None for step in steps]
    if any(not isinstance(row, dict) for row in rows):
        return None
    return rows


def _bodies(steps: list[Any]) -> tuple[list[int], list[Any]] | None:
    responses = [_response(step) for step in steps]
    if any(row is None for row in responses):
        return None
    statuses = [row.get("status") for row in responses]
    if any(
        not isinstance(status, int) or isinstance(status, bool) for status in statuses
    ):
        return None
    return statuses, [row.get("body") for row in responses]


def _repo_prefix(repo_name: str) -> str:
    return f"/api/v1/repos/corem_eval/{urllib.parse.quote(repo_name, safe='')}"


def _repo_valid(repo: Any, repo_name: str, bindings: Mapping[str, Any]) -> bool:
    return bool(
        isinstance(repo, dict)
        and repo.get("name") == bindings.get("repo") == repo_name
        and repo.get("private") is True
    )


def _file_goal(
    contract: Mapping[str, Any], steps: list[Any], bindings: Mapping[str, Any]
) -> bool:
    if len(steps) != 3:
        return False
    observed = _bodies(steps)
    requests = _requests(steps)
    if observed is None or requests is None:
        return False
    statuses, bodies = observed
    repo, created, fetched = bodies
    if not isinstance(created, dict) or not isinstance(fetched, dict):
        return False
    repo_name = contract.get("repo_name")
    file_path = contract.get("file_path")
    expected_content = contract.get("file_text")
    if not all(
        isinstance(value, str) for value in (repo_name, file_path, expected_content)
    ):
        return False
    created_content = created.get("content")
    create_json = requests[1].get("json")
    if not isinstance(created_content, dict) or not isinstance(create_json, dict):
        return False
    try:
        decoded = (
            base64.b64decode(str(fetched.get("content", "")), validate=False)
            .decode()
            .strip()
        )
        submitted = (
            base64.b64decode(str(create_json.get("content", "")), validate=True)
            .decode()
            .strip()
        )
    except (ValueError, UnicodeDecodeError):
        return False
    prefix = _repo_prefix(repo_name)
    return bool(
        statuses == [201, 201, 200]
        and _repo_valid(repo, repo_name, bindings)
        and created_content.get("path")
        == fetched.get("path")
        == bindings.get("file_path")
        == file_path
        and decoded == submitted == expected_content
        and requests[1].get("method") == "POST"
        and requests[2].get("method") == "GET"
        and _request_path(steps[1])
        == f"{prefix}/contents/{urllib.parse.quote(file_path, safe='')}"
        and _request_path(steps[2])
        == f"{prefix}/contents/{urllib.parse.quote(file_path, safe='')}"
    )


def _hook_goal(
    contract: Mapping[str, Any], steps: list[Any], bindings: Mapping[str, Any]
) -> bool:
    if len(steps) != 3:
        return False
    observed = _bodies(steps)
    requests = _requests(steps)
    if observed is None or requests is None:
        return False
    statuses, bodies = observed
    repo, created, fetched = bodies
    if not isinstance(created, dict) or not isinstance(fetched, dict):
        return False
    repo_name = contract.get("repo_name")
    expected_event = contract.get("hook_event")
    hook_id = created.get("id")
    create_json = requests[1].get("json")
    if not isinstance(repo_name, str) or not isinstance(expected_event, str):
        return False
    if not isinstance(hook_id, int) or isinstance(hook_id, bool):
        return False
    if not isinstance(create_json, dict):
        return False
    config = create_json.get("config")
    prefix = _repo_prefix(repo_name)
    return bool(
        statuses == [201, 201, 200]
        and _repo_valid(repo, repo_name, bindings)
        and fetched.get("id") == bindings.get("hook_id") == hook_id
        and created.get("type") == fetched.get("type") == "gitea"
        and created.get("active") is False
        and fetched.get("active") is False
        and created.get("events") == fetched.get("events") == [expected_event]
        and create_json.get("events") == [expected_event]
        and create_json.get("active") is False
        and isinstance(config, dict)
        and config.get("content_type") == "json"
        and config.get("url") == "https://example.invalid/corem-eval-hook"
        and requests[1].get("method") == "POST"
        and requests[2].get("method") == "GET"
        and _request_path(steps[1]) == f"{prefix}/hooks"
        and _request_path(steps[2]) == f"{prefix}/hooks/{hook_id}"
    )


def _topic_goal(
    contract: Mapping[str, Any], steps: list[Any], bindings: Mapping[str, Any]
) -> bool:
    if len(steps) != 5:
        return False
    observed = _bodies(steps)
    requests = _requests(steps)
    if observed is None or requests is None:
        return False
    statuses, bodies = observed
    repo, _, listed, _, final = bodies
    repo_name = contract.get("repo_name")
    topic = contract.get("topic")
    if not isinstance(repo_name, str) or not isinstance(topic, str):
        return False
    if not isinstance(listed, dict) or not isinstance(final, dict):
        return False
    prefix = _repo_prefix(repo_name)
    return bool(
        statuses == [201, 204, 200, 204, 200]
        and _repo_valid(repo, repo_name, bindings)
        and requests[1].get("json") == {"topics": [topic]}
        and listed.get("topics") == [topic]
        and bindings.get("topic") == topic
        and final.get("topics") == []
        and [row.get("method") for row in requests]
        == ["POST", "PUT", "GET", "DELETE", "GET"]
        and _request_path(steps[1]) == f"{prefix}/topics"
        and _request_path(steps[2]) == f"{prefix}/topics"
        and _request_path(steps[3])
        == f"{prefix}/topics/{urllib.parse.quote(topic, safe='')}"
        and _request_path(steps[4]) == f"{prefix}/topics"
    )


class IndependentV7CheckerV5(IndependentV7CheckerV4):
    def check(
        self, event: Mapping[str, Any], execution: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        contract = event.get("workflow_contract")
        if not isinstance(contract, dict):
            raise TypeError("v7 event requires a workflow contract")
        kind = contract.get("kind")
        goals = {
            "v7_gitea_file": _file_goal,
            "v7_gitea_hook": _hook_goal,
            "v7_gitea_topic": _topic_goal,
        }
        if kind not in goals:
            return super().check(event, execution)
        checked = dict(self.base.check(event, execution))
        steps = execution.get("steps")
        if not isinstance(steps, list):
            steps = []
        bindings = execution.get("bindings")
        if not isinstance(bindings, dict):
            bindings = {}
        expected_order = list(range(1, int(event.get("operation_count", 0)) + 1))
        for item in checked["binding_checks"]:
            if item.get("atom") == "binding:step_order":
                item["passed"] = [
                    step.get("step_index") if isinstance(step, dict) else None
                    for step in steps
                ] == expected_order
        for item in checked["communication_checks"]:
            if item.get("criterion") == "communication:workflow_summary":
                communication = execution.get("communication")
                summary = (
                    communication.get("outcome_summary")
                    if isinstance(communication, dict)
                    else None
                )
                item["passed"] = isinstance(summary, str) and (
                    f"{len(expected_order)}-step" in summary
                    or (len(expected_order) == 3 and "three-step" in summary)
                )
        checked["contract_id"] = "independent-v7-checker-v5"
        checked["goal_satisfied"] = bool(
            execution.get("completed") is True
            and all(item.get("passed") is True for item in checked["binding_checks"])
            and goals[kind](contract, steps, bindings)
        )
        return checked


def create_checker() -> IndependentV7CheckerV5:
    return IndependentV7CheckerV5()
