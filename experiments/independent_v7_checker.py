"""Independent semantic checker for the first v7 service-family pilots.

The checker reads the event contract and captured HTTP transcript, not the
runner's success flag alone. Additional task kinds must have explicit checks.
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Mapping
from typing import Any

from experiments.public_challenge_checker import PublicChallengeChecker


def _response(step: Any) -> dict[str, Any] | None:
    if not isinstance(step, dict):
        return None
    value = step.get("response")
    return value if isinstance(value, dict) else None


def _request_path(step: Any) -> str | None:
    if not isinstance(step, dict) or not isinstance(step.get("request"), dict):
        return None
    value = step["request"].get("path")
    return value if isinstance(value, str) else None


def _gitea_goal(
    contract: Mapping[str, Any], steps: list[Any], bindings: Mapping[str, Any]
) -> bool:
    if len(steps) != 3:
        return False
    first, second, third = (_response(step) for step in steps)
    if first is None or second is None or third is None:
        return False
    a, b, c = (row.get("body") for row in (first, second, third))
    if not all(isinstance(body, dict) for body in (a, b, c)):
        return False
    repo = contract.get("repo_name")
    title = contract.get("issue_title")
    issue_index = b.get("number")
    if not isinstance(repo, str) or not isinstance(title, str):
        return False
    if not isinstance(issue_index, int) or isinstance(issue_index, bool):
        return False
    full_name = f"corem_eval/{repo}"
    return bool(
        [first.get("status"), second.get("status"), third.get("status")]
        == [201, 201, 200]
        and a.get("name") == bindings.get("repo") == repo
        and a.get("full_name") == full_name
        and a.get("private") is True
        and b.get("title") == c.get("title") == title
        and b.get("number") == c.get("number") == bindings.get("issue_index")
        and isinstance(b.get("repository"), dict)
        and b["repository"].get("full_name") == full_name
        and isinstance(c.get("repository"), dict)
        and c["repository"].get("full_name") == full_name
        and _request_path(steps[1])
        == f"/api/v1/repos/corem_eval/{urllib.parse.quote(repo, safe='')}/issues"
        and _request_path(steps[2])
        == (
            f"/api/v1/repos/corem_eval/{urllib.parse.quote(repo, safe='')}"
            f"/issues/{issue_index}"
        )
    )


def _meili_goal(
    contract: Mapping[str, Any], steps: list[Any], bindings: Mapping[str, Any]
) -> bool:
    if len(steps) != 3:
        return False
    first, second, third = (_response(step) for step in steps)
    if first is None or second is None or third is None:
        return False
    a, b, c = (row.get("body") for row in (first, second, third))
    if not all(isinstance(body, dict) for body in (a, b, c)):
        return False
    uid = contract.get("index_uid")
    task_uid = a.get("taskUid")
    if not isinstance(uid, str) or not isinstance(task_uid, int):
        return False
    return bool(
        [first.get("status"), second.get("status"), third.get("status")]
        == [202, 200, 200]
        and a.get("indexUid") == b.get("indexUid") == c.get("uid") == uid
        and b.get("uid") == bindings.get("task_uid") == task_uid
        and b.get("status") == "succeeded"
        and b.get("error") is None
        and c.get("primaryKey") == "id"
        and bindings.get("index_uid") == uid
        and _request_path(steps[1]) == f"/tasks/{task_uid}"
        and _request_path(steps[2]) == f"/indexes/{urllib.parse.quote(uid, safe='')}"
    )


class IndependentV7Checker:
    def __init__(self) -> None:
        self.base = PublicChallengeChecker()

    def check(
        self, event: Mapping[str, Any], execution: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        checked = dict(self.base.check(event, execution))
        contract = event.get("workflow_contract")
        if not isinstance(contract, dict):
            raise TypeError("v7 event requires a workflow contract")
        steps = execution.get("steps")
        if not isinstance(steps, list):
            steps = []
        bindings = execution.get("bindings")
        if not isinstance(bindings, dict):
            bindings = {}
        kind = contract.get("kind")
        if kind == "v7_gitea_repo_issue":
            semantic_goal = _gitea_goal(contract, steps, bindings)
        elif kind == "v7_meili_index":
            semantic_goal = _meili_goal(contract, steps, bindings)
        else:
            raise ValueError(f"unimplemented v7 task kind: {kind}")
        binding_pass = all(
            item.get("passed") is True for item in checked["binding_checks"]
        )
        checked["contract_id"] = "independent-v7-pilot-checker-v1"
        checked["goal_satisfied"] = bool(
            execution.get("completed") is True and binding_pass and semantic_goal
        )
        return checked


def create_checker() -> IndependentV7Checker:
    return IndependentV7Checker()
