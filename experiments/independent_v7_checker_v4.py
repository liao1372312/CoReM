"""Checker extension for a four-step Gitea issue-comment workflow."""

from __future__ import annotations

import urllib.parse
from collections.abc import Mapping
from typing import Any

from experiments.independent_v7_checker import _request_path, _response
from experiments.independent_v7_checker_v3 import IndependentV7CheckerV3


def _issue_comment_goal(
    contract: Mapping[str, Any], steps: list[Any], bindings: Mapping[str, Any]
) -> bool:
    if len(steps) != 4:
        return False
    responses = [_response(step) for step in steps]
    if any(row is None for row in responses):
        return False
    repo, issue, created, fetched = (row.get("body") for row in responses)
    if not all(isinstance(body, dict) for body in (repo, issue, created, fetched)):
        return False
    repo_name = contract.get("repo_name")
    issue_title = contract.get("issue_title")
    comment_body = contract.get("comment_body")
    issue_index = issue.get("number")
    comment_id = created.get("id")
    if not all(
        isinstance(value, str) for value in (repo_name, issue_title, comment_body)
    ):
        return False
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (issue_index, comment_id)
    ):
        return False
    prefix = f"/api/v1/repos/corem_eval/{urllib.parse.quote(repo_name, safe='')}"
    requests = [
        step.get("request") if isinstance(step, dict) else None for step in steps
    ]
    if any(not isinstance(request, dict) for request in requests):
        return False
    issue_request_json = requests[1].get("json")
    comment_request_json = requests[2].get("json")
    if not isinstance(issue_request_json, dict) or not isinstance(
        comment_request_json, dict
    ):
        return False
    return bool(
        [row.get("status") for row in responses] == [201, 201, 201, 200]
        and repo.get("name") == bindings.get("repo") == repo_name
        and repo.get("private") is True
        and issue.get("title") == issue_title
        and issue_index == bindings.get("issue_index")
        and created.get("body") == fetched.get("body") == comment_body
        and comment_id == fetched.get("id") == bindings.get("comment_id")
        and requests[0].get("method") == "POST"
        and requests[1].get("method") == "POST"
        and requests[2].get("method") == "POST"
        and requests[3].get("method") == "GET"
        and issue_request_json.get("title") == issue_title
        and comment_request_json.get("body") == comment_body
        and _request_path(steps[1]) == f"{prefix}/issues"
        and _request_path(steps[2]) == f"{prefix}/issues/{issue_index}/comments"
        and _request_path(steps[3]) == f"{prefix}/issues/comments/{comment_id}"
    )


def _wiki_goal(
    contract: Mapping[str, Any], steps: list[Any], bindings: Mapping[str, Any]
) -> bool:
    if len(steps) != 3:
        return False
    responses = [_response(step) for step in steps]
    if any(row is None for row in responses):
        return False
    repo, created, fetched = (row.get("body") for row in responses)
    if not all(isinstance(body, dict) for body in (repo, created, fetched)):
        return False
    repo_name = contract.get("repo_name")
    title = contract.get("wiki_title")
    content = contract.get("content_base64")
    if not all(isinstance(value, str) for value in (repo_name, title, content)):
        return False
    sub_url = created.get("sub_url")
    if not isinstance(sub_url, str) or not sub_url:
        return False
    requests = [
        step.get("request") if isinstance(step, dict) else None for step in steps
    ]
    if any(not isinstance(request, dict) for request in requests):
        return False
    created_json = requests[1].get("json")
    if not isinstance(created_json, dict):
        return False
    prefix = f"/api/v1/repos/corem_eval/{urllib.parse.quote(repo_name, safe='')}"
    return bool(
        [row.get("status") for row in responses] == [201, 201, 200]
        and repo.get("name") == bindings.get("repo") == repo_name
        and repo.get("private") is True
        and created.get("title") == fetched.get("title") == title
        and bindings.get("wiki_sub_url") == sub_url
        and created.get("content_base64") == fetched.get("content_base64") == content
        and created_json.get("title") == title
        and created_json.get("content_base64") == content
        and requests[1].get("method") == "POST"
        and requests[2].get("method") == "GET"
        and _request_path(steps[1]) == f"{prefix}/wiki/new"
        and _request_path(steps[2])
        == f"{prefix}/wiki/page/{urllib.parse.quote(sub_url, safe='')}"
    )


def _project_goal(
    contract: Mapping[str, Any], steps: list[Any], bindings: Mapping[str, Any]
) -> bool:
    if len(steps) != 3:
        return False
    responses = [_response(step) for step in steps]
    if any(row is None for row in responses):
        return False
    repo, created, fetched = (row.get("body") for row in responses)
    if not all(isinstance(body, dict) for body in (repo, created, fetched)):
        return False
    repo_name = contract.get("repo_name")
    title = contract.get("project_title")
    project_id = created.get("id")
    if not isinstance(repo_name, str) or not isinstance(title, str):
        return False
    if not isinstance(project_id, int) or isinstance(project_id, bool):
        return False
    requests = [
        step.get("request") if isinstance(step, dict) else None for step in steps
    ]
    if any(not isinstance(request, dict) for request in requests):
        return False
    create_json = requests[1].get("json")
    if not isinstance(create_json, dict):
        return False
    prefix = f"/api/v1/repos/corem_eval/{urllib.parse.quote(repo_name, safe='')}"
    return bool(
        [row.get("status") for row in responses] == [201, 201, 200]
        and repo.get("name") == bindings.get("repo") == repo_name
        and repo.get("private") is True
        and created.get("title") == fetched.get("title") == title
        and fetched.get("id") == bindings.get("project_id") == project_id
        and fetched.get("state") == "open"
        and create_json.get("title") == title
        and requests[1].get("method") == "POST"
        and requests[2].get("method") == "GET"
        and _request_path(steps[1]) == f"{prefix}/projects"
        and _request_path(steps[2]) == f"{prefix}/projects/{project_id}"
    )


class IndependentV7CheckerV4(IndependentV7CheckerV3):
    def check(
        self, event: Mapping[str, Any], execution: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        contract = event.get("workflow_contract")
        if not isinstance(contract, dict):
            raise TypeError("v7 event requires a workflow contract")
        kind = contract.get("kind")
        if kind not in {
            "v7_gitea_issue_comment",
            "v7_gitea_wiki",
            "v7_gitea_project",
        }:
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
        semantic_goal = {
            "v7_gitea_issue_comment": _issue_comment_goal,
            "v7_gitea_wiki": _wiki_goal,
            "v7_gitea_project": _project_goal,
        }[kind](contract, steps, bindings)
        checked["contract_id"] = "independent-v7-checker-v4"
        checked["goal_satisfied"] = bool(
            execution.get("completed") is True
            and all(item.get("passed") is True for item in checked["binding_checks"])
            and semantic_goal
        )
        return checked


def create_checker() -> IndependentV7CheckerV4:
    return IndependentV7CheckerV4()
