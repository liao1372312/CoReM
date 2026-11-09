"""Versioned extension of the v7 checker for additional source-derived families."""

from __future__ import annotations

import urllib.parse
from collections.abc import Mapping
from typing import Any

from experiments.independent_v7_checker import (
    IndependentV7Checker,
    _request_path,
    _response,
)


def _gitea_label_goal(
    contract: Mapping[str, Any], steps: list[Any], bindings: Mapping[str, Any]
) -> bool:
    if len(steps) != 3:
        return False
    first, second, third = (_response(step) for step in steps)
    if first is None or second is None or third is None:
        return False
    repo, created, fetched = (row.get("body") for row in (first, second, third))
    if not all(isinstance(body, dict) for body in (repo, created, fetched)):
        return False
    repo_name = contract.get("repo_name")
    label_name = contract.get("label_name")
    label_id = created.get("id")
    if not isinstance(repo_name, str) or not isinstance(label_name, str):
        return False
    if not isinstance(label_id, int) or isinstance(label_id, bool):
        return False
    prefix = f"/api/v1/repos/corem_eval/{urllib.parse.quote(repo_name, safe='')}"
    return bool(
        [first.get("status"), second.get("status"), third.get("status")]
        == [201, 201, 200]
        and repo.get("name") == bindings.get("repo") == repo_name
        and repo.get("full_name") == f"corem_eval/{repo_name}"
        and repo.get("private") is True
        and created.get("name") == fetched.get("name") == label_name
        and fetched.get("id") == bindings.get("label_id") == label_id
        and _request_path(steps[1]) == f"{prefix}/labels"
        and _request_path(steps[2]) == f"{prefix}/labels/{label_id}"
    )


def _meili_document_goal(
    contract: Mapping[str, Any], steps: list[Any], bindings: Mapping[str, Any]
) -> bool:
    if len(steps) != 3:
        return False
    first, second, third = (_response(step) for step in steps)
    if first is None or second is None or third is None:
        return False
    task, task_result, document = (row.get("body") for row in (first, second, third))
    if not all(isinstance(body, dict) for body in (task, task_result, document)):
        return False
    uid = contract.get("index_uid")
    document_id = contract.get("document_id")
    title = contract.get("document_title")
    task_uid = task.get("taskUid")
    if not isinstance(uid, str) or not isinstance(document_id, int):
        return False
    if not isinstance(title, str) or not isinstance(task_uid, int):
        return False
    return bool(
        [first.get("status"), second.get("status"), third.get("status")]
        == [202, 200, 200]
        and task.get("indexUid") == task_result.get("indexUid") == uid
        and task_result.get("uid") == bindings.get("task_uid") == task_uid
        and task_result.get("status") == "succeeded"
        and task_result.get("error") is None
        and document.get("id") == document_id
        and document.get("title") == title
        and bindings.get("index_uid") == uid
        and _request_path(steps[1]) == f"/tasks/{task_uid}"
        and _request_path(steps[2])
        == f"/indexes/{urllib.parse.quote(uid, safe='')}/documents/{document_id}"
    )


def _gitea_milestone_goal(
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
    title = contract.get("milestone_title")
    state = contract.get("milestone_state")
    milestone_id = created.get("id")
    if not all(isinstance(value, str) for value in (repo_name, title, state)):
        return False
    if not isinstance(milestone_id, int) or isinstance(milestone_id, bool):
        return False
    prefix = f"/api/v1/repos/corem_eval/{urllib.parse.quote(repo_name, safe='')}"
    return bool(
        [row.get("status") for row in responses] == [201, 201, 200]
        and repo.get("name") == bindings.get("repo") == repo_name
        and repo.get("private") is True
        and created.get("title") == fetched.get("title") == title
        and created.get("state") == fetched.get("state") == state
        and fetched.get("id") == bindings.get("milestone_id") == milestone_id
        and _request_path(steps[1]) == f"{prefix}/milestones"
        and _request_path(steps[2]) == f"{prefix}/milestones/{milestone_id}"
    )


def _meili_search_goal(
    contract: Mapping[str, Any], steps: list[Any], bindings: Mapping[str, Any]
) -> bool:
    if len(steps) != 3:
        return False
    responses = [_response(step) for step in steps]
    if any(row is None for row in responses):
        return False
    task, task_result, search_result = (row.get("body") for row in responses)
    if not all(isinstance(body, dict) for body in (task, task_result, search_result)):
        return False
    uid = contract.get("index_uid")
    doc_id = contract.get("document_id")
    title = contract.get("document_title")
    query = contract.get("search_query")
    task_uid = task.get("taskUid")
    hits = search_result.get("hits")
    if not isinstance(uid, str) or not isinstance(doc_id, int):
        return False
    if not isinstance(title, str) or not isinstance(query, str):
        return False
    if not isinstance(task_uid, int) or not isinstance(hits, list):
        return False
    request = steps[2].get("request") if isinstance(steps[2], dict) else None
    return bool(
        [row.get("status") for row in responses] == [202, 200, 200]
        and task.get("indexUid") == task_result.get("indexUid") == uid
        and task_result.get("uid") == bindings.get("task_uid") == task_uid
        and task_result.get("status") == "succeeded"
        and task_result.get("error") is None
        and bindings.get("index_uid") == uid
        and _request_path(steps[1]) == f"/tasks/{task_uid}"
        and _request_path(steps[2])
        == f"/indexes/{urllib.parse.quote(uid, safe='')}/search"
        and isinstance(request, dict)
        and isinstance(request.get("json"), dict)
        and request["json"].get("q") == query
        and any(
            isinstance(hit, dict)
            and hit.get("id") == doc_id
            and hit.get("title") == title
            for hit in hits
        )
    )


class IndependentV7CheckerV2(IndependentV7Checker):
    def check(
        self, event: Mapping[str, Any], execution: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        contract = event.get("workflow_contract")
        if not isinstance(contract, dict):
            raise TypeError("v7 event requires a workflow contract")
        kind = contract.get("kind")
        if kind not in {
            "v7_gitea_label",
            "v7_gitea_milestone",
            "v7_meili_document",
            "v7_meili_search",
        }:
            return super().check(event, execution)
        checked = dict(self.base.check(event, execution))
        steps = execution.get("steps")
        if not isinstance(steps, list):
            steps = []
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
        bindings = execution.get("bindings")
        if not isinstance(bindings, dict):
            bindings = {}
        semantic_goal = {
            "v7_gitea_label": _gitea_label_goal,
            "v7_gitea_milestone": _gitea_milestone_goal,
            "v7_meili_document": _meili_document_goal,
            "v7_meili_search": _meili_search_goal,
        }[kind](contract, steps, bindings)
        checked["contract_id"] = "independent-v7-checker-v2"
        checked["goal_satisfied"] = bool(
            execution.get("completed") is True
            and all(item.get("passed") is True for item in checked["binding_checks"])
            and semantic_goal
        )
        return checked


def create_checker() -> IndependentV7CheckerV2:
    return IndependentV7CheckerV2()
