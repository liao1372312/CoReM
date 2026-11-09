"""Versioned checker extension for branch, release, and index-setting workflows."""

from __future__ import annotations

import urllib.parse
from collections.abc import Mapping
from typing import Any

from experiments.independent_v7_checker import _request_path, _response
from experiments.independent_v7_checker_v2 import IndependentV7CheckerV2


def _gitea_branch_goal(
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
    branch_name = contract.get("branch_name")
    if not isinstance(repo_name, str) or not isinstance(branch_name, str):
        return False
    prefix = f"/api/v1/repos/corem_eval/{urllib.parse.quote(repo_name, safe='')}"
    return bool(
        [row.get("status") for row in responses] == [201, 201, 200]
        and repo.get("name") == bindings.get("repo") == repo_name
        and repo.get("private") is True
        and created.get("name")
        == fetched.get("name")
        == bindings.get("branch_name")
        == branch_name
        and _request_path(steps[1]) == f"{prefix}/branches"
        and _request_path(steps[2])
        == f"{prefix}/branches/{urllib.parse.quote(branch_name, safe='')}"
    )


def _meili_stats_goal(
    contract: Mapping[str, Any], steps: list[Any], bindings: Mapping[str, Any]
) -> bool:
    if len(steps) != 3:
        return False
    responses = [_response(step) for step in steps]
    if any(row is None for row in responses):
        return False
    task, task_result, stats = (row.get("body") for row in responses)
    if not all(isinstance(body, dict) for body in (task, task_result, stats)):
        return False
    uid = contract.get("index_uid")
    expected_count = contract.get("document_count")
    task_uid = task.get("taskUid")
    if not isinstance(uid, str) or not isinstance(expected_count, int):
        return False
    if not isinstance(task_uid, int) or isinstance(task_uid, bool):
        return False
    return bool(
        [row.get("status") for row in responses] == [202, 200, 200]
        and task.get("indexUid") == task_result.get("indexUid") == uid
        and task_result.get("uid") == bindings.get("task_uid") == task_uid
        and task_result.get("status") == "succeeded"
        and task_result.get("error") is None
        and bindings.get("index_uid") == uid
        and stats.get("numberOfDocuments") == expected_count
        and stats.get("isIndexing") is False
        and _request_path(steps[1]) == f"/tasks/{task_uid}"
        and _request_path(steps[2])
        == f"/indexes/{urllib.parse.quote(uid, safe='')}/stats"
    )


def _gitea_release_goal(
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
    release_name = contract.get("release_name")
    tag_name = contract.get("tag_name")
    release_id = created.get("id")
    if not all(isinstance(value, str) for value in (repo_name, release_name, tag_name)):
        return False
    if not isinstance(release_id, int) or isinstance(release_id, bool):
        return False
    prefix = f"/api/v1/repos/corem_eval/{urllib.parse.quote(repo_name, safe='')}"
    return bool(
        [row.get("status") for row in responses] == [201, 201, 200]
        and repo.get("name") == bindings.get("repo") == repo_name
        and repo.get("private") is True
        and created.get("name") == fetched.get("name") == release_name
        and created.get("tag_name") == fetched.get("tag_name") == tag_name
        and created.get("draft") is False
        and fetched.get("draft") is False
        and fetched.get("id") == bindings.get("release_id") == release_id
        and _request_path(steps[1]) == f"{prefix}/releases"
        and _request_path(steps[2]) == f"{prefix}/releases/{release_id}"
    )


def _meili_setting_goal(
    contract: Mapping[str, Any], steps: list[Any], bindings: Mapping[str, Any]
) -> bool:
    if len(steps) != 5:
        return False
    responses = [_response(step) for step in steps]
    if any(row is None for row in responses):
        return False
    created, create_task, updated, update_task, final = (
        row.get("body") for row in responses
    )
    if not all(
        isinstance(body, dict) for body in (created, create_task, updated, update_task)
    ):
        return False
    uid = contract.get("index_uid")
    setting_path = {
        "v7_meili_synonyms": "synonyms",
        "v7_meili_stopwords": "stop-words",
        "v7_meili_filterable": "filterable-attributes",
        "v7_meili_sortable": "sortable-attributes",
    }.get(contract.get("kind"))
    if not isinstance(uid, str) or setting_path is None:
        return False
    update_request = steps[2].get("request") if isinstance(steps[2], dict) else None
    if not isinstance(update_request, dict):
        return False
    if setting_path == "synonyms":
        term = contract.get("term")
        synonyms = contract.get("synonyms")
        final_valid = bool(
            isinstance(term, str)
            and isinstance(synonyms, list)
            and all(isinstance(value, str) for value in synonyms)
            and isinstance(final, dict)
            and final.get(term) == synonyms
            and update_request.get("json") == {term: synonyms}
        )
    else:
        expected = contract.get("setting_values")
        final_valid = bool(
            isinstance(expected, list)
            and all(isinstance(value, str) for value in expected)
            and isinstance(final, list)
            and all(isinstance(value, str) for value in final)
            and len(final) == len(expected)
            and set(final) == set(expected)
            and update_request.get("json") == expected
        )
    task_create = created.get("taskUid")
    task_update = updated.get("taskUid")
    if (
        not isinstance(task_create, int)
        or isinstance(task_create, bool)
        or not isinstance(task_update, int)
        or isinstance(task_update, bool)
    ):
        return False
    endpoint = f"/indexes/{urllib.parse.quote(uid, safe='')}/settings/{setting_path}"
    return bool(
        [row.get("status") for row in responses] == [202, 200, 202, 200, 200]
        and created.get("indexUid") == create_task.get("indexUid") == uid
        and updated.get("indexUid") == update_task.get("indexUid") == uid
        and create_task.get("uid") == bindings.get("task_create") == task_create
        and update_task.get("uid") == bindings.get("task_settings") == task_update
        and create_task.get("status") == update_task.get("status") == "succeeded"
        and create_task.get("type") == "indexCreation"
        and update_task.get("type") == "settingsUpdate"
        and create_task.get("error") is None
        and update_task.get("error") is None
        and bindings.get("index_uid") == bindings.get("index_uid_confirm") == uid
        and final_valid
        and update_request.get("method") == "PUT"
        and _request_path(steps[1]) == f"/tasks/{task_create}"
        and _request_path(steps[2]) == endpoint
        and _request_path(steps[3]) == f"/tasks/{task_update}"
        and _request_path(steps[4]) == endpoint
    )


class IndependentV7CheckerV3(IndependentV7CheckerV2):
    def check(
        self, event: Mapping[str, Any], execution: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        contract = event.get("workflow_contract")
        if not isinstance(contract, dict):
            raise TypeError("v7 event requires a workflow contract")
        kind = contract.get("kind")
        if kind not in {
            "v7_gitea_branch",
            "v7_gitea_release",
            "v7_meili_stats",
            "v7_meili_synonyms",
            "v7_meili_stopwords",
            "v7_meili_filterable",
            "v7_meili_sortable",
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
            "v7_gitea_branch": _gitea_branch_goal,
            "v7_gitea_release": _gitea_release_goal,
            "v7_meili_stats": _meili_stats_goal,
            "v7_meili_synonyms": _meili_setting_goal,
            "v7_meili_stopwords": _meili_setting_goal,
            "v7_meili_filterable": _meili_setting_goal,
            "v7_meili_sortable": _meili_setting_goal,
        }[kind](contract, steps, bindings)
        checked["contract_id"] = "independent-v7-checker-v3"
        checked["goal_satisfied"] = bool(
            execution.get("completed") is True
            and all(item.get("passed") is True for item in checked["binding_checks"])
            and semantic_goal
        )
        return checked


def create_checker() -> IndependentV7CheckerV3:
    return IndependentV7CheckerV3()
