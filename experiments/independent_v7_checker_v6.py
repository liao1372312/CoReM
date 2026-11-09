"""Independent checker rules for Meilisearch deletion, multi-search, and facets."""

from __future__ import annotations

import urllib.parse
from collections.abc import Mapping
from typing import Any

from experiments.independent_v7_checker import _request_path
from experiments.independent_v7_checker_v5 import (
    IndependentV7CheckerV5,
    _bodies,
    _requests,
)


def _valid_task(task: Any, uid: str, task_uid: Any, task_type: str) -> bool:
    return bool(
        isinstance(task, dict)
        and isinstance(task_uid, int)
        and not isinstance(task_uid, bool)
        and task.get("uid") == task_uid
        and task.get("indexUid") == uid
        and task.get("status") == "succeeded"
        and task.get("type") == task_type
        and task.get("error") is None
    )


def _contains_scalar(value: Any, target: str) -> bool:
    if isinstance(value, dict):
        return any(_contains_scalar(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_contains_scalar(item, target) for item in value)
    return isinstance(value, (str, int, float)) and str(value) == target


def _runtime_handoffs_ok(
    contract: Mapping[str, Any], steps: list[Any], bindings: Mapping[str, Any]
) -> bool:
    handoffs = contract.get("handoffs")
    if not isinstance(handoffs, list) or len(handoffs) < 2:
        return False
    for handoff in handoffs:
        if not isinstance(handoff, dict):
            return False
        name, target = handoff.get("name"), handoff.get("target_step")
        if (
            not isinstance(name, str)
            or not isinstance(target, int)
            or not 1 <= target <= len(steps)
        ):
            return False
        value = bindings.get(name)
        step = steps[target - 1]
        request = step.get("request") if isinstance(step, dict) else None
        if value is None or not isinstance(request, dict):
            return False
        path = request.get("path")
        target_value = str(value)
        if not (
            isinstance(path, str)
            and urllib.parse.quote(target_value, safe="") in path
            or _contains_scalar(request.get("json"), target_value)
        ):
            return False
    return True


def _deletion_goal(
    contract: Mapping[str, Any], steps: list[Any], bindings: Mapping[str, Any]
) -> bool:
    if len(steps) != 5:
        return False
    observed = _bodies(steps)
    requests = _requests(steps)
    if observed is None or requests is None:
        return False
    statuses, bodies = observed
    add, add_task, delete, delete_task, final = bodies
    if not all(
        isinstance(item, dict) for item in (add, add_task, delete, delete_task, final)
    ):
        return False
    uid = contract.get("index_uid")
    keep_id = contract.get("keep_id")
    delete_id = contract.get("delete_id")
    keep_title = contract.get("keep_title")
    if not isinstance(uid, str) or not isinstance(keep_title, str):
        return False
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (keep_id, delete_id)
    ):
        return False
    add_uid = add.get("taskUid")
    delete_uid = delete.get("taskUid")
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (add_uid, delete_uid)
    ):
        return False
    add_docs = requests[0].get("json")
    if not isinstance(add_docs, list) or len(add_docs) != 2:
        return False
    document_ids = {row.get("id") for row in add_docs if isinstance(row, dict)}
    prefix = f"/indexes/{urllib.parse.quote(uid, safe='')}"
    return bool(
        statuses == [202, 200, 202, 200, 200]
        and add.get("indexUid") == delete.get("indexUid") == uid
        and _valid_task(add_task, uid, add_uid, "documentAdditionOrUpdate")
        and _valid_task(delete_task, uid, delete_uid, "documentDeletion")
        and bindings.get("task_add") == add_uid
        and bindings.get("index_uid") == uid
        and bindings.get("task_delete") == delete_uid
        and bindings.get("index_uid_confirm") == uid
        and document_ids == {keep_id, delete_id}
        and final.get("id") == keep_id
        and final.get("title") == keep_title
        and [row.get("method") for row in requests]
        == ["POST", "GET", "DELETE", "GET", "GET"]
        and _request_path(steps[0]) == f"{prefix}/documents?primaryKey=id"
        and _request_path(steps[1]) == f"/tasks/{add_uid}"
        and _request_path(steps[2]) == f"{prefix}/documents/{delete_id}"
        and _request_path(steps[3]) == f"/tasks/{delete_uid}"
        and _request_path(steps[4]) == f"{prefix}/documents/{keep_id}"
    )


def _multi_goal(
    contract: Mapping[str, Any], steps: list[Any], bindings: Mapping[str, Any]
) -> bool:
    if len(steps) != 5:
        return False
    observed = _bodies(steps)
    requests = _requests(steps)
    if observed is None or requests is None:
        return False
    statuses, bodies = observed
    first, first_task, second, second_task, final = bodies
    if not all(
        isinstance(item, dict)
        for item in (first, first_task, second, second_task, final)
    ):
        return False
    uid_a, uid_b = contract.get("index_a"), contract.get("index_b")
    term_a, term_b = contract.get("term_a"), contract.get("term_b")
    doc_a, doc_b = contract.get("document_a"), contract.get("document_b")
    if not all(isinstance(value, str) for value in (uid_a, uid_b, term_a, term_b)):
        return False
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (doc_a, doc_b)
    ):
        return False
    task_a, task_b = first.get("taskUid"), second.get("taskUid")
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (task_a, task_b)
    ):
        return False
    results = final.get("results")
    query_json = requests[4].get("json")
    if (
        not isinstance(results, list)
        or len(results) != 2
        or not isinstance(query_json, dict)
    ):
        return False
    if any(not isinstance(row, dict) for row in results):
        return False
    if any(not isinstance(row.get("hits"), list) for row in results):
        return False
    expected_queries = [
        {"indexUid": uid_a, "q": term_a},
        {"indexUid": uid_b, "q": term_b},
    ]
    return bool(
        statuses == [202, 200, 202, 200, 200]
        and first.get("indexUid") == uid_a
        and second.get("indexUid") == uid_b
        and _valid_task(first_task, uid_a, task_a, "documentAdditionOrUpdate")
        and _valid_task(second_task, uid_b, task_b, "documentAdditionOrUpdate")
        and bindings.get("task_a") == task_a
        and bindings.get("index_a") == uid_a
        and bindings.get("task_b") == task_b
        and bindings.get("index_b") == uid_b
        and query_json.get("queries") == expected_queries
        and results[0].get("indexUid") == uid_a
        and results[1].get("indexUid") == uid_b
        and any(
            hit.get("id") == doc_a
            for hit in results[0].get("hits", [])
            if isinstance(hit, dict)
        )
        and any(
            hit.get("id") == doc_b
            for hit in results[1].get("hits", [])
            if isinstance(hit, dict)
        )
        and [row.get("method") for row in requests]
        == ["POST", "GET", "POST", "GET", "POST"]
        and _request_path(steps[0])
        == f"/indexes/{urllib.parse.quote(uid_a, safe='')}/documents?primaryKey=id"
        and _request_path(steps[1]) == f"/tasks/{task_a}"
        and _request_path(steps[2])
        == f"/indexes/{urllib.parse.quote(uid_b, safe='')}/documents?primaryKey=id"
        and _request_path(steps[3]) == f"/tasks/{task_b}"
        and _request_path(steps[4]) == "/multi-search"
    )


def _facet_goal(
    contract: Mapping[str, Any], steps: list[Any], bindings: Mapping[str, Any]
) -> bool:
    if len(steps) != 5:
        return False
    observed = _bodies(steps)
    requests = _requests(steps)
    if observed is None or requests is None:
        return False
    statuses, bodies = observed
    add, add_task, setting, setting_task, final = bodies
    if not all(
        isinstance(item, dict) for item in (add, add_task, setting, setting_task, final)
    ):
        return False
    uid, category, term = (
        contract.get("index_uid"),
        contract.get("category"),
        contract.get("search_term"),
    )
    doc_id = contract.get("document_id")
    if not all(isinstance(value, str) for value in (uid, category, term)):
        return False
    if not isinstance(doc_id, int) or isinstance(doc_id, bool):
        return False
    add_uid, settings_uid = add.get("taskUid"), setting.get("taskUid")
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (add_uid, settings_uid)
    ):
        return False
    docs = requests[0].get("json")
    final_json = requests[4].get("json")
    if not isinstance(docs, list) or not isinstance(final_json, dict):
        return False
    distribution = final.get("facetDistribution")
    if not isinstance(distribution, dict) or not isinstance(
        distribution.get("category"), dict
    ):
        return False
    if not isinstance(final.get("hits"), list):
        return False
    prefix = f"/indexes/{urllib.parse.quote(uid, safe='')}"
    return bool(
        statuses == [202, 200, 202, 200, 200]
        and add.get("indexUid") == setting.get("indexUid") == uid
        and _valid_task(add_task, uid, add_uid, "documentAdditionOrUpdate")
        and _valid_task(setting_task, uid, settings_uid, "settingsUpdate")
        and bindings.get("task_add") == add_uid
        and bindings.get("index_uid") == uid
        and bindings.get("task_settings") == settings_uid
        and bindings.get("index_uid_confirm") == uid
        and any(row.get("id") == doc_id for row in docs if isinstance(row, dict))
        and requests[2].get("json") == ["category"]
        and final_json == {"q": term, "facets": ["category"]}
        and distribution["category"].get(category) == 1
        and any(
            hit.get("id") == doc_id
            for hit in final.get("hits", [])
            if isinstance(hit, dict)
        )
        and [row.get("method") for row in requests]
        == ["POST", "GET", "PUT", "GET", "POST"]
        and _request_path(steps[0]) == f"{prefix}/documents?primaryKey=id"
        and _request_path(steps[1]) == f"/tasks/{add_uid}"
        and _request_path(steps[2]) == f"{prefix}/settings/filterable-attributes"
        and _request_path(steps[3]) == f"/tasks/{settings_uid}"
        and _request_path(steps[4]) == f"{prefix}/search"
    )


class IndependentV7CheckerV6(IndependentV7CheckerV5):
    def check(
        self, event: Mapping[str, Any], execution: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        contract = event.get("workflow_contract")
        if not isinstance(contract, dict):
            raise TypeError("v7 event requires a workflow contract")
        kind = contract.get("kind")
        goals = {
            "v7_meili_deletion": _deletion_goal,
            "v7_meili_multi_search": _multi_goal,
            "v7_meili_facet": _facet_goal,
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
            elif item.get("atom") == "binding:runtime_handoffs":
                item["passed"] = _runtime_handoffs_ok(contract, steps, bindings)
        for item in checked["communication_checks"]:
            if item.get("criterion") == "communication:workflow_summary":
                communication = execution.get("communication")
                summary = (
                    communication.get("outcome_summary")
                    if isinstance(communication, dict)
                    else None
                )
                item["passed"] = (
                    isinstance(summary, str)
                    and f"{len(expected_order)}-step" in summary
                )
        checked["contract_id"] = "independent-v7-checker-v6"
        checked["goal_satisfied"] = bool(
            execution.get("completed") is True
            and all(item.get("passed") is True for item in checked["binding_checks"])
            and goals[kind](contract, steps, bindings)
        )
        return checked


def create_checker() -> IndependentV7CheckerV6:
    return IndependentV7CheckerV6()
