"""Independent Meilisearch document-update checker, extending v6 rules."""

from __future__ import annotations

import urllib.parse
from collections.abc import Mapping
from typing import Any

from experiments.independent_v7_checker import _request_path
from experiments.independent_v7_checker_v5 import _bodies, _requests
from experiments.independent_v7_checker_v6 import (
    IndependentV7CheckerV6,
    _runtime_handoffs_ok,
    _valid_task,
)


def _document_update_goal(
    contract: Mapping[str, Any], steps: list[Any], bindings: Mapping[str, Any]
) -> bool:
    if len(steps) != 5:
        return False
    observed = _bodies(steps)
    requests = _requests(steps)
    if observed is None or requests is None:
        return False
    statuses, bodies = observed
    added, add_task, updated, update_task, final = bodies
    if not all(
        isinstance(value, dict)
        for value in (added, add_task, updated, update_task, final)
    ):
        return False
    uid = contract.get("index_uid")
    doc_id = contract.get("document_id")
    title = contract.get("document_title")
    owner = contract.get("owner")
    if (
        not isinstance(uid, str)
        or not isinstance(title, str)
        or not isinstance(owner, str)
    ):
        return False
    if not isinstance(doc_id, int) or isinstance(doc_id, bool):
        return False
    add_uid, update_uid = added.get("taskUid"), updated.get("taskUid")
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (add_uid, update_uid)
    ):
        return False
    initial_payload = requests[0].get("json")
    update_payload = requests[2].get("json")
    if not isinstance(initial_payload, list) or not isinstance(update_payload, list):
        return False
    prefix = f"/indexes/{urllib.parse.quote(uid, safe='')}"
    return bool(
        statuses == [202, 200, 202, 200, 200]
        and added.get("indexUid") == updated.get("indexUid") == uid
        and _valid_task(add_task, uid, add_uid, "documentAdditionOrUpdate")
        and _valid_task(update_task, uid, update_uid, "documentAdditionOrUpdate")
        and bindings.get("task_add") == add_uid
        and bindings.get("index_uid") == uid
        and bindings.get("task_update") == update_uid
        and bindings.get("index_uid_confirm") == uid
        and len(initial_payload) == 1
        and isinstance(initial_payload[0], dict)
        and initial_payload[0].get("id") == doc_id
        and initial_payload[0].get("owner") == owner
        and update_payload == [{"id": doc_id, "title": title}]
        and final.get("id") == doc_id
        and final.get("title") == title
        and final.get("owner") == owner
        and [row.get("method") for row in requests]
        == ["POST", "GET", "PUT", "GET", "GET"]
        and _request_path(steps[0]) == f"{prefix}/documents?primaryKey=id"
        and _request_path(steps[1]) == f"/tasks/{add_uid}"
        and _request_path(steps[2]) == f"{prefix}/documents?primaryKey=id"
        and _request_path(steps[3]) == f"/tasks/{update_uid}"
        and _request_path(steps[4]) == f"{prefix}/documents/{doc_id}"
    )


def _index_swap_goal(
    contract: Mapping[str, Any], steps: list[Any], bindings: Mapping[str, Any]
) -> bool:
    if len(steps) != 7:
        return False
    observed = _bodies(steps)
    requests = _requests(steps)
    if observed is None or requests is None:
        return False
    statuses, bodies = observed
    first, first_task, second, second_task, swap, swap_task, final = bodies
    if not all(
        isinstance(value, dict)
        for value in (first, first_task, second, second_task, swap, swap_task, final)
    ):
        return False
    uid_a, uid_b = contract.get("index_a"), contract.get("index_b")
    doc_id, title = contract.get("document_id"), contract.get("document_title")
    if not isinstance(uid_a, str) or not isinstance(uid_b, str):
        return False
    if (
        not isinstance(title, str)
        or not isinstance(doc_id, int)
        or isinstance(doc_id, bool)
    ):
        return False
    task_a, task_b, task_swap = (
        first.get("taskUid"),
        second.get("taskUid"),
        swap.get("taskUid"),
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (task_a, task_b, task_swap)
    ):
        return False
    add_docs = requests[2].get("json")
    if not isinstance(add_docs, list) or len(add_docs) != 1:
        return False
    return bool(
        statuses == [202, 200, 202, 200, 202, 200, 200]
        and first.get("indexUid") == uid_a
        and second.get("indexUid") == uid_b
        and _valid_task(first_task, uid_a, task_a, "documentAdditionOrUpdate")
        and _valid_task(second_task, uid_b, task_b, "documentAdditionOrUpdate")
        and swap_task.get("uid") == task_swap
        and swap_task.get("type") == "indexSwap"
        and swap_task.get("status") == "succeeded"
        and swap_task.get("error") is None
        and bindings.get("task_a") == task_a
        and bindings.get("index_a") == uid_a
        and bindings.get("task_b") == task_b
        and bindings.get("index_b") == uid_b
        and bindings.get("task_swap") == task_swap
        and requests[4].get("json") == [{"indexes": [uid_a, uid_b]}]
        and isinstance(add_docs[0], dict)
        and add_docs[0].get("id") == doc_id
        and add_docs[0].get("title") == title
        and final.get("id") == doc_id
        and final.get("title") == title
        and [row.get("method") for row in requests]
        == ["POST", "GET", "POST", "GET", "POST", "GET", "GET"]
        and _request_path(steps[0])
        == f"/indexes/{urllib.parse.quote(uid_a, safe='')}/documents?primaryKey=id"
        and _request_path(steps[1]) == f"/tasks/{task_a}"
        and _request_path(steps[2])
        == f"/indexes/{urllib.parse.quote(uid_b, safe='')}/documents?primaryKey=id"
        and _request_path(steps[3]) == f"/tasks/{task_b}"
        and _request_path(steps[4]) == "/swap-indexes"
        and _request_path(steps[5]) == f"/tasks/{task_swap}"
        and _request_path(steps[6])
        == f"/indexes/{urllib.parse.quote(uid_a, safe='')}/documents/{doc_id}"
    )


def _index_delete_goal(
    contract: Mapping[str, Any], steps: list[Any], bindings: Mapping[str, Any]
) -> bool:
    if len(steps) != 7:
        return False
    observed = _bodies(steps)
    requests = _requests(steps)
    if observed is None or requests is None:
        return False
    statuses, bodies = observed
    (
        created_target,
        target_task,
        created_decoy,
        decoy_task,
        deleted,
        delete_task,
        final,
    ) = bodies
    if not all(
        isinstance(value, dict)
        for value in (
            created_target,
            target_task,
            created_decoy,
            decoy_task,
            deleted,
            delete_task,
            final,
        )
    ):
        return False
    target, decoy = contract.get("index_target"), contract.get("index_decoy")
    if not isinstance(target, str) or not isinstance(decoy, str) or target == decoy:
        return False
    target_uid = created_target.get("taskUid")
    decoy_uid = created_decoy.get("taskUid")
    delete_uid = deleted.get("taskUid")
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (target_uid, decoy_uid, delete_uid)
    ):
        return False
    return bool(
        statuses == [202, 200, 202, 200, 202, 200, 404]
        and _valid_task(target_task, target, target_uid, "indexCreation")
        and _valid_task(decoy_task, decoy, decoy_uid, "indexCreation")
        and _valid_task(delete_task, target, delete_uid, "indexDeletion")
        and bindings.get("task_target") == target_uid
        and bindings.get("index_target") == target
        and bindings.get("task_decoy") == decoy_uid
        and bindings.get("index_decoy") == decoy
        and bindings.get("task_delete") == delete_uid
        and created_target.get("indexUid") == target
        and created_decoy.get("indexUid") == decoy
        and deleted.get("indexUid") == target
        and requests[0].get("json") == {"uid": target, "primaryKey": "id"}
        and requests[2].get("json") == {"uid": decoy, "primaryKey": "id"}
        and final.get("code") == "index_not_found"
        and [row.get("method") for row in requests]
        == ["POST", "GET", "POST", "GET", "DELETE", "GET", "GET"]
        and _request_path(steps[0]) == "/indexes"
        and _request_path(steps[1]) == f"/tasks/{target_uid}"
        and _request_path(steps[2]) == "/indexes"
        and _request_path(steps[3]) == f"/tasks/{decoy_uid}"
        and _request_path(steps[4]) == f"/indexes/{urllib.parse.quote(target, safe='')}"
        and _request_path(steps[5]) == f"/tasks/{delete_uid}"
        and _request_path(steps[6]) == f"/indexes/{urllib.parse.quote(target, safe='')}"
    )


class IndependentV7CheckerV7(IndependentV7CheckerV6):
    def check(
        self, event: Mapping[str, Any], execution: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        contract = event.get("workflow_contract")
        if not isinstance(contract, dict):
            raise TypeError("v7 event requires a workflow contract")
        goals = {
            "v7_meili_document_update": _document_update_goal,
            "v7_meili_index_swap": _index_swap_goal,
            "v7_meili_index_delete": _index_delete_goal,
        }
        kind = contract.get("kind")
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
        checked["contract_id"] = "independent-v7-checker-v7"
        checked["goal_satisfied"] = bool(
            execution.get("completed") is True
            and all(item.get("passed") is True for item in checked["binding_checks"])
            and goals[kind](contract, steps, bindings)
        )
        return checked


def create_checker() -> IndependentV7CheckerV7:
    return IndependentV7CheckerV7()
