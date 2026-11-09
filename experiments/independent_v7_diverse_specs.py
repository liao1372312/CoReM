"""Prespecified Meilisearch workflow replacements for settings-only families."""

from __future__ import annotations

from typing import Any

DIVERSE_MEILI = (
    "meili_document_update",
    "meili_index_swap",
    "meili_index_delete",
)
TASK_DOC = (
    "https://www.meilisearch.com/docs/reference/api/async-task-management/get-task"
)
ADD_DOC = (
    "https://www.meilisearch.com/docs/reference/api/documents/add-or-replace-documents"
)
UPDATE_DOC = (
    "https://www.meilisearch.com/docs/reference/api/documents/add-or-update-documents"
)
GET_DOC = "https://www.meilisearch.com/docs/reference/api/documents/get-document"


def _capture(name: str, path: list[str]) -> dict[str, Any]:
    return {"name": name, "source": "body_path", "path": path}


def _handoff(name: str, target_step: int) -> dict[str, Any]:
    return {"name": name, "target_step": target_step}


def diverse_meili_spec(family: str, incident: int, positive: bool) -> dict[str, Any]:
    if family == "meili_index_swap":
        return _index_swap_spec(incident, positive)
    if family == "meili_index_delete":
        return _index_delete_spec(incident, positive)
    if family != "meili_document_update":
        raise ValueError(f"unknown diverse Meilisearch family: {family}")
    suffix = f"{incident:02d}"
    uid = f"v7stage_doc_update_{suffix}"
    expected = f"Release approved {suffix}"
    chosen = (
        expected
        if positive
        else (f"{expected} pending" if incident >= 5 else f"Draft release {suffix}")
    )
    plans = [
        {
            "method": "POST",
            "path": f"/indexes/{uid}/documents?primaryKey=id",
            "json": [{"id": 11, "title": f"Draft release {suffix}", "owner": "corem"}],
            "capture": _capture("task_add", ["taskUid"]),
        },
        {
            "method": "GET",
            "path": "/tasks/{task_add}",
            "capture": _capture("index_uid", ["indexUid"]),
        },
        {
            "method": "PUT",
            "path": "/indexes/{index_uid}/documents?primaryKey=id",
            "json": [{"id": 11, "title": chosen}],
            "capture": _capture("task_update", ["taskUid"]),
        },
        {
            "method": "GET",
            "path": "/tasks/{task_update}",
            "capture": _capture("index_uid_confirm", ["indexUid"]),
        },
        {"method": "GET", "path": "/indexes/{index_uid_confirm}/documents/11"},
    ]
    descriptions = [
        f"Add draft document 11 with owner corem to {uid}; capture the task ID.",
        "Await addition and capture its returned index UID.",
        f"Partially update document 11 title to '{chosen}' in that index; capture task ID.",
        "Await the document update and capture the confirmed index UID.",
        "Read document 11 by the confirmed index UID.",
        f"Verify title '{chosen}' and preserved owner corem.",
    ]
    return {
        "plans": plans,
        "descriptions": descriptions,
        "docs": [ADD_DOC, TASK_DOC, UPDATE_DOC, TASK_DOC, GET_DOC, GET_DOC],
        "contract": {
            "kind": "v7_meili_document_update",
            "index_uid": uid,
            "document_id": 11,
            "document_title": expected,
            "owner": "corem",
            "statuses": [202, 200, 202, 200, 200],
            "handoffs": [
                _handoff("task_add", 2),
                _handoff("index_uid", 3),
                _handoff("task_update", 4),
                _handoff("index_uid_confirm", 5),
            ],
        },
        "query": (
            f"In {uid}, add draft document 11 owned by corem, update its title to "
            f"'{expected}' without removing its owner, then read it back."
        ),
    }


def _index_swap_spec(incident: int, positive: bool) -> dict[str, Any]:
    suffix = f"{incident:02d}"
    uid_a = f"v7stage_swap_live_{suffix}"
    uid_b = f"v7stage_swap_next_{suffix}"
    expected = f"Release candidate {suffix}"
    chosen = (
        expected
        if positive
        else (f"{expected} pending" if incident >= 5 else f"Draft candidate {suffix}")
    )
    plans = [
        {
            "method": "POST",
            "path": f"/indexes/{uid_a}/documents?primaryKey=id",
            "json": [{"id": 11, "title": f"Old release {suffix}"}],
            "capture": _capture("task_a", ["taskUid"]),
        },
        {
            "method": "GET",
            "path": "/tasks/{task_a}",
            "capture": _capture("index_a", ["indexUid"]),
        },
        {
            "method": "POST",
            "path": f"/indexes/{uid_b}/documents?primaryKey=id",
            "json": [{"id": 22, "title": chosen}],
            "capture": _capture("task_b", ["taskUid"]),
        },
        {
            "method": "GET",
            "path": "/tasks/{task_b}",
            "capture": _capture("index_b", ["indexUid"]),
        },
        {
            "method": "POST",
            "path": "/swap-indexes",
            "json": [{"indexes": ["{index_a}", "{index_b}"]}],
            "capture": _capture("task_swap", ["taskUid"]),
        },
        {"method": "GET", "path": "/tasks/{task_swap}"},
        {"method": "GET", "path": "/indexes/{index_a}/documents/22"},
    ]
    descriptions = [
        f"Add old document 11 to live index {uid_a}; capture task ID.",
        "Wait for the live-index addition; capture returned index UID.",
        f"Add candidate document 22 titled '{chosen}' to {uid_b}; capture task ID.",
        "Wait for the candidate-index addition; capture returned index UID.",
        "Swap the two returned index UIDs and capture the swap task ID.",
        "Wait for the swap task to succeed.",
        "Read document 22 through the original live index UID.",
        f"Verify document 22 in the live index is '{chosen}'.",
    ]
    swap_doc = "https://www.meilisearch.com/docs/reference/api/indexes#swap-indexes"
    return {
        "plans": plans,
        "descriptions": descriptions,
        "docs": [
            ADD_DOC,
            TASK_DOC,
            ADD_DOC,
            TASK_DOC,
            swap_doc,
            TASK_DOC,
            GET_DOC,
            GET_DOC,
        ],
        "contract": {
            "kind": "v7_meili_index_swap",
            "index_a": uid_a,
            "index_b": uid_b,
            "document_id": 22,
            "document_title": expected,
            "statuses": [202, 200, 202, 200, 202, 200, 200],
            "handoffs": [
                _handoff("task_a", 2),
                _handoff("index_a", 5),
                _handoff("task_b", 4),
                _handoff("index_b", 5),
                _handoff("task_swap", 6),
                _handoff("index_a", 7),
            ],
        },
        "query": (
            f"Prepare live index {uid_a} and candidate index {uid_b}, swap them "
            f"after both indexing tasks, then read '{expected}' document 22 "
            "through the original live UID."
        ),
    }


def _index_delete_spec(incident: int, positive: bool) -> dict[str, Any]:
    suffix = f"{incident:02d}"
    target = f"v7stage_delete_target_{suffix}"
    decoy = f"v7stage_delete_decoy_{suffix}"
    deleted = target if positive else decoy
    binding = "index_target" if positive else "index_decoy"
    plans = [
        {
            "method": "POST",
            "path": "/indexes",
            "json": {"uid": target, "primaryKey": "id"},
            "capture": _capture("task_target", ["taskUid"]),
        },
        {
            "method": "GET",
            "path": "/tasks/{task_target}",
            "capture": _capture("index_target", ["indexUid"]),
        },
        {
            "method": "POST",
            "path": "/indexes",
            "json": {"uid": decoy, "primaryKey": "id"},
            "capture": _capture("task_decoy", ["taskUid"]),
        },
        {
            "method": "GET",
            "path": "/tasks/{task_decoy}",
            "capture": _capture("index_decoy", ["indexUid"]),
        },
        {
            "method": "DELETE",
            "path": f"/indexes/{{{binding}}}",
            "capture": _capture("task_delete", ["taskUid"]),
        },
        {"method": "GET", "path": "/tasks/{task_delete}"},
        {"method": "GET", "path": "/indexes/{index_target}"},
    ]
    descriptions = [
        f"Create target index {target} and capture its creation task ID.",
        "Wait for target creation and capture its returned index UID.",
        f"Create decoy index {decoy} and capture its creation task ID.",
        "Wait for decoy creation and capture its returned index UID.",
        f"Delete returned index {deleted} and capture its deletion task ID.",
        "Wait for the index deletion task to succeed.",
        "Read the target index UID after deletion.",
        f"Verify {'target is absent' if positive else 'target remains present'}.",
    ]
    index_doc = "https://www.meilisearch.com/docs/reference/api/indexes"
    return {
        "plans": plans,
        "descriptions": descriptions,
        "docs": [
            index_doc,
            TASK_DOC,
            index_doc,
            TASK_DOC,
            index_doc,
            TASK_DOC,
            index_doc,
            index_doc,
        ],
        "contract": {
            "kind": "v7_meili_index_delete",
            "index_target": target,
            "index_decoy": decoy,
            "statuses": [202, 200, 202, 200, 202, 200, 404],
            "handoffs": [
                _handoff("task_target", 2),
                _handoff("index_target", 5),
                _handoff("task_decoy", 4),
                _handoff("task_delete", 6),
                _handoff("index_target", 7),
            ],
        },
        "query": (
            f"Create indexes {target} and {decoy}, delete the returned target "
            "UID, await deletion, and verify target is absent."
        ),
    }
