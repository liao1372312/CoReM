"""Unscored, source-derived v7 workflow specifications for candidate staging."""

from __future__ import annotations

import base64
from typing import Any

GITEA_EXTRA = ("gitea_file", "gitea_hook", "gitea_topic")
MEILI_EXTRA = ("meili_deletion", "meili_multi_search", "meili_facet")


def _capture(name: str, path: list[str | int]) -> dict[str, Any]:
    return {"name": name, "source": "body_path", "path": path}


def _handoff(name: str, target_step: int) -> dict[str, Any]:
    return {"name": name, "target_step": target_step}


def extra_spec(family: str, incident: int, positive: bool) -> dict[str, Any]:
    """Return a complete positive or plausible complete negative workflow."""
    suffix = f"{incident:02d}"
    ambiguity = incident >= 5
    if family in GITEA_EXTRA:
        repo = f"v7stage-{family.removeprefix('gitea_')}-{suffix}"
        prefix = "/api/v1/repos/corem_eval/{repo}"
        plans = [
            {
                "method": "POST",
                "path": "/api/v1/user/repos",
                "json": {"name": repo, "private": True, "auto_init": True},
                "capture": _capture("repo", ["name"]),
            }
        ]
        descriptions = [
            f"Create private repository {repo} and capture its returned name."
        ]
        docs = ["https://docs.gitea.com/api/operations/create-current-user-repo/"]
        handoffs = [_handoff("repo", 2)]
        if family == "gitea_file":
            path = f"release-{suffix}.txt"
            expected = f"Release file approved {suffix}"
            chosen = (
                expected
                if positive
                else (f"{expected} pending" if ambiguity else f"Draft file {suffix}")
            )
            plans += [
                {
                    "method": "POST",
                    "path": f"{prefix}/contents/{path}",
                    "json": {
                        "content": base64.b64encode(chosen.encode()).decode(),
                        "message": "Add release file",
                    },
                    "capture": _capture("file_path", ["content", "path"]),
                },
                {"method": "GET", "path": f"{prefix}/contents/{{file_path}}"},
            ]
            descriptions += [
                f"Write {path} containing '{chosen}' and capture its returned path.",
                "Read the file by returned repository and file path.",
                f"Verify the file content is '{chosen}'.",
            ]
            docs += [
                "https://docs.gitea.com/api/operations/repo-create-file/",
                "https://docs.gitea.com/api/operations/repo-get-contents/",
            ]
            handoffs += [_handoff("file_path", 3)]
            contract = {
                "file_path": path,
                "file_text": expected,
                "statuses": [201, 201, 200],
            }
            query = (
                f"Create {repo}, write '{expected}' to {path}, and read the file back."
            )
        elif family == "gitea_hook":
            expected = "push"
            chosen = expected if positive else "issues"
            plans += [
                {
                    "method": "POST",
                    "path": f"{prefix}/hooks",
                    "json": {
                        "type": "gitea",
                        "active": False,
                        "events": [chosen],
                        "config": {
                            "url": "https://example.invalid/corem-eval-hook",
                            "content_type": "json",
                        },
                    },
                    "capture": _capture("hook_id", ["id"]),
                },
                {"method": "GET", "path": f"{prefix}/hooks/{{hook_id}}"},
            ]
            descriptions += [
                f"Create an inactive {chosen} webhook and capture its returned ID.",
                "Read the inactive webhook by returned repository and hook ID.",
                f"Verify its event is {chosen} and it is inactive.",
            ]
            docs += [
                "https://docs.gitea.com/api/operations/repo-create-hook/",
                "https://docs.gitea.com/api/operations/repo-get-hook/",
            ]
            handoffs += [_handoff("hook_id", 3)]
            contract = {"hook_event": expected, "statuses": [201, 201, 200]}
            query = (
                f"Create {repo}, register an inactive push webhook, and read it back."
            )
        else:
            expected = f"release-review-{suffix}"
            chosen = (
                expected
                if positive
                else (f"{expected}-hold" if ambiguity else f"draft-{suffix}")
            )
            plans += [
                {
                    "method": "PUT",
                    "path": f"{prefix}/topics",
                    "json": {"topics": [chosen]},
                },
                {
                    "method": "GET",
                    "path": f"{prefix}/topics",
                    "capture": _capture("topic", ["topics", 0]),
                },
                {"method": "DELETE", "path": f"{prefix}/topics/{{topic}}"},
                {"method": "GET", "path": f"{prefix}/topics"},
            ]
            descriptions += [
                f"Set topic {chosen} on the repository.",
                "Read the topic list and capture its first returned topic.",
                "Delete the returned topic, not a guessed topic name.",
                "Read the final topic list and verify it is empty.",
                f"Verify the captured topic was {chosen} before deletion.",
            ]
            docs += [
                "https://docs.gitea.com/api/operations/repo-update-topics/",
                "https://docs.gitea.com/api/operations/repo-list-topics/",
                "https://docs.gitea.com/api/operations/repo-delete-topic/",
                "https://docs.gitea.com/api/operations/repo-list-topics/",
            ]
            handoffs += [_handoff("topic", 4)]
            contract = {"topic": expected, "statuses": [201, 204, 200, 204, 200]}
            query = f"Create {repo}, set topic {expected}, read and delete that returned topic, then verify none remain."
        return {
            "plans": plans,
            "descriptions": descriptions,
            "docs": docs + [docs[-1]],
            "contract": {
                "kind": f"v7_{family}",
                "repo_name": repo,
                "handoffs": handoffs,
                **contract,
            },
            "query": query,
        }

    if family not in MEILI_EXTRA:
        raise ValueError(f"unknown extra v7 family: {family}")
    uid = f"v7stage_{family.removeprefix('meili_')}_{suffix}"
    task_doc = (
        "https://www.meilisearch.com/docs/reference/api/async-task-management/get-task"
    )
    add_doc = "https://www.meilisearch.com/docs/reference/api/documents/add-or-replace-documents"
    if family == "meili_deletion":
        keep_id, delete_id = 11, 22
        expected = f"Release handbook {suffix}"
        chosen = (
            expected
            if positive
            else (f"{expected} pending" if ambiguity else f"Draft handbook {suffix}")
        )
        plans = [
            {
                "method": "POST",
                "path": f"/indexes/{uid}/documents?primaryKey=id",
                "json": [
                    {"id": keep_id, "title": chosen},
                    {"id": delete_id, "title": f"Draft notice {suffix}"},
                ],
                "capture": _capture("task_add", ["taskUid"]),
            },
            {
                "method": "GET",
                "path": "/tasks/{task_add}",
                "capture": _capture("index_uid", ["indexUid"]),
            },
            {
                "method": "DELETE",
                "path": "/indexes/{index_uid}/documents/22",
                "capture": _capture("task_delete", ["taskUid"]),
            },
            {
                "method": "GET",
                "path": "/tasks/{task_delete}",
                "capture": _capture("index_uid_confirm", ["indexUid"]),
            },
            {"method": "GET", "path": "/indexes/{index_uid_confirm}/documents/11"},
        ]
        descriptions = [
            f"Add document 11 titled '{chosen}' and draft document 22 to {uid}.",
            "Wait for addition and capture its returned index UID.",
            "Delete draft document 22 from that returned index and capture task ID.",
            "Wait for deletion and capture its confirmed index UID.",
            "Read surviving document 11 from the confirmed index UID.",
            f"Verify document 11 retains title '{chosen}'.",
        ]
        docs = [
            add_doc,
            task_doc,
            "https://www.meilisearch.com/docs/reference/api/documents/delete-one-document",
            task_doc,
            "https://www.meilisearch.com/docs/reference/api/documents/get-document",
        ]
        contract = {
            "keep_id": keep_id,
            "delete_id": delete_id,
            "keep_title": expected,
            "index_uid": uid,
        }
        handoffs = [
            _handoff("task_add", 2),
            _handoff("index_uid", 3),
            _handoff("task_delete", 4),
            _handoff("index_uid_confirm", 5),
        ]
        query = f"In {uid}, add release handbook '{expected}' and a draft, delete the draft, then read the survivor."
    elif family == "meili_multi_search":
        uid_b = f"{uid}_b"
        term_a, term_b = f"handbook{suffix}", f"invoice{suffix}"
        chosen_b = (
            term_b
            if positive
            else (f"{term_b}pending" if ambiguity else f"receipt{suffix}")
        )
        plans = [
            {
                "method": "POST",
                "path": f"/indexes/{uid}/documents?primaryKey=id",
                "json": [{"id": 11, "title": f"Release {term_a}"}],
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
                "json": [{"id": 22, "title": f"Payment {term_b}"}],
                "capture": _capture("task_b", ["taskUid"]),
            },
            {
                "method": "GET",
                "path": "/tasks/{task_b}",
                "capture": _capture("index_b", ["indexUid"]),
            },
            {
                "method": "POST",
                "path": "/multi-search",
                "json": {
                    "queries": [
                        {"indexUid": "{index_a}", "q": term_a},
                        {"indexUid": "{index_b}", "q": chosen_b},
                    ]
                },
            },
        ]
        descriptions = [
            f"Add release {term_a} document 11 to {uid}; capture task ID.",
            "Wait for the first task and capture its returned index UID.",
            f"Add payment {term_b} document 22 to {uid_b}; capture task ID.",
            "Wait for the second task and capture its returned index UID.",
            f"Multi-search returned index UIDs for {term_a} and {chosen_b}.",
            f"Verify both search results contain documents 11 and 22 for {term_a} and {chosen_b}.",
        ]
        docs = [
            add_doc,
            task_doc,
            add_doc,
            task_doc,
            "https://www.meilisearch.com/docs/reference/api/multi_search/multi-search",
        ]
        contract = {
            "index_a": uid,
            "index_b": uid_b,
            "term_a": term_a,
            "term_b": term_b,
            "document_a": 11,
            "document_b": 22,
        }
        handoffs = [
            _handoff("task_a", 2),
            _handoff("index_a", 5),
            _handoff("task_b", 4),
            _handoff("index_b", 5),
        ]
        query = f"Index release {term_a} and payment {term_b} separately, wait for both tasks, and multi-search both returned indexes."
    else:
        expected = f"urgent{suffix}"
        chosen = (
            expected
            if positive
            else (f"{expected}pending" if ambiguity else f"routine{suffix}")
        )
        plans = [
            {
                "method": "POST",
                "path": f"/indexes/{uid}/documents?primaryKey=id",
                "json": [
                    {"id": 11, "title": f"Release report {suffix}", "category": chosen}
                ],
                "capture": _capture("task_add", ["taskUid"]),
            },
            {
                "method": "GET",
                "path": "/tasks/{task_add}",
                "capture": _capture("index_uid", ["indexUid"]),
            },
            {
                "method": "PUT",
                "path": "/indexes/{index_uid}/settings/filterable-attributes",
                "json": ["category"],
                "capture": _capture("task_settings", ["taskUid"]),
            },
            {
                "method": "GET",
                "path": "/tasks/{task_settings}",
                "capture": _capture("index_uid_confirm", ["indexUid"]),
            },
            {
                "method": "POST",
                "path": "/indexes/{index_uid_confirm}/search",
                "json": {"q": "report", "facets": ["category"]},
            },
        ]
        descriptions = [
            f"Add release report document 11 with category {chosen} to {uid}.",
            "Wait for addition and capture the returned index UID.",
            "Make category filterable on that returned index and capture task ID.",
            "Wait for the settings task and capture confirmed index UID.",
            "Search for report with category facet on the confirmed index.",
            f"Verify category {chosen} occurs once with document 11.",
        ]
        docs = [
            add_doc,
            task_doc,
            "https://www.meilisearch.com/docs/reference/api/settings#filterable-attributes",
            task_doc,
            "https://www.meilisearch.com/docs/reference/api/search/search-with-post",
        ]
        contract = {
            "index_uid": uid,
            "category": expected,
            "search_term": "report",
            "document_id": 11,
        }
        handoffs = [
            _handoff("task_add", 2),
            _handoff("index_uid", 3),
            _handoff("task_settings", 4),
            _handoff("index_uid_confirm", 5),
        ]
        query = f"Index a release report with category {expected} in {uid}, make category filterable, and verify its facet count."
    return {
        "plans": plans,
        "descriptions": descriptions,
        "docs": docs + [docs[-1]],
        "contract": {
            "kind": f"v7_{family}",
            "statuses": [202, 200, 202, 200, 200],
            "handoffs": handoffs,
            **contract,
        },
        "query": query,
    }
