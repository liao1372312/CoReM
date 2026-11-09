#!/usr/bin/env python3
"""Build oracle-only staging data for unscored Gitea/Meilisearch families.

The 16-family option is a candidate, not a sealed benchmark. Do not inspect
model rankings before validation and freezing or merge scored development families.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import secrets
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import (  # noqa: E402
    audit_real_dataset,
    parse_utc,
    sha256_file,
    write_checksums,
    write_json,
    write_jsonl,
)
from experiments.independent_v7_extra_specs import (  # noqa: E402
    GITEA_EXTRA,
    MEILI_EXTRA,
    extra_spec,
)
from experiments.independent_v7_diverse_specs import (  # noqa: E402
    DIVERSE_MEILI,
    diverse_meili_spec,
)

GITEA_VERSION = "28.1.0"
MEILI_VERSION = "1.55.0"
GITEA_RELEASE = "https://github.com/go-gitea/gitea/releases/tag/v28.1.0"
MEILI_RELEASE = "https://github.com/meilisearch/meilisearch/releases/tag/v1.55.0"
GITEA_REPO_DOC = "https://docs.gitea.com/api/operations/create-current-user-repo/"
MEILI_TASK_DOC = (
    "https://www.meilisearch.com/docs/reference/api/async-task-management/get-task"
)
GITEA_FAMILIES = (
    "gitea_branch",
    "gitea_release",
    "gitea_issue_comment",
    "gitea_wiki",
    "gitea_project",
)
MEILI_FAMILIES = (
    "meili_stats",
    "meili_synonyms",
    "meili_stopwords",
    "meili_filterable",
    "meili_sortable",
)
FAMILIES = GITEA_FAMILIES + MEILI_FAMILIES
CHECKER = ROOT / "experiments" / "independent_v7_checker_v4.py"
CHECKER_V6 = ROOT / "experiments" / "independent_v7_checker_v6.py"
CHECKER_V7 = ROOT / "experiments" / "independent_v7_checker_v7.py"


def _digest(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    )


def _capture(name: str, path: list[str]) -> dict[str, Any]:
    return {"name": name, "source": "body_path", "path": path}


def _repo_plan(name: str) -> dict[str, Any]:
    return {
        "method": "POST",
        "path": "/api/v1/user/repos",
        "json": {"name": name, "private": True, "auto_init": True},
        "capture": _capture("repo", ["name"]),
    }


def _meili_create(uid: str) -> dict[str, Any]:
    return {
        "method": "POST",
        "path": "/indexes",
        "json": {"uid": uid, "primaryKey": "id"},
        "capture": _capture("task_create", ["taskUid"]),
    }


def _meili_poll(name: str, capture_name: str) -> dict[str, Any]:
    return {
        "method": "GET",
        "path": f"/tasks/{{{name}}}",
        "capture": _capture(capture_name, ["indexUid"]),
    }


def _handoff(name: str, target_step: int) -> dict[str, Any]:
    return {"name": name, "target_step": target_step}


def _gitea_spec(family: str, incident: int, positive: bool) -> dict[str, Any]:
    suffix = f"{incident:02d}"
    repo = f"v7stage-{family.removeprefix('gitea_')}-{suffix}"
    plans = [_repo_plan(repo)]
    descriptions = [f"Create private repository {repo}; capture its returned name."]
    docs = [GITEA_REPO_DOC]
    common = {
        "repo_name": repo,
        "statuses": [201, 201, 200],
        "handoffs": [_handoff("repo", 2)],
    }
    ambiguity = incident >= 5
    if family == "gitea_branch":
        target = f"release-ready-{suffix}"
        chosen = (
            target
            if positive
            else (f"release-ready-{suffix}-hold" if ambiguity else f"draft-{suffix}")
        )
        plans.extend(
            [
                {
                    "method": "POST",
                    "path": "/api/v1/repos/corem_eval/{repo}/branches",
                    "json": {"new_branch_name": chosen, "old_branch_name": "main"},
                    "capture": _capture("branch_name", ["name"]),
                },
                {
                    "method": "GET",
                    "path": "/api/v1/repos/corem_eval/{repo}/branches/{branch_name}",
                },
            ]
        )
        docs += [
            "https://docs.gitea.com/api/operations/repo-create-branch/",
            "https://docs.gitea.com/api/operations/repo-get-branch/",
        ]
        descriptions += [
            f"Create branch {chosen} from main and capture the returned branch name.",
            "Fetch the branch using the returned repository and branch name.",
            f"Verify that the fetched branch is {chosen} in the private repository.",
        ]
        contract = {
            "kind": "v7_gitea_branch",
            "branch_name": target,
            **common,
        }
        contract["handoffs"].append(_handoff("branch_name", 3))
        query = f"Create private repository {repo}, make branch {target} from main, and read it back."
    elif family == "gitea_release":
        target = f"Ready release {suffix}"
        chosen = (
            target
            if positive
            else (f"{target} draft" if ambiguity else f"Draft release {suffix}")
        )
        tag = f"v1.0.{incident}"
        plans.extend(
            [
                {
                    "method": "POST",
                    "path": "/api/v1/repos/corem_eval/{repo}/releases",
                    "json": {
                        "tag_name": tag,
                        "name": chosen,
                        "draft": False,
                        "prerelease": False,
                        "target_commitish": "main",
                    },
                    "capture": _capture("release_id", ["id"]),
                },
                {
                    "method": "GET",
                    "path": "/api/v1/repos/corem_eval/{repo}/releases/{release_id}",
                },
            ]
        )
        docs += [
            "https://docs.gitea.com/api/operations/repo-create-release/",
            "https://docs.gitea.com/api/operations/repo-get-release/",
        ]
        descriptions += [
            f"Publish release {chosen} with tag {tag}; capture its returned ID.",
            "Read the published release by repository and returned release ID.",
            f"Verify release name {chosen}, tag {tag}, and non-draft state.",
        ]
        contract = {
            "kind": "v7_gitea_release",
            "release_name": target,
            "tag_name": tag,
            **common,
        }
        contract["handoffs"].append(_handoff("release_id", 3))
        query = f"Create {repo}, publish non-draft release {target} under tag {tag}, and verify it."
    elif family == "gitea_issue_comment":
        title = f"Release gate {suffix}"
        target = f"Reviewed for deployment {suffix}"
        chosen = (
            target
            if positive
            else (f"{target} pending" if ambiguity else f"Needs review {suffix}")
        )
        plans.extend(
            [
                {
                    "method": "POST",
                    "path": "/api/v1/repos/corem_eval/{repo}/issues",
                    "json": {"title": title},
                    "capture": _capture("issue_index", ["number"]),
                },
                {
                    "method": "POST",
                    "path": "/api/v1/repos/corem_eval/{repo}/issues/{issue_index}/comments",
                    "json": {"body": chosen},
                    "capture": _capture("comment_id", ["id"]),
                },
                {
                    "method": "GET",
                    "path": "/api/v1/repos/corem_eval/{repo}/issues/comments/{comment_id}",
                },
            ]
        )
        docs += [
            "https://docs.gitea.com/api/operations/issue-create-issue/",
            "https://docs.gitea.com/api/operations/issue-create-comment/",
            "https://docs.gitea.com/api/operations/issue-get-comment/",
        ]
        descriptions += [
            f"Create issue titled {title} and capture its issue number.",
            f"Post comment {chosen} to the returned issue; capture comment ID.",
            "Read the comment by its returned ID in the created repository.",
            f"Verify issue {title} received comment {chosen}.",
        ]
        contract = {
            "kind": "v7_gitea_issue_comment",
            "issue_title": title,
            "comment_body": target,
            "repo_name": repo,
            "statuses": [201, 201, 201, 200],
            "handoffs": [
                _handoff("repo", 2),
                _handoff("issue_index", 3),
                _handoff("comment_id", 4),
            ],
        }
        query = f"In private repository {repo}, open issue {title}, add comment '{target}', and read it back."
    elif family == "gitea_wiki":
        title = f"Release-Guide-{suffix}"
        target = f"Ready for release {suffix}"
        chosen = (
            target
            if positive
            else (f"{target} pending" if ambiguity else f"Draft only {suffix}")
        )
        encoded = base64.b64encode(chosen.encode()).decode()
        plans.extend(
            [
                {
                    "method": "POST",
                    "path": "/api/v1/repos/corem_eval/{repo}/wiki/new",
                    "json": {"title": title, "content_base64": encoded},
                    "capture": _capture("wiki_sub_url", ["sub_url"]),
                },
                {
                    "method": "GET",
                    "path": "/api/v1/repos/corem_eval/{repo}/wiki/page/{wiki_sub_url}",
                },
            ]
        )
        docs += [
            "https://docs.gitea.com/api/operations/repo-create-wiki-page/",
            "https://docs.gitea.com/api/operations/repo-get-wiki-page/",
        ]
        descriptions += [
            f"Create wiki page {title} with text '{chosen}'; capture its returned sub-URL.",
            "Read the wiki page by repository and returned sub-URL, not by guessed title.",
            f"Verify page {title} contains '{chosen}'.",
        ]
        contract = {
            "kind": "v7_gitea_wiki",
            "wiki_title": title,
            "content_base64": base64.b64encode(target.encode()).decode(),
            **common,
        }
        contract["handoffs"].append(_handoff("wiki_sub_url", 3))
        query = f"Create {repo}, write wiki page {title} containing '{target}', and read its returned URL."
    elif family == "gitea_project":
        target = f"Release board {suffix}"
        chosen = (
            target
            if positive
            else (f"{target} archive" if ambiguity else f"Draft board {suffix}")
        )
        plans.extend(
            [
                {
                    "method": "POST",
                    "path": "/api/v1/repos/corem_eval/{repo}/projects",
                    "json": {"title": chosen, "template_type": "none"},
                    "capture": _capture("project_id", ["id"]),
                },
                {
                    "method": "GET",
                    "path": "/api/v1/repos/corem_eval/{repo}/projects/{project_id}",
                },
            ]
        )
        docs += [
            "https://docs.gitea.com/api/operations/repo-create-project/",
            "https://docs.gitea.com/api/operations/repo-get-project/",
        ]
        descriptions += [
            f"Create open repository project {chosen}; capture its returned ID.",
            "Fetch the project using repository and returned project ID.",
            f"Verify project title {chosen} and open state.",
        ]
        contract = {"kind": "v7_gitea_project", "project_title": target, **common}
        contract["handoffs"].append(_handoff("project_id", 3))
        query = f"Create {repo}, open project {target}, and read that project by its returned ID."
    else:
        raise ValueError(f"unknown Gitea family: {family}")
    return {
        "plans": plans,
        "descriptions": descriptions,
        "docs": docs + [docs[-1]],
        "contract": contract,
        "query": query,
    }


def _meili_spec(family: str, incident: int, positive: bool) -> dict[str, Any]:
    suffix = f"{incident:02d}"
    uid = f"v7stage_{family.removeprefix('meili_')}_{suffix}"
    ambiguity = incident >= 5
    if family == "meili_stats":
        doc_count = 1 if positive else 2
        documents = [
            {"id": index, "title": f"Release audit {suffix} record {index}"}
            for index in range(1, doc_count + 1)
        ]
        plans = [
            {
                "method": "POST",
                "path": f"/indexes/{uid}/documents?primaryKey=id",
                "json": documents,
                "capture": _capture("task_uid", ["taskUid"]),
            },
            _meili_poll("task_uid", "index_uid"),
            {"method": "GET", "path": "/indexes/{index_uid}/stats"},
        ]
        descriptions = [
            f"Add exactly {doc_count} document(s) to index {uid}; capture the indexing task ID.",
            "Wait for the document task and capture its returned index UID.",
            "Read statistics for that returned index UID.",
            f"Verify the index contains exactly {doc_count} document(s).",
        ]
        docs = [
            "https://www.meilisearch.com/docs/reference/api/documents/add-or-replace-documents",
            MEILI_TASK_DOC,
            "https://www.meilisearch.com/docs/reference/api/indexes/get-stats-of-index",
        ]
        contract = {
            "kind": "v7_meili_stats",
            "index_uid": uid,
            "document_count": 1,
            "statuses": [202, 200, 200],
            "handoffs": [_handoff("task_uid", 2), _handoff("index_uid", 3)],
        }
        query = f"Index exactly one release-audit document in {uid}, wait, and verify its count by statistics."
    else:
        setting_path = {
            "meili_synonyms": "synonyms",
            "meili_stopwords": "stop-words",
            "meili_filterable": "filterable-attributes",
            "meili_sortable": "sortable-attributes",
        }.get(family)
        if setting_path is None:
            raise ValueError(f"unknown Meilisearch family: {family}")
        if family == "meili_synonyms":
            term = f"report{suffix}"
            expected = [f"briefing{suffix}"]
            chosen = (
                expected
                if positive
                else [f"briefing{suffix}pending" if ambiguity else f"summary{suffix}"]
            )
            payload: Any = {term: chosen}
            contract_values = {"term": term, "synonyms": expected}
            expected_text = f"synonym {term} -> {expected[0]}"
            chosen_text = f"synonym {term} -> {chosen[0]}"
        else:
            stem = {
                "meili_stopwords": ("quiet", "silent", "muted"),
                "meili_filterable": ("category", "rating", "internal"),
                "meili_sortable": ("price", "rating", "internal"),
            }[family]
            expected = [f"{stem[0]}{suffix}", f"{stem[1]}{suffix}"]
            alternative = (
                f"{stem[1]}{suffix}pending" if ambiguity else f"{stem[2]}{suffix}"
            )
            chosen = expected if positive else [expected[0], alternative]
            payload = chosen
            contract_values = {"setting_values": expected}
            expected_text = ", ".join(expected)
            chosen_text = ", ".join(chosen)
        plans = [
            _meili_create(uid),
            _meili_poll("task_create", "index_uid"),
            {
                "method": "PUT",
                "path": f"/indexes/{{index_uid}}/settings/{setting_path}",
                "json": payload,
                "capture": _capture("task_settings", ["taskUid"]),
            },
            _meili_poll("task_settings", "index_uid_confirm"),
            {
                "method": "GET",
                "path": f"/indexes/{{index_uid_confirm}}/settings/{setting_path}",
            },
        ]
        descriptions = [
            f"Create index {uid}; capture its creation task UID.",
            "Wait for index creation and capture its returned index UID.",
            f"Set {setting_path} to {chosen_text}; capture the update task UID.",
            "Wait for the settings task and capture the confirmed index UID.",
            f"Read {setting_path} from that confirmed index UID.",
            f"Verify {setting_path} contains {chosen_text}.",
        ]
        docs = [
            "https://www.meilisearch.com/docs/reference/api/indexes/create-index",
            MEILI_TASK_DOC,
            f"https://www.meilisearch.com/docs/reference/api/settings#{setting_path}",
            MEILI_TASK_DOC,
            f"https://www.meilisearch.com/docs/reference/api/settings#{setting_path}",
        ]
        contract = {
            "kind": f"v7_{family}",
            "index_uid": uid,
            **contract_values,
            "statuses": [202, 200, 202, 200, 200],
            "handoffs": [
                _handoff("task_create", 2),
                _handoff("index_uid", 3),
                _handoff("task_settings", 4),
                _handoff("index_uid_confirm", 5),
            ],
        }
        query = f"Create index {uid}, set {setting_path} to {expected_text}, wait for both tasks, and read it back."
    return {
        "plans": plans,
        "descriptions": descriptions,
        "docs": docs + [docs[-1]],
        "contract": contract,
        "query": query,
    }


def build_spec(family: str, incident: int, positive: bool) -> dict[str, Any]:
    if family in GITEA_FAMILIES:
        return _gitea_spec(family, incident, positive)
    if family in MEILI_FAMILIES:
        return _meili_spec(family, incident, positive)
    if family in GITEA_EXTRA + MEILI_EXTRA:
        return extra_spec(family, incident, positive)
    if family in DIVERSE_MEILI:
        return diverse_meili_spec(family, incident, positive)
    raise ValueError(f"unknown staging family: {family}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/public_independent_v7_staging_v1")
    )
    parser.add_argument(
        "--selections",
        type=Path,
        default=Path("experiments/results_independent_v7_staging_v1_selections"),
    )
    parser.add_argument(
        "--collection-end-utc",
        required=True,
        help="Explicit UTC source-collection timestamp, recorded verbatim for deterministic reruns.",
    )
    parser.add_argument(
        "--include-extra-six",
        action="store_true",
        help="Add six unscored operation types to make a 16-family candidate set.",
    )
    parser.add_argument(
        "--replace-settings-with-diverse",
        action="store_true",
        help="Replace three settings-only keys with diverse Meilisearch workflows.",
    )
    parser.add_argument(
        "--opaque-bundle-ids",
        action="store_true",
        help="Randomize positive side and replace bundle_a/b IDs with opaque tokens.",
    )
    parser.add_argument(
        "--assignment-salt",
        help="Reproduce an opaque-ID dataset after its evaluator salt is released.",
    )
    args = parser.parse_args()
    collection_end = parse_utc(args.collection_end_utc, "collection_end_utc")
    if args.dataset.exists() or args.selections.exists():
        parser.error("refusing to overwrite staging data")
    if args.assignment_salt and not args.opaque_bundle_ids:
        parser.error("--assignment-salt requires --opaque-bundle-ids")
    assignment_salt = (
        args.assignment_salt or secrets.token_hex(16)
        if args.opaque_bundle_ids
        else None
    )
    events: list[dict[str, Any]] = []
    memories: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    if args.replace_settings_with_diverse:
        families = (
            GITEA_FAMILIES
            + GITEA_EXTRA
            + ("meili_stats", "meili_synonyms")
            + MEILI_EXTRA
            + DIVERSE_MEILI
        )
    elif args.include_extra_six:
        families = GITEA_FAMILIES + GITEA_EXTRA + MEILI_FAMILIES + MEILI_EXTRA
    else:
        families = FAMILIES
    expanded = args.include_extra_six or args.replace_settings_with_diverse
    gitea_families = GITEA_FAMILIES + GITEA_EXTRA
    for family_index, family in enumerate(families):
        service = "gitea" if family in gitea_families else "meilisearch"
        version = GITEA_VERSION if service == "gitea" else MEILI_VERSION
        valid_from = (
            "2026-10-06T20:14:45Z" if service == "gitea" else "2026-10-12T09:21:50Z"
        )
        for incident in range(1, 9):
            event_id = f"v7-stage:{family}:event-{incident:03d}"
            oracle_spec = build_spec(family, incident, True)
            operation_count = len(oracle_spec["plans"])
            atoms = [f"role:step{step}" for step in range(1, operation_count + 1)] + [
                "role:verify"
            ]
            events.append(
                {
                    "event_id": event_id,
                    "incident_id": event_id,
                    "duplicate_group": None,
                    "provider_id": service,
                    "service_id": service,
                    "version_id": version,
                    "event_time": (
                        f"2026-10-18T{family_index // 2:02d}:"
                        f"{(family_index % 2) * 30 + incident:02d}:00Z"
                        if expanded
                        else f"2026-10-18T{family_index:02d}:{incident:02d}:00Z"
                    ),
                    "query": oracle_spec["query"],
                    "task_key": family,
                    "operation_count": operation_count,
                    "checker_contract_id": (
                        "independent-v7-checker-v7"
                        if family in DIVERSE_MEILI
                        else "independent-v7-checker-v6"
                        if family in MEILI_EXTRA
                        else "independent-v7-checker-v5"
                        if family in GITEA_EXTRA
                        else "independent-v7-checker-v4"
                        if family
                        in {"gitea_issue_comment", "gitea_wiki", "gitea_project"}
                        else "independent-v7-checker-v3"
                    ),
                    "required_binding_atoms": [
                        "binding:step_order",
                        "binding:runtime_handoffs",
                        "binding:status_codes",
                    ],
                    "required_evidence_atoms": atoms,
                    "communication_criteria": [
                        "communication:all_statuses",
                        "communication:workflow_summary",
                    ],
                    "workflow_contract": oracle_spec["contract"],
                    "observation_tier": "T2_static_public_release_controlled_staging",
                    "lexical_stratum": "ambiguous" if incident >= 5 else "easy",
                }
            )
            for bundle in ("bundle_a", "bundle_b"):
                if assignment_salt is not None:
                    positive_side = (
                        hashlib.sha256(
                            f"{assignment_salt}:{event_id}:positive".encode()
                        ).digest()[0]
                        & 1
                    )
                    positive = (bundle == "bundle_a") == (positive_side == 0)
                    opaque = hashlib.sha256(
                        f"{assignment_salt}:{event_id}:{bundle}:id".encode()
                    ).hexdigest()[:20]
                    bundle_id = f"candidate-{opaque}"
                else:
                    positive = (bundle == "bundle_a") == (incident % 2 == 1)
                    bundle_id = bundle
                spec = build_spec(family, incident, positive)
                member_ids = []
                for index, description in enumerate(spec["descriptions"], 1):
                    memory_id = f"v7-stage:{family}:event-{incident:03d}:{bundle_id}:record-{index}"
                    member_ids.append(memory_id)
                    record = {
                        "memory_id": memory_id,
                        "service_id": service,
                        "provider_id": service,
                        "version_scope": [version],
                        "valid_from": valid_from,
                        "valid_until": None,
                        "evidence_atoms": [atoms[index - 1]],
                        "provenance_uri": spec["docs"][index - 1],
                        "provenance_verified": True,
                        "role": "workflow_step"
                        if index <= operation_count
                        else "verification",
                        "step_index": index if index <= operation_count else None,
                        "storage_cost": 48,
                        "token_cost": 32,
                        "text": description,
                    }
                    if index <= operation_count:
                        record["plan"] = spec["plans"][index - 1]
                    record["content_digest"] = _digest(
                        {"text": description, "plan": record.get("plan")}
                    )
                    memories.append(record)
                edge_id = f"v7-stage:{family}:event-{incident:03d}:{bundle_id}"
                edges.append(
                    {
                        "edge_id": edge_id,
                        "members": member_ids,
                        "relation_type": f"{operation_count}-step-runtime-handoff",
                        "evidence_atoms": atoms,
                        "provenance": spec["docs"][0],
                        "verified": True,
                        "version_scope": [version],
                        "valid_from": valid_from,
                        "valid_until": None,
                    }
                )
                labels.append(
                    {
                        "run_id": edge_id,
                        "event_id": event_id,
                        "method": "oracle" if positive else "complete_negative",
                        "seed": 0,
                        "selected_memory_ids": member_ids,
                        "token_cost": 32 * len(member_ids),
                        "latency_ms": 0.0,
                    }
                )
                candidates.append(
                    {
                        "event_id": event_id,
                        "candidate_edge_id": edge_id,
                        "selected_memory_ids": member_ids,
                    }
                )
    if any(
        parse_utc(row["event_time"], "event_time") > collection_end for row in events
    ):
        parser.error("logical event_time must not exceed collection_end_utc")
    args.dataset.mkdir(parents=True)
    for filename, rows in (
        ("events.jsonl", events),
        ("memories.jsonl", memories),
        ("dependency_hyperedges.jsonl", edges),
    ):
        write_jsonl(args.dataset / filename, rows)
    write_json(
        args.dataset / "manifest.json",
        {
            "schema_version": "corem-real-static-v1",
            "dataset_name": f"CoReM v7 {len(families)}-family oracle-only staging",
            "benchmark_mode": "static_public_release",
            "natural_drift_claim": False,
            "real_world_observations": True,
            "event_time_semantics": "Controlled logical order; not observed production traffic",
            "claim_boundary": (
                "Unscored 16-family candidate; not sealed or evaluated for model rankings"
                if expanded
                else "Ten unscored staging families; not a sealed 16-family confirmation benchmark"
            ),
            "assignment_commitment_sha256": (
                hashlib.sha256(assignment_salt.encode()).hexdigest()
                if assignment_salt is not None
                else None
            ),
            "observation_tier": "T2_static_public_release_controlled_staging",
            "provenance": {
                "source_uri": f"{GITEA_RELEASE} ; {MEILI_RELEASE}",
                "license": "MIT (Gitea; Meilisearch Community Edition only)",
                "immutable_source_id": (
                    "gitea:ea93cfa4a300564ac457d3903469fea49ec5a4da37d0cebaa435f8e329ea64d0;"
                    "meilisearch:7e33f5889235f14290074a19a6d3574bd9b10fb2ee8731eabb7dafd0714b0513"
                ),
                "collection_start_utc": "2026-10-06T20:14:45Z",
                "collection_end_utc": args.collection_end_utc,
            },
            "execution": {
                "resettable": True,
                "runner_contract": "independent-v7-runner-v1",
            },
            "checker": {
                "independent_from_runner": True,
                "contract_id": (
                    "independent-v7-checker-v7"
                    if args.replace_settings_with_diverse
                    else "independent-v7-checker-v6"
                    if args.include_extra_six
                    else "independent-v7-checker-v4"
                ),
                "implementation_digest": f"sha256:{sha256_file(CHECKER_V7 if args.replace_settings_with_diverse else CHECKER_V6 if args.include_extra_six else CHECKER)}",
            },
            "files": {
                name: {"records": len(rows), "sha256": sha256_file(args.dataset / name)}
                for name, rows in (
                    ("events.jsonl", events),
                    ("memories.jsonl", memories),
                )
            },
        },
    )
    write_checksums(
        args.dataset,
        (
            "events.jsonl",
            "memories.jsonl",
            "dependency_hyperedges.jsonl",
            "manifest.json",
        ),
    )
    args.selections.mkdir(parents=True)
    if assignment_salt is not None:
        (args.selections / "assignment_salt.txt").write_text(
            assignment_salt + "\n", encoding="utf-8"
        )
    write_jsonl(
        args.selections / "splits.jsonl",
        [{"event_id": row["event_id"], "split": "test"} for row in events],
    )
    write_jsonl(args.selections / "selections.jsonl", labels)
    write_jsonl(args.selections / "candidate_selections.jsonl", candidates)
    write_jsonl(
        args.selections / "query_only_events.jsonl",
        [
            {
                key: row[key]
                for key in (
                    "event_id",
                    "service_id",
                    "provider_id",
                    "version_id",
                    "event_time",
                    "query",
                )
            }
            for row in events
        ],
    )
    selection_files = (
        "splits.jsonl",
        "selections.jsonl",
        "candidate_selections.jsonl",
        "query_only_events.jsonl",
    )
    write_checksums(
        args.selections,
        selection_files
        + (("assignment_salt.txt",) if assignment_salt is not None else ()),
    )
    report = audit_real_dataset(args.dataset)
    if not report.eligible:
        raise RuntimeError(f"staging dataset failed static audit: {report.errors}")
    print(
        f"wrote {len(events)} staging incidents, {len(labels)} oracle/negative rows, {len(memories)} records"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
