#!/usr/bin/env python3
"""Build four new-provider development families; never use as sealed evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import (  # noqa: E402
    audit_real_dataset,
    sha256_file,
    write_checksums,
    write_json,
    write_jsonl,
)

CHECKER = ROOT / "experiments" / "independent_v7_checker_v2.py"
ATOMS = ["role:step1", "role:step2", "role:step3", "role:verify"]
REPO_DOC = "https://docs.gitea.com/api/operations/create-current-user-repo/"
TASK_DOC = (
    "https://www.meilisearch.com/docs/reference/api/async-task-management/get-task"
)
ADD_DOC = (
    "https://www.meilisearch.com/docs/reference/api/documents/add-or-replace-documents"
)


def _hash(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    )


def _repo_plan(name: str) -> dict[str, Any]:
    return {
        "method": "POST",
        "path": "/api/v1/user/repos",
        "json": {"name": name, "private": True, "auto_init": True},
        "capture": {"name": "repo", "source": "body_path", "path": ["name"]},
    }


def _meili_add_plan(uid: str, title: str, doc_id: int) -> dict[str, Any]:
    return {
        "method": "POST",
        "path": f"/indexes/{uid}/documents?primaryKey=id",
        "json": [{"id": doc_id, "title": title}],
        "capture": {"name": "task_uid", "source": "body_path", "path": ["taskUid"]},
    }


def _meili_poll_plan() -> dict[str, Any]:
    return {
        "method": "GET",
        "path": "/tasks/{task_uid}",
        "capture": {"name": "index_uid", "source": "body_path", "path": ["indexUid"]},
    }


def _spec(family: str, bundle: str) -> dict[str, Any]:
    positive = (family in {"gitea_label", "meili_document"}) == (bundle == "bundle_a")
    if family == "gitea_label":
        repo = "v7-dev-label"
        name = "Urgent triage" if positive else "Routine triage"
        docs = [
            REPO_DOC,
            "https://docs.gitea.com/api/operations/issue-create-label/",
            "https://docs.gitea.com/api/operations/issue-get-label/",
        ]
        plans = [
            _repo_plan(repo),
            {
                "method": "POST",
                "path": "/api/v1/repos/corem_eval/{repo}/labels",
                "json": {"name": name, "color": "#ff0000"},
                "capture": {"name": "label_id", "source": "body_path", "path": ["id"]},
            },
            {
                "method": "GET",
                "path": "/api/v1/repos/corem_eval/{repo}/labels/{label_id}",
            },
        ]
        text = [
            "Create a private Git repository and capture its actual name.",
            f"Create repository label named {name} and capture its ID.",
            "Read the created label using its returned repository and label ID.",
            f"Verify the label is named {name} in the private repository.",
        ]
        contract = {
            "kind": "v7_gitea_label",
            "repo_name": repo,
            "label_name": "Urgent triage",
            "statuses": [201, 201, 200],
            "handoffs": [
                {"name": "repo", "target_step": 2},
                {"name": "label_id", "target_step": 3},
            ],
        }
        query = "Create a private repository with an urgent triage label and verify it by ID."
    elif family == "gitea_milestone":
        repo = "v7-dev-milestone"
        state = "open" if positive else "closed"
        docs = [
            REPO_DOC,
            "https://docs.gitea.com/api/operations/issue-create-milestone/",
            "https://docs.gitea.com/api/operations/issue-get-milestone/",
        ]
        plans = [
            _repo_plan(repo),
            {
                "method": "POST",
                "path": "/api/v1/repos/corem_eval/{repo}/milestones",
                "json": {"title": "Sprint gate", "state": state},
                "capture": {
                    "name": "milestone_id",
                    "source": "body_path",
                    "path": ["id"],
                },
            },
            {
                "method": "GET",
                "path": "/api/v1/repos/corem_eval/{repo}/milestones/{milestone_id}",
            },
        ]
        text = [
            "Create a private Git repository and capture its actual name.",
            f"Create a {state} milestone named Sprint gate and capture its ID.",
            "Read the milestone using the returned repository and milestone ID.",
            f"Verify the Sprint gate milestone is {state} in the private repository.",
        ]
        contract = {
            "kind": "v7_gitea_milestone",
            "repo_name": repo,
            "milestone_title": "Sprint gate",
            "milestone_state": "open",
            "statuses": [201, 201, 200],
            "handoffs": [
                {"name": "repo", "target_step": 2},
                {"name": "milestone_id", "target_step": 3},
            ],
        }
        query = "Create a private repository, open a Sprint gate milestone, and verify its state."
    elif family == "meili_document":
        uid = "v7_dev_document"
        title = "Release audit" if positive else "Routine inventory"
        docs = [
            ADD_DOC,
            TASK_DOC,
            "https://www.meilisearch.com/docs/reference/api/documents/get-document",
        ]
        plans = [
            _meili_add_plan(uid, title, 11),
            _meili_poll_plan(),
            {"method": "GET", "path": "/indexes/{index_uid}/documents/11"},
        ]
        text = [
            f"Add document 11 titled {title} to a search index and capture its task ID.",
            "Wait for the document task and recover its index identifier.",
            "Read document 11 from the index returned by the task.",
            f"Verify document 11 was indexed with title {title}.",
        ]
        contract = {
            "kind": "v7_meili_document",
            "index_uid": uid,
            "document_id": 11,
            "document_title": "Release audit",
            "statuses": [202, 200, 200],
            "handoffs": [
                {"name": "task_uid", "target_step": 2},
                {"name": "index_uid", "target_step": 3},
            ],
        }
        query = (
            "Index document 11 as Release audit, wait for indexing, and read it back."
        )
    elif family == "meili_search":
        uid = "v7_dev_search"
        term = "handbook" if positive else "inventory"
        docs = [
            ADD_DOC,
            TASK_DOC,
            "https://www.meilisearch.com/docs/reference/api/search/search-with-post",
        ]
        plans = [
            _meili_add_plan(uid, "Release audit handbook", 23),
            _meili_poll_plan(),
            {
                "method": "POST",
                "path": "/indexes/{index_uid}/search",
                "json": {"q": term},
            },
        ]
        text = [
            "Index document 23 titled Release audit handbook and capture its task ID.",
            "Wait for indexing and recover the index identifier.",
            f"Search the resulting index for {term}.",
            f"Verify the {term} search returns document 23.",
        ]
        contract = {
            "kind": "v7_meili_search",
            "index_uid": uid,
            "document_id": 23,
            "document_title": "Release audit handbook",
            "search_query": "handbook",
            "statuses": [202, 200, 200],
            "handoffs": [
                {"name": "task_uid", "target_step": 2},
                {"name": "index_uid", "target_step": 3},
            ],
        }
        query = "Index the release audit handbook, wait, then search for handbook to find it."
    else:
        raise ValueError(f"unknown development family: {family}")
    return {
        "positive": positive,
        "service": "gitea" if family.startswith("gitea") else "meilisearch",
        "plans": plans,
        "text": text,
        "docs": docs,
        "contract": contract,
        "query": query,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/public_independent_v7_development_v1"),
    )
    parser.add_argument(
        "--selections",
        type=Path,
        default=Path("experiments/results_independent_v7_development_v1_selections"),
    )
    args = parser.parse_args()
    if args.dataset.exists() or args.selections.exists():
        parser.error("refusing to overwrite existing development data")
    families = ("gitea_label", "gitea_milestone", "meili_document", "meili_search")
    events, memories, edges, labels, candidates = [], [], [], [], []
    for index, family in enumerate(families):
        event_id = f"v7-dev:{family}:event-001"
        for bundle in ("bundle_a", "bundle_b"):
            spec = _spec(family, bundle)
            service = spec["service"]
            version = "28.1.0" if service == "gitea" else "1.55.0"
            valid_from = (
                "2026-10-06T20:14:45Z" if service == "gitea" else "2026-10-12T09:21:50Z"
            )
            if bundle == "bundle_a":
                events.append(
                    {
                        "event_id": event_id,
                        "incident_id": event_id,
                        "duplicate_group": None,
                        "provider_id": service,
                        "service_id": service,
                        "version_id": version,
                        "event_time": f"2026-10-18T10:{index:02d}:00Z",
                        "query": spec["query"],
                        "task_key": family,
                        "operation_count": 3,
                        "checker_contract_id": "independent-v7-checker-v2",
                        "required_binding_atoms": [
                            "binding:step_order",
                            "binding:runtime_handoffs",
                            "binding:status_codes",
                        ],
                        "required_evidence_atoms": ATOMS,
                        "communication_criteria": [
                            "communication:all_statuses",
                            "communication:workflow_summary",
                        ],
                        "workflow_contract": spec["contract"],
                        "observation_tier": "T2_static_public_release_controlled_development",
                    }
                )
            member_ids = []
            for step, description in enumerate(spec["text"], 1):
                memory_id = f"v7-dev:{family}:{bundle}:record-{step}"
                member_ids.append(memory_id)
                record = {
                    "memory_id": memory_id,
                    "service_id": service,
                    "provider_id": service,
                    "version_scope": [version],
                    "valid_from": valid_from,
                    "valid_until": None,
                    "evidence_atoms": [ATOMS[step - 1]],
                    "provenance_uri": spec["docs"][min(step - 1, 2)],
                    "provenance_verified": True,
                    "role": "workflow_step" if step <= 3 else "verification",
                    "step_index": step if step <= 3 else None,
                    "storage_cost": 48,
                    "token_cost": 32,
                    "text": description,
                }
                if step <= 3:
                    record["plan"] = spec["plans"][step - 1]
                record["content_digest"] = _hash(
                    {"text": description, "plan": record.get("plan")}
                )
                memories.append(record)
            edges.append(
                {
                    "edge_id": f"v7-dev:{family}:{bundle}",
                    "members": member_ids,
                    "relation_type": "three-step-runtime-handoff",
                    "evidence_atoms": ATOMS,
                    "provenance": spec["docs"][0],
                    "verified": True,
                    "version_scope": [version],
                    "valid_from": valid_from,
                    "valid_until": None,
                }
            )
            labels.append(
                {
                    "run_id": f"v7-dev:{family}:{bundle}",
                    "event_id": event_id,
                    "method": "oracle" if spec["positive"] else "complete_negative",
                    "seed": 0,
                    "selected_memory_ids": member_ids,
                    "token_cost": 128,
                    "latency_ms": 0.0,
                }
            )
            candidates.append(
                {
                    "event_id": event_id,
                    "candidate_edge_id": f"v7-dev:{family}:{bundle}",
                    "selected_memory_ids": member_ids,
                }
            )
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
            "dataset_name": "CoReM v7 four-family development probe",
            "benchmark_mode": "static_public_release",
            "natural_drift_claim": False,
            "real_world_observations": True,
            "event_time_semantics": "Controlled local execution order, not observed traffic",
            "claim_boundary": "Four hand-authored oracle-tested development families; not independent confirmation",
            "observation_tier": "T2_static_public_release_controlled_development",
            "provenance": {
                "source_uri": (
                    "https://github.com/go-gitea/gitea/releases/tag/v28.1.0 ; "
                    "https://github.com/meilisearch/meilisearch/releases/tag/v1.55.0"
                ),
                "license": "MIT (Gitea; Meilisearch Community Edition only)",
                "immutable_source_id": (
                    "gitea:ea93cfa4a300564ac457d3903469fea49ec5a4da37d0cebaa435f8e329ea64d0;"
                    "meilisearch:7e33f5889235f14290074a19a6d3574bd9b10fb2ee8731eabb7dafd0714b0513"
                ),
                "collection_start_utc": "2026-10-06T20:14:45Z",
                "collection_end_utc": "2026-10-18T10:03:00Z",
            },
            "execution": {
                "resettable": True,
                "runner_contract": "independent-v7-runner-v1",
            },
            "checker": {
                "independent_from_runner": True,
                "contract_id": "independent-v7-checker-v2",
                "implementation_digest": f"sha256:{sha256_file(CHECKER)}",
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
    write_checksums(
        args.selections,
        (
            "splits.jsonl",
            "selections.jsonl",
            "candidate_selections.jsonl",
            "query_only_events.jsonl",
        ),
    )
    audit = audit_real_dataset(args.dataset)
    if not audit.eligible:
        raise RuntimeError(f"development dataset failed static audit: {audit.errors}")
    print(
        f"wrote {len(events)} development events and {len(labels)} labeled executions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
