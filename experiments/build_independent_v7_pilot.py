#!/usr/bin/env python3
"""Build two source-pinned v7 infrastructure pilots, not confirmatory data."""

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

CHECKER = ROOT / "experiments" / "independent_v7_checker.py"
ATOM_NAMES = ["role:step1", "role:step2", "role:step3", "role:verify"]
GITEA_DOCS = (
    "https://docs.gitea.com/api/operations/create-current-user-repo/",
    "https://docs.gitea.com/api/operations/issue-create-issue/",
    "https://docs.gitea.com/api/operations/issue-get-issue/",
)
MEILI_DOCS = (
    "https://www.meilisearch.com/docs/reference/api/indexes/create-index",
    "https://www.meilisearch.com/docs/reference/api/async-task-management/get-task",
    "https://www.meilisearch.com/docs/reference/api/indexes/get-index",
)


def _digest(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    )


def _memory(
    service: str,
    version: str,
    bundle_id: str,
    index: int,
    plan: dict[str, Any] | None,
    description: str,
) -> dict[str, Any]:
    memory_id = f"v7-pilot:{service}:{bundle_id}:record-{index}"
    row = {
        "memory_id": memory_id,
        "service_id": service,
        "provider_id": "gitea" if service == "gitea" else "meilisearch",
        "version_scope": [version],
        "valid_from": "2026-10-06T20:14:45Z"
        if service == "gitea"
        else "2026-10-12T09:21:50Z",
        "valid_until": None,
        "evidence_atoms": [ATOM_NAMES[index - 1]],
        "provenance_uri": (GITEA_DOCS if service == "gitea" else MEILI_DOCS)[
            min(index - 1, 2)
        ],
        "provenance_verified": True,
        "role": "workflow_step" if index <= 3 else "verification",
        "step_index": index if index <= 3 else None,
        "storage_cost": 48,
        "token_cost": 32,
        "text": description,
    }
    if plan is not None:
        row["plan"] = plan
    row["content_digest"] = _digest({"text": description, "plan": plan})
    return row


def _family(service: str, bundle_id: str) -> list[dict[str, Any]]:
    # The correct bundle alternates across services; identifiers never encode labels.
    negative = (service == "gitea" and bundle_id == "bundle_a") or (
        service == "meilisearch" and bundle_id == "bundle_b"
    )
    if service == "gitea":
        plans = [
            {
                "method": "POST",
                "path": "/api/v1/user/repos",
                "json": {"name": "v7-pilot", "private": True, "auto_init": True},
                "capture": {"name": "repo", "source": "body_path", "path": ["name"]},
            },
            {
                "method": "POST",
                "path": "/api/v1/repos/corem_eval/{repo}/issues",
                "json": {
                    "title": "Unrelated issue" if negative else "Review service handoff"
                },
                "capture": {
                    "name": "issue_index",
                    "source": "body_path",
                    "path": ["number"],
                },
            },
            {
                "method": "GET",
                "path": "/api/v1/repos/corem_eval/{repo}/issues/{issue_index}",
            },
        ]
        descriptions = [
            "Create a private Git repository and capture its server-assigned name.",
            (
                "Create an unrelated issue titled Unrelated issue and capture its number."
                if negative
                else "Create the requested review issue titled Review service handoff and capture its number."
            ),
            "Read back the issue by its returned repository and issue number.",
            "Check repository privacy and that the final issue has the requested title.",
        ]
        version = "28.1.0"
    else:
        plans = [
            {
                "method": "POST",
                "path": "/indexes",
                "json": {
                    "uid": "v7_pilot",
                    "primaryKey": "code" if negative else "id",
                },
                "capture": {
                    "name": "task_uid",
                    "source": "body_path",
                    "path": ["taskUid"],
                },
            },
            {
                "method": "GET",
                "path": "/tasks/{task_uid}",
                "capture": {
                    "name": "index_uid",
                    "source": "body_path",
                    "path": ["indexUid"],
                },
            },
            {"method": "GET", "path": "/indexes/{index_uid}"},
        ]
        descriptions = [
            (
                "Create a searchable index with document primary key code and capture its asynchronous task ID."
                if negative
                else "Create a searchable index with document primary key id and capture its asynchronous task ID."
            ),
            "Poll the returned task until completion and recover its index identifier.",
            "Read back the created index using the observed index identifier.",
            "Check successful index creation and the requested primary key.",
        ]
        version = "1.55.0"
    return [
        _memory(
            service,
            version,
            bundle_id,
            index,
            plans[index - 1] if index <= 3 else None,
            description,
        )
        for index, description in enumerate(descriptions, 1)
    ]


def _event(service: str, minute: int) -> dict[str, Any]:
    contract = (
        {
            "kind": "v7_gitea_repo_issue",
            "repo_name": "v7-pilot",
            "issue_title": "Review service handoff",
            "statuses": [201, 201, 200],
            "handoffs": [
                {"name": "repo", "target_step": 2},
                {"name": "issue_index", "target_step": 3},
            ],
        }
        if service == "gitea"
        else {
            "kind": "v7_meili_index",
            "index_uid": "v7_pilot",
            "statuses": [202, 200, 200],
            "handoffs": [
                {"name": "task_uid", "target_step": 2},
                {"name": "index_uid", "target_step": 3},
            ],
        }
    )
    return {
        "event_id": f"v7-pilot:{service}:event-001",
        "incident_id": f"v7-pilot:{service}:incident-001",
        "duplicate_group": None,
        "provider_id": service,
        "service_id": service,
        "version_id": "28.1.0" if service == "gitea" else "1.55.0",
        "event_time": f"2026-10-18T10:{minute:02d}:00Z",
        "query": (
            "Create a private repository, open the requested review issue in it, and verify the issue by reading it back."
            if service == "gitea"
            else "Create an index with document key id, wait for creation, and verify the resulting index."
        ),
        "task_key": "repo_issue" if service == "gitea" else "index_creation",
        "operation_count": 3,
        "checker_contract_id": "independent-v7-pilot-checker-v1",
        "required_binding_atoms": [
            "binding:step_order",
            "binding:runtime_handoffs",
            "binding:status_codes",
        ],
        "required_evidence_atoms": ATOM_NAMES,
        "communication_criteria": [
            "communication:all_statuses",
            "communication:workflow_summary",
        ],
        "workflow_contract": contract,
        "observation_tier": "T2_static_public_release_controlled_pilot",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/public_independent_v7_pilot_v6")
    )
    parser.add_argument(
        "--selections",
        type=Path,
        default=Path("experiments/results_independent_v7_pilot_v6_selections"),
    )
    args = parser.parse_args()
    if args.dataset.exists() or args.selections.exists():
        parser.error("refusing to overwrite a pilot dataset or selection directory")
    events = [_event("gitea", 0), _event("meilisearch", 1)]
    memories = [
        record
        for service in ("gitea", "meilisearch")
        for bundle_id in ("bundle_a", "bundle_b")
        for record in _family(service, bundle_id)
    ]
    edges = []
    selections = []
    candidates = []
    for event in events:
        service = event["service_id"]
        for bundle_id in ("bundle_a", "bundle_b"):
            members = [
                row["memory_id"]
                for row in memories
                if row["service_id"] == service and f":{bundle_id}:" in row["memory_id"]
            ]
            edges.append(
                {
                    "edge_id": f"v7-pilot:{service}:{bundle_id}",
                    "members": members,
                    "relation_type": "three-step-runtime-handoff",
                    "evidence_atoms": ATOM_NAMES,
                    "provenance": (GITEA_DOCS if service == "gitea" else MEILI_DOCS)[0],
                    "verified": True,
                    "version_scope": [event["version_id"]],
                    "valid_from": "2026-10-06T20:14:45Z"
                    if service == "gitea"
                    else "2026-10-12T09:21:50Z",
                    "valid_until": None,
                }
            )
            selections.append(
                {
                    "run_id": f"v7-pilot:{bundle_id}:{service}",
                    "event_id": event["event_id"],
                    "method": "oracle"
                    if (
                        (service == "gitea" and bundle_id == "bundle_b")
                        or (service == "meilisearch" and bundle_id == "bundle_a")
                    )
                    else "complete_negative",
                    "seed": 0,
                    "selected_memory_ids": members,
                    "token_cost": 128,
                    "latency_ms": 0.0,
                }
            )
            candidates.append(
                {
                    "event_id": event["event_id"],
                    "candidate_edge_id": f"v7-pilot:{service}:{bundle_id}",
                    "selected_memory_ids": members,
                }
            )
    args.dataset.mkdir(parents=True)
    write_jsonl(args.dataset / "events.jsonl", events)
    write_jsonl(args.dataset / "memories.jsonl", memories)
    write_jsonl(args.dataset / "dependency_hyperedges.jsonl", edges)
    source_files = {}
    for name, rows in (("events.jsonl", events), ("memories.jsonl", memories)):
        source_files[name] = {
            "records": len(rows),
            "sha256": sha256_file(args.dataset / name),
        }
    write_json(
        args.dataset / "manifest.json",
        {
            "schema_version": "corem-real-static-v1",
            "dataset_name": "CoReM v7 two-family infrastructure pilot",
            "benchmark_mode": "static_public_release",
            "natural_drift_claim": False,
            "real_world_observations": True,
            "event_time_semantics": "Controlled local execution order, not observed traffic",
            "claim_boundary": "Two hand-authored three-step source-derived pilot families; not a confirmatory result",
            "observation_tier": "T2_static_public_release_controlled_pilot",
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
                "collection_end_utc": "2026-10-18T10:00:00Z",
            },
            "execution": {
                "resettable": True,
                "runner_contract": "independent-v7-runner-v1",
            },
            "checker": {
                "independent_from_runner": True,
                "contract_id": "independent-v7-pilot-checker-v1",
                "implementation_digest": f"sha256:{sha256_file(CHECKER)}",
            },
            "files": source_files,
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
        [{"event_id": event["event_id"], "split": "test"} for event in events],
    )
    write_jsonl(args.selections / "selections.jsonl", selections)
    write_jsonl(args.selections / "candidate_selections.jsonl", candidates)
    write_jsonl(
        args.selections / "query_only_events.jsonl",
        [
            {
                key: event[key]
                for key in (
                    "event_id",
                    "service_id",
                    "provider_id",
                    "version_id",
                    "event_time",
                    "query",
                )
            }
            for event in events
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
        raise RuntimeError(f"generated pilot did not pass static audit: {audit.errors}")
    print(
        f"wrote {len(events)} pilot events and {len(selections)} label-neutral oracle/negative selections"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
