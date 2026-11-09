#!/usr/bin/env python3
"""Build the two-step public release-reconstruction workflow dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import (
    SCHEMA_VERSION,
    sha256_file,
    write_checksums,
    write_json,
    write_jsonl,
)
from scripts.build_public_versioned_service_dataset import SOURCES, iso, parse_utc


def digest_object(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def workflow_specs(service: str, version: str) -> list[dict[str, Any]]:
    if service == "json-server":
        modern = version == "1.0.0-beta.3"
        id_type = "string" if modern else "number"
        return [
            {
                "task_key": "paginate_fetch",
                "query": "Fetch a two-item first page, pass its first post identifier into a second request, and verify that post.",
                "plans": [
                    {
                        "method": "GET",
                        "path": f"/posts?_page=1&{'_per_page' if modern else '_limit'}=2",
                        "capture": {
                            "name": "post_id",
                            "source": "body_path",
                            "path": ["data", 0, "id"] if modern else [0, "id"],
                        },
                    },
                    {"method": "GET", "path": "/posts/{post_id}"},
                ],
                "contract": {
                    "kind": "json_paginate_fetch",
                    "statuses": [200, 200],
                    "handoff_name": "post_id",
                    "id_type": id_type,
                },
            },
            {
                "task_key": "create_fetch",
                "query": "Create an Echo post, pass the returned identifier into a fetch, and verify the persisted record.",
                "plans": [
                    {
                        "method": "POST",
                        "path": "/posts",
                        "json": {"title": "Echo", "views": 7},
                        "capture": {
                            "name": "created_id",
                            "source": "body_path",
                            "path": ["id"],
                        },
                    },
                    {"method": "GET", "path": "/posts/{created_id}"},
                ],
                "contract": {
                    "kind": "json_create_fetch",
                    "statuses": [201, 200],
                    "handoff_name": "created_id",
                    "id_type": id_type,
                },
            },
            {
                "task_key": "relationship_fetch",
                "query": "Expand comment 1 to its related post, pass the expanded post identifier into a fetch, and verify the relation.",
                "plans": [
                    {
                        "method": "GET",
                        "path": (
                            "/comments/1?_embed=post"
                            if modern
                            else "/comments/1?_expand=post"
                        ),
                        "capture": {
                            "name": "post_id",
                            "source": "body_path",
                            "path": ["post", "id"],
                        },
                    },
                    {"method": "GET", "path": "/posts/{post_id}"},
                ],
                "contract": {
                    "kind": "json_relationship_fetch",
                    "statuses": [200, 200],
                    "handoff_name": "post_id",
                    "id_type": id_type,
                },
            },
            {
                "task_key": "update_fetch",
                "query": "Update post 1, pass the updated identifier into a fetch, and verify the stored title.",
                "plans": [
                    {
                        "method": "PATCH",
                        "path": "/posts/1",
                        "json": {"title": "updated"},
                        "capture": {
                            "name": "post_id",
                            "source": "body_path",
                            "path": ["id"],
                        },
                    },
                    {"method": "GET", "path": "/posts/{post_id}"},
                ],
                "contract": {
                    "kind": "json_update_fetch",
                    "statuses": [200, 200],
                    "handoff_name": "post_id",
                    "id_type": id_type,
                },
            },
        ]

    trailers_status = 200 if version == "v2.15.0" else 404
    return [
        {
            "task_key": "uuid_reflect",
            "query": "Obtain a server UUID, pass it into a second endpoint path, and verify the reflected handoff.",
            "plans": [
                {
                    "method": "GET",
                    "path": "/uuid",
                    "capture": {
                        "name": "uuid",
                        "source": "body_path",
                        "path": ["uuid"],
                    },
                },
                {"method": "GET", "path": "/anything/{uuid}"},
            ],
            "contract": {
                "kind": "httpbin_uuid_reflect",
                "statuses": [200, 200],
                "handoff_name": "uuid",
            },
        },
        {
            "task_key": "base64_reflect",
            "query": "Decode the CoReM payload, pass the observed text into a second endpoint path, and verify the reflection.",
            "plans": [
                {
                    "method": "GET",
                    "path": "/base64/Q29SZU0=",
                    "capture": {"name": "decoded", "source": "text"},
                },
                {"method": "GET", "path": "/anything/{decoded}"},
            ],
            "contract": {
                "kind": "httpbin_base64_reflect",
                "statuses": [200, 200],
                "handoff_name": "decoded",
            },
        },
        {
            "task_key": "header_reflect",
            "query": "Obtain the release-proof response header value, pass it into a second endpoint path, and verify it.",
            "plans": [
                {
                    "method": "GET",
                    "path": "/response-headers?X-Corem-Token=release-proof",
                    "capture": {
                        "name": "release_token",
                        "source": "body_path",
                        "path": ["X-Corem-Token"],
                    },
                },
                {"method": "GET", "path": "/anything/{release_token}"},
            ],
            "contract": {
                "kind": "httpbin_header_reflect",
                "statuses": [200, 200],
                "handoff_name": "release_token",
            },
        },
        {
            "task_key": "trailer_status_reflect",
            "query": "Probe trailer availability, pass the observed status into a second endpoint path, and verify release-specific behavior.",
            "plans": [
                {
                    "method": "GET",
                    "path": "/trailers",
                    "capture": {
                        "name": "availability_status",
                        "source": "status",
                    },
                },
                {
                    "method": "GET",
                    "path": "/anything/status-{availability_status}",
                },
            ],
            "contract": {
                "kind": "httpbin_trailer_status_reflect",
                "statuses": [trailers_status, 200],
                "handoff_name": "availability_status",
                "trailers_status": trailers_status,
            },
        },
    ]


def build_records() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    events: list[dict[str, Any]] = []
    memories: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    hyperedges: list[dict[str, Any]] = []
    for service, source in SOURCES.items():
        versions = list(source["versions"])
        transition_time = source["versions"][versions[1]]["released"]
        transitions.append(
            {
                "transition_id": f"workflow:{service}:{versions[0]}->{versions[1]}",
                "provider_id": source["provider_id"],
                "service_id": service,
                "from_version_id": versions[0],
                "to_version_id": versions[1],
                "transition_time": transition_time,
                "evidence_uri": source["change_evidence"],
                "independently_observed": True,
                "source_commit_from": source["versions"][versions[0]]["commit"],
                "source_commit_to": source["versions"][versions[1]]["commit"],
            }
        )
        for version_index, version in enumerate(versions):
            release = source["versions"][version]
            valid_until = transition_time if version_index == 0 else None
            specs = workflow_specs(service, version)
            for spec in specs:
                base_id = f"workflow:{service}:{version}:{spec['task_key']}"
                atoms = [
                    f"step1:{service}:{spec['task_key']}",
                    f"step2:{service}:{spec['task_key']}",
                    f"verify:{service}:{spec['task_key']}",
                ]
                common = {
                    "service_id": service,
                    "provider_id": source["provider_id"],
                    "version_scope": [version],
                    "valid_from": release["released"],
                    "valid_until": valid_until,
                    "provenance_uri": source["change_evidence"],
                    "provenance_verified": True,
                    "source_commit": release["commit"],
                    "operation_ids": [spec["task_key"]],
                    "token_cost": 48,
                    "storage_cost": 64,
                }
                members = []
                for step_index, plan in enumerate(spec["plans"], start=1):
                    memory = {
                        **common,
                        "memory_id": f"{base_id}:step{step_index}",
                        "role": "workflow_step",
                        "step_index": step_index,
                        "text": (
                            f"{spec['query']} Step {step_index}: use "
                            f"{plan['method']} {plan['path']} for {service} {version}."
                        ),
                        "evidence_atoms": [atoms[step_index - 1]],
                        "plan": plan,
                    }
                    memory["content_digest"] = digest_object(memory)
                    memories.append(memory)
                    members.append(memory["memory_id"])
                verification = {
                    **common,
                    "memory_id": f"{base_id}:verification",
                    "role": "verification",
                    "step_index": 3,
                    "text": (
                        f"{spec['query']} Verify the two-step runtime handoff and "
                        f"release-specific postconditions for {service} {version}."
                    ),
                    "evidence_atoms": [atoms[2]],
                    "verification_contract": spec["contract"],
                }
                verification["content_digest"] = digest_object(verification)
                memories.append(verification)
                members.append(verification["memory_id"])
                hyperedges.append(
                    {
                        "edge_id": f"dependency:{base_id}",
                        "members": members,
                        "evidence_atoms": atoms,
                        "relation_type": "two-step-runtime-handoff",
                        "version_scope": [version],
                        "valid_from": release["released"],
                        "valid_until": valid_until,
                        "provenance": source["change_evidence"],
                        "verified": True,
                    }
                )

            release_time = parse_utc(release["released"])
            for index in range(20):
                spec = specs[index % len(specs)]
                event_time = release_time + timedelta(hours=6 * (index + 1))
                if valid_until is not None and event_time >= parse_utc(valid_until):
                    event_time = release_time + timedelta(minutes=10 * (index + 1))
                event_id = f"workflow:{service}:{version}:event-{index + 1:02d}"
                events.append(
                    {
                        "event_id": event_id,
                        "incident_id": f"incident:{event_id}",
                        "duplicate_group": None,
                        "provider_id": source["provider_id"],
                        "service_id": service,
                        "version_id": version,
                        "event_time": iso(event_time),
                        "query": spec["query"],
                        "task_key": spec["task_key"],
                        "operation_count": 2,
                        "workflow_contract": spec["contract"],
                        "required_binding_atoms": [
                            "binding:step_order",
                            "binding:runtime_handoff",
                            "binding:status_codes",
                        ],
                        "required_evidence_atoms": [
                            f"step1:{service}:{spec['task_key']}",
                            f"step2:{service}:{spec['task_key']}",
                            f"verify:{service}:{spec['task_key']}",
                        ],
                        "communication_criteria": [
                            "communication:both_statuses",
                            "communication:workflow_summary",
                        ],
                        "checker_contract_id": "public-versioned-workflows-checker-v1",
                        "observation_tier": "T2_public_release_reconstruction",
                    }
                )
    return events, memories, transitions, hyperedges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    events, memories, transitions, hyperedges = build_records()
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "events.jsonl", events)
    write_jsonl(args.output / "memories.jsonl", memories)
    write_jsonl(args.output / "version_transitions.jsonl", transitions)
    write_jsonl(args.output / "dependency_hyperedges.jsonl", hyperedges)
    write_json(args.output / "source_releases.json", SOURCES)
    checker_path = REPOSITORY_ROOT / "experiments" / "public_workflows_checker.py"
    checker_digest = f"sha256:{sha256_file(checker_path)}"
    files = {
        filename: {
            "records": len(rows),
            "sha256": sha256_file(args.output / filename),
        }
        for filename, rows in (
            ("events.jsonl", events),
            ("memories.jsonl", memories),
            ("version_transitions.jsonl", transitions),
        )
    }
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_name": "CoReM Public Versioned Two-Step Workflows T2",
            "real_world_observations": True,
            "observation_tier": "T2_public_release_reconstruction",
            "claim_boundary": (
                "Two-step offline workflows against genuine pinned public releases; "
                "not live production traffic or natural deployment telemetry."
            ),
            "event_time_semantics": (
                "Deterministic logical replay order anchored to official release "
                "timestamps; execution wall time is recorded separately."
            ),
            "workflow_contract": {
                "operations_per_incident": 2,
                "runtime_handoff_required": True,
                "expected_events": len(events),
            },
            "provenance": {
                "source_uri": (
                    "https://github.com/typicode/json-server ; "
                    "https://github.com/mccutchen/go-httpbin"
                ),
                "license": "MIT (both upstream projects)",
                "immutable_source_id": (
                    "json-server:78ea713,6aa56d9;go-httpbin:7053398,34a21a3"
                ),
                "collection_start_utc": min(row["event_time"] for row in events),
                "collection_end_utc": max(row["event_time"] for row in events),
            },
            "execution": {
                "resettable": True,
                "runner_contract": "public-versioned-two-step-workflows-runner-v1",
            },
            "checker": {
                "independent_from_runner": True,
                "contract_id": "public-versioned-workflows-checker-v1",
                "implementation_digest": checker_digest,
            },
            "files": files,
        },
    )
    write_checksums(
        args.output,
        (
            "dependency_hyperedges.jsonl",
            "events.jsonl",
            "manifest.json",
            "memories.jsonl",
            "source_releases.json",
            "version_transitions.jsonl",
        ),
    )
    print(
        f"wrote {len(events)} two-step events, {len(memories)} memories, "
        f"and {len(transitions)} transitions to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
