#!/usr/bin/env python3
"""Build the pinned public-service release-reconstruction dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime, timedelta
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

SOURCES = {
    "json-server": {
        "provider_id": "typicode",
        "repository": "https://github.com/typicode/json-server",
        "license": "MIT",
        "versions": {
            "0.17.4": {
                "released": "2023-09-29T13:36:58.964Z",
                "commit": "78ea71375666d49145734689c097654c54f90686",
                "artifact": "json-server@0.17.4",
                "integrity": "sha512-bGBb0WtFuAKbgI7JV3A864irWnMZSvBYRJbohaOuatHwKSRFUfqtQlrYMrB6WbalXy/cJabyjlb7JkHli6dYjQ==",
            },
            "1.0.0-beta.3": {
                "released": "2024-09-24T10:33:52.359Z",
                "commit": "6aa56d9581488d9bcd1baf42c4c97b293cd9ee99",
                "artifact": "json-server@1.0.0-beta.3",
                "integrity": "sha512-DwE69Ep5ccwIJZBUIWEENC30Yj8bwr4Ax9W9VoIWAYnB8Sj4ReptscO8/DRHv/nXwVlmb3Bk73Ls86+VZdYkkA==",
            },
        },
        "release_evidence": "https://www.npmjs.com/package/json-server?activeTab=versions",
        "change_evidence": "https://github.com/typicode/json-server/blob/v1/README.md#notable-differences-with-v017",
    },
    "go-httpbin": {
        "provider_id": "mccutchen",
        "repository": "https://github.com/mccutchen/go-httpbin",
        "license": "MIT",
        "versions": {
            "v2.14.1": {
                "released": "2024-08-29T21:41:28Z",
                "commit": "705339886f092da0f5851bb31e07f29bba4b6835",
                "artifact": "ghcr.io/mccutchen/go-httpbin@sha256:2612f203b1be154524b0dbb65212a158776f0643e2f6e1dad45984fc93413d97",
                "integrity": "sha256:2612f203b1be154524b0dbb65212a158776f0643e2f6e1dad45984fc93413d97",
            },
            "v2.15.0": {
                "released": "2024-09-17T04:51:38Z",
                "commit": "34a21a3ef81df5aa42185fcae2e5a608404ac6a0",
                "artifact": "ghcr.io/mccutchen/go-httpbin@sha256:24528cf5229d0b70065ac27e6c9e4d96f5452a84a3ce4433e56573c18d96827a",
                "integrity": "sha256:24528cf5229d0b70065ac27e6c9e4d96f5452a84a3ce4433e56573c18d96827a",
            },
        },
        "release_evidence": "https://github.com/mccutchen/go-httpbin/releases",
        "change_evidence": "https://github.com/mccutchen/go-httpbin/releases/tag/v2.15.0",
    },
}


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def iso(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def digest_object(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def task_specs(service: str, version: str) -> list[dict[str, Any]]:
    if service == "json-server":
        modern = version == "1.0.0-beta.3"
        identifier_type = "string" if modern else "number"
        return [
            {
                "task_key": "paginate",
                "query": "Return the first page of exactly two posts and report the result shape.",
                "plan": {
                    "method": "GET",
                    "path": f"/posts?_page=1&{'_per_page' if modern else '_limit'}=2",
                },
                "contract": {
                    "kind": "json_pagination",
                    "status": 200,
                    "container": "paged" if modern else "list",
                    "count": 2,
                    "id_type": identifier_type,
                },
            },
            {
                "task_key": "identifier",
                "query": "Fetch post 1 and confirm its title and identifier type.",
                "plan": {"method": "GET", "path": "/posts/1"},
                "contract": {
                    "kind": "json_record",
                    "status": 200,
                    "title": "alpha",
                    "id_type": identifier_type,
                },
            },
            {
                "task_key": "create",
                "query": "Create a post titled Echo with seven views and confirm the created record.",
                "plan": {
                    "method": "POST",
                    "path": "/posts",
                    "json": {"title": "Echo", "views": 7},
                },
                "contract": {
                    "kind": "json_create",
                    "status": 201,
                    "title": "Echo",
                    "views": 7,
                    "id_type": identifier_type,
                },
            },
            {
                "task_key": "comments",
                "query": "List the two comments belonging to post 1 and confirm identifier types.",
                "plan": {"method": "GET", "path": "/comments?postId=1"},
                "contract": {
                    "kind": "json_comments",
                    "status": 200,
                    "count": 2,
                    "id_type": identifier_type,
                },
            },
        ]

    modern = version == "v2.15.0"
    return [
        {
            "task_key": "anything",
            "query": "Send a GET probe with mode test and verify the reflected method, path, and query.",
            "plan": {"method": "GET", "path": "/anything/probe?mode=test"},
            "contract": {
                "kind": "httpbin_anything",
                "status": 200,
                "method": "GET",
                "path_suffix": "/anything/probe?mode=test",
                "query_value": "test",
            },
        },
        {
            "task_key": "no_content",
            "query": "Request an intentional 204 response and confirm that no body is returned.",
            "plan": {"method": "GET", "path": "/status/204"},
            "contract": {"kind": "status_empty", "status": 204},
        },
        {
            "task_key": "base64",
            "query": "Decode the base64 endpoint payload and confirm it returns CoReM.",
            "plan": {"method": "GET", "path": "/base64/Q29SZU0="},
            "contract": {"kind": "plain_text", "status": 200, "text": "CoReM"},
        },
        {
            "task_key": "trailers",
            "query": "Probe the trailers endpoint and report its release-specific availability.",
            "plan": {"method": "GET", "path": "/trailers"},
            "contract": {
                "kind": "trailers",
                "status": 200 if modern else 404,
                "available": modern,
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
                "transition_id": f"{service}:{versions[0]}->{versions[1]}",
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
            specs = task_specs(service, version)
            valid_until = transition_time if version_index == 0 else None
            for spec in specs:
                action_atom = f"action:{service}:{spec['task_key']}"
                verify_atom = f"verify:{service}:{spec['task_key']}"
                base_id = f"{service}:{version}:{spec['task_key']}"
                common = {
                    "service_id": service,
                    "provider_id": source["provider_id"],
                    "version_scope": [version],
                    "valid_from": release["released"],
                    "valid_until": valid_until,
                    "provenance_uri": source["change_evidence"],
                    "provenance_verified": True,
                    "source_commit": release["commit"],
                }
                action = {
                    **common,
                    "memory_id": f"{base_id}:action",
                    "role": "action",
                    "text": f"{spec['query']} Use {spec['plan']['method']} {spec['plan']['path']} for {service} {version}.",
                    "token_cost": 48,
                    "storage_cost": 64,
                    "evidence_atoms": [action_atom],
                    "operation_ids": [spec["task_key"]],
                    "plan": spec["plan"],
                }
                action["content_digest"] = digest_object(
                    {
                        key: value
                        for key, value in action.items()
                        if key != "content_digest"
                    }
                )
                verification = {
                    **common,
                    "memory_id": f"{base_id}:verification",
                    "role": "verification",
                    "text": f"{spec['query']} Verify status {spec['contract']['status']} and the declared response postconditions for {service} {version}.",
                    "token_cost": 48,
                    "storage_cost": 64,
                    "evidence_atoms": [verify_atom],
                    "operation_ids": [spec["task_key"]],
                    "verification_contract": spec["contract"],
                }
                verification["content_digest"] = digest_object(
                    {
                        key: value
                        for key, value in verification.items()
                        if key != "content_digest"
                    }
                )
                memories.extend((action, verification))
                hyperedges.append(
                    {
                        "edge_id": f"dependency:{base_id}",
                        "members": [action["memory_id"], verification["memory_id"]],
                        "evidence_atoms": [action_atom, verify_atom],
                        "relation_type": "action-verification",
                        "version_scope": [version],
                        "valid_from": release["released"],
                        "valid_until": valid_until,
                        "provenance": source["change_evidence"],
                        "verified": True,
                    }
                )

            release_time = parse_utc(release["released"])
            # Twenty independent replay requests per release: five repetitions
            # of four task types. Times are logical replay order, not claimed
            # production collection timestamps.
            for index in range(20):
                spec = specs[index % len(specs)]
                event_time = release_time + timedelta(hours=6 * (index + 1))
                if valid_until is not None and event_time >= parse_utc(valid_until):
                    event_time = release_time + timedelta(minutes=10 * (index + 1))
                event_id = f"{service}:{version}:event-{index + 1:02d}"
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
                        "task_contract": spec["contract"],
                        "required_binding_atoms": [
                            "binding:http_method",
                            "binding:http_path",
                            "binding:status_code",
                        ],
                        "required_evidence_atoms": [
                            f"action:{service}:{spec['task_key']}",
                            f"verify:{service}:{spec['task_key']}",
                        ],
                        "communication_criteria": [
                            "communication:explicit_status",
                            "communication:outcome_summary",
                        ],
                        "checker_contract_id": "public-versioned-services-checker-v1",
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
    checker_path = REPOSITORY_ROOT / "experiments/public_services_checker.py"
    checker_digest = (
        f"sha256:{sha256_file(checker_path)}"
        if checker_path.is_file()
        else "sha256:checker-not-yet-installed"
    )
    files = {
        filename: {"records": len(rows), "sha256": sha256_file(args.output / filename)}
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
            "dataset_name": "CoReM Public Versioned Services T2",
            "real_world_observations": True,
            "observation_tier": "T2_public_release_reconstruction",
            "claim_boundary": "Offline replay against genuine pinned public releases; not live production traffic or natural deployment telemetry.",
            "event_time_semantics": "Deterministic logical replay order anchored to official release timestamps; wall-clock executions are recorded separately.",
            "provenance": {
                "source_uri": "https://github.com/typicode/json-server ; https://github.com/mccutchen/go-httpbin",
                "license": "MIT (both upstream projects)",
                "immutable_source_id": "json-server:78ea713,6aa56d9;go-httpbin:7053398,34a21a3",
                "collection_start_utc": min(row["event_time"] for row in events),
                "collection_end_utc": max(row["event_time"] for row in events),
            },
            "execution": {
                "resettable": True,
                "runner_contract": "public-versioned-services-runner-v1",
            },
            "checker": {
                "independent_from_runner": True,
                "contract_id": "public-versioned-services-checker-v1",
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
        f"wrote {len(events)} events, {len(memories)} memories, and {len(transitions)} transitions to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
