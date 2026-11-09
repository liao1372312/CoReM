#!/usr/bin/env python3
"""Build the frozen four-provider v5 public release challenge dataset."""

from __future__ import annotations

import argparse
import hashlib
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
from scripts.build_public_challenge_dataset import (
    QUERY_VARIANTS,
    ROLE_ATOMS,
    digest_object,
)
from scripts.build_public_challenge_dataset_v2 import (
    CREATE_QUERIES,
    transformed_records,
)
from scripts.build_public_versioned_service_dataset import SOURCES, iso, parse_utc

WIREMOCK_SHA256 = {
    "3.9.1": "723a880d50d3b0a145af0df07e578c2cb85e77feb2231e6991c9a1366926912c",
    "3.13.2": "d097b19bd483c5038479b13a5c71e9faf8f2f5106584f0c120a7770ab0bdb367",
}
MOCKSERVER_SHA256 = {
    "5.14.0": "2a11b20118fc2904a9408f6bcb98b63ed1f26e6db84447cbb99a6f108c2246e6",
    "5.15.0": "bf5cf50cb5bf8ef99ea372e78b5ceddcc6eb6a051f3b5b1d3e62c7b812c92aeb",
}

EXTRA_SOURCES: dict[str, dict[str, Any]] = {
    "wiremock": {
        "provider_id": "wiremock",
        "repository": "https://github.com/wiremock/wiremock",
        "license": "Apache-2.0",
        "release_evidence": "https://github.com/wiremock/wiremock/releases",
        "change_evidence": "https://github.com/wiremock/wiremock/releases/tag/3.13.2",
        "versions": {
            "3.9.1": {
                "released": "2024-07-24T15:16:17Z",
                "commit": "2ba8a3e7df0017d19c6463852481e273eeb4aac1",
                "artifact": "https://repo.maven.apache.org/maven2/org/wiremock/wiremock-standalone/3.9.1/wiremock-standalone-3.9.1.jar",
                "integrity": f"sha256:{WIREMOCK_SHA256['3.9.1']}",
            },
            "3.13.2": {
                "released": "2025-11-14T14:39:57Z",
                "commit": "88587aa13b4899da080538e1b21bec0da105491e",
                "artifact": "https://repo.maven.apache.org/maven2/org/wiremock/wiremock-standalone/3.13.2/wiremock-standalone-3.13.2.jar",
                "integrity": f"sha256:{WIREMOCK_SHA256['3.13.2']}",
            },
        },
    },
    "mockserver": {
        "provider_id": "mock-server",
        "repository": "https://github.com/mock-server/mockserver",
        "license": "Apache-2.0",
        "release_evidence": "https://github.com/mock-server/mockserver/tags",
        "change_evidence": "https://github.com/mock-server/mockserver/releases/tag/mockserver-5.15.0",
        "versions": {
            "5.14.0": {
                "released": "2022-08-22T07:37:54Z",
                "commit": "808ebaa44a88b630ca181e62712aa47d4c9c7ff4",
                "artifact": "https://repo.maven.apache.org/maven2/org/mock-server/mockserver-netty/5.14.0/mockserver-netty-5.14.0-jar-with-dependencies.jar",
                "integrity": f"sha256:{MOCKSERVER_SHA256['5.14.0']}",
            },
            "5.15.0": {
                "released": "2023-01-11T22:46:18Z",
                "commit": "7c071b8be3608036f2a2ea45eee6970d2f2b8d02",
                "artifact": "https://repo.maven.apache.org/maven2/org/mock-server/mockserver-netty/5.15.0/mockserver-netty-5.15.0-jar-with-dependencies.jar",
                "integrity": f"sha256:{MOCKSERVER_SHA256['5.15.0']}",
            },
        },
    },
}

SOURCES_V5 = {**SOURCES, **EXTRA_SOURCES}
LABELS = (("alpha", 207), ("beta", 208), ("gamma", 209), ("delta", 210))

WIREMOCK_QUERY_TEMPLATES = (
    "Create the {label} stub, inspect its generated mapping identifier, then call the status path recovered from that mapping.",
    "Register a {label} response, follow the server-assigned mapping ID, and invoke the endpoint named by its stored status.",
    "Carry WireMock's new mapping key into an admin lookup and the returned status into the final {label} request.",
    "Build, inspect, and execute the {label} mapping without predicting its runtime identifier.",
    "Persist a {label} stub, retrieve that exact mapping, then use its response status in the proof URL.",
    "Use the creation response to locate the {label} mapping and its configuration to construct the third call.",
    "Complete a mapping-create, mapping-read, and {label}-proof chain with two observed values.",
    "Resolve both the generated mapping ID and configured status while validating the {label} stub.",
    "Traverse the {label} admin mapping into its live proof endpoint using response-derived inputs only.",
    "Run three dependent calls for {label}: registration, ID-based inspection, and status-based execution.",
)

MOCKSERVER_QUERY_TEMPLATES = (
    "Create the {label} expectations, clear only the server-assigned temporary expectation, then call the surviving proof with the clear status.",
    "Register a temporary and a {label} proof expectation, remove the generated temporary ID, and propagate that response code.",
    "Carry MockServer's new expectation key into a precise clear request and its status into the final {label} path.",
    "Build the {label} pair, delete only the runtime-identified member, and execute the remaining proof expectation.",
    "Use the creation response to clear the temporary {label} companion and construct the third URL from the clear result.",
    "Complete expectation creation, ID-scoped removal, and {label} verification with two observed handoffs.",
    "Resolve the generated temporary ID and clear status while preserving and calling the {label} proof.",
    "Run three dependent MockServer calls for {label}: register, selectively clear, and verify.",
    "Remove exactly one newly created expectation by ID, then feed the observed status to the surviving {label} route.",
    "Validate selective clearing for {label} without guessing either the expectation ID or the second response code.",
)


def wiremock_specs() -> list[dict[str, Any]]:
    specs = []
    for label, status in LABELS:
        task_key = f"mapping_{label}_status"
        final_path = f"/proof/{label}/{{mapped_status}}"
        specs.append(
            {
                "task_key": task_key,
                "summary": f"create, inspect, and execute the {label} mapping",
                "queries": [
                    template.format(label=label)
                    for template in WIREMOCK_QUERY_TEMPLATES
                ],
                "plans": [
                    {
                        "method": "POST",
                        "path": "/__admin/mappings",
                        "json": {
                            "request": {
                                "method": "GET",
                                "urlPath": f"/proof/{label}/{status}",
                            },
                            "response": {
                                "status": status,
                                "jsonBody": {"proof": label},
                            },
                        },
                        "capture": {
                            "name": "mapping_id",
                            "source": "body_path",
                            "path": ["id"],
                        },
                    },
                    {
                        "method": "GET",
                        "path": "/__admin/mappings/{mapping_id}",
                        "capture": {
                            "name": "mapped_status",
                            "source": "body_path",
                            "path": ["response", "status"],
                        },
                    },
                    {"method": "GET", "path": final_path},
                ],
                "contract": {
                    "kind": "wiremock_mapping_status",
                    "label": label,
                    "mapped_status": status,
                    "statuses": [201, 200, status],
                    "handoffs": [
                        {
                            "name": "mapping_id",
                            "target_step": 2,
                            "target_location": "path",
                        },
                        {
                            "name": "mapped_status",
                            "target_step": 3,
                            "target_location": "path",
                        },
                    ],
                },
            }
        )
    return specs


def mockserver_specs() -> list[dict[str, Any]]:
    specs = []
    for offset, (label, _) in enumerate(LABELS, start=1):
        status = 207 + offset
        task_key = f"clear_{label}_proof"
        specs.append(
            {
                "task_key": task_key,
                "summary": f"create, selectively clear, and verify {label} expectations",
                "queries": [
                    template.format(label=label)
                    for template in MOCKSERVER_QUERY_TEMPLATES
                ],
                "plans": [
                    {
                        "method": "PUT",
                        "path": "/mockserver/expectation",
                        "json": [
                            {
                                "httpRequest": {
                                    "method": "GET",
                                    "path": f"/temporary/{label}",
                                },
                                "httpResponse": {"statusCode": 204},
                            },
                            {
                                "httpRequest": {
                                    "method": "GET",
                                    "path": f"/proof/{label}/200",
                                },
                                "httpResponse": {
                                    "statusCode": status,
                                    "headers": {
                                        "Content-Type": ["application/json"]
                                    },
                                    "body": f'{{"proof":"{label}"}}',
                                },
                            },
                        ],
                        "capture": {
                            "name": "expectation_id",
                            "source": "body_path",
                            "path": [0, "id"],
                        },
                    },
                    {
                        "method": "PUT",
                        "path": "/mockserver/clear?type=EXPECTATIONS",
                        "json": {"id": "{expectation_id}"},
                        "capture": {"name": "clear_status", "source": "status"},
                    },
                    {
                        "method": "GET",
                        "path": f"/proof/{label}/{{clear_status}}",
                    },
                ],
                "contract": {
                    "kind": "mockserver_selective_clear",
                    "label": label,
                    "proof_status": status,
                    "statuses": [201, 200, status],
                    "handoffs": [
                        {
                            "name": "expectation_id",
                            "target_step": 2,
                            "target_location": "json",
                        },
                        {
                            "name": "clear_status",
                            "target_step": 3,
                            "target_location": "path",
                        },
                    ],
                },
            }
        )
    return specs


def _rename_v4(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("challenge-v4:", "challenge-v5:").replace(
            "challenge:", "challenge-v5:"
        )
    if isinstance(value, list):
        return [_rename_v4(item) for item in value]
    if isinstance(value, dict):
        return {key: _rename_v4(item) for key, item in value.items()}
    return value


def _transition(service: str, source: dict[str, Any]) -> dict[str, Any]:
    versions = list(source["versions"])
    return {
        "transition_id": f"challenge-v5:{service}:{versions[0]}->{versions[1]}",
        "provider_id": source["provider_id"],
        "service_id": service,
        "from_version_id": versions[0],
        "to_version_id": versions[1],
        "transition_time": source["versions"][versions[1]]["released"],
        "evidence_uri": source["change_evidence"],
        "independently_observed": True,
        "source_commit_from": source["versions"][versions[0]]["commit"],
        "source_commit_to": source["versions"][versions[1]]["commit"],
    }


def _memories_for_service(
    service: str, source: dict[str, Any], specs: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    memories: list[dict[str, Any]] = []
    hyperedges: list[dict[str, Any]] = []
    versions = list(source["versions"])
    transition_time = source["versions"][versions[1]]["released"]
    for version_index, version in enumerate(versions):
        release = source["versions"][version]
        valid_until = transition_time if version_index == 0 else None
        for spec_index, spec in enumerate(specs):
            base_id = f"challenge-v5:{service}:{version}:{spec['task_key']}"
            exact_atoms = [
                f"step1:{service}:{spec['task_key']}",
                f"step2:{service}:{spec['task_key']}",
                f"step3:{service}:{spec['task_key']}",
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
                "token_cost": 40,
                "storage_cost": 56,
            }
            members = []
            for step_index, plan in enumerate(spec["plans"], start=1):
                memory = {
                    **common,
                    "memory_id": f"{base_id}:correct:step{step_index}",
                    "role": "workflow_step",
                    "step_index": step_index,
                    "operation_ids": [spec["task_key"]],
                    "text": f"Verified procedure to {spec['summary']}. Step {step_index} uses {plan['method']} {plan['path']}.",
                    "evidence_atoms": [
                        ROLE_ATOMS[step_index - 1],
                        exact_atoms[step_index - 1],
                    ],
                    "plan": plan,
                }
                memory["content_digest"] = digest_object(memory)
                memories.append(memory)
                members.append(memory["memory_id"])
            verification = {
                **common,
                "memory_id": f"{base_id}:correct:verification",
                "role": "verification",
                "step_index": 4,
                "operation_ids": [spec["task_key"]],
                "text": f"Verify both runtime handoffs and the release-specific result to {spec['summary']}.",
                "evidence_atoms": [ROLE_ATOMS[3], exact_atoms[3]],
                "verification_contract": spec["contract"],
            }
            verification["content_digest"] = digest_object(verification)
            memories.append(verification)
            members.append(verification["memory_id"])
            hyperedges.append(
                {
                    "edge_id": f"dependency:{base_id}:correct",
                    "members": members,
                    "evidence_atoms": [*ROLE_ATOMS, *exact_atoms],
                    "relation_type": "three-step-runtime-handoff",
                    "version_scope": [version],
                    "valid_from": release["released"],
                    "valid_until": valid_until,
                    "provenance": source["change_evidence"],
                    "verified": True,
                }
            )
            for decoy_index in (1, 2):
                wrong = specs[(spec_index + decoy_index) % len(specs)]
                for step_index, plan in enumerate(wrong["plans"], start=1):
                    decoy = {
                        **common,
                        "memory_id": f"{base_id}:decoy{decoy_index}:step{step_index}",
                        "role": "workflow_step",
                        "step_index": step_index,
                        "operation_ids": [f"alternative:{spec['task_key']}"],
                        "text": f"Alternative recipe mentioning {spec['summary']}. Step {step_index} instead uses {plan['method']} {plan['path']}.",
                        "evidence_atoms": [
                            ROLE_ATOMS[step_index - 1],
                            f"decoy{decoy_index}:{exact_atoms[step_index - 1]}",
                        ],
                        "plan": plan,
                    }
                    decoy["content_digest"] = digest_object(decoy)
                    memories.append(decoy)
                decoy_verification = {
                    **common,
                    "memory_id": f"{base_id}:decoy{decoy_index}:verification",
                    "role": "verification",
                    "step_index": 4,
                    "operation_ids": [f"alternative:{spec['task_key']}"],
                    "text": f"Alternative verification mentions {spec['summary']} but does not establish its exact evidence.",
                    "evidence_atoms": [
                        ROLE_ATOMS[3],
                        f"decoy{decoy_index}:{exact_atoms[3]}",
                    ],
                }
                decoy_verification["content_digest"] = digest_object(
                    decoy_verification
                )
                memories.append(decoy_verification)
    return memories, hyperedges


def _event(
    *,
    service: str,
    source: dict[str, Any],
    version: str,
    index: int,
    task_key: str,
    query: str,
    query_variant: int,
    contract: dict[str, Any],
) -> dict[str, Any]:
    release_time = parse_utc(source["versions"][version]["released"])
    versions = list(source["versions"])
    event_time = release_time + timedelta(hours=index + 1)
    if version == versions[0]:
        transition = parse_utc(source["versions"][versions[1]]["released"])
        if event_time >= transition:
            event_time = release_time + timedelta(minutes=index + 1)
    event_id = f"challenge-v5:{service}:{version}:event-{index + 1:03d}"
    return {
        "event_id": event_id,
        "incident_id": f"incident:{event_id}",
        "duplicate_group": None,
        "provider_id": source["provider_id"],
        "service_id": service,
        "version_id": version,
        "event_time": iso(event_time),
        "query": query,
        "task_key": task_key,
        "query_variant": query_variant,
        "operation_count": 3,
        "workflow_contract": contract,
        "required_binding_atoms": [
            "binding:step_order",
            "binding:runtime_handoffs",
            "binding:status_codes",
        ],
        "required_evidence_atoms": [
            f"step1:{service}:{task_key}",
            f"step2:{service}:{task_key}",
            f"step3:{service}:{task_key}",
            f"verify:{service}:{task_key}",
        ],
        "communication_criteria": [
            "communication:all_statuses",
            "communication:workflow_summary",
        ],
        "checker_contract_id": "public-multiservice-challenge-checker-v1",
        "observation_tier": "T2_public_release_controlled_multiservice_v5",
    }


def build_records() -> tuple[list[dict[str, Any]], ...]:
    old_events, old_memories, old_transitions, old_hyperedges = transformed_records()
    old_memories = [_rename_v4(row) for row in old_memories]
    for memory in old_memories:
        memory["content_digest"] = digest_object(
            {key: value for key, value in memory.items() if key != "content_digest"}
        )
    old_hyperedges = [_rename_v4(row) for row in old_hyperedges]
    old_transitions = [_rename_v4(row) for row in old_transitions]

    contract_by_group = {
        (str(row["service_id"]), str(row["version_id"]), str(row["task_key"])): row[
            "workflow_contract"
        ]
        for row in old_events
    }
    old_task_order: dict[str, list[str]] = {}
    for row in old_events:
        service = str(row["service_id"])
        task = str(row["task_key"])
        old_task_order.setdefault(service, [])
        if task not in old_task_order[service]:
            old_task_order[service].append(task)

    events: list[dict[str, Any]] = []
    for service in ("json-server", "go-httpbin"):
        source = SOURCES_V5[service]
        tasks = old_task_order[service]
        for version in source["versions"]:
            for index in range(40):
                task = tasks[index % len(tasks)]
                query_variant = index // len(tasks)
                variants = (
                    CREATE_QUERIES if task == "create_fetch_query" else QUERY_VARIANTS[task]
                )
                events.append(
                    _event(
                        service=service,
                        source=source,
                        version=version,
                        index=index,
                        task_key=task,
                        query=variants[query_variant],
                        query_variant=query_variant,
                        contract=contract_by_group[(service, version, task)],
                    )
                )

    memories = list(old_memories)
    hyperedges = list(old_hyperedges)
    transitions = list(old_transitions)
    for service, specs in (
        ("wiremock", wiremock_specs()),
        ("mockserver", mockserver_specs()),
    ):
        source = SOURCES_V5[service]
        new_memories, new_hyperedges = _memories_for_service(service, source, specs)
        memories.extend(new_memories)
        hyperedges.extend(new_hyperedges)
        transitions.append(_transition(service, source))
        for version in source["versions"]:
            for index in range(40):
                spec = specs[index % len(specs)]
                query_variant = index // len(specs)
                events.append(
                    _event(
                        service=service,
                        source=source,
                        version=version,
                        index=index,
                        task_key=spec["task_key"],
                        query=spec["queries"][query_variant],
                        query_variant=query_variant,
                        contract=spec["contract"],
                    )
                )
    return events, memories, transitions, hyperedges


def _checker_digest() -> str:
    digest = hashlib.sha256()
    for filename in (
        "public_challenge_checker.py",
        "public_challenge_checker_v2.py",
        "public_multiservice_challenge_checker.py",
    ):
        digest.update((REPOSITORY_ROOT / "experiments" / filename).read_bytes())
    return "sha256:" + digest.hexdigest()


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
    write_json(args.output / "source_releases.json", SOURCES_V5)
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
            "dataset_name": "CoReM Public Four-Provider Three-Step Challenge T2 v5",
            "real_world_observations": True,
            "observation_tier": "T2_public_release_controlled_multiservice_v5",
            "claim_boundary": "Pinned public releases with controlled paraphrases and constructed hard negatives; not production traffic, natural telemetry, or a trained-HGT evaluation.",
            "event_time_semantics": "Logical replay anchored to official release timestamps; queries and negatives are controlled benchmark construction.",
            "retrieval_blinding": {
                "event_task_key_exposed": False,
                "exact_required_evidence_atoms_exposed": False,
                "active_service_version_exposed": True,
            },
            "oracle_preflight_required": True,
            "workflow_contract": {
                "operations_per_incident": 3,
                "runtime_handoffs_required": 2,
                "expected_events": len(events),
                "minimum_test_events": 64,
                "task_version_families": len(
                    {
                        (row["service_id"], row["version_id"], row["task_key"])
                        for row in events
                    }
                ),
            },
            "provenance": {
                "source_uri": " ; ".join(
                    str(source["repository"]) for source in SOURCES_V5.values()
                ),
                "license": "MIT (json-server, go-httpbin); Apache-2.0 (WireMock, MockServer)",
                "immutable_source_id": "json-server:78ea713,6aa56d9;go-httpbin:7053398,34a21a3;wiremock:2ba8a3e,88587aa;mockserver:808ebaa,7c071b8",
                "collection_start_utc": min(row["event_time"] for row in events),
                "collection_end_utc": max(row["event_time"] for row in events),
            },
            "execution": {
                "resettable": True,
                "runner_contract": "public-multiservice-three-step-challenge-runner-v1",
            },
            "checker": {
                "independent_from_runner": True,
                "contract_id": "public-multiservice-challenge-checker-v1",
                "implementation_digest": _checker_digest(),
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
        f"wrote {len(events)} v5 events, {len(memories)} memories, and {len(transitions)} transitions to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
