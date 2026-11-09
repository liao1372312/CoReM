#!/usr/bin/env python3
"""Build the query-blind three-step public release challenge dataset."""

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

ROLE_ATOMS = ("role:step1", "role:step2", "role:step3", "role:verify")


def digest_object(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


QUERY_VARIANTS = {
    "paginate_title_query": [
        "Start from a short page of posts, follow the first identifier, then locate the record again using its observed title.",
        "List two posts, inspect the leading item, and use the title returned by that lookup for a final filtered search.",
        "Traverse page to post to title-filtered collection while preserving the identifier and title handoffs.",
        "Read the first entry from a two-result page, fetch that entry, and confirm it through a title query.",
        "Resolve a paginated post into its detail response and finish by searching for the detail response's title.",
        "Use a two-item listing to choose a post, observe its title from the detail endpoint, and query that title.",
        "Chain a compact post page into a detail call and a final title-based collection lookup.",
        "Take the first ID exposed by pagination, retrieve its title, and prove the title maps back to the same post.",
        "Without guessing values, derive a post key from a two-row page and a search term from the fetched post.",
        "Perform page, detail, and filtered-list calls; each later request must consume the value returned just before it.",
    ],
    "create_update_fetch": [
        "Create a draft record, use its returned key to rename it Echo, and retrieve the updated object.",
        "Persist a Draft post with seven views, patch the new resource to Echo, then fetch it by the patched identifier.",
        "Carry a server-assigned post ID through creation, update, and final read-back.",
        "Insert a draft, change only its title after observing the ID, and confirm the stored Echo record.",
        "Run a create-to-patch-to-fetch chain for a seven-view post, using runtime identifiers throughout.",
        "Obtain an ID by posting a Draft, update that exact resource to Echo, and read the result.",
        "Build and verify an Echo record by chaining POST, PATCH, and GET rather than assuming its key.",
        "Let the service allocate a post key, propagate it through a title revision, and verify persistence.",
        "Create a seven-view placeholder, transform its title to Echo, and retrieve the same server-created resource.",
        "Complete an insert, mutation, and read-back sequence where both later paths depend on observed IDs.",
    ],
    "relationship_title_query": [
        "Resolve comment one to its embedded post, fetch that post, and use the returned title in a final search.",
        "Follow the comment-to-post relationship, inspect the related post, then locate it through its observed title.",
        "Traverse comment, related post, and title-filtered collection without supplying the post ID or title in advance.",
        "Expand the first comment's parent, read the parent resource, and confirm it with a title query.",
        "Use relationship expansion to discover a post key and its detail response to discover the final search term.",
        "Begin at comment 1, carry the linked post identifier into a detail request, then query the resulting title.",
        "Chain a relationship lookup into a post fetch and a title-based collection check.",
        "Discover the parent post from the comment response and prove the same record is returned by title filtering.",
        "Derive both post ID and title at runtime while moving from comment to post to search results.",
        "Execute three dependent reads: relationship resolution, post detail, and filtered verification.",
    ],
    "update_views_query": [
        "Rename post one to updated, read its view count, and use that count in a final collection query.",
        "Patch the first post, fetch the changed resource, then filter posts by the views observed from that fetch.",
        "Carry the updated post key into a detail call and its returned view total into a search.",
        "Mutate post 1's title, inspect the persisted object, and verify it via a runtime view-count filter.",
        "Run update, read-back, and views-query steps with both later inputs obtained from responses.",
        "Change the title of post one, obtain its views from the detail endpoint, and query that numeric value.",
        "Confirm an update through a three-call chain ending in a collection filtered by observed views.",
        "After patching the known post, propagate its ID and then its view count through two dependent requests.",
        "Use the update response to choose a resource and the resource response to construct a views filter.",
        "Perform mutation, detail verification, and numeric filtering without hard-coding the intermediate view value.",
    ],
    "uuid_status_chain": [
        "Request a server UUID, reflect it through an arbitrary path, and report the reflected call's status in a third path.",
        "Generate a UUID, pass that exact value to an echo endpoint, then propagate the echo status.",
        "Chain identifier generation, path reflection, and status reflection using only observed runtime values.",
        "Obtain a random service identifier, verify its reflected URL, and encode the second response status in the last request.",
        "Complete UUID-to-echo-to-status routing without predicting either handoff.",
        "Use the UUID response as the next path segment and the resulting HTTP code as the final segment.",
        "Build a three-call trace from generated identifier through echo response to status-labelled endpoint.",
        "Carry a fresh UUID into an anything route, then carry that route's status into another request.",
        "Demonstrate two runtime handoffs: UUID to path and reflected status to final path.",
        "Execute generation, reflection, and status confirmation while preserving the server-observed UUID.",
    ],
    "base64_status_chain": [
        "Decode the supplied payload, reflect the decoded text, and pass the reflection status to a final endpoint.",
        "Turn the encoded value into text, use that text as a path segment, then propagate the second call's status.",
        "Chain base64 decoding, arbitrary-path reflection, and HTTP-status reflection.",
        "Observe the decoded CoReM value, verify it in an echo URL, and encode the echo status in the last request.",
        "Complete a decode-to-echo-to-status workflow using response-derived values only.",
        "Feed the base64 response text into an anything route and its status into a third route.",
        "Run three dependent calls beginning with payload decoding and ending with a status-labelled path.",
        "Use decoded text for the second request and the second response code for the third.",
        "Demonstrate text and status handoffs after decoding the service payload.",
        "Recover CoReM from base64, reflect it, and confirm the reflection through a final status path.",
    ],
    "header_status_chain": [
        "Read the release-proof response header, reflect its value, and route the reflection status into a third request.",
        "Obtain the custom header token, use it as an arbitrary path segment, then propagate the HTTP code.",
        "Chain response-header extraction, token reflection, and status reflection.",
        "Discover release-proof from the service response, verify it in the next URL, and encode that call's status.",
        "Complete header-to-echo-to-status routing without inserting an unobserved token.",
        "Pass the returned Corem header value into an anything endpoint and its status into another endpoint.",
        "Execute three calls whose handoffs are response header then response code.",
        "Use the response-provided release token for the second path and the second status for the final path.",
        "Demonstrate a header-derived path followed by a status-derived path.",
        "Resolve, reflect, and confirm the release-proof token across a three-operation chain.",
    ],
    "trailer_status_chain": [
        "Probe trailer availability, reflect the observed status, and use the reflection status in a final path.",
        "Call the trailers endpoint, carry its release-specific HTTP code into an echo, then propagate the echo code.",
        "Chain availability probing with two status-derived requests.",
        "Observe whether trailers are available, verify that code in the next URL, and finish with the next status.",
        "Complete a trailer-to-status-to-final workflow without assuming release behavior.",
        "Use the trailer response code for an anything path and the second response code for a final path.",
        "Execute three dependent calls that expose and propagate release-specific trailer availability.",
        "Carry the first HTTP status into the second request and the second status into the third.",
        "Demonstrate two status handoffs starting from the version-sensitive trailers route.",
        "Probe, reflect, and finalize trailer availability through a three-operation chain.",
    ],
}


def challenge_specs(service: str, version: str) -> list[dict[str, Any]]:
    if service == "json-server":
        modern = version == "1.0.0-beta.3"
        id_type = "string" if modern else "number"
        page_path = f"/posts?_page=1&{'_per_page' if modern else '_limit'}=2"
        page_capture = ["data", 0, "id"] if modern else [0, "id"]
        relation_path = (
            "/comments/1?_embed=post" if modern else "/comments/1?_expand=post"
        )
        return [
            {
                "task_key": "paginate_title_query",
                "summary": "derive a post and title from pagination before a filtered verification",
                "plans": [
                    {
                        "method": "GET",
                        "path": page_path,
                        "capture": {
                            "name": "post_id",
                            "source": "body_path",
                            "path": page_capture,
                        },
                    },
                    {
                        "method": "GET",
                        "path": "/posts/{post_id}",
                        "capture": {
                            "name": "post_title",
                            "source": "body_path",
                            "path": ["title"],
                        },
                    },
                    {"method": "GET", "path": "/posts?title={post_title}"},
                ],
                "contract": {
                    "kind": "json_paginate_title_query",
                    "statuses": [200, 200, 200],
                    "handoffs": [
                        {"name": "post_id", "target_step": 2},
                        {"name": "post_title", "target_step": 3},
                    ],
                    "id_type": id_type,
                },
            },
            {
                "task_key": "create_update_fetch",
                "summary": "create a Draft post, update it to Echo, and fetch the same runtime resource",
                "plans": [
                    {
                        "method": "POST",
                        "path": "/posts",
                        "json": {"title": "Draft", "views": 7},
                        "capture": {
                            "name": "created_id",
                            "source": "body_path",
                            "path": ["id"],
                        },
                    },
                    {
                        "method": "PATCH",
                        "path": "/posts/{created_id}",
                        "json": {"title": "Echo"},
                        "capture": {
                            "name": "updated_id",
                            "source": "body_path",
                            "path": ["id"],
                        },
                    },
                    {"method": "GET", "path": "/posts/{updated_id}"},
                ],
                "contract": {
                    "kind": "json_create_update_fetch",
                    "statuses": [201, 200, 200],
                    "handoffs": [
                        {"name": "created_id", "target_step": 2},
                        {"name": "updated_id", "target_step": 3},
                    ],
                    "id_type": id_type,
                },
            },
            {
                "task_key": "relationship_title_query",
                "summary": "follow comment one to its post and use the fetched title for verification",
                "plans": [
                    {
                        "method": "GET",
                        "path": relation_path,
                        "capture": {
                            "name": "post_id",
                            "source": "body_path",
                            "path": ["post", "id"],
                        },
                    },
                    {
                        "method": "GET",
                        "path": "/posts/{post_id}",
                        "capture": {
                            "name": "post_title",
                            "source": "body_path",
                            "path": ["title"],
                        },
                    },
                    {"method": "GET", "path": "/posts?title={post_title}"},
                ],
                "contract": {
                    "kind": "json_relationship_title_query",
                    "statuses": [200, 200, 200],
                    "handoffs": [
                        {"name": "post_id", "target_step": 2},
                        {"name": "post_title", "target_step": 3},
                    ],
                    "id_type": id_type,
                },
            },
            {
                "task_key": "update_views_query",
                "summary": "update post one and use its fetched view count in a final collection query",
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
                    {
                        "method": "GET",
                        "path": "/posts/{post_id}",
                        "capture": {
                            "name": "views",
                            "source": "body_path",
                            "path": ["views"],
                        },
                    },
                    {"method": "GET", "path": "/posts?views={views}"},
                ],
                "contract": {
                    "kind": "json_update_views_query",
                    "statuses": [200, 200, 200],
                    "handoffs": [
                        {"name": "post_id", "target_step": 2},
                        {"name": "views", "target_step": 3},
                    ],
                    "id_type": id_type,
                },
            },
        ]

    trailers_status = 200 if version == "v2.15.0" else 404
    return [
        {
            "task_key": "uuid_status_chain",
            "summary": "obtain a UUID, reflect it, and propagate the reflection status",
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
                {
                    "method": "GET",
                    "path": "/anything/{uuid}",
                    "capture": {"name": "reflected_status", "source": "status"},
                },
                {"method": "GET", "path": "/anything/status-{reflected_status}"},
            ],
            "contract": {
                "kind": "httpbin_uuid_status_chain",
                "statuses": [200, 200, 200],
                "handoffs": [
                    {"name": "uuid", "target_step": 2},
                    {"name": "reflected_status", "target_step": 3},
                ],
            },
        },
        {
            "task_key": "base64_status_chain",
            "summary": "decode CoReM, reflect the text, and propagate the reflection status",
            "plans": [
                {
                    "method": "GET",
                    "path": "/base64/Q29SZU0=",
                    "capture": {"name": "decoded", "source": "text"},
                },
                {
                    "method": "GET",
                    "path": "/anything/{decoded}",
                    "capture": {"name": "reflected_status", "source": "status"},
                },
                {"method": "GET", "path": "/anything/status-{reflected_status}"},
            ],
            "contract": {
                "kind": "httpbin_base64_status_chain",
                "statuses": [200, 200, 200],
                "handoffs": [
                    {"name": "decoded", "target_step": 2},
                    {"name": "reflected_status", "target_step": 3},
                ],
            },
        },
        {
            "task_key": "header_status_chain",
            "summary": "extract release-proof from a response header, reflect it, and propagate the status",
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
                {
                    "method": "GET",
                    "path": "/anything/{release_token}",
                    "capture": {"name": "reflected_status", "source": "status"},
                },
                {"method": "GET", "path": "/anything/status-{reflected_status}"},
            ],
            "contract": {
                "kind": "httpbin_header_status_chain",
                "statuses": [200, 200, 200],
                "handoffs": [
                    {"name": "release_token", "target_step": 2},
                    {"name": "reflected_status", "target_step": 3},
                ],
            },
        },
        {
            "task_key": "trailer_status_chain",
            "summary": "probe trailer availability and propagate two observed statuses",
            "plans": [
                {
                    "method": "GET",
                    "path": "/trailers",
                    "capture": {"name": "availability_status", "source": "status"},
                },
                {
                    "method": "GET",
                    "path": "/anything/status-{availability_status}",
                    "capture": {"name": "reflected_status", "source": "status"},
                },
                {"method": "GET", "path": "/anything/final-{reflected_status}"},
            ],
            "contract": {
                "kind": "httpbin_trailer_status_chain",
                "statuses": [trailers_status, 200, 200],
                "handoffs": [
                    {"name": "availability_status", "target_step": 2},
                    {"name": "reflected_status", "target_step": 3},
                ],
                "trailers_status": trailers_status,
            },
        },
    ]


def build_records() -> tuple[list[dict[str, Any]], ...]:
    events: list[dict[str, Any]] = []
    memories: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    hyperedges: list[dict[str, Any]] = []
    for service, source in SOURCES.items():
        versions = list(source["versions"])
        transition_time = source["versions"][versions[1]]["released"]
        transitions.append(
            {
                "transition_id": f"challenge:{service}:{versions[0]}->{versions[1]}",
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
            specs = challenge_specs(service, version)
            for spec_index, spec in enumerate(specs):
                base_id = f"challenge:{service}:{version}:{spec['task_key']}"
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
                    "text": f"Verify all runtime handoffs and release-specific outcomes to {spec['summary']}.",
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
                        "text": f"Alternative verification notes mention {spec['summary']} but do not establish its exact evidence.",
                        "evidence_atoms": [
                            ROLE_ATOMS[3],
                            f"decoy{decoy_index}:{exact_atoms[3]}",
                        ],
                    }
                    decoy_verification["content_digest"] = digest_object(
                        decoy_verification
                    )
                    memories.append(decoy_verification)

            release_time = parse_utc(release["released"])
            for index in range(20):
                spec = specs[index % len(specs)]
                occurrence = index // len(specs)
                query_variant = version_index * 5 + occurrence
                event_time = release_time + timedelta(hours=6 * (index + 1))
                if valid_until is not None and event_time >= parse_utc(valid_until):
                    event_time = release_time + timedelta(minutes=10 * (index + 1))
                event_id = f"challenge:{service}:{version}:event-{index + 1:02d}"
                events.append(
                    {
                        "event_id": event_id,
                        "incident_id": f"incident:{event_id}",
                        "duplicate_group": None,
                        "provider_id": source["provider_id"],
                        "service_id": service,
                        "version_id": version,
                        "event_time": iso(event_time),
                        "query": QUERY_VARIANTS[spec["task_key"]][query_variant],
                        "task_key": spec["task_key"],
                        "query_variant": query_variant,
                        "operation_count": 3,
                        "workflow_contract": spec["contract"],
                        "required_binding_atoms": [
                            "binding:step_order",
                            "binding:runtime_handoffs",
                            "binding:status_codes",
                        ],
                        "required_evidence_atoms": [
                            f"step1:{service}:{spec['task_key']}",
                            f"step2:{service}:{spec['task_key']}",
                            f"step3:{service}:{spec['task_key']}",
                            f"verify:{service}:{spec['task_key']}",
                        ],
                        "communication_criteria": [
                            "communication:all_statuses",
                            "communication:workflow_summary",
                        ],
                        "checker_contract_id": "public-versioned-challenge-checker-v1",
                        "observation_tier": "T2_public_release_controlled_challenge",
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
    checker_path = REPOSITORY_ROOT / "experiments" / "public_challenge_checker.py"
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
            "dataset_name": "CoReM Public Three-Step Query-Blind Challenge T2",
            "real_world_observations": True,
            "observation_tier": "T2_public_release_controlled_challenge",
            "claim_boundary": "Genuine pinned releases with controlled paraphrases and constructed hard negatives; not production traffic or natural deployment telemetry.",
            "event_time_semantics": "Deterministic logical replay order anchored to official releases; query variants and negatives are controlled benchmark construction.",
            "retrieval_blinding": {
                "event_task_key_exposed": False,
                "exact_required_evidence_atoms_exposed": False,
                "active_service_version_exposed": True,
            },
            "workflow_contract": {
                "operations_per_incident": 3,
                "runtime_handoffs_required": 2,
                "expected_events": len(events),
            },
            "provenance": {
                "source_uri": "https://github.com/typicode/json-server ; https://github.com/mccutchen/go-httpbin",
                "license": "MIT (both upstream projects)",
                "immutable_source_id": "json-server:78ea713,6aa56d9;go-httpbin:7053398,34a21a3",
                "collection_start_utc": min(row["event_time"] for row in events),
                "collection_end_utc": max(row["event_time"] for row in events),
            },
            "execution": {
                "resettable": True,
                "runner_contract": "public-versioned-three-step-challenge-runner-v1",
            },
            "checker": {
                "independent_from_runner": True,
                "contract_id": "public-versioned-challenge-checker-v1",
                "implementation_digest": f"sha256:{sha256_file(checker_path)}",
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
        f"wrote {len(events)} three-step events, {len(memories)} memories, and {len(transitions)} transitions to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
