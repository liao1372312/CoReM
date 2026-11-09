#!/usr/bin/env python3
"""Build v4 of the three-step challenge after an oracle-feasibility repair."""

from __future__ import annotations

import argparse
import hashlib
import sys
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
from scripts.build_public_challenge_dataset import build_records, digest_object
from scripts.build_public_versioned_service_dataset import SOURCES

OLD_TASK = "create_update_fetch"
NEW_TASK = "create_fetch_query"
OLD_SUMMARY = (
    "create a Draft post, update it to Echo, and fetch the same runtime resource"
)
NEW_SUMMARY = (
    "create an Echo post, fetch it, and use the observed title in a final query"
)
CREATE_QUERIES = [
    "Create an Echo record, follow its server-assigned key, and use the fetched title for a final collection lookup.",
    "Insert a seven-view Echo post, retrieve that runtime resource, then query posts by the observed title.",
    "Carry a generated post ID into a detail call and the detail title into a filtered search.",
    "Persist Echo, read the exact new object, and confirm it through a title-based collection request.",
    "Run create, fetch, and title-query operations without predicting the identifier or search term.",
    "Let the service allocate an Echo post key, inspect that post, and search using its returned title.",
    "Build a three-call create-to-detail-to-query chain for an Echo post with seven views.",
    "Use the creation response for the second path and the fetched title for the third request.",
    "Derive both the resource key and title at runtime while creating and verifying an Echo record.",
    "Complete POST, detail GET, and filtered GET calls whose later inputs come from earlier responses.",
]


def _rename(value: Any) -> Any:
    if isinstance(value, str):
        return (
            value.replace("challenge:", "challenge-v4:")
            .replace(OLD_TASK, NEW_TASK)
            .replace(OLD_SUMMARY, NEW_SUMMARY)
        )
    if isinstance(value, list):
        return [_rename(item) for item in value]
    if isinstance(value, dict):
        return {key: _rename(item) for key, item in value.items()}
    return value


def _repair_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if (
        plan.get("method") == "POST"
        and plan.get("path") == "/posts"
        and isinstance(plan.get("json"), dict)
        and plan["json"].get("title") == "Draft"
    ):
        repaired = dict(plan)
        repaired["json"] = {**plan["json"], "title": "Echo"}
        return repaired
    if plan.get("method") == "PATCH" and plan.get("path") == "/posts/{created_id}":
        return {
            "method": "GET",
            "path": "/posts/{created_id}",
            "capture": {
                "name": "post_title",
                "source": "body_path",
                "path": ["title"],
            },
        }
    if plan.get("method") == "GET" and plan.get("path") == "/posts/{updated_id}":
        return {"method": "GET", "path": "/posts?title={post_title}"}
    return plan


def _contract(id_type: str) -> dict[str, Any]:
    return {
        "kind": "json_create_fetch_query",
        "statuses": [201, 200, 200],
        "handoffs": [
            {"name": "created_id", "target_step": 2},
            {"name": "post_title", "target_step": 3},
        ],
        "id_type": id_type,
    }


def transformed_records() -> tuple[list[dict[str, Any]], ...]:
    original = build_records()
    events, memories, transitions, hyperedges = tuple(
        [_rename(row) for row in rows] for rows in original
    )
    for memory in memories:
        plan = memory.get("plan")
        if isinstance(plan, dict):
            memory["plan"] = _repair_plan(plan)
        verification = memory.get("verification_contract")
        if (
            isinstance(verification, dict)
            and verification.get("kind") == f"json_{NEW_TASK}"
        ):
            memory["verification_contract"] = _contract(str(verification["id_type"]))
        memory["content_digest"] = digest_object(
            {key: value for key, value in memory.items() if key != "content_digest"}
        )
    for event in events:
        event["checker_contract_id"] = "public-versioned-challenge-checker-v2"
        event["observation_tier"] = "T2_public_release_controlled_challenge_v4"
        if event["task_key"] == NEW_TASK:
            event["query"] = CREATE_QUERIES[int(event["query_variant"])]
            event["workflow_contract"] = _contract(
                "string" if event["version_id"] == "1.0.0-beta.3" else "number"
            )
    return events, memories, transitions, hyperedges


def _checker_digest() -> str:
    digest = hashlib.sha256()
    for filename in ("public_challenge_checker.py", "public_challenge_checker_v2.py"):
        digest.update((REPOSITORY_ROOT / "experiments" / filename).read_bytes())
    return "sha256:" + digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    events, memories, transitions, hyperedges = transformed_records()
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "events.jsonl", events)
    write_jsonl(args.output / "memories.jsonl", memories)
    write_jsonl(args.output / "version_transitions.jsonl", transitions)
    write_jsonl(args.output / "dependency_hyperedges.jsonl", hyperedges)
    write_json(args.output / "source_releases.json", SOURCES)
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
            "dataset_name": "CoReM Public Three-Step Query-Blind Challenge T2 v4",
            "real_world_observations": True,
            "observation_tier": "T2_public_release_controlled_challenge_v4",
            "claim_boundary": "Genuine pinned releases with controlled paraphrases and hard negatives; v4 is a post-v3 exploratory repair, not production traffic.",
            "event_time_semantics": "Logical replay anchored to official releases; queries and negatives are controlled benchmark construction.",
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
                "contract_id": "public-versioned-challenge-checker-v2",
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
        f"wrote {len(events)} oracle-gated events, {len(memories)} memories, and {len(transitions)} transitions to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
