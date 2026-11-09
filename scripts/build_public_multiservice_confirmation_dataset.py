#!/usr/bin/env python3
"""Build a fresh-query confirmation set from the frozen v5 service releases."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import (  # noqa: E402
    SCHEMA_VERSION,
    load_json,
    load_jsonl,
    parse_utc,
    sha256_file,
    write_checksums,
    write_json,
    write_jsonl,
)

QUERY_FRAMES = (
    "{}",
    "Use only values returned by earlier calls when you {}",
    "Recover every dependent input at runtime and {}",
    "For a three-operation replay, {}",
)

JSON_SERVER_CORES = {
    "create_fetch_query": (
        "create an Echo post with seven views, fetch its assigned resource, and search by the fetched title.",
        "insert the requested post, follow the identifier returned by creation, and filter the collection using the later title.",
        "persist an Echo record, read back that exact object, and verify it through a title-based query.",
        "perform create, detail-read, and filtered-list calls while forwarding the generated key and observed title.",
        "add the seven-view Echo item, inspect the server-created record, and locate it again from its response title.",
    ),
    "paginate_title_query": (
        "list a two-item post page, open the first returned identifier, and filter posts by the detail title.",
        "take the leading resource from a short page, fetch it, and use its observed title in the final collection query.",
        "navigate from pagination to the first post and back to a title-filtered listing using response values.",
        "read two posts, resolve the first result's detail, and confirm that object through its returned title.",
        "derive a post key from a compact page, derive a search term from the detail, and run the final lookup.",
    ),
    "relationship_title_query": (
        "resolve comment one to its related post, fetch that post, and search the collection by its returned title.",
        "follow the first comment's parent identifier, inspect the parent, and use the observed title as a filter.",
        "move from comment relationship to post detail to title query without pre-filling either handoff.",
        "discover the linked post through comment one, retrieve it, and verify it with a response-derived title search.",
        "perform relationship resolution, parent detail, and filtered verification while forwarding both runtime values.",
    ),
    "update_views_query": (
        "rename post one to updated, read back the changed post, and filter the collection by its returned view count.",
        "patch the first post's title, fetch the resulting resource, and query posts using the observed views value.",
        "apply the title update, follow the response identifier to detail, and forward the detail view count into a search.",
        "mutate post one, verify the stored object, and construct the last filter from its runtime numeric field.",
        "execute update, detail-read, and views-filter calls without guessing the intermediate identifier or count.",
    ),
}

GO_HTTPBIN_CORES = {
    "base64_status_chain": (
        "decode the payload, reflect the returned text in an arbitrary path, and reflect that call's status in the last path.",
        "obtain the decoded value, pass it to an echo route, and use the echo response code for the final request.",
        "chain base64 decoding into path reflection and then into status reflection with two runtime handoffs.",
        "recover the service text, verify it through an anything endpoint, and encode the verification status in a third URL.",
        "run decode, echo, and status-labelled calls while forwarding the decoded body and second response code.",
    ),
    "header_status_chain": (
        "read the release-proof response header, reflect that token in a path, and pass the reflection status to the final path.",
        "obtain the custom header value, use it with an anything endpoint, and forward that endpoint's HTTP code.",
        "chain header extraction, token reflection, and status reflection without supplying the token in advance.",
        "discover the release marker from the first response, verify it in a second URL, and reuse the second status in a third.",
        "perform header-to-path and status-to-path handoffs across three dependent requests.",
    ),
    "trailer_status_chain": (
        "probe the trailers endpoint, reflect its observed status, and pass the reflection status to a final path.",
        "call the version-sensitive trailer route, use its HTTP code in an anything URL, and forward the next code once more.",
        "chain the trailer availability response into two successive status-derived requests.",
        "observe the trailer call's code, verify that value through path reflection, and finish with the verification code.",
        "execute a trailer probe followed by two dependent paths without assuming either response status.",
    ),
    "uuid_status_chain": (
        "request a server UUID, reflect the returned identifier in a path, and pass the reflection status to the final path.",
        "generate a UUID, use that exact runtime value with an anything endpoint, and forward the resulting HTTP code.",
        "chain identifier generation, identifier reflection, and status reflection without predicting either handoff.",
        "obtain a fresh service identifier, verify it in the next URL, and encode the verification status in a third URL.",
        "perform UUID-to-path and status-to-path handoffs across three dependent requests.",
    ),
}


def proof_cores(service: str, proof: str) -> tuple[str, ...]:
    if service == "wiremock":
        return (
            f"create the {proof} mapping, inspect it through the returned mapping ID, and call its proof path using the stored status.",
            f"register a {proof} stub, follow the server-assigned key to its configuration, and forward the response status into the live URL.",
            f"chain {proof} mapping creation, ID-based lookup, and status-derived proof execution.",
            f"persist the {proof} response, retrieve that exact mapping, and construct the last request from its configured status.",
            f"perform registration, admin inspection, and {proof} verification without guessing the mapping identifier or status.",
        )
    return (
        f"create the temporary and {proof} proof expectations, clear only the returned temporary ID, and call the survivor with the clear status.",
        f"register the {proof} pair, remove the server-identified temporary expectation, and forward that response code to the proof route.",
        f"chain {proof} expectation creation, identifier-scoped clearing, and status-derived verification.",
        f"build the {proof} expectations, delete only the runtime-selected member, and construct the final proof URL from the clear result.",
        f"perform {proof} registration, selective removal, and survivor execution without guessing the generated key or response code.",
    )


def query_variants(service: str, task_key: str) -> list[str]:
    if service == "json-server":
        cores = JSON_SERVER_CORES[task_key]
    elif service == "go-httpbin":
        cores = GO_HTTPBIN_CORES[task_key]
    else:
        proof = task_key.removeprefix("mapping_").removesuffix("_status")
        if service == "mockserver":
            proof = task_key.removeprefix("clear_").removesuffix("_proof")
        cores = proof_cores(service, proof)
    return [frame.format(core) for frame in QUERY_FRAMES for core in cores]


def iso(value: Any) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def checker_digest() -> str:
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
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "public_multiservice_challenge_v5",
    )
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2

    base_events = load_jsonl(args.base_dataset / "events.jsonl")
    memories = load_jsonl(args.base_dataset / "memories.jsonl")
    transitions = load_jsonl(args.base_dataset / "version_transitions.jsonl")
    hyperedges = load_jsonl(args.base_dataset / "dependency_hyperedges.jsonl")
    sources = load_json(args.base_dataset / "source_releases.json")
    base_manifest = load_json(args.base_dataset / "manifest.json")

    exemplars: dict[tuple[str, str, str], dict[str, Any]] = {}
    task_order: dict[tuple[str, str], list[str]] = defaultdict(list)
    old_queries = {str(row["query"]) for row in base_events}
    for event in base_events:
        group = (
            str(event["service_id"]),
            str(event["version_id"]),
            str(event["task_key"]),
        )
        exemplars.setdefault(group, event)
        service_version = group[:2]
        if group[2] not in task_order[service_version]:
            task_order[service_version].append(group[2])

    events = []
    for service, source in sources.items():
        for version_index, (version, metadata) in enumerate(source["versions"].items()):
            tasks = task_order[(service, version)]
            for index in range(80):
                task_key = tasks[index % len(tasks)]
                query_variant = index // len(tasks)
                query = query_variants(service, task_key)[query_variant]
                if query in old_queries:
                    raise ValueError(f"confirmation query duplicates v5: {query}")
                release_time = parse_utc(metadata["released"], "released")
                event_time = release_time + (
                    timedelta(minutes=index + 1)
                    if version_index == 0
                    else timedelta(hours=index + 1)
                )
                exemplar = exemplars[(service, version, task_key)]
                event_id = (
                    f"challenge-v6-confirm:{service}:{version}:event-{index + 1:03d}"
                )
                events.append(
                    {
                        **{
                            key: value
                            for key, value in exemplar.items()
                            if key
                            not in {
                                "event_id",
                                "incident_id",
                                "duplicate_group",
                                "event_time",
                                "query",
                                "query_variant",
                                "observation_tier",
                            }
                        },
                        "event_id": event_id,
                        "incident_id": f"incident:{event_id}",
                        "duplicate_group": None,
                        "event_time": iso(event_time),
                        "query": query,
                        "query_variant": query_variant,
                        "observation_tier": "T2_public_release_fresh_query_confirmation_v6",
                    }
                )

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "events.jsonl", events)
    write_jsonl(args.output / "memories.jsonl", memories)
    write_jsonl(args.output / "version_transitions.jsonl", transitions)
    write_jsonl(args.output / "dependency_hyperedges.jsonl", hyperedges)
    write_json(args.output / "source_releases.json", sources)
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
            "dataset_name": "CoReM Four-Provider Fresh-Query Confirmation T2 v6",
            "real_world_observations": True,
            "observation_tier": "T2_public_release_fresh_query_confirmation_v6",
            "claim_boundary": (
                "Pinned public releases with newly authored controlled queries. The "
                "set tests query robustness on the same providers, versions, task "
                "families, memories, and checker as v5; it is not unseen-provider "
                "generalization or production traffic."
            ),
            "event_time_semantics": (
                "Logical replay anchored to official release timestamps; all query "
                "strings are absent from v5 and were frozen before test execution."
            ),
            "parent_dataset": str(args.base_dataset.resolve()),
            "parent_manifest_sha256": sha256_file(args.base_dataset / "manifest.json"),
            "retrieval_blinding": base_manifest["retrieval_blinding"],
            "oracle_preflight_required": True,
            "workflow_contract": {
                "operations_per_incident": 3,
                "runtime_handoffs_required": 2,
                "expected_events": len(events),
                "minimum_test_events": 128,
                "task_version_families": len(
                    {
                        (row["service_id"], row["version_id"], row["task_key"])
                        for row in events
                    }
                ),
                "queries_per_task_version_family": 20,
            },
            "provenance": {
                **base_manifest["provenance"],
                "collection_start_utc": min(row["event_time"] for row in events),
                "collection_end_utc": max(row["event_time"] for row in events),
            },
            "execution": base_manifest["execution"],
            "checker": {
                **base_manifest["checker"],
                "implementation_digest": checker_digest(),
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
        f"wrote {len(events)} fresh-query events with {len(memories)} frozen memories "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
