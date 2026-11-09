#!/usr/bin/env python3
"""Fail closed on structural prerequisites for a prospective v7 sealed set.

This audits dataset design, not model success or independent oracle execution.
It must run before selections or test outcomes are inspected.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import load_jsonl, sha256_file, write_json  # noqa: E402

QUERY_FIELDS = {
    "event_id",
    "event_time",
    "provider_id",
    "query",
    "service_id",
    "version_id",
}
CANDIDATE_FIELDS = {"event_id", "candidate_edge_id", "selected_memory_ids"}
PROVIDERS = {"gitea": "28.1.0", "meilisearch": "1.55.0"}
ANSWER_WORDS = ("oracle", "correct", "negative", "goal_satisfied", "success")


def audit(
    events: list[dict[str, Any]],
    memories: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    query_view: list[dict[str, Any]],
    excluded_families: set[tuple[str, str]],
) -> dict[str, Any]:
    issues: list[str] = []
    event_ids = [str(row.get("event_id")) for row in events]
    if any(
        not isinstance(row.get("event_id"), str) or not row["event_id"]
        for row in events
    ):
        issues.append("event IDs must be nonempty strings")
    if len(event_ids) != len(set(event_ids)):
        issues.append("event IDs are not unique")
    by_id = {str(row.get("event_id")): row for row in events}
    if len(events) < 128:
        issues.append(f"requires >=128 incidents; found {len(events)}")
    families: dict[str, set[str]] = defaultdict(set)
    family_sizes: Counter[tuple[str, str]] = Counter()
    for event in events:
        event_id = str(event.get("event_id"))
        provider = event.get("provider_id")
        family = event.get("task_key")
        if provider not in PROVIDERS or event.get("service_id") != provider:
            issues.append(f"{event_id}: unpinned or mismatched provider")
            continue
        if event.get("version_id") != PROVIDERS[provider]:
            issues.append(f"{event_id}: unpinned service version")
        if not isinstance(family, str) or not family:
            issues.append(f"{event_id}: missing task family")
            continue
        families[provider].add(family)
        family_sizes[(provider, family)] += 1
        if (provider, family) in excluded_families:
            issues.append(
                f"{event_id}: family overlaps inspected pilot/development data"
            )
        count = event.get("operation_count")
        contract = event.get("workflow_contract")
        if not isinstance(count, int) or isinstance(count, bool) or count < 3:
            issues.append(f"{event_id}: requires >=3 real operations")
            continue
        if not isinstance(contract, dict):
            issues.append(f"{event_id}: missing workflow contract")
            continue
        handoffs = contract.get("handoffs")
        statuses = contract.get("statuses")
        if (
            not isinstance(handoffs, list)
            or len(handoffs) < 2
            or len(
                {str(item.get("name")) for item in handoffs if isinstance(item, dict)}
            )
            < 2
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("name"), str)
                or not isinstance(item.get("target_step"), int)
                or isinstance(item.get("target_step"), bool)
                or not 2 <= item["target_step"] <= count
                for item in handoffs
            )
        ):
            issues.append(f"{event_id}: requires >=2 valid runtime handoffs")
        if (
            not isinstance(statuses, list)
            or len(statuses) != count
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or not (
                    200 <= value < 300
                    or (
                        contract.get("kind") == "v7_meili_index_delete"
                        and count == 7
                        and index == count - 1
                        and value == 404
                    )
                )
                for index, value in enumerate(statuses)
            )
        ):
            issues.append(f"{event_id}: status contract does not match operations")
    for provider in PROVIDERS:
        if len(families[provider]) < 8:
            issues.append(
                f"{provider}: requires >=8 untouched families; found {len(families[provider])}"
            )
    if len(family_sizes) < 16:
        issues.append(f"requires >=16 provider-family pairs; found {len(family_sizes)}")
    for (provider, family), size in sorted(family_sizes.items()):
        if size < 8:
            issues.append(f"{provider}/{family}: requires >=8 incidents; found {size}")

    memory_by_id = {str(row.get("memory_id")): row for row in memories}
    if len(memory_by_id) != len(memories):
        issues.append("memory IDs are not unique")
    query_by_id = {str(row.get("event_id")): row for row in query_view}
    if len(query_by_id) != len(query_view) or set(query_by_id) != set(by_id):
        issues.append("query-only view must have exactly one row per event")
    for event_id, event in by_id.items():
        view = query_by_id.get(event_id)
        if view is None:
            continue
        if set(view) != QUERY_FIELDS or view != {
            key: event.get(key) for key in QUERY_FIELDS
        }:
            issues.append(
                f"{event_id}: query view contains missing/extra or altered fields"
            )

    candidates_by_event: Counter[str] = Counter()
    candidate_ids: set[tuple[str, str]] = set()
    for row in candidates:
        event_id = str(row.get("event_id"))
        edge_id = str(row.get("candidate_edge_id"))
        candidates_by_event[event_id] += 1
        if set(row) != CANDIDATE_FIELDS:
            issues.append(f"{event_id}: candidate input contains non-selection fields")
        if event_id not in by_id:
            issues.append(f"{event_id}: candidate references absent event")
            continue
        key = (event_id, edge_id)
        if key in candidate_ids:
            issues.append(f"{event_id}: duplicate candidate ID")
        candidate_ids.add(key)
        ids = row.get("selected_memory_ids")
        if not isinstance(ids, list) or not ids or len(ids) != len(set(map(str, ids))):
            issues.append(f"{event_id}: invalid candidate memory IDs")
            continue
        if any(word in edge_id.lower() for word in ANSWER_WORDS) or any(
            any(word in str(memory_id).lower() for word in ANSWER_WORDS)
            for memory_id in ids
        ):
            issues.append(f"{event_id}: answer vocabulary in candidate identifiers")
        if any(
            re.search(r"(?:^|:)bundle_[ab](?::|$)", value.lower())
            for value in (edge_id, *(str(memory_id) for memory_id in ids))
        ):
            issues.append(f"{event_id}: candidate identifiers expose bundle side")
        event = by_id[event_id]
        required_atoms = event.get("required_evidence_atoms")
        if not isinstance(required_atoms, list):
            issues.append(f"{event_id}: missing required evidence atoms")
            required_atoms = []
        required = set(map(str, required_atoms))
        observed: set[str] = set()
        for memory_id in ids:
            memory = memory_by_id.get(str(memory_id))
            if memory is None:
                issues.append(f"{event_id}: missing candidate memory {memory_id}")
                continue
            scopes = memory.get("version_scope")
            if (
                memory.get("service_id") != event.get("service_id")
                or not isinstance(scopes, list)
                or event.get("version_id") not in scopes
            ):
                issues.append(f"{event_id}: candidate uses wrong provider/version")
            memory_atoms = memory.get("evidence_atoms")
            if isinstance(memory_atoms, list):
                observed.update(map(str, memory_atoms))
        if not required.issubset(observed):
            issues.append(f"{event_id}: candidate is not evidence-complete")
    for event_id in by_id:
        if candidates_by_event[event_id] < 2:
            issues.append(f"{event_id}: requires >=2 complete candidates")
    return {
        "status": "structurally_ready_for_oracle_gate" if not issues else "not_ready",
        "metrics": {
            "incidents": len(events),
            "families_by_provider": {p: len(families[p]) for p in PROVIDERS},
            "candidate_rows": len(candidates),
            "excluded_family_count": len(excluded_families),
        },
        "issues": sorted(set(issues)),
        "scope_note": (
            "Structural checks cannot establish family diversity, oracle validity, "
            "negative validity, sealed execution, or a learned-component advantage."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--query-view", type=Path, required=True)
    parser.add_argument("--exclude-dataset", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing audit")
    excluded: set[tuple[str, str]] = set()
    for path in args.exclude_dataset:
        excluded.update(
            (str(row.get("provider_id")), str(row.get("task_key")))
            for row in load_jsonl(path / "events.jsonl")
        )
    inputs = {
        "events": args.dataset / "events.jsonl",
        "memories": args.dataset / "memories.jsonl",
        "candidates": args.candidates,
        "query_view": args.query_view,
    }
    result = audit(
        load_jsonl(inputs["events"]),
        load_jsonl(inputs["memories"]),
        load_jsonl(inputs["candidates"]),
        load_jsonl(inputs["query_view"]),
        excluded,
    )
    result["input_sha256"] = {name: sha256_file(path) for name, path in inputs.items()}
    result["excluded_dataset_events_sha256"] = {
        str(path): sha256_file(path / "events.jsonl") for path in args.exclude_dataset
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result["metrics"], sort_keys=True))
    print(result["status"], f"issues={len(result['issues'])}")
    return 0 if not result["issues"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
