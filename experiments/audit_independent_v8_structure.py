#!/usr/bin/env python3
"""Fail-closed structural gate for a prospective untouched v8 benchmark.

This checks dataset eligibility, not oracle validity or method performance.
No result from this audit alone is a scored confirmation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_jsonl, parse_utc, sha256_file, write_json  # noqa: E402
from corem.v8_candidates import validate_candidate  # noqa: E402

VERSIONS = {"qdrant": "1.15.5", "openbao": "2.7.0"}
QUERY_FIELDS = {"event_id", "service_id", "provider_id", "version_id", "event_time", "query"}
DATA_FILES = (
    "events.jsonl",
    "query_only_events.jsonl",
    "memories.jsonl",
    "dependency_hyperedges.jsonl",
    "candidate_selections.jsonl",
)


def audit(
    events: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    excluded_families: set[tuple[str, str]],
    *,
    token_budget: int = 160,
) -> dict[str, Any]:
    issues: list[str] = []
    event_by_id = {str(row.get("event_id")): row for row in events}
    query_by_id = {str(row.get("event_id")): row for row in query_rows}
    memories = {str(row.get("memory_id")): row for row in memory_rows}
    edges = {str(row.get("edge_id")): row for row in edge_rows}
    if len(events) < 128:
        issues.append(f"requires at least 128 incidents; found {len(events)}")
    if len(event_by_id) != len(events) or len(query_by_id) != len(query_rows):
        issues.append("event or query IDs are duplicated")
    if len(memories) != len(memory_rows) or len(edges) != len(edge_rows):
        issues.append("memory or dependency-edge IDs are duplicated")
    if set(query_by_id) != set(event_by_id):
        issues.append("query-only view does not match event IDs")
    families: dict[str, set[str]] = defaultdict(set)
    family_sizes: Counter[tuple[str, str]] = Counter()
    time_by_provider: dict[str, list[Any]] = defaultdict(list)
    for event_id, event in event_by_id.items():
        provider = event.get("provider_id")
        family = event.get("task_key")
        if (
            provider not in VERSIONS
            or event.get("service_id") != provider
            or event.get("version_id") != VERSIONS.get(provider)
        ):
            issues.append(f"{event_id}: provider/service/version is not pinned")
            continue
        if not isinstance(family, str) or not family:
            issues.append(f"{event_id}: missing operation family")
            continue
        families[provider].add(family)
        family_sizes[(provider, family)] += 1
        if (provider, family) in excluded_families or any(
            value in family.lower() for value in ("pilot", "development", "staging", "alias")
        ):
            issues.append(f"{event_id}: family overlaps inspected development")
        if event.get("observation_tier") != "independent_candidate":
            issues.append(f"{event_id}: not marked as an independent candidate")
        view = query_by_id.get(event_id)
        if view is None or set(view) != QUERY_FIELDS or view != {
            key: event.get(key) for key in QUERY_FIELDS
        }:
            issues.append(f"{event_id}: query-only view is missing, altered, or leaking fields")
        try:
            time_by_provider[provider].append(parse_utc(event["event_time"], "event_time"))
        except (KeyError, TypeError, ValueError) as error:
            issues.append(f"{event_id}: invalid event time: {error}")
        operation_count = event.get("operation_count")
        contract = event.get("workflow_contract")
        if (
            not isinstance(operation_count, int)
            or isinstance(operation_count, bool)
            or operation_count < 3
            or not isinstance(contract, dict)
            or contract.get("kind") != "assertions_v1"
            or not isinstance(contract.get("trace_statuses"), list)
            or len(contract["trace_statuses"]) != operation_count
            or not isinstance(contract.get("state_checks"), list)
            or not contract["state_checks"]
        ):
            issues.append(f"{event_id}: incomplete real-operation/checker contract")
    for provider in VERSIONS:
        if len(families[provider]) < 8:
            issues.append(f"{provider}: requires eight fresh families, found {len(families[provider])}")
    if len(family_sizes) < 16:
        issues.append(f"requires sixteen provider-family pairs, found {len(family_sizes)}")
    for (provider, family), count in sorted(family_sizes.items()):
        if count < 8:
            issues.append(f"{provider}/{family}: requires eight incidents, found {count}")
    for provider, timestamps in time_by_provider.items():
        if len(timestamps) != len(set(timestamps)):
            issues.append(f"{provider}: duplicate event timestamps")

    candidates_by_event: Counter[str] = Counter()
    seen_candidates: set[tuple[str, str]] = set()
    for candidate in candidates:
        event_id = str(candidate.get("event_id"))
        if event_id not in event_by_id:
            issues.append(f"{event_id}: candidate references absent event")
            continue
        candidate_key = (event_id, str(candidate.get("candidate_edge_id")))
        if candidate_key in seen_candidates:
            issues.append(f"{event_id}: duplicate candidate edge")
        seen_candidates.add(candidate_key)
        candidates_by_event[event_id] += 1
        try:
            event = event_by_id[event_id]
            validate_candidate(candidate, event, memories, edges)
            selected = candidate["selected_memory_ids"]
            cost = sum(int(memories[memory_id]["token_cost"]) for memory_id in selected)
            if cost > token_budget:
                issues.append(f"{event_id}: candidate exceeds token budget")
            atoms = set().union(
                *(set(map(str, memories[memory_id].get("evidence_atoms", []))) for memory_id in selected)
            )
            if not set(map(str, event.get("required_evidence_atoms", []))).issubset(atoms):
                issues.append(f"{event_id}: candidate is not evidence-complete")
        except (KeyError, TypeError, ValueError) as error:
            issues.append(f"{event_id}: invalid candidate: {error}")
    for event_id in event_by_id:
        if candidates_by_event[event_id] != 2:
            issues.append(f"{event_id}: requires exactly two complete candidates")
    return {
        "status": "structurally_ready_for_oracle_gate" if not issues else "not_ready",
        "metrics": {
            "events": len(events),
            "candidate_rows": len(candidates),
            "families_by_provider": {provider: len(families[provider]) for provider in VERSIONS},
            "excluded_family_count": len(excluded_families),
        },
        "issues": sorted(set(issues)),
        "scope_note": (
            "Structural checks do not establish oracle/negative validity, "
            "task independence, semantic difficulty, or model advantage."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--exclude-dataset", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=160)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing audit")
    if args.token_budget < 1:
        parser.error("token budget must be positive")
    excluded = {
        (str(row.get("provider_id")), str(row.get("task_key")))
        for directory in args.exclude_dataset
        for row in load_jsonl(directory / "events.jsonl")
    }
    rows = {name: load_jsonl(args.dataset / name) for name in DATA_FILES}
    result = audit(
        rows["events.jsonl"],
        rows["query_only_events.jsonl"],
        rows["memories.jsonl"],
        rows["dependency_hyperedges.jsonl"],
        rows["candidate_selections.jsonl"],
        excluded,
        token_budget=args.token_budget,
    )
    result["input_sha256"] = {
        name: sha256_file(args.dataset / name) for name in DATA_FILES
    }
    result["excluded_dataset_sha256"] = {
        str(path): sha256_file(path / "events.jsonl") for path in args.exclude_dataset
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result["metrics"], sort_keys=True))
    print(result["status"], f"issues={len(result['issues'])}")
    return 0 if not result["issues"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
