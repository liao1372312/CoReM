#!/usr/bin/env python3
"""Audit v9 public inputs without reading private selections or outcomes."""

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

from corem.real_eval import load_jsonl, parse_utc, sha256_file, write_json  # noqa: E402
from corem.v8_candidates import validate_candidate  # noqa: E402
from experiments.independent_v9_runner import VERSIONS  # noqa: E402
from experiments.score_complete_candidate_bm25 import QUERY_FIELDS  # noqa: E402

DATA_FILES = (
    "events.jsonl", "query_only_events.jsonl", "memories.jsonl",
    "dependency_hyperedges.jsonl", "candidate_selections.jsonl", "manifest.json",
)


def _unique(rows: list[dict[str, Any]], key: str, issues: list[str]) -> dict[str, dict[str, Any]]:
    mapped = {str(row.get(key)): row for row in rows}
    if len(mapped) != len(rows):
        issues.append(f"duplicate {key}")
    return mapped


def audit(dataset: Path, budget: int) -> dict[str, Any]:
    issues: list[str] = []
    event_rows = load_jsonl(dataset / "events.jsonl")
    query_rows = load_jsonl(dataset / "query_only_events.jsonl")
    memory_rows = load_jsonl(dataset / "memories.jsonl")
    edge_rows = load_jsonl(dataset / "dependency_hyperedges.jsonl")
    candidate_rows = load_jsonl(dataset / "candidate_selections.jsonl")
    events = _unique(event_rows, "event_id", issues)
    queries = _unique(query_rows, "event_id", issues)
    memories = _unique(memory_rows, "memory_id", issues)
    edges = _unique(edge_rows, "edge_id", issues)
    family_counts: Counter[tuple[str, str]] = Counter()
    times: dict[str, set[Any]] = defaultdict(set)
    if len(events) != 128 or len(queries) != 128 or len(candidate_rows) != 256:
        issues.append("expected 128 events, 128 query views, and 256 candidate rows")
    if set(events) != set(queries):
        issues.append("query and full event IDs differ")
    for event_id, event in events.items():
        provider = str(event.get("provider_id"))
        family = str(event.get("task_key"))
        family_counts[provider, family] += 1
        if event.get("service_id") != provider or event.get("version_id") != VERSIONS.get(provider):
            issues.append(f"{event_id}: unpinned provider/version")
        if event.get("observation_tier") not in {
            "prospective_unscored_staging",
            "prospective_outcome_hidden_confirmation",
        }:
            issues.append(f"{event_id}: unexpected observation tier")
        query = queries.get(event_id)
        if query is None or set(query) != QUERY_FIELDS or query != {key: event.get(key) for key in QUERY_FIELDS}:
            issues.append(f"{event_id}: leaking or inconsistent query-only projection")
        contract = event.get("workflow_contract")
        if (
            not isinstance(contract, dict) or contract.get("kind") != "assertions_v1"
            or not isinstance(contract.get("state_checks"), list) or not contract["state_checks"]
            or len(contract.get("trace_statuses", [])) != event.get("operation_count")
        ):
            issues.append(f"{event_id}: incomplete live-state checker contract")
        try:
            stamp = parse_utc(event["event_time"], "event_time")
            if stamp in times[provider]:
                issues.append(f"{provider}: duplicated event time")
            times[provider].add(stamp)
        except (KeyError, TypeError, ValueError) as error:
            issues.append(f"{event_id}: invalid event time: {error}")
    if len(family_counts) != 16 or Counter(provider for provider, _ in family_counts) != {"typesense": 8, "solr": 8}:
        issues.append("requires eight distinct operation families per new provider")
    if any(count != 8 for count in family_counts.values()):
        issues.append("each provider/family requires exactly eight incidents")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_keys = set()
    for row in candidate_rows:
        event_id = str(row.get("event_id"))
        if event_id not in events:
            issues.append(f"{event_id}: candidate has no event")
            continue
        key = event_id, str(row.get("candidate_edge_id"))
        if key in candidate_keys:
            issues.append(f"{event_id}: duplicate candidate edge")
        candidate_keys.add(key)
        try:
            validate_candidate(row, events[event_id], memories, edges)
            selected = [memories[item] for item in row["selected_memory_ids"]]
            if sum(int(item["token_cost"]) for item in selected) > budget:
                issues.append(f"{event_id}: candidate over budget")
            atoms = set().union(*(set(map(str, item.get("evidence_atoms", []))) for item in selected))
            if not set(events[event_id].get("required_evidence_atoms", [])).issubset(atoms):
                issues.append(f"{event_id}: incomplete evidence atoms")
            grouped[event_id].append(row)
        except (KeyError, TypeError, ValueError) as error:
            issues.append(f"{event_id}: invalid candidate: {error}")
    for event_id in events:
        pair = grouped[event_id]
        if len(pair) != 2:
            issues.append(f"{event_id}: expected two complete candidates")
            continue
        plans = []
        bags = []
        for row in pair:
            records = [memories[item] for item in row["selected_memory_ids"]]
            ordered = sorted((item for item in records if item.get("role") == "workflow_step"), key=lambda item: item["step_index"])
            if len(ordered) != events[event_id].get("operation_count"):
                issues.append(f"{event_id}: wrong number of executable operations")
            serialized = " ".join(json.dumps(item.get("plan", {}), sort_keys=True) for item in ordered)
            plans.append(serialized)
            bags.append(Counter(re.findall(r"[a-z0-9_]+", serialized.casefold())))
        if plans[0] == plans[1]:
            issues.append(f"{event_id}: identical executable plans")
        if bags[0] != bags[1]:
            issues.append(f"{event_id}: unequal candidate literal bags")
    return {
        "status": "structurally_ready_for_physical_gate" if not issues else "not_ready",
        "claim_boundary": "This audit does not establish task validity, independent method performance, or external generalization.",
        "metrics": {"events": len(events), "candidate_rows": len(candidate_rows),
                    "family_counts": {f"{provider}/{family}": count for (provider, family), count in sorted(family_counts.items())}},
        "issues": sorted(set(issues)),
        "token_budget": budget,
        "input_sha256": {name: sha256_file(dataset / name) for name in DATA_FILES},
        "script_sha256": sha256_file(Path(__file__)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists() or args.token_budget < 1:
        parser.error("output must be new and budget positive")
    result = audit(args.dataset, args.token_budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(result["status"], f"issues={len(result['issues'])}")
    return 0 if not result["issues"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
