"""Public-only structural checks for the unscored v9 staging set."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from corem.real_eval import load_jsonl
from corem.v8_candidates import validate_candidate


def test_v9_staging_candidates_are_complete_and_literal_balanced() -> None:
    dataset = Path("data/public_independent_v9_staging_v1")
    events = {row["event_id"]: row for row in load_jsonl(dataset / "query_only_events.jsonl")}
    memories = {row["memory_id"]: row for row in load_jsonl(dataset / "memories.jsonl")}
    edges = {row["edge_id"]: row for row in load_jsonl(dataset / "dependency_hyperedges.jsonl")}
    grouped = defaultdict(list)
    for row in load_jsonl(dataset / "candidate_selections.jsonl"):
        validate_candidate(row, events[row["event_id"]], memories, edges)
        grouped[row["event_id"]].append(row)
    assert len(events) == len(grouped) == 16
    assert Counter(row["service_id"] for row in events.values()) == {"typesense": 8, "solr": 8}
    for event_id, candidates in grouped.items():
        assert len(candidates) == 2, event_id
        bags = []
        serialized = []
        for candidate in candidates:
            rows = [memories[item] for item in candidate["selected_memory_ids"]]
            assert sum(row["token_cost"] for row in rows) == 144, event_id
            text = " ".join(json.dumps(row.get("plan", {}), sort_keys=True) for row in rows)
            serialized.append(text)
            bags.append(Counter(re.findall(r"[a-z0-9_]+", text.casefold())))
        assert bags[0] == bags[1], event_id
        assert serialized[0] != serialized[1], event_id


def test_v9_six_family_staging_remains_label_free_and_literal_balanced() -> None:
    dataset = Path("data/public_independent_v9_staging_v3")
    events = {row["event_id"]: row for row in load_jsonl(dataset / "query_only_events.jsonl")}
    memories = {row["memory_id"]: row for row in load_jsonl(dataset / "memories.jsonl")}
    edges = {row["edge_id"]: row for row in load_jsonl(dataset / "dependency_hyperedges.jsonl")}
    grouped = defaultdict(list)
    for row in load_jsonl(dataset / "candidate_selections.jsonl"):
        validate_candidate(row, events[row["event_id"]], memories, edges)
        grouped[row["event_id"]].append(row)
    assert len(events) == len(grouped) == 48
    assert Counter(row["service_id"] for row in events.values()) == {"typesense": 24, "solr": 24}
    assert len({row["event_id"].split(":")[1] for row in events.values()}) == 6
    for event_id, candidates in grouped.items():
        assert len(candidates) == 2, event_id
        bags = []
        plans = []
        for candidate in candidates:
            rows = [memories[item] for item in candidate["selected_memory_ids"]]
            assert sum(row["token_cost"] for row in rows) <= 192, event_id
            plan_text = " ".join(json.dumps(row.get("plan", {}), sort_keys=True) for row in rows)
            plans.append(plan_text)
            bags.append(Counter(re.findall(r"[a-z0-9_]+", plan_text.casefold())))
        assert bags[0] == bags[1], event_id
        assert plans[0] != plans[1], event_id


def test_v9_ten_family_staging_has_balanced_complete_candidates() -> None:
    dataset = Path("data/public_independent_v9_staging_v4")
    events = {row["event_id"]: row for row in load_jsonl(dataset / "query_only_events.jsonl")}
    memories = {row["memory_id"]: row for row in load_jsonl(dataset / "memories.jsonl")}
    edges = {row["edge_id"]: row for row in load_jsonl(dataset / "dependency_hyperedges.jsonl")}
    grouped = defaultdict(list)
    for row in load_jsonl(dataset / "candidate_selections.jsonl"):
        validate_candidate(row, events[row["event_id"]], memories, edges)
        grouped[row["event_id"]].append(row)
    assert len(events) == len(grouped) == 80
    assert Counter(row["service_id"] for row in events.values()) == {"typesense": 40, "solr": 40}
    assert len({row["event_id"].split(":")[1] for row in events.values()}) == 10
    for event_id, candidates in grouped.items():
        assert len(candidates) == 2, event_id
        bags = []
        plans = []
        for candidate in candidates:
            rows = [memories[item] for item in candidate["selected_memory_ids"]]
            assert sum(row["token_cost"] for row in rows) <= 192, event_id
            plan_text = " ".join(json.dumps(row.get("plan", {}), sort_keys=True) for row in rows)
            bags.append(Counter(re.findall(r"[a-z0-9_]+", plan_text.casefold())))
            plans.append(plan_text)
        assert bags[0] == bags[1], event_id
        assert plans[0] != plans[1], event_id


def test_v9_sixteen_family_staging_has_complete_matched_pairs() -> None:
    dataset = Path("data/public_independent_v9_staging_v6")
    events = {row["event_id"]: row for row in load_jsonl(dataset / "query_only_events.jsonl")}
    memories = {row["memory_id"]: row for row in load_jsonl(dataset / "memories.jsonl")}
    edges = {row["edge_id"]: row for row in load_jsonl(dataset / "dependency_hyperedges.jsonl")}
    grouped = defaultdict(list)
    for row in load_jsonl(dataset / "candidate_selections.jsonl"):
        validate_candidate(row, events[row["event_id"]], memories, edges)
        grouped[row["event_id"]].append(row)
    assert len(events) == len(grouped) == 128
    assert Counter(row["service_id"] for row in events.values()) == {"typesense": 64, "solr": 64}
    assert len({row["event_id"].split(":")[1] for row in events.values()}) == 16
    for event_id, candidates in grouped.items():
        assert len(candidates) == 2, event_id
        bags = []
        plans = []
        for candidate in candidates:
            rows = [memories[item] for item in candidate["selected_memory_ids"]]
            assert sum(row["token_cost"] for row in rows) <= 256, event_id
            plan_text = " ".join(json.dumps(row.get("plan", {}), sort_keys=True) for row in rows)
            bags.append(Counter(re.findall(r"[a-z0-9_]+", plan_text.casefold())))
            plans.append(plan_text)
        assert bags[0] == bags[1], event_id
        assert plans[0] != plans[1], event_id
