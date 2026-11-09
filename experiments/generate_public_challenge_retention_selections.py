#!/usr/bin/env python3
"""Build past-only active banks and frozen test selections for retention study."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.model import SetUtilityModel
from corem.real_eval import (
    load_jsonl,
    parse_utc,
    write_checksums,
    write_json,
    write_jsonl,
)
from corem.retrieval import BundleBeamRetriever, RetrievalConfig
from corem.types import Request
from experiments.generate_public_challenge_selections import GENERIC_REQUIRED_ATOMS
from experiments.generate_public_workflow_selections import (
    DEFAULT_SEEDS,
    bm25_scores,
    jitter,
    make_graph,
)

METHODS = (
    "group_aware_retention",
    "independent_relevance",
    "item_recency",
    "no_memory",
)
FORECAST_EVENTS_PER_SERVICE = 16
FORECAST_DECAY = 0.90
RETRIEVAL_BUDGET = 160


def _forecast_events(
    events: list[dict[str, Any]], assignments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    split_by_event = {str(row["event_id"]): str(row["split"]) for row in assignments}
    by_service: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if split_by_event[str(event["event_id"])] == "test":
            continue
        by_service[str(event["service_id"])].append(event)
    selected: list[dict[str, Any]] = []
    for service_events in by_service.values():
        service_events.sort(key=lambda row: (row["event_time"], row["event_id"]))
        selected.extend(service_events[-FORECAST_EVENTS_PER_SERVICE:])
    return selected


def _item_scores(
    forecast: list[dict[str, Any]], memories: list[dict[str, Any]]
) -> dict[str, float]:
    scores = {str(row["memory_id"]): 0.0 for row in memories}
    by_service: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in forecast:
        by_service[str(event["service_id"])].append(event)
    for service_events in by_service.values():
        service_events.sort(key=lambda row: (row["event_time"], row["event_id"]))
        for index, event in enumerate(service_events):
            event_time = parse_utc(event["event_time"], "event_time").timestamp()
            candidates = [
                row
                for row in memories
                if row["service_id"] == event["service_id"]
                and event["version_id"] in row["version_scope"]
                and parse_utc(row["valid_from"], "valid_from").timestamp() <= event_time
            ]
            lexical = bm25_scores(str(event["query"]), candidates)
            age = len(service_events) - 1 - index
            weight = FORECAST_DECAY**age
            for memory_id, value in lexical.items():
                scores[memory_id] += weight * value
    return scores


def _rank_jitter(seed: int, label: str, item_id: str) -> float:
    return jitter(seed, f"retention:{label}", item_id)


def _group_bank(
    hyperedges: list[dict[str, Any]],
    memories_by_id: dict[str, dict[str, Any]],
    scores: dict[str, float],
    seed: int,
    budget: int,
) -> tuple[list[str], int, int]:
    ranked = sorted(
        hyperedges,
        key=lambda edge: (
            -sum(scores[str(member)] for member in edge["members"])
            - _rank_jitter(seed, "group", str(edge["edge_id"])),
            str(edge["edge_id"]),
        ),
    )
    selected: list[str] = []
    selected_set: set[str] = set()
    cost = 0
    groups = 0
    for edge in ranked:
        members = [str(member) for member in edge["members"]]
        additions = [member for member in members if member not in selected_set]
        group_cost = sum(
            int(memories_by_id[item]["storage_cost"]) for item in additions
        )
        if cost + group_cost > budget:
            continue
        selected.extend(additions)
        selected_set.update(additions)
        cost += group_cost
        groups += 1
    return sorted(selected), cost, groups


def _item_bank(
    memories: list[dict[str, Any]],
    scores: dict[str, float],
    seed: int,
    budget: int,
    *,
    recency: bool,
) -> tuple[list[str], int, int]:
    def priority(row: dict[str, Any]) -> tuple[float, str]:
        memory_id = str(row["memory_id"])
        if recency:
            value = parse_utc(row["valid_from"], "valid_from").timestamp()
        else:
            value = scores[memory_id]
        value += _rank_jitter(seed, "recency" if recency else "relevance", memory_id)
        return -value, memory_id

    selected: list[str] = []
    cost = 0
    for row in sorted(memories, key=priority):
        item_cost = int(row["storage_cost"])
        if cost + item_cost <= budget:
            selected.append(str(row["memory_id"]))
            cost += item_cost
    return sorted(selected), cost, 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-budget", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    if args.storage_budget <= 0 or len(set(args.seeds)) != len(args.seeds):
        print("storage budget must be positive and seeds unique")
        return 2

    events = load_jsonl(args.dataset / "events.jsonl")
    memories = load_jsonl(args.dataset / "memories.jsonl")
    hyperedges = load_jsonl(args.dataset / "dependency_hyperedges.jsonl")
    assignments = load_jsonl(args.splits)
    split_by_event = {str(row["event_id"]): str(row["split"]) for row in assignments}
    test_events = sorted(
        (event for event in events if split_by_event[str(event["event_id"])] == "test"),
        key=lambda row: (row["event_time"], row["event_id"]),
    )
    forecast = _forecast_events(events, assignments)
    scores = _item_scores(forecast, memories)
    memories_by_id = {str(row["memory_id"]): row for row in memories}
    graph = make_graph(memories, hyperedges)
    retriever = BundleBeamRetriever(
        SetUtilityModel(),
        RetrievalConfig(
            retrieval_budget=RETRIEVAL_BUDGET,
            beam_width=16,
            max_depth=4,
            candidate_pool_size=100,
            dependency_hops=2,
            max_closure_size=8,
            enforce_dependency_closure=True,
        ),
    )

    banks: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for seed in args.seeds:
        for method in METHODS:
            if method == "group_aware_retention":
                bank, bank_cost, complete_groups = _group_bank(
                    hyperedges,
                    memories_by_id,
                    scores,
                    seed,
                    args.storage_budget,
                )
            elif method == "independent_relevance":
                bank, bank_cost, complete_groups = _item_bank(
                    memories,
                    scores,
                    seed,
                    args.storage_budget,
                    recency=False,
                )
            elif method == "item_recency":
                bank, bank_cost, complete_groups = _item_bank(
                    memories,
                    scores,
                    seed,
                    args.storage_budget,
                    recency=True,
                )
            else:
                bank, bank_cost, complete_groups = [], 0, 0
            bank_set = frozenset(bank)
            if bank_cost > args.storage_budget:
                raise RuntimeError(f"{method} exceeds the storage budget")
            banks.append(
                {
                    "method": method,
                    "seed": seed,
                    "storage_budget": args.storage_budget,
                    "retained_memory_ids": bank,
                    "storage_cost": bank_cost,
                    "complete_groups_selected": complete_groups,
                }
            )
            for event in test_events:
                event_id = str(event["event_id"])
                event_timestamp = parse_utc(
                    event["event_time"], "event_time"
                ).timestamp()
                candidates = [
                    memory_id
                    for memory_id in bank
                    if memories_by_id[memory_id]["service_id"] == event["service_id"]
                    and parse_utc(
                        memories_by_id[memory_id]["valid_from"], "valid_from"
                    ).timestamp()
                    <= event_timestamp
                ]
                started = time.perf_counter()
                if method == "no_memory":
                    selected: list[str] = []
                    token_cost = 0
                    status = "fallback"
                    evaluations = 0
                else:
                    request = Request(
                        request_id=event_id,
                        text=str(event["query"]),
                        required_atoms=GENERIC_REQUIRED_ATOMS,
                        operation_ids=frozenset(),
                        timestamp=event_timestamp,
                        allowed_scopes=frozenset((str(event["service_id"]),)),
                        active_versions=frozenset((str(event["version_id"]),)),
                    )
                    result = retriever.retrieve(request, candidates, graph)
                    selected = sorted(result.selected_ids)
                    token_cost = result.token_cost
                    status = result.status
                    evaluations = result.evaluations
                latency_ms = (time.perf_counter() - started) * 1000.0
                if not set(selected).issubset(bank_set):
                    raise RuntimeError("retrieval escaped the retained bank")
                selections.append(
                    {
                        "run_id": f"{method}:{args.storage_budget}:{seed}:{event_id}",
                        "event_id": event_id,
                        "seed": seed,
                        "method": method,
                        "selected_memory_ids": selected,
                        "token_cost": token_cost,
                        "latency_ms": latency_ms,
                        "llm_calls": 0,
                        "active_bank_cost": bank_cost,
                        "retrieval_status": status,
                        "evaluations": evaluations,
                    }
                )

    forecast_digest = hashlib.sha256(
        "\n".join(sorted(str(row["event_id"]) for row in forecast)).encode()
    ).hexdigest()
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "retained_banks.jsonl", banks)
    write_jsonl(args.output / "selections.jsonl", selections)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": "corem-public-challenge-retention-v1",
            "dataset": str(args.dataset.resolve()),
            "methods": list(METHODS),
            "seeds": args.seeds,
            "storage_budget": args.storage_budget,
            "retrieval_budget": RETRIEVAL_BUDGET,
            "selection_rows": len(selections),
            "bank_rows": len(banks),
            "forecast": {
                "test_events_used": False,
                "events_per_service": FORECAST_EVENTS_PER_SERVICE,
                "recency_decay": FORECAST_DECAY,
                "event_count": len(forecast),
                "event_ids_sha256": forecast_digest,
            },
            "claim_boundary": "Past-only controlled active-bank comparison; not online or natural-production retention.",
        },
    )
    write_checksums(
        args.output, ("manifest.json", "retained_banks.jsonl", "selections.jsonl")
    )
    print(
        f"wrote {len(selections)} selections and {len(banks)} banks at "
        f"Bm={args.storage_budget}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
