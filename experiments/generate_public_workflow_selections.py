#!/usr/bin/env python3
"""Generate five-seed retrieval exports for public two-step workflows."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
import time
from collections import Counter
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
from corem.types import DependencyHyperedge, EvidenceGraph, MemoryRecord, Request

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.:/?=&{}-]+")
METHODS = (
    "no_memory",
    "service_bm25",
    "recency_bm25",
    "version_bm25",
    "corem_reference",
)
DEFAULT_SEEDS = (2027, 2028, 2029, 2030, 2031)


def tokens(text: str) -> list[str]:
    return [value.lower() for value in TOKEN_PATTERN.findall(text)]


def bm25_scores(query: str, rows: list[dict[str, Any]]) -> dict[str, float]:
    documents = {
        str(row["memory_id"]): tokens(str(row.get("text", ""))) for row in rows
    }
    if not documents:
        return {}
    query_terms = tokens(query)
    document_frequency = Counter(
        term for document in documents.values() for term in set(document)
    )
    average_length = sum(map(len, documents.values())) / len(documents)
    scores: dict[str, float] = {}
    for memory_id, document in documents.items():
        counts = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = counts[term]
            if not frequency:
                continue
            inverse = math.log(
                1.0
                + (len(documents) - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            denominator = frequency + 1.5 * (
                1.0 - 0.75 + 0.75 * len(document) / max(1.0, average_length)
            )
            score += inverse * frequency * 2.5 / denominator
        scores[memory_id] = score
    return scores


def jitter(seed: int, event_id: str, memory_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{event_id}:{memory_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / (2**64 - 1) * 1e-9


def budgeted(ranked: list[dict[str, Any]], budget: int) -> tuple[list[str], int]:
    selected: list[str] = []
    cost = 0
    for row in ranked:
        item_cost = int(row["token_cost"])
        if cost + item_cost <= budget:
            selected.append(str(row["memory_id"]))
            cost += item_cost
    return selected, cost


def make_graph(
    memories: list[dict[str, Any]], hyperedges: list[dict[str, Any]]
) -> EvidenceGraph:
    graph = EvidenceGraph()
    provenance_by_id: dict[str, str] = {}
    for row in memories:
        provenance = str(row.get("provenance_uri") or row.get("provenance") or "")
        if not provenance:
            raise ValueError(f"memory {row.get('memory_id')} has no provenance")
        provenance_by_id[str(row["memory_id"])] = provenance
        graph.add_record(
            MemoryRecord(
                record_id=str(row["memory_id"]),
                text=str(row["text"]),
                evidence_atoms=frozenset(map(str, row["evidence_atoms"])),
                operation_ids=frozenset(map(str, row.get("operation_ids", []))),
                token_cost=int(row["token_cost"]),
                storage_cost=int(row["storage_cost"]),
                timestamp=parse_utc(row["valid_from"], "valid_from").timestamp(),
                provenance=provenance,
                service_scope=frozenset((str(row["service_id"]),)),
                version_scope=frozenset(map(str, row["version_scope"])),
                valid_from=parse_utc(row["valid_from"], "valid_from").timestamp(),
                valid_until=(
                    parse_utc(row["valid_until"], "valid_until").timestamp()
                    if row.get("valid_until")
                    else None
                ),
                provenance_verified=row.get("provenance_verified") is True,
            )
        )
    for row in hyperedges:
        edge_provenance = str(
            row.get("provenance") or row.get("provenance_uri") or ""
        )
        if not edge_provenance:
            member_provenance = sorted(
                {
                    provenance_by_id[str(member_id)]
                    for member_id in row["members"]
                    if str(member_id) in provenance_by_id
                }
            )
            if not member_provenance:
                raise ValueError(f"hyperedge {row.get('edge_id')} has no provenance")
            edge_provenance = ";".join(member_provenance)
        graph.add_hyperedge(
            DependencyHyperedge(
                edge_id=str(row["edge_id"]),
                members=frozenset(map(str, row["members"])),
                evidence_atoms=frozenset(map(str, row["evidence_atoms"])),
                relation_type=str(row["relation_type"]),
                provenance=edge_provenance,
                verified=row.get("verified") is True,
                version_scope=frozenset(map(str, row["version_scope"])),
                valid_from=parse_utc(row["valid_from"], "valid_from").timestamp(),
                valid_until=(
                    parse_utc(row["valid_until"], "valid_until").timestamp()
                    if row.get("valid_until")
                    else None
                ),
            )
        )
    return graph


def ranked_rows(
    rows: list[dict[str, Any]],
    scores: dict[str, float],
    *,
    seed: int,
    event_id: str,
    recency_weight: float = 0.0,
) -> list[dict[str, Any]]:
    max_score = max(scores.values(), default=1.0) or 1.0
    timestamps = [
        parse_utc(row["valid_from"], "valid_from").timestamp() for row in rows
    ]
    newest = max(timestamps, default=0.0)
    oldest = min(timestamps, default=0.0)
    span = max(1.0, newest - oldest)

    def score(row: dict[str, Any]) -> float:
        memory_id = str(row["memory_id"])
        lexical = scores[memory_id] / max_score
        recent = (
            parse_utc(row["valid_from"], "valid_from").timestamp() - oldest
        ) / span
        return (
            (1.0 - recency_weight) * lexical
            + recency_weight * recent
            + jitter(seed, event_id, memory_id)
        )

    return sorted(rows, key=lambda row: (-score(row), str(row["memory_id"])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=144)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    if len(set(args.seeds)) != len(args.seeds):
        print("seeds must be unique")
        return 2

    events = load_jsonl(args.dataset / "events.jsonl")
    memories = load_jsonl(args.dataset / "memories.jsonl")
    hyperedges = load_jsonl(args.dataset / "dependency_hyperedges.jsonl")
    graph = make_graph(memories, hyperedges)
    retriever = BundleBeamRetriever(
        SetUtilityModel(),
        RetrievalConfig(
            retrieval_budget=args.token_budget,
            beam_width=8,
            max_depth=3,
            candidate_pool_size=40,
            dependency_hops=2,
            max_closure_size=6,
            enforce_dependency_closure=True,
        ),
    )
    rows: list[dict[str, Any]] = []
    ordered_events = sorted(
        events,
        key=lambda row: (
            parse_utc(row["event_time"], "event_time"),
            str(row["event_id"]),
        ),
    )
    for seed in args.seeds:
        for event in ordered_events:
            event_id = str(event["event_id"])
            event_timestamp = parse_utc(event["event_time"], "event_time").timestamp()
            available = [
                row
                for row in memories
                if row.get("service_id") == event.get("service_id")
                and parse_utc(row["valid_from"], "valid_from").timestamp()
                <= event_timestamp
            ]
            active_version = [
                row
                for row in available
                if event.get("version_id") in row.get("version_scope", [])
            ]
            lexical = bm25_scores(str(event["query"]), available)
            version_lexical = bm25_scores(str(event["query"]), active_version)
            for method in METHODS:
                started = time.perf_counter()
                status = "ok"
                evaluations = 0
                if method == "no_memory":
                    selected, cost = [], 0
                    status = "fallback"
                elif method == "service_bm25":
                    ranked = ranked_rows(
                        available,
                        lexical,
                        seed=seed,
                        event_id=event_id,
                    )
                    selected, cost = budgeted(ranked, args.token_budget)
                    evaluations = len(available)
                elif method == "recency_bm25":
                    ranked = ranked_rows(
                        available,
                        lexical,
                        seed=seed,
                        event_id=event_id,
                        recency_weight=0.20,
                    )
                    selected, cost = budgeted(ranked, args.token_budget)
                    evaluations = len(available)
                elif method == "version_bm25":
                    ranked = ranked_rows(
                        active_version,
                        version_lexical,
                        seed=seed,
                        event_id=event_id,
                    )
                    selected, cost = budgeted(ranked, args.token_budget)
                    evaluations = len(active_version)
                else:
                    request = Request(
                        request_id=event_id,
                        text=str(event["query"]),
                        required_atoms=frozenset(
                            map(str, event["required_evidence_atoms"])
                        ),
                        operation_ids=frozenset((str(event["task_key"]),)),
                        timestamp=event_timestamp,
                        allowed_scopes=frozenset((str(event["service_id"]),)),
                        active_versions=frozenset((str(event["version_id"]),)),
                    )
                    result = retriever.retrieve(
                        request,
                        [str(row["memory_id"]) for row in available],
                        graph,
                    )
                    selected = sorted(result.selected_ids)
                    cost = result.token_cost
                    status = result.status
                    evaluations = result.evaluations
                elapsed = (time.perf_counter() - started) * 1000.0
                rows.append(
                    {
                        "run_id": f"{method}:{seed}:{event_id}",
                        "event_id": event_id,
                        "seed": seed,
                        "method": method,
                        "selected_memory_ids": selected,
                        "token_cost": cost,
                        "latency_ms": elapsed,
                        "llm_calls": 0,
                        "active_bank_cost": sum(
                            int(row["storage_cost"]) for row in available
                        ),
                        "retrieval_status": status,
                        "evaluations": evaluations,
                    }
                )

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "selections.jsonl", rows)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": "corem-public-workflow-selections-v1",
            "dataset": str(args.dataset.resolve()),
            "methods": list(METHODS),
            "seeds": args.seeds,
            "token_budget": args.token_budget,
            "selection_rows": len(rows),
            "label_blind": True,
            "seed_policy": (
                "Seeds affect only deterministic SHA-256 tie-breaking; all rows "
                "are executed and no seed is selected by outcome."
            ),
            "method_note": (
                "corem_reference uses the repository implementation with default "
                "untrained set utility; it is not a trained full-model estimate."
            ),
        },
    )
    write_checksums(args.output, ("manifest.json", "selections.jsonl"))
    print(f"wrote {len(rows)} five-seed selections to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
