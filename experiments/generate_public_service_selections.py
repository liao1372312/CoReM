#!/usr/bin/env python3
"""Generate label-blind baselines and CoReM-reference selections."""

from __future__ import annotations

import argparse
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

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.:/?=&-]+")
METHODS = ("no_memory", "service_bm25", "recency_bm25", "corem_reference")


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
    for row in memories:
        graph.add_record(
            MemoryRecord(
                record_id=str(row["memory_id"]),
                text=str(row["text"]),
                evidence_atoms=frozenset(map(str, row["evidence_atoms"])),
                operation_ids=frozenset(map(str, row.get("operation_ids", []))),
                token_cost=int(row["token_cost"]),
                storage_cost=int(row["storage_cost"]),
                timestamp=parse_utc(row["valid_from"], "valid_from").timestamp(),
                provenance=str(row["provenance_uri"]),
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
        graph.add_hyperedge(
            DependencyHyperedge(
                edge_id=str(row["edge_id"]),
                members=frozenset(map(str, row["members"])),
                evidence_atoms=frozenset(map(str, row["evidence_atoms"])),
                relation_type=str(row["relation_type"]),
                provenance=str(row["provenance"]),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2

    events = load_jsonl(args.dataset / "events.jsonl")
    memories = load_jsonl(args.dataset / "memories.jsonl")
    hyperedges = load_jsonl(args.dataset / "dependency_hyperedges.jsonl")
    graph = make_graph(memories, hyperedges)
    model = SetUtilityModel()
    retriever = BundleBeamRetriever(
        model,
        RetrievalConfig(
            retrieval_budget=args.token_budget,
            beam_width=8,
            max_depth=3,
            candidate_pool_size=32,
            dependency_hops=2,
            max_closure_size=4,
            enforce_dependency_closure=True,
        ),
    )
    rows: list[dict[str, Any]] = []
    for event in sorted(
        events,
        key=lambda row: (
            parse_utc(row["event_time"], "event_time"),
            str(row["event_id"]),
        ),
    ):
        event_timestamp = parse_utc(event["event_time"], "event_time").timestamp()
        available = [
            row
            for row in memories
            if row.get("service_id") == event.get("service_id")
            and parse_utc(row["valid_from"], "valid_from").timestamp()
            <= event_timestamp
        ]
        lexical = bm25_scores(str(event["query"]), available)
        max_lexical = max(lexical.values(), default=1.0) or 1.0
        newest = max(
            (
                parse_utc(row["valid_from"], "valid_from").timestamp()
                for row in available
            ),
            default=event_timestamp,
        )
        oldest = min(
            (
                parse_utc(row["valid_from"], "valid_from").timestamp()
                for row in available
            ),
            default=event_timestamp,
        )
        span = max(1.0, newest - oldest)
        for method in METHODS:
            started = time.perf_counter()
            status = "ok"
            evaluations = 0
            if method == "no_memory":
                selected, cost = [], 0
                status = "fallback"
            elif method == "service_bm25":
                ranked = sorted(
                    available,
                    key=lambda row: (
                        -lexical[str(row["memory_id"])],
                        str(row["memory_id"]),
                    ),
                )
                selected, cost = budgeted(ranked, args.token_budget)
                evaluations = len(available)
            elif method == "recency_bm25":
                ranked = sorted(
                    available,
                    key=lambda row: (
                        -(
                            0.8 * lexical[str(row["memory_id"])] / max_lexical
                            + 0.2
                            * (
                                parse_utc(row["valid_from"], "valid_from").timestamp()
                                - oldest
                            )
                            / span
                        ),
                        str(row["memory_id"]),
                    ),
                )
                selected, cost = budgeted(ranked, args.token_budget)
                evaluations = len(available)
            else:
                request = Request(
                    request_id=str(event["event_id"]),
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
                    request, [str(row["memory_id"]) for row in available], graph
                )
                selected = sorted(result.selected_ids)
                cost = result.token_cost
                status = result.status
                evaluations = result.evaluations
            elapsed = (time.perf_counter() - started) * 1000.0
            rows.append(
                {
                    "run_id": f"{method}:{args.seed}:{event['event_id']}",
                    "event_id": event["event_id"],
                    "seed": args.seed,
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
            "schema_version": "corem-public-service-selections-v1",
            "dataset": str(args.dataset.resolve()),
            "methods": list(METHODS),
            "seeds": [args.seed],
            "token_budget": args.token_budget,
            "selection_rows": len(rows),
            "label_blind": True,
            "method_note": "corem_reference uses the repository implementation with default untrained set utility; it is not a trained full-model estimate.",
        },
    )
    write_checksums(args.output, ("manifest.json", "selections.jsonl"))
    print(f"wrote {len(rows)} selections to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
