#!/usr/bin/env python3
"""Generate mechanism-isolating selections for the public v5 challenge."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.model import SetUtilityModel, _dot
from corem.real_eval import (
    load_jsonl,
    parse_utc,
    write_checksums,
    write_json,
    write_jsonl,
)
from corem.retrieval import Bundle, BundleBeamRetriever, RetrievalConfig
from corem.types import EvidenceGraph, Request
from experiments.generate_public_challenge_selections import GENERIC_REQUIRED_ATOMS
from experiments.generate_public_workflow_selections import make_graph

METHODS = (
    "corem_query_only",
    "corem_additive_utility",
    "corem_no_hyperedges",
    "corem_item_only",
    "corem_no_closure",
)


class AdditiveUtilityModel(SetUtilityModel):
    """A1: retain the graph and bundles but remove set interaction terms."""

    def logit(
        self,
        request: Request,
        selected_ids: Iterable[str],
        graph: EvidenceGraph,
    ) -> float:
        selected = frozenset(selected_ids)
        graph._require_records(selected)
        value = _dot(self.base_weights, self._base_features(request))
        for record_id in selected:
            value += _dot(
                self.unary_weights,
                self._unary_features(request, graph.records[record_id], graph),
            )
        return value


class ItemOnlyRetriever(BundleBeamRetriever):
    """A4: expand singleton records only, without dependency-bundle moves."""

    def build_bundles(
        self,
        request: Request,
        bank: frozenset[str],
        graph: EvidenceGraph,
    ) -> tuple[Bundle, ...]:
        eligible = [key for key in bank if graph.is_admissible(key, request)]
        ranked = sorted(
            eligible,
            key=lambda key: (self.model.semantic_similarity(request, key, graph), key),
            reverse=True,
        )
        return tuple(
            Bundle(f"record:{key}", frozenset((key,)))
            for key in ranked[: self.config.candidate_pool_size]
        )


def request_for(event: dict[str, Any]) -> Request:
    return Request(
        request_id=str(event["event_id"]),
        text=str(event["query"]),
        required_atoms=GENERIC_REQUIRED_ATOMS,
        operation_ids=frozenset(),
        timestamp=parse_utc(str(event["event_time"]), "event_time").timestamp(),
        allowed_scopes=frozenset((str(event["service_id"]),)),
        active_versions=frozenset((str(event["version_id"]),)),
    )


def config(token_budget: int, *, closure: bool) -> RetrievalConfig:
    return RetrievalConfig(
        retrieval_budget=token_budget,
        beam_width=16,
        max_depth=4,
        candidate_pool_size=100,
        dependency_hops=2,
        max_closure_size=8,
        enforce_dependency_closure=closure,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2

    events = load_jsonl(args.dataset / "events.jsonl")
    memories = load_jsonl(args.dataset / "memories.jsonl")
    hyperedges = load_jsonl(args.dataset / "dependency_hyperedges.jsonl")
    full_graph = make_graph(memories, hyperedges)
    no_hyperedge_graph = make_graph(memories, [])
    retrievers: dict[str, tuple[BundleBeamRetriever, EvidenceGraph]] = {
        "corem_query_only": (
            BundleBeamRetriever(
                SetUtilityModel(), config(args.token_budget, closure=True)
            ),
            full_graph,
        ),
        "corem_additive_utility": (
            BundleBeamRetriever(
                AdditiveUtilityModel(), config(args.token_budget, closure=True)
            ),
            full_graph,
        ),
        "corem_no_hyperedges": (
            BundleBeamRetriever(
                SetUtilityModel(), config(args.token_budget, closure=True)
            ),
            no_hyperedge_graph,
        ),
        "corem_item_only": (
            ItemOnlyRetriever(
                SetUtilityModel(), config(args.token_budget, closure=False)
            ),
            full_graph,
        ),
        "corem_no_closure": (
            BundleBeamRetriever(
                SetUtilityModel(), config(args.token_budget, closure=False)
            ),
            full_graph,
        ),
    }

    rows: list[dict[str, Any]] = []
    ordered_events = sorted(
        events,
        key=lambda row: (
            parse_utc(row["event_time"], "event_time"),
            str(row["event_id"]),
        ),
    )
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
        request = request_for(event)
        for method in METHODS:
            retriever, graph = retrievers[method]
            started = time.perf_counter()
            result = retriever.retrieve(
                request,
                [str(row["memory_id"]) for row in available],
                graph,
            )
            rows.append(
                {
                    "run_id": f"{method}:{args.seed}:{event_id}",
                    "event_id": event_id,
                    "seed": args.seed,
                    "method": method,
                    "selected_memory_ids": sorted(result.selected_ids),
                    "token_cost": result.token_cost,
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                    "llm_calls": 0,
                    "active_bank_cost": sum(
                        int(row["storage_cost"]) for row in available
                    ),
                    "retrieval_status": result.status,
                    "evaluations": result.evaluations,
                }
            )

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "selections.jsonl", rows)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": "corem-public-multiservice-ablation-selections-v1",
            "dataset": str(args.dataset.resolve()),
            "methods": list(METHODS),
            "seed": args.seed,
            "token_budget": args.token_budget,
            "selection_rows": len(rows),
            "label_blind": True,
            "ablation_contracts": {
                "corem_additive_utility": (
                    "A1: retain hyperedges, dependency bundles, and the encoder; "
                    "remove complement and redundancy contributions from set utility."
                ),
                "corem_no_hyperedges": (
                    "A2: remove dependency hyperedges from graph input, bundle "
                    "construction, closure, and set utility."
                ),
                "corem_item_only": (
                    "A4: keep the same graph and utility but permit singleton search "
                    "moves only, with no automatic closure."
                ),
                "corem_no_closure": (
                    "A5: retain explicit dependency bundles but disable automatic "
                    "closure of arbitrary search states."
                ),
            },
            "seed_note": (
                "The CoReM-family selectors are deterministic and do not consume the "
                "seed; one seed avoids presenting duplicate executions as independent."
            ),
        },
    )
    write_checksums(args.output, ("manifest.json", "selections.jsonl"))
    print(f"wrote {len(rows)} mechanism-ablation selections to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
