#!/usr/bin/env python3
"""Run CoReM E6 exact/approximate retrieval and retention diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_jsonl, write_checksums, write_json, write_jsonl  # noqa: E402
from corem.retention import GroupAwareRetention, RetentionConfig  # noqa: E402
from corem.retrieval import BundleBeamRetriever, RetrievalConfig  # noqa: E402
from corem.types import EvidenceGraph, Request  # noqa: E402
from experiments.run_expanded_hsc_corem import (  # noqa: E402
    FamilyView,
    LexicalResidualModel,
    build_graph,
    fit_model,
    verify_checksums,
)


def objective(
    model: LexicalResidualModel,
    request: Request,
    selected: frozenset[str],
    graph: EvidenceGraph,
    budget: int,
) -> float:
    probability = model.probability(request, selected, graph)
    cost = sum(graph.records[item].token_cost for item in selected)
    return probability - 0.05 * cost / budget


def exact_retrieve(
    model: LexicalResidualModel,
    view: FamilyView,
    candidate_ids: tuple[str, ...],
    graph: EvidenceGraph,
    budget: int,
) -> tuple[frozenset[str], float]:
    best_set = frozenset()
    best_value = -math.inf
    for count in range(1, len(candidate_ids) + 1):
        for chosen in itertools.combinations(candidate_ids, count):
            selected = frozenset(
                record_id for candidate_id in chosen for record_id in view.members[candidate_id]
            )
            cost = sum(graph.records[item].token_cost for item in selected)
            if cost > budget:
                continue
            value = objective(model, view.request, selected, graph, budget)
            if (value, -cost, sorted(selected)) > (
                best_value,
                -sum(graph.records[item].token_cost for item in best_set),
                sorted(best_set),
            ):
                best_set, best_value = selected, value
    if not best_set:
        raise ValueError("small exact instance has no feasible complete set")
    return best_set, best_value


def greedy_retrieve(
    model: LexicalResidualModel,
    view: FamilyView,
    candidate_ids: tuple[str, ...],
    graph: EvidenceGraph,
    budget: int,
) -> tuple[frozenset[str], float]:
    selected = frozenset()
    value = -math.inf
    remaining = set(candidate_ids)
    while remaining:
        proposals = []
        for candidate_id in remaining:
            proposal = selected.union(view.members[candidate_id])
            cost = sum(graph.records[item].token_cost for item in proposal)
            if cost <= budget:
                proposals.append(
                    (objective(model, view.request, proposal, graph, budget), candidate_id, proposal)
                )
        if not proposals:
            break
        proposal_value, candidate_id, proposal = max(
            proposals, key=lambda item: (item[0], item[1])
        )
        if selected and proposal_value <= value:
            break
        selected, value = frozenset(proposal), proposal_value
        remaining.remove(candidate_id)
    return selected, value


def feasible_retrieval(
    selected: frozenset[str],
    view: FamilyView,
    graph: EvidenceGraph,
    budget: int,
) -> bool:
    cost = sum(graph.records[item].token_cost for item in selected)
    complete_candidates = [
        candidate_id
        for candidate_id in view.candidate_ids[:4]
        if view.members[candidate_id].issubset(selected)
    ]
    union = frozenset(
        record_id
        for candidate_id in complete_candidates
        for record_id in view.members[candidate_id]
    )
    return bool(selected) and selected == union and cost <= budget


def bank_value(
    model: LexicalResidualModel,
    view: FamilyView,
    bank: frozenset[str],
    graph: EvidenceGraph,
    retrieval_budget: int,
) -> float:
    result = BundleBeamRetriever(
        model,  # type: ignore[arg-type]
        RetrievalConfig(
            retrieval_budget=retrieval_budget,
            beam_width=8,
            max_depth=2,
            candidate_pool_size=32,
            dependency_hops=2,
            max_closure_size=7,
            context_cost_weight=0.05,
        ),
    ).retrieve(view.request, bank, graph)
    return result.objective if result.status == "ok" else 0.0


def exact_retain(
    model: LexicalResidualModel,
    view: FamilyView,
    candidate_ids: tuple[str, ...],
    graph: EvidenceGraph,
    storage_budget: int,
    retrieval_budget: int,
) -> tuple[frozenset[str], float]:
    best_bank = frozenset()
    best_value = -math.inf
    for count in range(1, len(candidate_ids) + 1):
        for chosen in itertools.combinations(candidate_ids, count):
            bank = frozenset(
                record_id
                for candidate_id in chosen
                for record_id in view.members[candidate_id]
            )
            cost = sum(graph.records[item].storage_cost for item in bank)
            if cost > storage_budget:
                continue
            value = bank_value(model, view, bank, graph, retrieval_budget)
            if (value, -cost, sorted(bank)) > (
                best_value,
                -sum(graph.records[item].storage_cost for item in best_bank),
                sorted(best_bank),
            ):
                best_bank, best_value = bank, value
    return best_bank, best_value


def normalized_gap(exact: float, observed: float) -> float:
    return max(0.0, exact - observed) / max(abs(exact), 1e-9) * 100.0


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["target"]), str(row["solver"]))].append(row)
    return [
        {
            "target": target,
            "solver": solver,
            "instances": len(members),
            "mean_gap_pct": statistics.mean(float(row["gap_pct"]) for row in members),
            "feasible_pct": 100.0
            * statistics.mean(bool(row["feasible"]) for row in members),
            "mean_time_ms": statistics.mean(float(row["time_ms"]) for row in members),
        }
        for (target, solver), members in sorted(grouped.items())
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    verify_checksums(args.dataset)
    instances = load_jsonl(args.dataset / "instances.jsonl")
    candidates_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(args.dataset / "candidates.jsonl"):
        candidates_by_family[str(row["family_id"])].append(row)
    labels = {
        str(row["family_id"]): str(row["gold_candidate_id"])
        for row in load_jsonl(args.dataset / "labels.jsonl")
    }
    train_instances = []
    test_instances = []
    for instance in instances:
        family_id = str(instance["family_id"])
        if instance["split"] == "train":
            gold = labels[family_id]
            negative = next(item for item in instance["candidate_ids"] if item != gold)
            train_instances.append({**instance, "candidate_ids": [gold, negative]})
        elif instance["split"] == "test" and int(instance["depth"]) in (2, 3):
            test_instances.append(instance)
    train_graph, train_views, train_qos = build_graph(
        train_instances, candidates_by_family, labels, graph_enabled=True
    )
    test_graph, test_views, test_qos = build_graph(
        test_instances, candidates_by_family, labels, graph_enabled=True
    )
    model, fit_info = fit_model(
        train_graph,
        train_views,
        seed=args.seed,
        graph_enabled=True,
        set_enabled=True,
        lexical_enabled=True,
        training_fraction=1.0,
        qos_scores={**train_qos, **test_qos},
    )
    views = sorted(test_views.values(), key=lambda item: item.family_id)[:12]
    if len(views) != 12:
        raise ValueError("expected 12 small held-out instances")

    rows: list[dict[str, Any]] = []
    for view in views:
        candidate_ids = view.candidate_ids[:4]
        bank = frozenset(
            record_id for candidate_id in candidate_ids for record_id in view.members[candidate_id]
        )
        if not 8 <= len(bank) <= 12:
            raise ValueError("E6 retrieval instance must contain 8--12 records")
        retrieval_budget = 2 * view.depth * 24
        started = time.perf_counter()
        exact_set, exact_value = exact_retrieve(
            model, view, candidate_ids, test_graph, retrieval_budget
        )
        rows.append(
            {
                "family_id": view.family_id,
                "target": "Retrieve",
                "solver": "Exact enumeration",
                "gap_pct": 0.0,
                "feasible": feasible_retrieval(exact_set, view, test_graph, retrieval_budget),
                "time_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
        started = time.perf_counter()
        greedy_set, greedy_value = greedy_retrieve(
            model, view, candidate_ids, test_graph, retrieval_budget
        )
        rows.append(
            {
                "family_id": view.family_id,
                "target": "Retrieve",
                "solver": "Bundle greedy",
                "gap_pct": normalized_gap(exact_value, greedy_value),
                "feasible": feasible_retrieval(greedy_set, view, test_graph, retrieval_budget),
                "time_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
        for beam_width in (1, 4, 8):
            retriever = BundleBeamRetriever(
                model,  # type: ignore[arg-type]
                RetrievalConfig(
                    retrieval_budget=retrieval_budget,
                    beam_width=beam_width,
                    max_depth=2,
                    candidate_pool_size=32,
                    dependency_hops=2,
                    max_closure_size=7,
                    context_cost_weight=0.05,
                ),
            )
            started = time.perf_counter()
            result = retriever.retrieve(view.request, bank, test_graph)
            elapsed = (time.perf_counter() - started) * 1000.0
            rows.append(
                {
                    "family_id": view.family_id,
                    "target": "Retrieve",
                    "solver": f"Beam Kb={beam_width}",
                    "gap_pct": normalized_gap(exact_value, result.objective),
                    "feasible": result.status == "ok"
                    and feasible_retrieval(result.selected_ids, view, test_graph, retrieval_budget),
                    "time_ms": elapsed,
                }
            )

        storage_budget = retrieval_budget
        started = time.perf_counter()
        exact_bank, exact_bank_value = exact_retain(
            model,
            view,
            candidate_ids,
            test_graph,
            storage_budget,
            view.depth * 24,
        )
        rows.append(
            {
                "family_id": view.family_id,
                "target": "Retain",
                "solver": "Exact enumeration",
                "gap_pct": 0.0,
                "feasible": sum(test_graph.records[item].storage_cost for item in exact_bank)
                <= storage_budget,
                "time_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
        started = time.perf_counter()
        ranked_items = sorted(
            bank,
            key=lambda item: (
                -model.semantic_similarity(view.request, item, test_graph),
                item,
            ),
        )
        independent: set[str] = set()
        cost = 0
        for item in ranked_items:
            item_cost = test_graph.records[item].storage_cost
            if cost + item_cost <= storage_budget:
                independent.add(item)
                cost += item_cost
        independent_bank = frozenset(independent)
        independent_value = bank_value(
            model, view, independent_bank, test_graph, view.depth * 24
        )
        rows.append(
            {
                "family_id": view.family_id,
                "target": "Retain",
                "solver": "Independent item",
                "gap_pct": normalized_gap(exact_bank_value, independent_value),
                "feasible": cost <= storage_budget,
                "time_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
        retriever = BundleBeamRetriever(
            model,  # type: ignore[arg-type]
            RetrievalConfig(
                retrieval_budget=view.depth * 24,
                beam_width=8,
                max_depth=2,
                candidate_pool_size=32,
                dependency_hops=2,
                max_closure_size=7,
                context_cost_weight=0.05,
            ),
        )
        retainer = GroupAwareRetention(
            retriever,
            RetentionConfig(
                storage_budget=storage_budget,
                max_proposals=8,
                eviction_shortlist_size=8,
                max_eviction_group=7,
                max_proposal_group=7,
                relaxation_steps=2,
            ),
        )
        initial = view.members[candidate_ids[0]]
        started = time.perf_counter()
        retained = retainer.optimize(
            initial, bank.difference(initial), (view.request,), test_graph
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        observed_value = bank_value(
            model, view, retained.retained_ids, test_graph, view.depth * 24
        )
        rows.append(
            {
                "family_id": view.family_id,
                "target": "Retain",
                "solver": "Group swap",
                "gap_pct": normalized_gap(exact_bank_value, observed_value),
                "feasible": retained.storage_cost <= storage_budget,
                "time_ms": elapsed,
            }
        )

    summary = summarize(rows)
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "instance_results.jsonl", rows)
    write_json(
        args.output / "result.json",
        {
            "status": "complete_corem_e6",
            "schema_version": "corem-expanded-e6-v1",
            "dataset": str(args.dataset.resolve()),
            "dataset_checksums_sha256": hashlib.sha256(
                (args.dataset / "checksums.sha256").read_bytes()
            ).hexdigest(),
            "seed": args.seed,
            "instances": len(views),
            "fit": fit_info,
            "summary": summary,
            "claim_boundary": (
                "Small HSC candidate graphs only; exact enumeration is a diagnostic "
                "reference and supplies no general approximation guarantee."
            ),
        },
    )
    write_checksums(args.output, ("instance_results.jsonl", "result.json"))
    print(f"completed E6 on {len(views)} held-out small instances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
