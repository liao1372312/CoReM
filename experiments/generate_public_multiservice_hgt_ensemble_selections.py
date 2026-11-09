#!/usr/bin/env python3
"""Apply a frozen HGT/BM25 ensemble to label-blind bundle candidates."""

from __future__ import annotations

import argparse
import hashlib
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import (  # noqa: E402
    load_jsonl,
    parse_utc,
    write_checksums,
    write_json,
    write_jsonl,
)
from experiments.evaluate_public_multiservice_hgt_ensemble import load_models  # noqa: E402
from experiments.generate_public_multiservice_challenge_ablations import (  # noqa: E402
    request_for,
)
from experiments.generate_public_workflow_selections import (  # noqa: E402
    bm25_scores,
    make_graph,
)
from experiments.train_public_multiservice_hgt import VARIANTS  # noqa: E402

METHOD = "hgt_rank_ensemble"
TRAINING_RELATION_TYPE = "three-step-runtime-handoff"
TRANSFER_RELATION_TYPES = frozenset({"role-balanced-v9-staging-workflow"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--candidate-selections", type=Path, required=True)
    parser.add_argument(
        "--event-view",
        type=Path,
        help="Optional query-only event view; excludes checker and answer fields.",
    )
    parser.add_argument("--model-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default=METHOD)
    parser.add_argument("--state-variant", choices=VARIANTS, default="hgt_corem")
    parser.add_argument(
        "--model-variant",
        choices=VARIANTS,
        help="Inference architecture; defaults to --state-variant.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--provider",
        help="Optionally restrict the target split to one provider.",
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--neural-weight", type=float, default=0.9)
    parser.add_argument(
        "--lexical-aggregation",
        choices=("max", "sum", "mean"),
        default="max",
        help="Aggregate record BM25 within a complete bundle; max preserves old runs.",
    )
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--text-hash-dimension", type=int, default=64)
    parser.add_argument("--message-layers", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lexical-features", action="store_true")
    parser.add_argument(
        "--relation-vocabulary-policy",
        choices=("native", "collapse_to_training"),
        default="native",
        help="Label-free zero-shot projection of new handoff lengths to the frozen training relation.",
    )
    parser.add_argument(
        "--require-label-free-candidates",
        action="store_true",
        help="Reject candidate inputs carrying methods, outcomes, or other fields.",
    )
    parser.add_argument(
        "--tie-break-policy",
        choices=("legacy_memory_id", "seeded_hash"),
        default="legacy_memory_id",
        help="Use seeded_hash for new sealed evaluations; legacy preserves old runs.",
    )
    parser.add_argument(
        "--near-tie-epsilon",
        type=float,
        default=1e-6,
        help="Report scores this close as indistinguishable (no outcome lookup).",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_label_free_candidate(row: dict[str, Any]) -> None:
    allowed = {"event_id", "candidate_edge_id", "selected_memory_ids"}
    if set(row) != allowed:
        raise ValueError(
            "sealed candidate input must have only event_id, "
            "candidate_edge_id, and selected_memory_ids"
        )
    if any(
        forbidden in str(row).lower()
        for forbidden in ("oracle", "complete_negative", "goal_satisfied")
    ):
        raise ValueError("sealed candidate input contains answer vocabulary")
    if re.search(r"(?:^|:)bundle_[ab](?::|\b)", str(row).lower()):
        raise ValueError("sealed candidate input exposes a predictable bundle side")


def validate_query_only_event(row: dict[str, Any]) -> None:
    allowed = {
        "event_id",
        "service_id",
        "provider_id",
        "version_id",
        "event_time",
        "query",
    }
    if set(row) != allowed:
        raise ValueError("sealed event view must contain only query and scope fields")


def lexical_bundle_score(
    lexical: dict[str, float], selected: frozenset[str], aggregation: str
) -> float:
    values = [float(lexical[memory_id]) for memory_id in selected]
    if not values:
        raise ValueError("cannot score an empty candidate bundle")
    if aggregation == "max":
        return max(values)
    if aggregation == "sum":
        return sum(values)
    if aggregation == "mean":
        return statistics.mean(values)
    raise ValueError(f"unknown lexical aggregation: {aggregation}")


def project_handoff_relation_vocabulary(
    hyperedges: list[dict[str, Any]], policy: str
) -> list[dict[str, Any]]:
    """Map unseen operation-length relation names onto the frozen v5 schema."""
    if policy == "native":
        return hyperedges
    if policy != "collapse_to_training":
        raise ValueError(f"unknown relation vocabulary policy: {policy}")
    projected = []
    for edge in hyperedges:
        relation = str(edge["relation_type"])
        if (
            relation != TRAINING_RELATION_TYPE
            and relation not in TRANSFER_RELATION_TYPES
            and not re.fullmatch(r"\d+-step-runtime-handoff", relation)
        ):
            raise ValueError(f"cannot project unrecognized relation: {relation}")
        projected.append({**edge, "relation_type": TRAINING_RELATION_TYPE})
    return projected


def rank_candidates(
    rows: list[dict[str, Any]], event_id: str, seed: int, policy: str
) -> list[dict[str, Any]]:
    """Break exact score ties without consulting labels or execution results."""
    if policy == "seeded_hash":

        def key(row: dict[str, Any]) -> tuple[float, str]:
            opaque = "|".join(map(str, row["selected_memory_ids"]))
            digest = hashlib.sha256(f"{seed}|{event_id}|{opaque}".encode()).hexdigest()
            return float(row["score"]), digest
    elif policy == "legacy_memory_id":

        def key(row: dict[str, Any]) -> tuple[float, str]:
            return float(row["score"]), "|".join(map(str, row["selected_memory_ids"]))
    else:
        raise ValueError(f"unsupported tie-break policy: {policy}")
    return sorted(rows, key=key, reverse=True)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    if not 0.0 <= args.neural_weight <= 1.0:
        raise ValueError("neural weight must be between zero and one")
    if args.torch_threads < 1:
        raise ValueError("torch threads must be positive")
    if args.near_tie_epsilon < 0:
        raise ValueError("near-tie epsilon must be nonnegative")
    torch.set_num_threads(args.torch_threads)

    event_rows = load_jsonl(args.event_view or args.dataset / "events.jsonl")
    if args.event_view is not None:
        for row in event_rows:
            validate_query_only_event(row)
    events = {str(row["event_id"]): row for row in event_rows}
    target_ids = {
        str(row["event_id"])
        for row in load_jsonl(args.splits)
        if row["split"] == args.split
        and (
            args.provider is None
            or events[str(row["event_id"])]["provider_id"] == args.provider
        )
    }
    if not target_ids:
        raise ValueError("target split/provider has no events")
    memories = load_jsonl(args.dataset / "memories.jsonl")
    hyperedges = load_jsonl(args.dataset / "dependency_hyperedges.jsonl")
    full_graph = make_graph(memories, hyperedges)
    effective_model_variant = args.model_variant or args.state_variant
    if effective_model_variant == "hgt_no_hyperedges":
        model_graph = make_graph(memories, [])
    elif args.relation_vocabulary_policy == "native":
        model_graph = full_graph
    else:
        model_graph = make_graph(
            memories,
            project_handoff_relation_vocabulary(
                hyperedges, args.relation_vocabulary_policy
            ),
        )
    models = load_models(
        args.model_dirs,
        model_graph,
        dimension=args.dimension,
        text_hash_dimension=args.text_hash_dimension,
        message_layers=args.message_layers,
        device=args.device,
        lexical_features=args.lexical_features,
        state_variant=args.state_variant,
        model_variant=effective_model_variant,
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(args.candidate_selections):
        if args.require_label_free_candidates:
            validate_label_free_candidate(row)
        event_id = str(row["event_id"])
        if event_id in target_ids:
            grouped[event_id].append(row)
    if set(grouped) != target_ids:
        missing = sorted(target_ids.difference(grouped))
        raise ValueError(f"candidate enumeration misses target events: {missing[:3]}")

    output = []
    score_rows = []
    for event_id, candidates in sorted(grouped.items()):
        started = time.perf_counter()
        event = events[event_id]
        request = request_for(event)
        timestamp = parse_utc(event["event_time"], "event_time").timestamp()
        active = [
            row
            for row in memories
            if row.get("service_id") == event.get("service_id")
            and event.get("version_id") in row.get("version_scope", [])
            and parse_utc(row["valid_from"], "valid_from").timestamp() <= timestamp
        ]
        lexical = bm25_scores(request.text, active)
        candidate_scores = []
        for candidate in candidates:
            selected = frozenset(map(str, candidate["selected_memory_ids"]))
            if not full_graph.is_evidence_complete(request, selected):
                raise ValueError(f"incomplete candidate: {candidate['run_id']}")
            raw_bm25 = max(lexical[memory_id] for memory_id in selected)
            bundle_bm25 = lexical_bundle_score(
                lexical, selected, args.lexical_aggregation
            )
            probabilities = [
                model.probability(request, selected, model_graph)
                for _, model, _ in models
            ]
            candidate_scores.append(
                {
                    "event_id": event_id,
                    "candidate_edge_id": candidate.get("candidate_edge_id"),
                    "selected_memory_ids": sorted(selected),
                    "ensemble_probability": statistics.mean(probabilities),
                    "seed_probabilities": [
                        {"seed": seed, "probability": probability}
                        for (seed, _, _), probability in zip(models, probabilities)
                    ],
                    "seed_probability_std": (
                        statistics.stdev(probabilities)
                        if len(probabilities) > 1
                        else 0.0
                    ),
                    "bm25_max": raw_bm25,
                    "bm25_raw": bundle_bm25,
                }
            )
        low = min(float(row["bm25_raw"]) for row in candidate_scores)
        high = max(float(row["bm25_raw"]) for row in candidate_scores)
        for row in candidate_scores:
            bm25_normalized = (
                (float(row["bm25_raw"]) - low) / (high - low) if high > low else 0.0
            )
            row["bm25_normalized"] = bm25_normalized
            row["score"] = (
                args.neural_weight * float(row["ensemble_probability"])
                + (1.0 - args.neural_weight) * bm25_normalized
            )
        ranked = rank_candidates(
            candidate_scores, event_id, args.seed, args.tie_break_policy
        )
        selected = ranked[0]
        top_margin = (
            float(ranked[0]["score"]) - float(ranked[1]["score"])
            if len(ranked) > 1
            else None
        )
        selected_ids = list(selected["selected_memory_ids"])
        cost = sum(
            full_graph.records[memory_id].token_cost for memory_id in selected_ids
        )
        output.append(
            {
                "run_id": f"{args.method}:{args.seed}:{event_id}",
                "event_id": event_id,
                "seed": args.seed,
                "method": args.method,
                "selected_memory_ids": selected_ids,
                "token_cost": cost,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "llm_calls": 0,
                "active_bank_cost": sum(int(row["storage_cost"]) for row in active),
                "retrieval_status": "ok",
                "evaluations": len(candidates),
                "ensemble_probability": selected["ensemble_probability"],
                "bm25_normalized": selected["bm25_normalized"],
                "selection_score": selected["score"],
                "top_score_margin": top_margin,
                "near_tie": (
                    top_margin is not None and top_margin <= args.near_tie_epsilon
                ),
            }
        )
        score_rows.extend(candidate_scores)

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "selections.jsonl", output)
    write_jsonl(args.output / "candidate_scores.jsonl", score_rows)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": "corem-public-hgt-ensemble-selections-v1",
            "dataset": str(args.dataset.resolve()),
            "event_view": (
                {
                    "path": str(args.event_view.resolve()),
                    "sha256": sha256(args.event_view),
                }
                if args.event_view is not None
                else None
            ),
            "target_split": args.split,
            "provider": args.provider,
            "method": args.method,
            "seed": args.seed,
            "state_variant": args.state_variant,
            "model_variant": effective_model_variant,
            "model_seeds": [seed for seed, _, _ in models],
            "model_states": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for _, _, path in models
            ],
            "neural_weight": args.neural_weight,
            "bm25_weight": 1.0 - args.neural_weight,
            "lexical_aggregation": args.lexical_aggregation,
            "relation_vocabulary_policy": args.relation_vocabulary_policy,
            "training_relation_type": TRAINING_RELATION_TYPE,
            "torch_threads": args.torch_threads,
            "tie_break_policy": args.tie_break_policy,
            "near_tie_epsilon": args.near_tie_epsilon,
            "selection_rows": len(output),
            "label_blind": True,
            "label_free_candidate_input_enforced": args.require_label_free_candidates,
            "frozen_selection_contract": (
                "Average independently trained model probabilities and combine them "
                "with per-event normalized BM25 using the explicitly supplied frozen "
                "residual weight. Rank only the same pre-enumerated, current-version, "
                "role-complete bundles. State and inference variants are recorded "
                "separately so head-off diagnostics cannot be mistaken for retraining."
            ),
            "outcomes_loaded": False,
        },
    )
    write_checksums(
        args.output,
        ("candidate_scores.jsonl", "manifest.json", "selections.jsonl"),
    )
    print(f"wrote {len(output)} frozen ensemble selections to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
