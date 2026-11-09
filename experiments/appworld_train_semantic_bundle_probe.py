#!/usr/bin/env python3
"""Train-only AppWorld bundle-retrieval diagnostic with strong semantic controls.

Reads a protected local train view and writes only aggregate proxy metrics.
Never use its API-recall numbers as state-based AppWorld task success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "data" / "private_corem"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.appworld_train_retrieval_probe import (  # noqa: E402
    _aggregate,
    _choose_memories,
    _paired_generator_interval,
    _rank_memory_apis,
)
from experiments.generate_public_workflow_selections import bm25_scores  # noqa: E402


def _select_bundles(
    candidates: list[str],
    similarity: dict[str, float],
    api_sets: dict[str, set[str]],
    api_evidence: dict[str, float],
    count: int,
    beta: float,
    gamma: float,
) -> list[str]:
    """Finite greedy bundle selection; gold query APIs are never read here."""
    selected: list[str] = []
    covered: set[str] = set()
    for _ in range(min(count, len(candidates))):
        remaining = sorted(set(candidates) - set(selected))

        def utility(memory_id: str) -> tuple[float, str]:
            bundle = api_sets[memory_id]
            new = bundle - covered
            doc_gain = sum(max(api_evidence[api], 0.0) for api in new) / len(bundle)
            overlap = len(bundle & covered) / len(bundle)
            return (
                similarity[memory_id] + beta * doc_gain - gamma * overlap,
                memory_id,
            )

        chosen = max(remaining, key=utility)
        selected.append(chosen)
        covered.update(api_sets[chosen])
    return selected


def _pad_apis(
    selected: list[str],
    scores: dict[str, float],
    api_sets: dict[str, set[str]],
    doc_ranking: list[str],
    budget: int,
) -> set[str]:
    predicted = _rank_memory_apis(selected, scores, api_sets, budget)
    for api in doc_ranking:
        if len(predicted) >= budget:
            break
        predicted.add(api)
    if len(predicted) != budget:
        raise ValueError("API budget could not be filled")
    return predicted


def main() -> int:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-view", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--api-budget", type=int, default=20)
    parser.add_argument("--memory-count", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.25)
    args = parser.parse_args()
    private_view = args.private_view.resolve()
    output = args.output.resolve()
    if not private_view.is_relative_to(
        PRIVATE_ROOT.resolve()
    ) or not output.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error("protected inputs and outputs must stay under data/private_corem")
    if output.exists():
        parser.error("refusing to overwrite an existing result")
    if args.api_budget < 1 or args.memory_count < 1 or min(args.beta, args.gamma) < 0:
        parser.error("invalid budget or algorithm weight")

    payload = json.loads(private_view.read_text())
    if payload.get("status") != "protected_local_only_train_development":
        raise ValueError("not a train-only protected view")
    tasks = payload["tasks"]
    docs = payload["api_docs"]
    if len(tasks) != 90 or len(docs) != 457:
        raise ValueError("unexpected benchmark release or incomplete view")
    task_ids = [str(row["task_id"]) for row in tasks]
    doc_ids = [str(row["api"]) for row in docs]
    if len(set(task_ids)) != len(tasks) or len(set(doc_ids)) != len(docs):
        raise ValueError("duplicate task or API IDs")
    api_sets = {str(row["task_id"]): set(row["required_apis"]) for row in tasks}
    if any(not value or not value <= set(doc_ids) for value in api_sets.values()):
        raise ValueError("empty or undocumented API set")

    model = SentenceTransformer(str(args.model_dir.resolve()), device=args.device)
    task_vectors = model.encode(
        [row["instruction"] for row in tasks],
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    doc_vectors = model.encode(
        [row["text"] for row in docs],
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if not np.isfinite(task_vectors).all() or not np.isfinite(doc_vectors).all():
        raise ValueError("nonfinite semantic embeddings")
    task_similarities = task_vectors @ task_vectors.T
    doc_similarities = task_vectors @ doc_vectors.T
    doc_rows = [{"memory_id": row["api"], "text": row["text"]} for row in docs]
    methods: dict[str, list[dict[str, float]]] = {
        name: []
        for name in (
            "api_doc_bm25",
            "api_doc_semantic",
            "memory_bm25_padded",
            "memory_semantic_padded",
            "bundle_full",
            "bundle_no_doc_evidence",
            "bundle_no_overlap_penalty",
        )
    }
    for query_index, query in enumerate(tasks):
        query_id = query["task_id"]
        gold = api_sets[query_id]
        candidate_rows = [
            row for row in tasks if row["generator_id"] != query["generator_id"]
        ]
        if len(candidate_rows) != len(tasks) - 3 or any(
            row["generator_id"] == query["generator_id"] for row in candidate_rows
        ):
            raise ValueError("generator holdout failed")
        candidates = [row["task_id"] for row in candidate_rows]
        lexical = bm25_scores(
            query["instruction"],
            [
                {"memory_id": row["task_id"], "text": row["instruction"]}
                for row in candidate_rows
            ],
        )
        semantic = {
            task_id: float(task_similarities[query_index, index])
            for index, task_id in enumerate(task_ids)
            if task_id in candidates
        }
        evidence = {
            doc_id: float(doc_similarities[query_index, index])
            for index, doc_id in enumerate(doc_ids)
        }
        if any(not math.isfinite(value) for value in semantic.values()):
            raise ValueError("nonfinite semantic similarities")
        doc_lexical = bm25_scores(query["instruction"], doc_rows)
        lexical_doc_rank = sorted(doc_ids, key=lambda api: (-doc_lexical[api], api))
        semantic_doc_rank = sorted(doc_ids, key=lambda api: (-evidence[api], api))
        chosen = {
            "memory_bm25_padded": _choose_memories(
                lexical, api_sets, args.memory_count, None
            ),
            "memory_semantic_padded": _choose_memories(
                semantic, api_sets, args.memory_count, None
            ),
            "bundle_full": _select_bundles(
                candidates,
                semantic,
                api_sets,
                evidence,
                args.memory_count,
                args.beta,
                args.gamma,
            ),
            "bundle_no_doc_evidence": _select_bundles(
                candidates,
                semantic,
                api_sets,
                evidence,
                args.memory_count,
                0.0,
                args.gamma,
            ),
            "bundle_no_overlap_penalty": _select_bundles(
                candidates,
                semantic,
                api_sets,
                evidence,
                args.memory_count,
                args.beta,
                0.0,
            ),
        }
        predicted = {
            "api_doc_bm25": set(lexical_doc_rank[: args.api_budget]),
            "api_doc_semantic": set(semantic_doc_rank[: args.api_budget]),
            "memory_bm25_padded": _pad_apis(
                chosen["memory_bm25_padded"],
                lexical,
                api_sets,
                lexical_doc_rank,
                args.api_budget,
            ),
        }
        for name in (
            "memory_semantic_padded",
            "bundle_full",
            "bundle_no_doc_evidence",
            "bundle_no_overlap_penalty",
        ):
            predicted[name] = _pad_apis(
                chosen[name], semantic, api_sets, semantic_doc_rank, args.api_budget
            )
        for name, api_predictions in predicted.items():
            hit = len(gold & api_predictions)
            methods[name].append(
                {
                    "recall": hit / len(gold),
                    "precision": hit / len(api_predictions),
                    "api_count": float(len(api_predictions)),
                }
            )

    generators = [str(row["generator_id"]) for row in tasks]
    comparisons = {
        f"bundle_full_minus_{control}": _paired_generator_interval(
            methods["bundle_full"], methods[control], generators
        )
        for control in (
            "memory_semantic_padded",
            "memory_bm25_padded",
            "bundle_no_doc_evidence",
            "bundle_no_overlap_penalty",
        )
    }
    output_payload = {
        "status": "train_only_component_proxy_not_appworld_success",
        "benchmark": payload["benchmark"],
        "split": "train_only_leave_generator_out",
        "train_tasks": len(tasks),
        "generator_groups": len(set(generators)),
        "documented_apis": len(doc_ids),
        "api_budget": args.api_budget,
        "memory_count": args.memory_count,
        "beta": args.beta,
        "gamma": args.gamma,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "model_weights_sha256": hashlib.sha256(
            (args.model_dir / "model.safetensors").read_bytes()
        ).hexdigest(),
        "private_view_sha256": hashlib.sha256(private_view.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "methods": {name: _aggregate(rows) for name, rows in methods.items()},
        "paired_differences": comparisons,
        "protected_task_content_exported": False,
        "dev_or_test_task_loaded": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
