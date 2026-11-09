#!/usr/bin/env python3
"""Private AppWorld train-only, generator-held-out API retrieval diagnostic.

This is a proxy component metric, never an AppWorld task-success result. No
protected instructions, solutions, task IDs, or API text are exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.generate_public_workflow_selections import bm25_scores  # noqa: E402


def _choose_memories(
    scores: dict[str, float],
    api_sets: dict[str, set[str]],
    count: int,
    diversity: float | None,
) -> list[str]:
    chosen: list[str] = []
    maximum = max(scores.values(), default=0.0)
    for _ in range(count):
        remaining = sorted(set(scores) - set(chosen))
        if not remaining:
            break

        def value(memory_id: str) -> tuple[float, str]:
            lexical = scores[memory_id] / maximum if maximum > 0 else 0.0
            if diversity is None or not chosen:
                return lexical, memory_id
            overlaps = [
                len(api_sets[memory_id] & api_sets[prior])
                / max(1, len(api_sets[memory_id] | api_sets[prior]))
                for prior in chosen
            ]
            return lexical - diversity * max(overlaps), memory_id

        chosen.append(max(remaining, key=value))
    return chosen


def _rank_memory_apis(
    selected: list[str],
    scores: dict[str, float],
    api_sets: dict[str, set[str]],
    budget: int,
) -> set[str]:
    votes: Counter[str] = Counter()
    for memory_id in selected:
        for api in api_sets[memory_id]:
            votes[api] += max(scores[memory_id], 0.0) + 1e-9
    return set(sorted(votes, key=lambda api: (-votes[api], api))[:budget])


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: round(sum(row[key] for row in rows) / len(rows), 6)
        for key in sorted(rows[0])
    }


def _paired_generator_interval(
    left: list[dict[str, float]],
    right: list[dict[str, float]],
    generators: list[str],
    repetitions: int = 2000,
) -> dict[str, object]:
    if not (len(left) == len(right) == len(generators)):
        raise ValueError("paired rows or groups do not align")
    groups: dict[str, list[float]] = {}
    for left_row, right_row, generator in zip(left, right, generators, strict=True):
        groups.setdefault(generator, []).append(
            left_row["recall"] - right_row["recall"]
        )
    group_names = sorted(groups)
    randomizer = random.Random(2027)
    draws = []
    for _ in range(repetitions):
        sampled = randomizer.choices(group_names, k=len(group_names))
        differences = [difference for name in sampled for difference in groups[name]]
        draws.append(sum(differences) / len(differences))
    draws.sort()
    observed = [
        row["recall"] - control["recall"]
        for row, control in zip(left, right, strict=True)
    ]
    return {
        "mean": round(sum(observed) / len(observed), 6),
        "cluster_bootstrap_95_interval": [
            round(draws[int(0.025 * repetitions)], 6),
            round(draws[int(0.975 * repetitions) - 1], 6),
        ],
        "wins": sum(value > 1e-12 for value in observed),
        "losses": sum(value < -1e-12 for value in observed),
        "ties": sum(abs(value) <= 1e-12 for value in observed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-budget", type=int, default=20)
    parser.add_argument("--memory-count", type=int, default=3)
    parser.add_argument("--diversity", type=float, default=0.7)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing result")
    if args.api_budget < 1 or args.memory_count < 1 or args.diversity < 0:
        parser.error("invalid budget, memory count, or diversity")

    from appworld import load_task_ids, update_root
    from appworld.task import Task

    update_root(str(args.root.resolve()))
    train_ids = load_task_ids("train")
    tasks = [
        Task.load(
            task_id,
            load_ground_truth=True,
            ground_truth_mode="full",
            include_api_response_schemas=False,
        )
        for task_id in train_ids
    ]
    if not tasks or any(task.ground_truth is None for task in tasks):
        raise ValueError("train tasks lack ground truth")
    api_sets = {task.id: set(task.ground_truth.required_apis) for task in tasks}
    docs = [
        {
            "memory_id": f"{app}.{name}",
            "text": " ".join(
                [
                    app,
                    name.replace("_", " "),
                    str(doc["description"]),
                    " ".join(str(parameter["name"]) for parameter in doc["parameters"]),
                ]
            ),
        }
        for app, app_docs in tasks[0].api_docs.items()
        for name, doc in app_docs.items()
    ]
    doc_ids = {row["memory_id"] for row in docs}
    if len(doc_ids) != len(docs):
        raise ValueError("duplicate API documentation")
    metrics: dict[str, list[dict[str, float]]] = {
        name: []
        for name in (
            "api_doc_bm25",
            "memory_bm25",
            "memory_diverse",
            "memory_bm25_padded",
            "memory_diverse_padded",
        )
    }
    ceilings: list[float] = []
    for query in tasks:
        gold = api_sets[query.id]
        if not gold or not gold <= doc_ids:
            raise ValueError("empty or undocumented required API set")
        candidate_tasks = [
            task for task in tasks if task.generator_id != query.generator_id
        ]
        if len(candidate_tasks) >= len(tasks):
            raise ValueError("generator holdout failed")
        candidates = [
            {"memory_id": task.id, "text": task.instruction} for task in candidate_tasks
        ]
        scores = bm25_scores(query.instruction, candidates)
        if set(scores) != {task.id for task in candidate_tasks}:
            raise ValueError("BM25 candidate mismatch")
        seen = set().union(*(api_sets[task.id] for task in candidate_tasks))
        ceilings.append(len(gold & seen) / len(gold))
        rankings = {
            "memory_bm25": _choose_memories(scores, api_sets, args.memory_count, None),
            "memory_diverse": _choose_memories(
                scores, api_sets, args.memory_count, args.diversity
            ),
        }
        predicted = {
            name: _rank_memory_apis(selected, scores, api_sets, args.api_budget)
            for name, selected in rankings.items()
        }
        doc_scores = bm25_scores(query.instruction, docs)
        doc_ranking = sorted(doc_scores, key=lambda api: (-doc_scores[api], api))
        predicted["api_doc_bm25"] = set(doc_ranking[: args.api_budget])
        for name in ("memory_bm25", "memory_diverse"):
            padded = set(predicted[name])
            for api in doc_ranking:
                if len(padded) >= args.api_budget:
                    break
                padded.add(api)
            predicted[name + "_padded"] = padded
        for name, api_predictions in predicted.items():
            overlap = len(api_predictions & gold)
            metrics[name].append(
                {
                    "recall": overlap / len(gold),
                    "precision": overlap / len(api_predictions)
                    if api_predictions
                    else 0.0,
                    "api_count": float(len(api_predictions)),
                }
            )

    output = {
        "status": "development_proxy_only_not_task_success",
        "benchmark": "AppWorld 0.1.3.post1",
        "split": "train_only_leave_generator_out",
        "train_tasks": len(tasks),
        "generator_groups": len({task.generator_id for task in tasks}),
        "documented_apis": len(doc_ids),
        "api_budget": args.api_budget,
        "memory_count": args.memory_count,
        "diversity_penalty": args.diversity,
        "train_split_sha256": hashlib.sha256(
            (args.root / "data/datasets/train.txt").read_bytes()
        ).hexdigest(),
        "memory_seen_api_recall_ceiling": round(sum(ceilings) / len(ceilings), 6),
        "methods": {name: _aggregate(rows) for name, rows in metrics.items()},
        "paired_diverse_minus_bm25_at_20_apis": _paired_generator_interval(
            metrics["memory_diverse_padded"],
            metrics["memory_bm25_padded"],
            [task.generator_id for task in tasks],
        ),
        "protected_task_content_exported": False,
        "probe_loaded_dev_or_test_ground_truth": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
