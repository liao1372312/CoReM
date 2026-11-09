#!/usr/bin/env python3
"""Complete all locally identifiable tau3 paper analyses without claim inflation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.common import (  # noqa: E402
    binary_metrics,
    holm_adjust,
    json_dump,
    mean,
    paired_randomization_pvalue,
    percentile,
    write_csv,
    write_table_bundle,
)


TOKEN_RE = re.compile(r"[\w./:-]+", re.UNICODE)
RETRIEVAL_BUDGETS = (256, 512, 1024, 2048)
STORAGE_BUDGETS = (8_000, 16_000, 32_000, 64_000)
RETENTION_SEEDS = (17, 29, 41, 53, 67)
DRIFT_LEVELS = (0.0, 0.1, 0.2, 0.3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/tau3_corem"))
    parser.add_argument(
        "--multiseed",
        type=Path,
        default=Path(
            "experiments/results_tau3_multistep/tau3_multistep_multiseed.json"
        ),
    )
    parser.add_argument(
        "--retrieval-replays",
        type=Path,
        default=Path(
            "experiments/results_tau3_retrieval_ablations/"
            "retrieval_ablation_replays.json"
        ),
    )
    parser.add_argument(
        "--external-baselines",
        type=Path,
        default=Path("experiments/results_tau3_external_baselines/result.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results_tau3_paper_suite"),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def hashed_vector(text: str, dimension: int = 256) -> list[float]:
    vector = [0.0] * dimension
    for token in tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        vector[int.from_bytes(digest[:8], "big") % dimension] += (
            1.0 if digest[8] & 1 else -1.0
        )
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def bm25_scores(query: str, documents: dict[str, str]) -> dict[str, float]:
    document_tokens = {key: tokens(value) for key, value in documents.items()}
    count = len(document_tokens)
    average_length = mean(float(len(value)) for value in document_tokens.values())
    frequencies = Counter(token for value in document_tokens.values() for token in set(value))
    query_terms = set(tokens(query))
    scores: dict[str, float] = {}
    for key, words in document_tokens.items():
        term_counts = Counter(words)
        score = 0.0
        for term in sorted(query_terms):
            document_frequency = frequencies[term]
            inverse = math.log(1.0 + (count - document_frequency + 0.5) / (document_frequency + 0.5))
            frequency = term_counts[term]
            denominator = frequency + 1.5 * (
                1.0 - 0.75 + 0.75 * len(words) / max(1.0, average_length)
            )
            score += inverse * frequency * 2.5 / denominator if denominator else 0.0
        scores[key] = score
    return scores


def item_scores(
    task: dict[str, Any], memory_ids: Iterable[str], memories: dict[str, dict[str, Any]]
) -> dict[str, dict[str, float]]:
    documents = {key: str(memories[key]["text"]) for key in memory_ids}
    lexical = bm25_scores(str(task["query"]), documents)
    query_vector = hashed_vector(str(task["query"]))
    dense = {
        key: cosine(query_vector, hashed_vector(text))
        for key, text in documents.items()
    }

    def ranks(values: dict[str, float]) -> dict[str, int]:
        ordered = sorted(values, key=lambda key: (values[key], key), reverse=True)
        return {key: index + 1 for index, key in enumerate(ordered)}

    lexical_rank = ranks(lexical)
    dense_rank = ranks(dense)
    hybrid = {
        key: 1.0 / (60 + lexical_rank[key]) + 1.0 / (60 + dense_rank[key])
        for key in documents
    }
    return {"bm25": lexical, "dense_hash": dense, "hybrid_rrf": hybrid}


def aggregate_item_score(
    selected: Sequence[str], scores: dict[str, float], power: float
) -> float:
    return sum(scores[key] for key in selected) / (len(selected) ** power)


def selected_cost(row: dict[str, Any], memories: dict[str, dict[str, Any]]) -> int:
    return sum(int(memories[key]["token_cost"]) for key in row["selected_memory_ids"])


def declared_complete(row: dict[str, Any], declared: dict[str, frozenset[str]]) -> bool:
    return declared[row["task_id"]].issubset(row["selected_memory_ids"])


def declared_coverage(row: dict[str, Any], declared: dict[str, frozenset[str]]) -> float:
    group = declared[row["task_id"]]
    return len(group.intersection(row["selected_memory_ids"])) / len(group)


def task_pairwise_accuracy(rows: Sequence[dict[str, Any]], scores: Sequence[float]) -> float:
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        grouped[row["task_id"]].append((int(row["outcome"]), float(score)))
    pairs = []
    for members in grouped.values():
        positives = [score for label, score in members if label]
        negatives = [score for label, score in members if not label]
        for positive in positives:
            for negative in negatives:
                pairs.append(1.0 if positive > negative else 0.5 if positive == negative else 0.0)
    return mean(pairs)


def select_rows(
    rows: Sequence[dict[str, Any]],
    scores: Sequence[float],
    memories: dict[str, dict[str, Any]],
    *,
    budget: int | None = None,
) -> list[tuple[dict[str, Any] | None, float]]:
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        if budget is None or selected_cost(row, memories) <= budget:
            grouped[row["task_id"]].append((row, float(score)))
    task_ids = sorted({row["task_id"] for row in rows})
    selected = []
    for task_id in task_ids:
        choices = grouped.get(task_id, [])
        if not choices:
            selected.append((None, float("-inf")))
            continue
        selected.append(
            max(
                choices,
                key=lambda item: (
                    item[1],
                    -selected_cost(item[0], memories),
                    item[0]["example_id"],
                ),
            )
        )
    return selected


def policy_summary(
    method: str,
    selected: Sequence[tuple[dict[str, Any] | None, float]],
    memories: dict[str, dict[str, Any]],
    declared: dict[str, frozenset[str]],
) -> dict[str, Any]:
    rows = [row for row, _ in selected if row is not None]
    task_count = len(selected)
    return {
        "method": method,
        "tasks": task_count,
        "feasible_rate": len(rows) / task_count,
        "verified_logged_candidate_success": sum(int(row["outcome"]) for row in rows) / task_count,
        "declared_group_complete": sum(declared_complete(row, declared) for row in rows) / task_count,
        "mean_tokens": sum(selected_cost(row, memories) for row in rows) / task_count,
    }


def fit_text_baselines(
    tasks: dict[str, dict[str, Any]],
    outcomes: list[dict[str, Any]],
    memories: dict[str, dict[str, Any]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    scores_by_example: dict[str, dict[str, float]] = defaultdict(dict)
    outcomes_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        outcomes_by_task[row["task_id"]].append(row)
    for task_id, rows in outcomes_by_task.items():
        memory_ids = sorted({key for row in rows for key in row["selected_memory_ids"]})
        by_method = item_scores(tasks[task_id], memory_ids, memories)
        for method, scores in by_method.items():
            for power in (0.0, 0.5, 1.0, 1.5):
                label = f"{method}:p{power}"
                for row in rows:
                    scores_by_example[row["example_id"]][label] = aggregate_item_score(
                        row["selected_memory_ids"], scores, power
                    )
    selected_power = {}
    validation = [row for row in outcomes if row["split"] == "val"]
    for method in ("bm25", "dense_hash", "hybrid_rrf"):
        candidates = []
        for power in (0.0, 0.5, 1.0, 1.5):
            label = f"{method}:p{power}"
            selected = select_rows(
                validation,
                [scores_by_example[row["example_id"]][label] for row in validation],
                memories,
            )
            candidates.append(
                (
                    mean(float(row["outcome"]) for row, _ in selected if row is not None),
                    -power,
                    power,
                )
            )
        selected_power[method] = max(candidates)[-1]
    return selected_power, scores_by_example


def ensemble_probabilities(result: dict[str, Any], method: str) -> list[float]:
    count = len(result["per_seed"][0]["methods"][method]["test_probabilities"])
    return [
        mean(
            float(run["methods"][method]["test_probabilities"][index])
            for run in result["per_seed"]
        )
        for index in range(count)
    ]


def predictor_table(
    test: list[dict[str, Any]],
    result: dict[str, Any],
    declared: dict[str, frozenset[str]],
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    labels = [int(row["outcome"]) for row in test]
    scores = {
        "declared_complete_rule": [float(declared_complete(row, declared)) for row in test],
        "declared_coverage_rule": [declared_coverage(row, declared) for row in test],
        "set_size_rule": [
            len(row["selected_memory_ids"])
            / max(
                len(member["selected_memory_ids"])
                for member in test
                if member["task_id"] == row["task_id"]
            )
            for row in test
        ],
    }
    for method in result["aggregate"]:
        scores[method] = ensemble_probabilities(result, method)
    rows = []
    for method, values in scores.items():
        metrics = binary_metrics(labels, values, 0.5)
        rows.append(
            {
                "method": method,
                **{key: metrics[key] for key in ("roc_auc", "balanced_accuracy", "brier", "ece_10", "log_loss")},
                "task_pairwise_accuracy": task_pairwise_accuracy(test, values),
            }
        )
    return rows, scores


def policy_experiments(
    test: list[dict[str, Any]],
    result: dict[str, Any],
    memories: dict[str, dict[str, Any]],
    declared: dict[str, frozenset[str]],
    text_power: dict[str, float],
    text_scores: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[float]]]:
    summary: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    score_map: dict[str, list[float]] = {}

    deterministic: dict[str, list[float]] = {
        "declared_group_rule": [
            2.0 * float(declared_complete(row, declared)) + declared_coverage(row, declared)
            for row in test
        ],
        "largest_set_diagnostic": [float(len(row["selected_memory_ids"])) for row in test],
    }
    for method, power in text_power.items():
        label = f"{method}:p{power}"
        deterministic[f"{method}_logged_set"] = [
            text_scores[row["example_id"]][label] for row in test
        ]
    for method in result["aggregate"]:
        deterministic[f"{method}_ensemble"] = ensemble_probabilities(result, method)

    for method, scores in deterministic.items():
        score_map[method] = scores
        selected = select_rows(test, scores, memories)
        summary.append(policy_summary(method, selected, memories, declared))
        for row, score in selected:
            assert row is not None
            raw.append(
                {
                    "method": method,
                    "seed": "ensemble_or_deterministic",
                    "task_id": row["task_id"],
                    "example_id": row["example_id"],
                    "score": score,
                    "outcome": row["outcome"],
                    "token_cost": selected_cost(row, memories),
                    "declared_group_complete": int(declared_complete(row, declared)),
                }
            )

    for method in result["aggregate"]:
        seed_rates = []
        seed_group = []
        seed_tokens = []
        for run in result["per_seed"]:
            scores = [float(value) for value in run["methods"][method]["test_probabilities"]]
            selected = select_rows(test, scores, memories)
            seed_summary = policy_summary(method, selected, memories, declared)
            seed_rates.append(seed_summary["verified_logged_candidate_success"])
            seed_group.append(seed_summary["declared_group_complete"])
            seed_tokens.append(seed_summary["mean_tokens"])
            for row, score in selected:
                assert row is not None
                raw.append(
                    {
                        "method": method,
                        "seed": run["seed"],
                        "task_id": row["task_id"],
                        "example_id": row["example_id"],
                        "score": score,
                        "outcome": row["outcome"],
                        "token_cost": selected_cost(row, memories),
                        "declared_group_complete": int(declared_complete(row, declared)),
                    }
                )
        summary.append(
            {
                "method": f"{method}_seed_mean",
                "tasks": len({row["task_id"] for row in test}),
                "feasible_rate": 1.0,
                "verified_logged_candidate_success": mean(seed_rates),
                "seed_sample_std": statistics.stdev(seed_rates),
                "declared_group_complete": mean(seed_group),
                "mean_tokens": mean(seed_tokens),
            }
        )
    random_rates = []
    for seed in result["configuration"]["seeds"]:
        rng = random.Random(int(seed))
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in test:
            grouped[row["task_id"]].append(row)
        random_rates.append(mean(float(rng.choice(rows)["outcome"]) for rows in grouped.values()))
    summary.append(
        {
            "method": "random_logged_candidate_seed_mean",
            "tasks": len({row["task_id"] for row in test}),
            "feasible_rate": 1.0,
            "verified_logged_candidate_success": mean(random_rates),
            "seed_sample_std": statistics.stdev(random_rates),
            "declared_group_complete": None,
            "mean_tokens": None,
        }
    )
    return summary, raw, score_map


def budget_experiments(
    test: list[dict[str, Any]],
    score_map: dict[str, list[float]],
    memories: dict[str, dict[str, Any]],
    declared: dict[str, frozenset[str]],
) -> list[dict[str, Any]]:
    methods = (
        "hgt_corem_ensemble",
        "hgt_additive_encoder_matched_ensemble",
        "hgt_no_hyperedges_ensemble",
        "stable_hash_reference_ensemble",
        "declared_group_rule",
        "bm25_logged_set",
        "dense_hash_logged_set",
        "hybrid_rrf_logged_set",
    )
    output = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in test:
        grouped[row["task_id"]].append(row)
    for budget in RETRIEVAL_BUDGETS:
        positive_feasible = mean(
            float(
                any(
                    int(row["outcome"]) == 1 and selected_cost(row, memories) <= budget
                    for row in rows
                )
            )
            for rows in grouped.values()
        )
        for method in methods:
            selected = select_rows(test, score_map[method], memories, budget=budget)
            summary = policy_summary(method, selected, memories, declared)
            output.append(
                {
                    "budget": budget,
                    "positive_candidate_feasible_rate": positive_feasible,
                    "interpretation": "infeasible" if positive_feasible < 0.1 else "descriptive_sensitivity",
                    **summary,
                }
            )
    return output


def retention_experiments(
    test: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    memories: dict[str, dict[str, Any]],
    declared: dict[str, frozenset[str]],
    full_scores: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outcomes_by_task: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(test, full_scores):
        outcomes_by_task[row["task_id"]].append((row, score))
    target_groups: dict[str, frozenset[str]] = {}
    for task_id, rows in outcomes_by_task.items():
        eligible = [
            row for row, _ in rows if declared[task_id].issubset(row["selected_memory_ids"])
        ]
        target = min(
            eligible,
            key=lambda row: (selected_cost(row, memories), len(row["selected_memory_ids"]), row["example_id"]),
        )
        target_groups[task_id] = frozenset(target["selected_memory_ids"])
    owner = {key: task_id for task_id, group in target_groups.items() for key in group}
    all_group_storage = sum(
        int(memories[key]["storage_cost"]) for key in owner
    )

    def storage(bank: frozenset[str]) -> int:
        return sum(int(memories[key]["storage_cost"]) for key in bank)

    def group_bank(seen: set[str], scores: dict[str, float], budget: int) -> frozenset[str]:
        bank: set[str] = set()
        used = 0
        ranked = sorted(
            seen,
            key=lambda task_id: (
                scores.get(task_id, 0.0)
                / max(1, sum(int(memories[key]["storage_cost"]) for key in target_groups[task_id])),
                task_id,
            ),
            reverse=True,
        )
        for task_id in ranked:
            group = target_groups[task_id]
            cost = sum(int(memories[key]["storage_cost"]) for key in group)
            if used + cost <= budget:
                bank.update(group)
                used += cost
        return frozenset(bank)

    def item_bank(seen: set[str], scores: dict[str, float], budget: int) -> frozenset[str]:
        records = [key for task_id in seen for key in target_groups[task_id]]
        ranked = sorted(
            records,
            key=lambda key: (
                scores.get(owner[key], 0.0) / int(memories[key]["storage_cost"]),
                key,
            ),
            reverse=True,
        )
        bank: set[str] = set()
        used = 0
        for key in ranked:
            cost = int(memories[key]["storage_cost"])
            if used + cost <= budget:
                bank.add(key)
                used += cost
        return frozenset(bank)

    methods = (
        "group_aware_adaptive",
        "a6_independent_priority",
        "a7_singleton_swaps",
        "a8_uniform_forecast",
        "item_lru",
    )
    accumulators: dict[tuple[int, int, str], dict[str, Any]] = {}
    windows: list[dict[str, Any]] = []
    task_rows = list(tasks.values())
    for seed in RETENTION_SEEDS:
        rng = random.Random(seed)
        for budget in STORAGE_BUDGETS:
            banks = {method: frozenset() for method in methods}
            seen: set[str] = set()
            recent: deque[str] = deque(maxlen=100)
            last_seen: dict[str, int] = {}
            singleton_banks: set[str] = set()
            window_counts: dict[tuple[int, str], list[float]] = defaultdict(list)
            for event in range(1_000):
                window = event // 100
                domain_weights = (
                    {"airline": 0.45, "retail": 0.40, "telecom": 0.15}
                    if window < 5
                    else {"airline": 0.10, "retail": 0.20, "telecom": 0.70}
                )
                weights = [
                    domain_weights[task["domain"]]
                    / sum(other["domain"] == task["domain"] for other in task_rows)
                    for task in task_rows
                ]
                task = rng.choices(task_rows, weights=weights, k=1)[0]
                task_id = task["task_id"]
                for method in methods:
                    bank = banks[method]
                    choices = [
                        (row, score)
                        for row, score in outcomes_by_task[task_id]
                        if frozenset(row["selected_memory_ids"]).issubset(bank)
                        and selected_cost(row, memories) <= 1024
                    ]
                    if choices:
                        selected_row, _ = max(
                            choices,
                            key=lambda item: (
                                item[1],
                                -selected_cost(item[0], memories),
                                item[0]["example_id"],
                            ),
                        )
                        success = float(selected_row["outcome"])
                    else:
                        success = 0.0
                    group_available = float(target_groups[task_id].issubset(bank))
                    key = (seed, budget, method)
                    accumulator = accumulators.setdefault(
                        key,
                        {
                            "success": [],
                            "group": [],
                            "occupancy": [],
                            "churn": [],
                            "update_ms": [],
                        },
                    )
                    accumulator["success"].append(success)
                    accumulator["group"].append(group_available)
                    window_counts[(window, method)].append(success)

                seen.add(task_id)
                recent.append(task_id)
                last_seen[task_id] = event
                frequency = Counter(recent)
                adaptive_scores = {key: float(frequency[key]) + 0.01 for key in seen}
                uniform_scores = {key: 1.0 for key in seen}
                lru_scores = {key: float(last_seen[key] + 1) for key in seen}
                updated: dict[str, frozenset[str]] = {}
                started = time.perf_counter()
                updated["group_aware_adaptive"] = group_bank(seen, adaptive_scores, budget)
                group_ms = (time.perf_counter() - started) * 1000.0
                started = time.perf_counter()
                updated["a6_independent_priority"] = item_bank(seen, adaptive_scores, budget)
                independent_ms = (time.perf_counter() - started) * 1000.0
                started = time.perf_counter()
                for record_id in sorted(target_groups[task_id]):
                    singleton_banks.add(record_id)
                    while storage(frozenset(singleton_banks)) > budget:
                        victim = min(
                            singleton_banks,
                            key=lambda key: (
                                adaptive_scores.get(owner[key], 0.0)
                                / int(memories[key]["storage_cost"]),
                                key,
                            ),
                        )
                        singleton_banks.remove(victim)
                updated["a7_singleton_swaps"] = frozenset(singleton_banks)
                singleton_ms = (time.perf_counter() - started) * 1000.0
                started = time.perf_counter()
                updated["a8_uniform_forecast"] = group_bank(seen, uniform_scores, budget)
                uniform_ms = (time.perf_counter() - started) * 1000.0
                started = time.perf_counter()
                updated["item_lru"] = item_bank(seen, lru_scores, budget)
                lru_ms = (time.perf_counter() - started) * 1000.0
                update_times = {
                    "group_aware_adaptive": group_ms,
                    "a6_independent_priority": independent_ms,
                    "a7_singleton_swaps": singleton_ms,
                    "a8_uniform_forecast": uniform_ms,
                    "item_lru": lru_ms,
                }
                for method in methods:
                    key = (seed, budget, method)
                    before = banks[method]
                    after = updated[method]
                    accumulators[key]["churn"].append(len(before.symmetric_difference(after)))
                    accumulators[key]["occupancy"].append(storage(after) / budget)
                    accumulators[key]["update_ms"].append(update_times[method])
                    banks[method] = after
            for (window, method), values in window_counts.items():
                windows.append(
                    {
                        "seed": seed,
                        "budget": budget,
                        "window": window,
                        "phase": "pre_shift" if window < 5 else "post_shift",
                        "method": method,
                        "controlled_logged_replay_success": mean(values),
                    }
                )
    summary = []
    for (seed, budget, method), values in sorted(accumulators.items()):
        summary.append(
            {
                "seed": seed,
                "storage_budget": budget,
                "method": method,
                "events": len(values["success"]),
                "controlled_logged_replay_success": mean(values["success"]),
                "post_shift_success": mean(values["success"][500:]),
                "retained_complement_availability": mean(values["group"]),
                "mean_bank_occupancy": mean(values["occupancy"]),
                "mean_churn_records": mean(values["churn"]),
                "p95_update_latency_ms": percentile(values["update_ms"], 0.95),
                "bank_non_binding": budget >= all_group_storage,
            }
        )
    aggregate = []
    for budget in STORAGE_BUDGETS:
        for method in methods:
            members = [
                row for row in summary
                if row["storage_budget"] == budget and row["method"] == method
            ]
            aggregate.append(
                {
                    "storage_budget": budget,
                    "method": method,
                    "seeds": len(members),
                    "controlled_logged_replay_success": mean(
                        float(row["controlled_logged_replay_success"]) for row in members
                    ),
                    "seed_sample_std": statistics.stdev(
                        float(row["controlled_logged_replay_success"]) for row in members
                    ),
                    "post_shift_success": mean(float(row["post_shift_success"]) for row in members),
                    "retained_complement_availability": mean(
                        float(row["retained_complement_availability"]) for row in members
                    ),
                    "mean_bank_occupancy": mean(float(row["mean_bank_occupancy"]) for row in members),
                    "mean_churn_records": mean(float(row["mean_churn_records"]) for row in members),
                    "p95_update_latency_ms": mean(float(row["p95_update_latency_ms"]) for row in members),
                    "bank_non_binding": all(bool(row["bank_non_binding"]) for row in members),
                }
            )
    return aggregate, windows


def drift_experiments(
    test: list[dict[str, Any]],
    full_scores: list[float],
) -> list[dict[str, Any]]:
    by_task: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    operations = set()
    for row, score in zip(test, full_scores):
        by_task[row["task_id"]].append((row, score))
        operations.update(row["selected_operation_ids"])
    operation_list = sorted(operations)
    output = []
    for drift_type in ("schema_break", "silent_semantic", "unavailability", "qos_delay"):
        for intensity in DRIFT_LEVELS:
            for seed in RETENTION_SEEDS:
                rng = random.Random(f"{drift_type}:{intensity}:{seed}")
                affected = set(rng.sample(operation_list, round(intensity * len(operation_list))))
                for method in ("unaware", "scope_mask"):
                    successes = []
                    stale_uses = []
                    delay_ms = []
                    for choices in by_task.values():
                        candidates = choices
                        detectable = drift_type in {"schema_break", "unavailability"}
                        if method == "scope_mask" and detectable:
                            candidates = [
                                item
                                for item in choices
                                if not affected.intersection(item[0]["selected_operation_ids"])
                            ]
                        if not candidates:
                            successes.append(0.0)
                            stale_uses.append(0.0)
                            delay_ms.append(0.0)
                            continue
                        row, _ = max(
                            candidates,
                            key=lambda item: (item[1], -len(item[0]["selected_memory_ids"]), item[0]["example_id"]),
                        )
                        stale = bool(affected.intersection(row["selected_operation_ids"]))
                        stale_uses.append(float(stale and drift_type != "qos_delay"))
                        if drift_type in {"schema_break", "silent_semantic", "unavailability"} and stale:
                            successes.append(0.0)
                        else:
                            successes.append(float(row["outcome"]))
                        delay_ms.append(100.0 * len(affected.intersection(row["selected_operation_ids"])) if drift_type == "qos_delay" else 0.0)
                    output.append(
                        {
                            "drift_type": drift_type,
                            "intensity": intensity,
                            "seed": seed,
                            "method": method,
                            "tasks": len(successes),
                            "controlled_contract_success": mean(successes),
                            "stale_use_rate": mean(stale_uses),
                            "mean_injected_delay_ms": mean(delay_ms),
                            "evidence_label": "controlled_derived_stress_not_execution",
                        }
                    )
    return output


def stratified_results(
    test: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    scores: dict[str, list[float]],
) -> list[dict[str, Any]]:
    output = []
    strata: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(test):
        depth = len(tasks[row["task_id"]]["gold_operation_ids"])
        label = str(depth) if depth in (2, 3, 4) else "5+" if depth >= 5 else "1"
        strata[label].append(index)
    for stratum, indices in sorted(strata.items()):
        labels = [int(test[index]["outcome"]) for index in indices]
        methods = [
            "hgt_corem",
            "hgt_additive_encoder_matched",
            "hgt_no_hyperedges",
            "stable_hash_reference",
            "declared_coverage_rule",
        ]
        if "hgt_no_redundancy" in scores:
            methods.insert(3, "hgt_no_redundancy")
        for method in methods:
            values = [scores[method][index] for index in indices]
            output.append(
                {
                    "declared_operation_count": stratum,
                    "method": method,
                    "candidate_sets": len(indices),
                    "tasks": len({test[index]["task_id"] for index in indices}),
                    "roc_auc": (
                        binary_metrics(labels, values, 0.5)["roc_auc"]
                        if 0 < sum(labels) < len(labels)
                        else None
                    ),
                    "task_pairwise_accuracy": task_pairwise_accuracy(
                        [test[index] for index in indices], values
                    ),
                }
            )
    return output


def write_documents(
    output: Path,
    predictor: list[dict[str, Any]],
    policies: list[dict[str, Any]],
    ablations: list[dict[str, Any]],
    blocked: list[dict[str, Any]],
    result: dict[str, Any],
    external_result: dict[str, Any] | None,
) -> None:
    by_predictor = {row["method"]: row for row in predictor}
    by_policy = {row["method"]: row for row in policies}
    rule_auc = float(by_predictor["declared_coverage_rule"]["roc_auc"])
    if rule_auc >= 0.95:
        rule_interpretation = (
            "Because this no-training rule is nearly sufficient for the generated "
            "labels, the benchmark supports relation-completeness prediction but does "
            "not establish that learned high-order interactions outperform the explicit rule."
        )
    else:
        rule_interpretation = (
            "The counterexamples reduce the sufficiency of this no-training rule, so "
            "the benchmark can distinguish complete-but-harmful sets from merely incomplete sets."
        )
    external_finding = (
        "Adapted external controls were freshly replayed on the same 43 held-out "
        "tasks. These are explicitly not native-system reproductions."
        if external_result is not None
        else "External systems remain blocked rather than receiving fabricated values."
    )
    findings = f"""# Findings

## Main result

The corrected run excludes gold operation IDs from request features. Full CoReM test ROC-AUC is {by_predictor['hgt_corem']['roc_auc']:.4f}; the encoder-matched A1 is {by_predictor['hgt_additive_encoder_matched']['roc_auc']:.4f}, and A2 without hyperedges is {by_predictor['hgt_no_hyperedges']['roc_auc']:.4f}.

The declared-group coverage rule reaches ROC-AUC {rule_auc:.4f}. {rule_interpretation}

The logged-candidate policy result for the full ensemble is {by_policy['hgt_corem_ensemble']['verified_logged_candidate_success']:.4f}. This is selection among already executed candidate sets, not a free-form composer evaluation.

## Scope

- Set labels are execution-verified deterministic final states.
- Retention order and workload shift are controlled, because tau3 has no natural timestamps and memories are task-specific.
- Drift outputs are derived controlled stress tests, not independently executed service revisions.
- {external_finding}
- Native external-system runs and natural drift remain blocked rather than receiving fabricated values.
"""
    (output / "findings.md").write_text(findings, encoding="utf-8")
    has_a3 = "hgt_no_redundancy" in result["aggregate"]
    a3_contract = (
        "A3 disables redundancy terms while retaining complements and hyperedges; "
        "duplicate pairs are present in this dataset."
        if has_a3
        else "A3 is not identifiable because the dataset has zero duplicate pairs."
    )
    plan = f"""# Experiment Log

## SkillOpt route card

- **best_skill:** full empirical analysis adapted to predictive algorithm evaluation.
- **selection split:** validation only selected 20 epochs and text-score aggregation powers.
- **heldout gate:** the official test split was read only by the frozen final and analysis runners.
- **accepted patterns:** verified set-outcome prediction, logged-candidate policy selection, matched A1/A2, budget sweeps, controlled retention, controlled drift diagnostics, and explicitly labelled external adapters.
- **rejected patterns:** causal memory contribution, natural-drift claims, and relabelling adapters as native-system runs.
- **leakage correction:** `gold_operation_ids` are excluded from all request objects; the old result is archived as an audit.

## Ablation contract

- A1 keeps HGT hyperedges and relation training but disables complement and redundancy set terms.
- A2 removes hyperedges from the encoder graph and set utility.
- {a3_contract}
- A4/A5 use fresh environment replay when the separate replay artifact is present.
- A6-A8 are controlled retention experiments with identical storage budgets and streams.

## Execution

- Seeds: {result['configuration']['seeds']}
- GPU: {result['runtime']['gpu_name']}
- Dataset: {result['paper_suite_dataset']['tasks']} tasks / {result['paper_suite_dataset']['candidate_sets']} replayed candidate sets.
- Primary label: deterministic final-state success; communication quality is excluded.
"""
    (output / "EXPERIMENT_LOG.md").write_text(plan, encoding="utf-8")
    readme = """# tau3 Paper Experiment Suite

This directory contains every locally identifiable experiment requested by the paper protocol. Read `findings.md` first. Tables and raw summaries distinguish execution-verified set outcomes, controlled streams, derived stress tests, and blocked experiments. No proxy is relabelled as natural longitudinal or live service evidence.

- `tables/`: publication-oriented CSV, Markdown, and LaTeX summaries.
- `raw/`: selected candidates and controlled window-level results.
- `result.json`: machine-readable headline results and reproducibility hashes.
- `EXPERIMENT_LOG.md`: analysis route, ablation contract, and evidence boundaries.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    write_table_bundle(output / "tables", "table1_predictors", predictor, caption="Execution-verified candidate-set prediction")
    write_table_bundle(output / "tables", "table2_logged_candidate_policies", policies, caption="Policy selection among execution-verified logged candidates")
    write_table_bundle(output / "tables", "table3_ablations", ablations, caption="Mechanism ablations with evidence labels")
    write_table_bundle(output / "tables", "table9_experiment_status", blocked, caption="Completion status and remaining evidence gaps")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    result = json.loads(args.multiseed.read_text(encoding="utf-8"))
    if result["configuration"].get("gold_operation_ids_used_as_request_features") is not False:
        raise ValueError("refusing a result that does not certify gold-operation exclusion")
    tasks_list = load_jsonl(args.dataset / "retrieval_tasks.jsonl")
    tasks = {row["task_id"]: row for row in tasks_list}
    outcomes = load_jsonl(args.dataset / "set_outcomes.jsonl")
    test = [row for row in outcomes if row["split"] == "test"]
    memories = {row["memory_id"]: row for row in load_jsonl(args.dataset / "memory_records.jsonl")}
    duplicate_path = args.dataset / "duplicate_groups.jsonl"
    duplicate_groups = load_jsonl(duplicate_path) if duplicate_path.is_file() else []
    declared = {
        row["task_id"]: frozenset(row["members"])
        for row in load_jsonl(args.dataset / "dependency_hyperedges.jsonl")
        if row.get("admissible_default") is True
    }
    if len(test) != len(result["per_seed"][0]["methods"]["hgt_corem"]["test_probabilities"]):
        raise ValueError("result probabilities do not align with the test rows")
    if any(row.get("event_time") is not None for row in tasks_list):
        raise ValueError("controlled-stream label assumes missing natural timestamps")
    external_result = (
        json.loads(args.external_baselines.read_text(encoding="utf-8"))
        if args.external_baselines.is_file()
        else None
    )

    text_power, text_scores = fit_text_baselines(tasks, outcomes, memories)
    predictor, predictor_scores = predictor_table(test, result, declared)
    policies, policy_raw, policy_scores = policy_experiments(
        test, result, memories, declared, text_power, text_scores
    )
    budget = budget_experiments(test, policy_scores, memories, declared)
    retention, retention_windows = retention_experiments(
        test,
        {key: value for key, value in tasks.items() if value["split"] == "test"},
        memories,
        declared,
        predictor_scores["hgt_corem"],
    )
    drift = drift_experiments(test, predictor_scores["hgt_corem"])
    stratified = stratified_results(test, tasks, predictor_scores)

    policy_by_method = {row["method"]: row for row in policies}
    full_vsr = policy_by_method["hgt_corem_ensemble"]["verified_logged_candidate_success"]
    ablations = [
        {
            "variant": "Full CoReM",
            "effect_metric": "logged-candidate success",
            "value": full_vsr,
            "delta_from_matched_full": 0.0,
            "evidence": "execution-verified logged candidates",
        },
        {
            "variant": "A1 additive utility",
            "effect_metric": "logged-candidate success",
            "value": policy_by_method["hgt_additive_encoder_matched_ensemble"]["verified_logged_candidate_success"],
            "delta_from_matched_full": policy_by_method["hgt_additive_encoder_matched_ensemble"]["verified_logged_candidate_success"] - full_vsr,
            "evidence": "encoder-matched, execution-verified logged candidates",
        },
        {
            "variant": "A2 no complement hyperedges",
            "effect_metric": "logged-candidate success",
            "value": policy_by_method["hgt_no_hyperedges_ensemble"]["verified_logged_candidate_success"],
            "delta_from_matched_full": policy_by_method["hgt_no_hyperedges_ensemble"]["verified_logged_candidate_success"] - full_vsr,
            "evidence": "execution-verified logged candidates",
        },
    ]
    if "hgt_no_redundancy_ensemble" in policy_by_method:
        a3_value = policy_by_method["hgt_no_redundancy_ensemble"][
            "verified_logged_candidate_success"
        ]
        ablations.append(
            {
                "variant": "A3 no redundancy",
                "effect_metric": "logged-candidate success",
                "value": a3_value,
                "delta_from_matched_full": a3_value - full_vsr,
                "evidence": "execution-verified logged candidates with duplicate pairs",
            }
        )
    else:
        ablations.append(
            {
                "variant": "A3 no redundancy",
                "effect_metric": "not identifiable",
                "value": None,
                "delta_from_matched_full": None,
                "evidence": "blocked: zero duplicate pairs",
            }
        )
    if args.retrieval_replays.is_file():
        replay = json.loads(args.retrieval_replays.read_text(encoding="utf-8"))
        replay_summary = {row["method"]: row for row in replay["summary"]}
        replay_full = replay_summary["full_corem"]["fresh_replay_vsr_mean"]
        for number, key, label in (
            (4, "a4_item_only", "item-only search"),
            (5, "a5_no_evidence_closure", "no evidence closure"),
        ):
            ablations.append(
                {
                    "variant": f"A{number} {label}",
                    "effect_metric": "fresh replay VSR",
                    "value": replay_summary[key]["fresh_replay_vsr_mean"],
                    "delta_from_matched_full": replay_summary[key]["fresh_replay_vsr_mean"] - replay_full,
                    "evidence": "fresh deterministic environment replay",
                }
            )
    else:
        for number, label in ((4, "item-only search"), (5, "no evidence closure")):
            ablations.append(
                {
                    "variant": f"A{number} {label}",
                    "effect_metric": "pending separate replay",
                    "value": None,
                    "delta_from_matched_full": None,
                    "evidence": "selection runner not yet replayed",
                }
            )
    retention_default = {
        row["method"]: row
        for row in retention
        if row["storage_budget"] == 32_000
    }
    retention_full = retention_default["group_aware_adaptive"]["controlled_logged_replay_success"]
    for number, key, label in (
        (6, "a6_independent_priority", "independent retention"),
        (7, "a7_singleton_swaps", "no group swaps"),
        (8, "a8_uniform_forecast", "uniform forecast"),
    ):
        ablations.append(
            {
                "variant": f"A{number} {label}",
                "effect_metric": "controlled-stream logged-replay success",
                "value": retention_default[key]["controlled_logged_replay_success"],
                "delta_from_matched_full": retention_default[key]["controlled_logged_replay_success"] - retention_full,
                "evidence": "controlled synthetic order; task-specific memories",
            }
        )

    blocked = []
    if external_result is not None:
        blocked.append(
            {
                "experiment": "adapted MAGMA/ACGM/HyperMem/HGMem/AgeMem",
                "status": "complete",
                "reason": "same bank/budget; fresh replay; explicitly not native reproductions",
            }
        )
    else:
        blocked.append(
            {
                "experiment": "adapted MAGMA/ACGM/HyperMem/HGMem/AgeMem",
                "status": "blocked",
                "reason": "adapter result artifact is absent",
            }
        )
    blocked.extend([
        {
            "experiment": "native MAGMA/ACGM/HyperMem/HGMem/AgeMem",
            "status": "blocked",
            "reason": "incompatible original tasks, model endpoints, or RL infrastructure",
        },
        {
            "experiment": "natural chronological retention",
            "status": "blocked",
            "reason": "no trustworthy event timestamps; every memory is task-specific",
        },
        {
            "experiment": "natural service drift and recovery",
            "status": "blocked",
            "reason": "one pinned service version and no observed version transitions",
        },
        {
            "experiment": "communication-quality VSR",
            "status": "blocked",
            "reason": "labels cover deterministic final state only",
        },
    ])

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "raw").mkdir(exist_ok=True)
    (args.output / "tables").mkdir(exist_ok=True)
    write_csv(args.output / "raw" / "policy_selections.csv", policy_raw)
    write_csv(args.output / "raw" / "retention_windows.csv", retention_windows)
    write_table_bundle(args.output / "tables", "table4_retrieval_budgets", budget, caption="Retrieval-budget sensitivity on executed candidate sets")
    write_table_bundle(args.output / "tables", "table5_retention_budgets", retention, caption="Controlled storage-budget and retention comparison")
    write_table_bundle(args.output / "tables", "table6_controlled_drift", drift, caption="Controlled derived drift stress (not natural drift)")
    write_table_bundle(args.output / "tables", "table7_depth_strata", stratified, caption="Predictive performance by declared operation count")

    comparisons = {}
    raw_pvalues = {}
    comparison_references = [
        "hgt_additive_encoder_matched",
        "hgt_no_hyperedges",
    ]
    if "hgt_no_redundancy" in result["aggregate"]:
        comparison_references.append("hgt_no_redundancy")
    for reference in comparison_references:
        differences = []
        for run in result["per_seed"]:
            full_selected = select_rows(
                test,
                run["methods"]["hgt_corem"]["test_probabilities"],
                memories,
            )
            ref_selected = select_rows(
                test,
                run["methods"][reference]["test_probabilities"],
                memories,
            )
            differences.append(
                mean(float(row["outcome"]) for row, _ in full_selected if row is not None)
                - mean(float(row["outcome"]) for row, _ in ref_selected if row is not None)
            )
        raw_pvalues[reference] = paired_randomization_pvalue(differences, seed=2027)
        comparisons[reference] = {
            "mean_seed_paired_success_difference": mean(differences),
            "seed_difference_sample_std": statistics.stdev(differences),
            "raw_pvalue": raw_pvalues[reference],
        }
    adjusted = holm_adjust(raw_pvalues)
    for reference in comparisons:
        comparisons[reference]["holm_pvalue"] = adjusted[reference]

    zero_drift = [
        row for row in drift
        if row["intensity"] == 0.0 and row["method"] == "unaware"
    ]
    zero_drift_ok = all(
        abs(float(row["controlled_contract_success"]) - full_vsr) < 1e-12
        for row in zero_drift
    )
    output_result = {
        "status": "complete_with_adapted_external_and_explicitly_blocked_natural_tracks",
        "evidence_scope": {
            "candidate_labels": "fresh execution-verified deterministic final state",
            "policy": "offline selection among logged executed candidates",
            "retention": "controlled synthetic-order logged replay",
            "drift": "controlled derived stress, not independent execution",
            "external_baselines": (
                "fresh execution replay of explicit method-inspired adaptations"
                if external_result is not None
                else "not available"
            ),
        },
        "dataset": {
            "tasks": len(tasks),
            "candidate_sets": len(outcomes),
            "test_tasks": len({row["task_id"] for row in test}),
            "test_candidate_sets": len(test),
            "duplicate_pairs": len(duplicate_groups),
            "natural_timestamps": 0,
        },
        "validation_selected": {
            "epochs": result["configuration"]["epochs"],
            "text_aggregation_powers": text_power,
        },
        "predictor_summary": predictor,
        "policy_summary": policies,
        "ablations": ablations,
        "policy_significance": comparisons,
        "controlled_retention_default_budget": list(retention_default.values()),
        "controlled_drift_zero_condition_reproduces_baseline": zero_drift_ok,
        "blocked": blocked,
        "external_baselines": external_result,
        "elapsed_seconds": time.perf_counter() - started,
        "input_sha256": {
            str(args.multiseed): sha256_file(args.multiseed),
            str(args.dataset / "manifest.json"): sha256_file(args.dataset / "manifest.json"),
            str(Path(__file__)): sha256_file(Path(__file__)),
            **(
                {str(args.external_baselines): sha256_file(args.external_baselines)}
                if external_result is not None
                else {}
            ),
        },
    }
    result["paper_suite_dataset"] = output_result["dataset"]
    json_dump(args.output / "result.json", output_result)
    write_documents(
        args.output,
        predictor,
        policies,
        ablations,
        blocked,
        result,
        external_result,
    )
    files = sorted(
        path for path in args.output.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    (args.output / "checksums.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(args.output)}\n" for path in files),
        encoding="utf-8",
    )
    print(json.dumps({
        "predictor": predictor,
        "policies": policies,
        "ablations": ablations,
        "zero_drift_ok": zero_drift_ok,
    }, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
