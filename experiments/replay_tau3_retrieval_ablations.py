#!/usr/bin/env python3
"""Freshly replay selected A4/A5 action sets in official tau3 environments."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger
from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import RewardType

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.build_tau3_corem_dataset import (  # noqa: E402
    DOMAINS,
    initialize_environment,
    matches_target,
    replay,
)
from experiments.tau3_selection_ids import selected_action_spec  # noqa: E402



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selections",
        type=Path,
        default=Path(
            "experiments/results_tau3_retrieval_ablations/"
            "retrieval_ablation_selections.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results_tau3_retrieval_ablations"),
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/tau3_corem"))
    return parser.parse_args()


def perturb(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return "__corem_controlled_unknown__"
    if isinstance(value, int):
        return -999_999
    if isinstance(value, float):
        return -999_999.0
    if isinstance(value, list):
        return [] if value else ["__corem_controlled_unknown__"]
    if isinstance(value, dict):
        return {key: perturb(item) for key, item in value.items()}
    if value is None:
        return "__corem_controlled_unknown__"
    return str(value) + "__corem_controlled_unknown__"


def replay_action_spec(
    constructor: Any,
    task: Any,
    indices: tuple[int, ...],
    extras: tuple[int, ...],
    stale: tuple[int, ...],
) -> dict[str, Any]:
    environment = initialize_environment(constructor, task)
    actions = task.evaluation_criteria.actions or []
    call_specs: list[tuple[str, int, ToolCall]] = []
    current = set(indices)
    stale_set = set(stale)
    for index in sorted(current.union(stale_set)):
        action = actions[index]
        if index in current:
            call_specs.append(
                (
                    "current",
                    index,
                    ToolCall(
                        id=f"retrieval:{task.id}:{index}",
                        name=action.name,
                        arguments=action.arguments,
                        requestor=action.requestor,
                    ),
                )
            )
        if index in stale_set:
            call_specs.append(
                (
                    "controlled_stale",
                    index,
                    ToolCall(
                        id=f"retrieval:{task.id}:stale:{index}",
                        name=action.name,
                        arguments=perturb(action.arguments),
                        requestor=action.requestor,
                    ),
                )
            )
    for index in extras:
        action = actions[index]
        call_specs.append(
            (
                "controlled_extra",
                index,
                ToolCall(
                    id=f"retrieval:{task.id}:extra:{index}",
                    name=action.name,
                    arguments=perturb(action.arguments),
                    requestor=action.requestor,
                ),
            )
        )
    errors = []
    call_checks = []
    for position, (kind, source_index, call) in enumerate(call_specs):
        response = environment.get_response(call)
        call_checks.append(
            {
                "position": position,
                "kind": kind,
                "source_action_index": source_index,
                "passed": not bool(response.error),
            }
        )
        if response.error:
            errors.append(
                {
                    "index": position,
                    "kind": kind,
                    "source_action_index": source_index,
                    "name": call.name,
                    "requestor": call.requestor,
                    "response": response.content,
                }
            )
    criteria = task.evaluation_criteria
    assertion_results = []
    assertion_errors = []
    if RewardType.ENV_ASSERTION in set(criteria.reward_basis):
        for assertion in criteria.env_assertions or []:
            try:
                assertion_results.append(
                    environment.run_env_assertion(assertion, raise_assertion_error=False)
                )
            except Exception as error:  # noqa: BLE001 - upstream assertion defect
                assertion_errors.append(
                    {
                        "env_type": assertion.env_type,
                        "func_name": assertion.func_name,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    return {
        "agent_db_hash": environment.get_db_hash(),
        "user_db_hash": environment.get_user_db_hash(),
        "env_assertion_results": assertion_results,
        "env_assertion_errors": assertion_errors,
        "errors": errors,
        "call_checks": call_checks,
    }


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    source = json.loads(args.selections.read_text(encoding="utf-8"))
    selections = [row for run in source["runs"] for row in run["selections"]]
    known_memory_ids = {
        json.loads(line)["memory_id"]
        for line in (args.dataset / "memory_records.jsonl").open(encoding="utf-8")
        if line.strip()
    }
    unknown = sorted(
        {
            memory_id
            for row in selections
            for memory_id in row["selected_memory_ids"]
            if memory_id not in known_memory_ids
        }
    )
    if unknown:
        raise ValueError(f"selection contains {len(unknown)} memories absent from dataset")
    task_catalog: dict[tuple[str, str], Any] = {}
    constructors = {}
    for domain, (constructor, get_tasks, _) in DOMAINS.items():
        constructors[domain] = constructor
        for task in get_tasks(None):
            task_catalog[(domain, str(task.id))] = task

    target_cache: dict[tuple[str, str], dict[str, Any]] = {}
    replay_cache: dict[
        tuple[str, str, tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        dict[str, Any],
    ] = {}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for position, row in enumerate(selections, start=1):
        domain = str(row["domain"])
        source_task_id = str(row["source_task_id"])
        task = task_catalog[(domain, source_task_id)]
        constructor = constructors[domain]
        task_key = (domain, source_task_id)
        if task_key not in target_cache:
            full = tuple(range(len(task.evaluation_criteria.actions or [])))
            target_cache[task_key] = replay(constructor, task, full)
        indices, extras, stale = selected_action_spec(row["selected_memory_ids"])
        replay_key = (domain, source_task_id, indices, extras, stale)
        if replay_key not in replay_cache:
            replay_cache[replay_key] = replay_action_spec(
                constructor, task, indices, extras, stale
            )
        result = replay_cache[replay_key]
        target = target_cache[task_key]
        rows.append(
            {
                **row,
                "selected_action_indices": list(indices),
                "selected_controlled_extra_indices": list(extras),
                "selected_controlled_stale_indices": list(stale),
                "call_checks": result["call_checks"],
                "outcome": int(matches_target(task, result, target)),
                "execution_error_count": len(result["errors"]),
                "assertion_error_count": len(result["env_assertion_errors"]),
                "fresh_environment_replay": True,
            }
        )
        if position % 100 == 0:
            print(f"replayed {position}/{len(selections)} selections", flush=True)

    per_seed: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in rows:
        per_seed[(str(row["method"]), int(row["seed"]))].append(int(row["outcome"]))
    summary = []
    for method in sorted({key[0] for key in per_seed}):
        seed_rates = [
            statistics.mean(values)
            for (candidate, _), values in per_seed.items()
            if candidate == method
        ]
        method_rows = [row for row in rows if row["method"] == method]
        summary.append(
            {
                "method": method,
                "seeds": len(seed_rates),
                "task_seed_n": len(method_rows),
                "fresh_replay_vsr_mean": statistics.mean(seed_rates),
                "fresh_replay_vsr_sample_std": (
                    statistics.stdev(seed_rates) if len(seed_rates) > 1 else 0.0
                ),
                "fallback_rate": statistics.mean(
                    float(row["status"] == "fallback") for row in method_rows
                ),
                "mean_tokens": statistics.mean(
                    float(row["token_cost"]) for row in method_rows
                ),
                "p95_latency_ms": sorted(
                    float(row["latency_ms"]) for row in method_rows
                )[int(0.95 * (len(method_rows) - 1))],
            }
        )
    result = {
        "status": "complete",
        "scope": "fresh deterministic final-state replay of selected action sets",
        "source_selection_sha256": hashlib.sha256(
            args.selections.read_bytes()
        ).hexdigest(),
        "dataset": str(args.dataset),
        "dataset_manifest_sha256": hashlib.sha256(
            (args.dataset / "manifest.json").read_bytes()
        ).hexdigest(),
        "unique_task_targets_replayed": len(target_cache),
        "unique_selected_sets_replayed": len(replay_cache),
        "selection_rows": len(rows),
        "selection_configuration": source.get("configuration", {}),
        "summary": summary,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "The candidate bank is restricted to each official reference trajectory.",
            "Labels cover deterministic final-state criteria, not communication quality.",
            "The stream is static and contains no natural service drift.",
        ],
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "retrieval_ablation_replays.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksums = []
    for artifact in sorted(args.output.glob("*.json")):
        checksums.append(
            f"{hashlib.sha256(artifact.read_bytes()).hexdigest()}  {artifact.name}\n"
        )
    (args.output / "checksums.sha256").write_text(
        "".join(checksums), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
