#!/usr/bin/env python3
"""Add executed complete-group counterexamples to a tau3-CoReM dataset.

Each new candidate contains the successful official reference trajectory plus one
extra same-operation call with deterministically perturbed arguments. The call is
executed in a fresh environment. Only candidates that fail the official final-state
contract are retained, creating complete-declared-group failures without using an
outcome-derived graph edge.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import RewardType
from loguru import logger

from build_tau3_corem_dataset import (
    DOMAINS,
    compact_text,
    initialize_environment,
    matches_target,
    replay,
    sha256_file,
    token_estimate,
    tool_catalog,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/tau3_corem"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/tau3_corem_counterexamples")
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        )


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


def replay_with_extra(
    constructor: Any,
    task: Any,
    action_index: int,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    environment = initialize_environment(constructor, task)
    actions = task.evaluation_criteria.actions or []
    errors = []
    calls = [
        ToolCall(
            id=f"counterexample:{task.id}:{index}",
            name=action.name,
            arguments=action.arguments,
            requestor=action.requestor,
        )
        for index, action in enumerate(actions)
    ]
    source = actions[action_index]
    calls.append(
        ToolCall(
            id=f"counterexample:{task.id}:extra:{action_index}",
            name=source.name,
            arguments=arguments,
            requestor=source.requestor,
        )
    )
    for position, call in enumerate(calls):
        response = environment.get_response(call)
        if response.error:
            errors.append(
                {
                    "index": position,
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
            except Exception as error:  # noqa: BLE001 - source task defect
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
    }


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    source = args.input.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    shutil.copytree(source, output)
    tasks = load_jsonl(output / "retrieval_tasks.jsonl")
    memories = {
        row["memory_id"]: row for row in load_jsonl(output / "memory_records.jsonl")
    }
    source_tasks: dict[tuple[str, str], Any] = {}
    constructors = {}
    catalogs = {}
    for domain, (constructor, get_tasks, _) in DOMAINS.items():
        constructors[domain] = constructor
        catalogs[domain] = tool_catalog(constructor(), domain)
        for task in get_tasks(None):
            source_tasks[(domain, str(task.id))] = task

    added = Counter()
    replay_count = 0
    for position, row in enumerate(tasks, start=1):
        domain = row["domain"]
        task = source_tasks[(domain, str(row["source_task_id"]))]
        constructor = constructors[domain]
        actions = task.evaluation_criteria.actions or []
        target = replay(constructor, task, tuple(range(len(actions))))
        replay_count += 1
        candidates = []
        for index, action in enumerate(actions):
            catalog = catalogs[domain][(action.requestor, action.name)]
            candidates.append((int(catalog["mutates_state"]), bool(action.arguments), -index, index))
        chosen: tuple[int, dict[str, Any], dict[str, Any]] | None = None
        for _, _, _, action_index in sorted(candidates, reverse=True):
            arguments = perturb(copy.deepcopy(actions[action_index].arguments))
            result = replay_with_extra(constructor, task, action_index, arguments)
            replay_count += 1
            if not matches_target(task, result, target):
                chosen = (action_index, arguments, result)
                break
        if chosen is None:
            added["no_failing_perturbation"] += 1
            continue
        action_index, arguments, result = chosen
        action = actions[action_index]
        normalized_task_id = row["task_id"]
        memory_id = f"mem:{normalized_task_id}:controlled-extra-{action_index}"
        operation_id = f"tau3:{domain}:{action.requestor}:{action.name}"
        argument_text = json.dumps(
            arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        text = compact_text(
            row["query"],
            "Additional retry",
            action.requestor,
            action.name,
            argument_text,
        )
        token_cost = token_estimate(text)
        append_jsonl(
            output / "memory_records.jsonl",
            {
                "memory_id": memory_id,
                "source": "tau3_corem_controlled_counterexample",
                "text": text,
                "operation_ids": [operation_id],
                "evidence_atoms": [
                    row["required_atoms"][0],
                    f"controlled-extra:{normalized_task_id}:{action_index}",
                ],
                "token_cost": token_cost,
                "storage_cost": token_cost + 4,
                "event_time": None,
                "provenance": f"controlled-argument-perturbation:{domain}:{task.id}:{action_index}",
                "provenance_verified": True,
                "observed_outcome": None,
                "split": row["split"],
                "admissible": True,
                "eligible_for_outcome_training": False,
                "version_scope": row["active_versions"],
            },
        )
        full_ids = [
            f"mem:{normalized_task_id}:a{index}" for index in range(len(actions))
        ]
        append_jsonl(
            output / "set_outcomes.jsonl",
            {
                "example_id": f"outcome:{normalized_task_id}:controlled-complete-fail",
                "task_id": normalized_task_id,
                "source_task_id": row["source_task_id"],
                "source": "tau3_corem_controlled_counterexample",
                "domain": domain,
                "query": row["query"],
                "required_atoms": row["required_atoms"],
                "verified_atoms": row["verified_atoms"],
                "gold_operation_ids": row["gold_operation_ids"],
                "selected_operation_ids": [
                    f"tau3:{domain}:{item.requestor}:{item.name}" for item in actions
                ] + [operation_id],
                "selected_memory_ids": full_ids + [memory_id],
                "outcome": 0,
                "split": row["split"],
                "event_time": None,
                "active_versions": row["active_versions"],
                "scope": "multi_operation_state_criteria",
                "label_basis": sorted(
                    item.value for item in set(task.evaluation_criteria.reward_basis).intersection(
                        {RewardType.DB, RewardType.ENV_ASSERTION}
                    )
                ),
                "actually_executed": True,
                "independently_verified": True,
                "eligible_for_training": True,
                "eligible_for_interaction_training": True,
                "variants": ["controlled_complete_group_with_perturbed_extra_call"],
                "execution_error_count": len(result["errors"]),
                "assertion_error_count": len(result["env_assertion_errors"]),
                "target_db_match": (
                    result["agent_db_hash"] == target["agent_db_hash"]
                    and result["user_db_hash"] == target["user_db_hash"]
                ),
                "environment_assertions_pass": (
                    bool(result["env_assertion_results"])
                    and all(result["env_assertion_results"])
                )
                if RewardType.ENV_ASSERTION in set(task.evaluation_criteria.reward_basis)
                else None,
            },
        )
        append_jsonl(
            output / "duplicate_groups.jsonl",
            {
                "group_id": f"duplicate:{normalized_task_id}:controlled-{action_index}",
                "memory_ids": [
                    f"mem:{normalized_task_id}:a{action_index}",
                    memory_id,
                ],
                "source": "controlled_same_operation_argument_perturbation",
            },
        )
        memories[memory_id] = {"memory_id": memory_id}
        added[f"split:{row['split']}"] += 1
        added["memory_records"] += 1
        added["set_outcomes"] += 1
        added["duplicate_groups"] += 1
        if position % 25 == 0:
            print(f"processed {position}/{len(tasks)} tasks", flush=True)

    quality_path = output / "quality_report.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["schema_version"] = "2.2.0"
    quality["counts"]["memory_records.jsonl"] = (
        quality["counts"].get("memory_records.jsonl", 0) + added["memory_records"]
    )
    quality["counts"]["set_outcomes.jsonl"] = (
        quality["counts"].get("set_outcomes.jsonl", 0) + added["set_outcomes"]
    )
    quality["counts"]["duplicate_groups.jsonl"] = (
        quality["counts"].get("duplicate_groups.jsonl", 0)
        + added["duplicate_groups"]
    )
    for split in ("train", "val", "test"):
        quality["outcome_counts_by_split"][split] += added[f"split:{split}"]
    quality["outcome_class_counts"]["0"] += added["set_outcomes"]
    quality["fresh_environment_replays"] += replay_count
    quality["controlled_counterexamples"] = {
        "attempted_tasks": len(tasks),
        "complete_group_failures_added": added["set_outcomes"],
        "tasks_without_failing_perturbation": added["no_failing_perturbation"],
        "construction": "successful official trajectory plus executed perturbed extra call",
    }
    quality["verified_invariants"]["controlled_counterexamples_are_freshly_replayed"] = True
    quality_path.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme_path = output / "README.md"
    readme_path.write_text(
        readme_path.read_text(encoding="utf-8")
        + "\n## Controlled counterexample extension\n\n"
        + "This 2.2 extension adds one freshly executed complete-declared-group failure per eligible task when a deterministic perturbed extra call violates the official final-state contract. The extra record is paired with its same-operation source in `duplicate_groups.jsonl`. This is a controlled T3 stress track, not natural service behavior.\n",
        encoding="utf-8",
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["name"] = "tau3_corem_execution_verified_counterexamples"
    manifest["schema_version"] = "2.2.0"
    files = {}
    for path in sorted(output.iterdir()):
        if path.name in {"manifest.json", "checksums.sha256"}:
            continue
        metadata = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if path.suffix == ".jsonl":
            metadata["records"] = sum(
                bool(line.strip()) for line in path.open(encoding="utf-8")
            )
        files[path.name] = metadata
    manifest["files"] = files
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_paths = sorted(
        path for path in output.iterdir() if path.name != "checksums.sha256"
    )
    (output / "checksums.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    print(json.dumps({"added": dict(added), "replays": replay_count}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
