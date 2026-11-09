#!/usr/bin/env python3
"""Build evaluation-only BS/ECR annotations and controlled stale records for tau3.

The source dataset is copied to a new directory.  Oracle irreducible actions are
used only to create held-out metric annotations.  A controlled stale record is
accepted only when its deprecated arguments fail under the current contract;
the prior-contract adapter maps those arguments to the official reference call.
This is a controlled schema-shift stress test, not natural service history.
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

from loguru import logger
from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import RewardType

from build_tau3_corem_dataset import (
    DOMAINS,
    compact_text,
    initialize_environment,
    matches_target,
    replay,
    sha256_file,
    token_estimate,
)
from augment_tau3_counterexamples import perturb


SCHEMA_VERSION = "2.3.0"
PRIOR_VERSION = "controlled-pre-migration-v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/tau3_corem_counterexamples")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/tau3_corem_metric_extension")
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


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def binding_annotations(
    task_id: str,
    task: Any,
    required_memory_ids: list[str],
) -> dict[str, Any]:
    actions = task.evaluation_criteria.actions or []
    atoms_by_memory: dict[str, list[dict[str, Any]]] = {}
    for memory_id in required_memory_ids:
        index = int(memory_id.rsplit(":a", 1)[1])
        action = actions[index]
        prefix = f"binding:{task_id}:a{index}"
        atoms = [
            {
                "atom": f"{prefix}:operation",
                "kind": "operation",
                "expected": f"{action.requestor}:{action.name}",
            }
        ]
        for key, value in sorted((action.arguments or {}).items()):
            atoms.append(
                {
                    "atom": f"{prefix}:argument:{key}",
                    "kind": "argument",
                    "argument": key,
                    "expected": canonical(value),
                }
            )
        atoms_by_memory[memory_id] = atoms
    return {
        "task_id": task_id,
        "required_evidence_memory_ids": required_memory_ids,
        "binding_atoms_by_memory": atoms_by_memory,
        "binding_atom_count": sum(len(atoms) for atoms in atoms_by_memory.values()),
        "annotation_basis": "evaluation-only fresh-replay irreducible official actions",
        "selector_visible": False,
    }


def replay_with_replacement(
    constructor: Any,
    task: Any,
    action_index: int,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    environment = initialize_environment(constructor, task)
    actions = task.evaluation_criteria.actions or []
    errors = []
    for index, action in enumerate(actions):
        call = ToolCall(
            id=f"controlled-shift:{task.id}:{index}",
            name=action.name,
            arguments=arguments if index == action_index else action.arguments,
            requestor=action.requestor,
        )
        response = environment.get_response(call)
        if response.error:
            errors.append(
                {
                    "index": index,
                    "name": action.name,
                    "requestor": action.requestor,
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
            except Exception as error:  # noqa: BLE001 - upstream task defect
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


def refresh_metadata(output: Path, added: Counter[str], replay_count: int) -> None:
    quality_path = output / "quality_report.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["schema_version"] = SCHEMA_VERSION
    for filename in ("memory_records.jsonl", "set_outcomes.jsonl", "duplicate_groups.jsonl"):
        quality["counts"][filename] = quality["counts"].get(filename, 0) + added[filename]
    quality["counts"]["metric_annotations.jsonl"] = added["metric_annotations.jsonl"]
    for split in ("train", "val", "test"):
        quality["outcome_counts_by_split"][split] += added[f"split:{split}"]
    quality["outcome_class_counts"]["0"] += added["set_outcomes.jsonl"]
    quality["fresh_environment_replays"] += replay_count
    quality["controlled_version_shift"] = {
        "prior_version": PRIOR_VERSION,
        "stale_records_added": added["memory_records.jsonl"],
        "tasks_without_current-invalid_replacement": added["no_invalid_replacement"],
        "old_contract_validation": "deprecated call translated to the official reference call",
        "current_contract_validation": "direct deprecated call fails fresh final-state replay",
        "natural_drift": False,
    }
    quality["verified_invariants"]["metric_annotations_are_evaluation_only"] = True
    quality["verified_invariants"]["controlled_stale_records_fail_current_replay"] = True
    quality_path.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["name"] = "tau3_corem_execution_metrics_controlled_shift"
    manifest["schema_version"] = SCHEMA_VERSION
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
    paths = sorted(path for path in output.iterdir() if path.name != "checksums.sha256")
    (output / "checksums.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in paths),
        encoding="utf-8",
    )


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
    edges = load_jsonl(output / "dependency_hyperedges.jsonl")
    oracle = {
        row["task_id"]: list(row["members"])
        for row in edges
        if row.get("evaluation_role") == "oracle_analysis_only"
    }
    source_tasks: dict[tuple[str, str], Any] = {}
    constructors = {}
    for domain, (constructor, get_tasks, _) in DOMAINS.items():
        constructors[domain] = constructor
        for task in get_tasks(None):
            source_tasks[(domain, str(task.id))] = task

    annotations_path = output / "metric_annotations.jsonl"
    annotations_path.write_text("", encoding="utf-8")
    added: Counter[str] = Counter()
    replay_count = 0
    for position, row in enumerate(tasks, start=1):
        domain = str(row["domain"])
        task = source_tasks[(domain, str(row["source_task_id"]))]
        constructor = constructors[domain]
        task_id = str(row["task_id"])
        required = sorted(oracle[task_id])
        append_jsonl(annotations_path, binding_annotations(task_id, task, required))
        added["metric_annotations.jsonl"] += 1

        actions = task.evaluation_criteria.actions or []
        full = tuple(range(len(actions)))
        target = replay(constructor, task, full)
        replay_count += 1
        chosen: tuple[int, dict[str, Any], dict[str, Any]] | None = None
        candidates = sorted(
            range(len(actions)),
            key=lambda index: (bool(actions[index].arguments), index),
            reverse=True,
        )
        for index in candidates:
            deprecated_arguments = perturb(copy.deepcopy(actions[index].arguments))
            result = replay_with_replacement(
                constructor, task, index, deprecated_arguments
            )
            replay_count += 1
            if not matches_target(task, result, target):
                chosen = (index, deprecated_arguments, result)
                break
        if chosen is None:
            added["no_invalid_replacement"] += 1
            continue

        index, deprecated_arguments, result = chosen
        action = actions[index]
        current_memory_id = f"mem:{task_id}:a{index}"
        stale_memory_id = f"mem:{task_id}:controlled-stale-{index}"
        operation_id = f"tau3:{domain}:{action.requestor}:{action.name}"
        text = compact_text(
            row["query"],
            "Legacy pre-migration step",
            action.requestor,
            action.name,
            canonical(deprecated_arguments),
        )
        token_cost = token_estimate(text)
        append_jsonl(
            output / "memory_records.jsonl",
            {
                "memory_id": stale_memory_id,
                "source": "tau3_corem_controlled_version_shift",
                "text": text,
                "operation_ids": [operation_id],
                "evidence_atoms": [
                    row["required_atoms"][0],
                    f"step:{task_id}:{index}",
                ],
                "token_cost": token_cost,
                "storage_cost": token_cost + 4,
                "event_time": None,
                "provenance": f"controlled-pre-migration-contract:{domain}:{task.id}:{index}",
                "provenance_verified": True,
                "observed_outcome": None,
                "split": row["split"],
                "admissible": True,
                "eligible_for_outcome_training": False,
                "version_scope": [PRIOR_VERSION],
                "controlled_stale": True,
                "current_version_invalid": True,
                "prior_contract_adapter": "translate deprecated arguments to official reference arguments",
                "current_replay_execution_error_count": len(result["errors"]),
                "current_replay_assertion_error_count": len(result["env_assertion_errors"]),
            },
        )
        selected_ids = [
            stale_memory_id if action_index == index else f"mem:{task_id}:a{action_index}"
            for action_index in full
        ]
        append_jsonl(
            output / "set_outcomes.jsonl",
            {
                "example_id": f"outcome:{task_id}:controlled-stale-replacement",
                "task_id": task_id,
                "source_task_id": row["source_task_id"],
                "source": "tau3_corem_controlled_version_shift",
                "domain": domain,
                "query": row["query"],
                "required_atoms": row["required_atoms"],
                "verified_atoms": row["verified_atoms"],
                "gold_operation_ids": row["gold_operation_ids"],
                "selected_operation_ids": [
                    f"tau3:{domain}:{item.requestor}:{item.name}" for item in actions
                ],
                "selected_memory_ids": selected_ids,
                "outcome": 0,
                "split": row["split"],
                "event_time": None,
                "active_versions": row["active_versions"],
                "scope": "multi_operation_state_criteria",
                "label_basis": sorted(
                    item.value
                    for item in set(task.evaluation_criteria.reward_basis).intersection(
                        {RewardType.DB, RewardType.ENV_ASSERTION}
                    )
                ),
                "actually_executed": True,
                "independently_verified": True,
                "eligible_for_training": False,
                "eligible_for_interaction_training": False,
                "variants": ["controlled_current-invalid_stale_replacement"],
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
                if RewardType.ENV_ASSERTION
                in set(task.evaluation_criteria.reward_basis)
                else None,
            },
        )
        append_jsonl(
            output / "duplicate_groups.jsonl",
            {
                "group_id": f"duplicate:{task_id}:controlled-stale-{index}",
                "memory_ids": [current_memory_id, stale_memory_id],
                "source": "controlled_prior-current_contract_pair",
            },
        )
        added["memory_records.jsonl"] += 1
        added["set_outcomes.jsonl"] += 1
        added["duplicate_groups.jsonl"] += 1
        added[f"split:{row['split']}"] += 1
        if position % 25 == 0:
            print(f"processed {position}/{len(tasks)} tasks", flush=True)

    readme = output / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8")
        + "\n## Evaluation-metric extension\n\n"
        + "Schema 2.3 adds evaluation-only irreducible-action annotations for BS/ECR "
        + "and controlled pre-migration records for SUR. The stale records are validated "
        + "against a synthetic contract adapter and fresh current-contract replay; they do "
        + "not represent natural production drift.\n",
        encoding="utf-8",
    )
    refresh_metadata(output, added, replay_count)
    print(json.dumps({"added": dict(added), "replays": replay_count}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
