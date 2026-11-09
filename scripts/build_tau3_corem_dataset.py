#!/usr/bin/env python3
"""Build an execution-verified CoReM dataset from public tau3-bench tasks.

The builder replays reference tool trajectories in the official environments,
then searches for a one-deletion-irreducible successful subset. Every emitted
set label is obtained by a fresh replay and the task's deterministic final-state
criteria: agent/user database hashes, environment assertions, or both.
Communication, action matching, and natural-language criteria are out of scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import RewardType, Task
from tau2.domains.airline.environment import (
    get_environment as get_airline_environment,
)
from tau2.domains.airline.environment import get_tasks as get_airline_tasks
from tau2.domains.airline.environment import get_tasks_split as get_airline_splits
from tau2.domains.retail.environment import get_environment as get_retail_environment
from tau2.domains.retail.environment import get_tasks as get_retail_tasks
from tau2.domains.retail.environment import get_tasks_split as get_retail_splits
from tau2.domains.telecom.environment import (
    get_environment as get_telecom_environment,
)
from tau2.domains.telecom.environment import get_tasks as get_telecom_tasks
from tau2.domains.telecom.environment import get_tasks_split as get_telecom_splits


SCHEMA_VERSION = "2.1.0"
LABEL_SCOPE = "multi_operation_state_criteria"
SOURCE_URL = "https://github.com/sierra-research/tau2-bench"
SOURCE_RELEASE = "1.0.1"
DOMAINS: dict[
    str,
    tuple[
        Callable[..., Any],
        Callable[[str | None], list[Task]],
        Callable[[], dict[str, list[str]]],
    ],
] = {
    "airline": (get_airline_environment, get_airline_tasks, get_airline_splits),
    "retail": (get_retail_environment, get_retail_tasks, get_retail_splits),
    "telecom": (get_telecom_environment, get_telecom_tasks, get_telecom_splits),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("third_party/tau2-bench"),
        help="checked-out official tau2-bench/tau3-bench repository",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tau3_corem"),
        help="new dataset directory; must not already exist",
    )
    parser.add_argument(
        "--domains",
        default="airline,retail,telecom",
        help="comma-separated domains from airline, retail, telecom",
    )
    return parser.parse_args()


def stable_digest(value: str, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def validation_split(official_train_position: int) -> bool:
    return official_train_position % 3 == 0


def compact_text(*parts: Any) -> str:
    return " ".join(
        str(part).strip().replace("\n", " ")
        for part in parts
        if part is not None and str(part).strip()
    )


def token_estimate(text: str) -> int:
    return max(1, len(text.split()))


def json_line(handle: TextIO, value: dict[str, Any]) -> None:
    handle.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_commit(source: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def task_query(task: Task) -> str:
    instructions = task.user_scenario.instructions
    return compact_text(
        instructions.reason_for_call,
        instructions.task_instructions,
        instructions.known_info,
        instructions.unknown_info,
    )


def initialize_environment(constructor: Callable[..., Any], task: Task) -> Any:
    environment = constructor()
    state = task.initial_state
    environment.set_state(
        initialization_data=state.initialization_data if state else None,
        initialization_actions=state.initialization_actions if state else None,
        message_history=list(state.message_history or []) if state else [],
    )
    return environment


def replay(
    constructor: Callable[..., Any], task: Task, action_indices: tuple[int, ...]
) -> dict[str, Any]:
    environment = initialize_environment(constructor, task)
    actions = task.evaluation_criteria.actions or []
    errors: list[dict[str, Any]] = []
    for index in action_indices:
        action = actions[index]
        call = ToolCall(
            id=f"replay:{task.id}:{index}",
            name=action.name,
            arguments=action.arguments,
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
    assertion_results: list[bool] = []
    assertion_errors: list[dict[str, Any]] = []
    criteria = task.evaluation_criteria
    if criteria is not None and RewardType.ENV_ASSERTION in criteria.reward_basis:
        for assertion in criteria.env_assertions or []:
            try:
                assertion_results.append(
                    environment.run_env_assertion(
                        assertion,
                        raise_assertion_error=False,
                    )
                )
            except Exception as error:  # noqa: BLE001  # source task defect
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


def matches_target(
    task: Task,
    result: dict[str, Any],
    target: dict[str, Any],
) -> bool:
    criteria = task.evaluation_criteria
    if criteria is None or result["errors"] or result["env_assertion_errors"]:
        return False
    reward_basis = set(criteria.reward_basis)
    db_match = (
        result["agent_db_hash"] == target["agent_db_hash"]
        and result["user_db_hash"] == target["user_db_hash"]
    )
    assertions_pass = bool(result["env_assertion_results"]) and all(
        result["env_assertion_results"]
    )
    return (
        (RewardType.DB not in reward_basis or db_match)
        and (RewardType.ENV_ASSERTION not in reward_basis or assertions_pass)
    )


def tool_catalog(environment: Any, domain: str) -> dict[tuple[str, str], dict[str, Any]]:
    catalog: dict[tuple[str, str], dict[str, Any]] = {}
    tool_groups = [("assistant", environment.get_tools())]
    if environment.user_tools is not None:
        tool_groups.append(("user", environment.get_user_tools()))
    for requestor, tools in tool_groups:
        toolkit = environment.tools if requestor == "assistant" else environment.user_tools
        for tool in tools:
            schema = tool.openai_schema["function"]
            catalog[(requestor, tool.name)] = {
                "operation_id": f"tau3:{domain}:{requestor}:{tool.name}",
                "service_id": f"tau3:{domain}:{requestor}",
                "source": "tau3_bench_public",
                "name": tool.name,
                "description": schema.get("description", ""),
                "category": domain,
                "method": "TOOL",
                "endpoint": f"{requestor}://{domain}/{tool.name}",
                "required_input_atoms": sorted(
                    schema.get("parameters", {}).get("required", [])
                ),
                "output_atoms": [],
                "tool_type": str(toolkit.tool_type(tool.name).value),
                "mutates_state": bool(toolkit.tool_mutates_state(tool.name)),
                "provenance": {
                    "source": SOURCE_URL,
                    "domain": domain,
                    "requestor": requestor,
                },
            }
    return catalog


class Builder:
    def __init__(self, source: Path, output: Path, domains: Iterable[str]) -> None:
        self.source = source.resolve()
        self.output = output.resolve()
        self.domains = tuple(domains)
        self.commit = source_commit(self.source)
        self.temp: Path | None = None
        self.handles: dict[str, TextIO] = {}
        self.operations: dict[str, dict[str, Any]] = {}
        self.counts: Counter[str] = Counter()
        self.split_tasks: Counter[str] = Counter()
        self.split_outcomes: Counter[str] = Counter()
        self.outcomes: Counter[int] = Counter()
        self.exclusions: Counter[str] = Counter()
        self.replay_count = 0

    def __enter__(self) -> "Builder":
        if self.output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {self.output}")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.temp = Path(
            tempfile.mkdtemp(prefix=".tau3_corem_", dir=self.output.parent)
        )
        for filename in (
            "memory_records.jsonl",
            "retrieval_tasks.jsonl",
            "dependency_hyperedges.jsonl",
            "set_outcomes.jsonl",
            "duplicate_groups.jsonl",
        ):
            self.handles[filename] = (self.temp / filename).open(
                "w", encoding="utf-8"
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles.values():
            handle.close()
        if exc_type is not None and self.temp is not None:
            shutil.rmtree(self.temp)

    def write(self, filename: str, row: dict[str, Any]) -> None:
        json_line(self.handles[filename], row)
        self.counts[filename] += 1

    def run_replay(
        self,
        constructor: Callable[..., Any],
        task: Task,
        indices: tuple[int, ...],
        cache: dict[tuple[int, ...], dict[str, Any]],
    ) -> dict[str, Any]:
        if indices not in cache:
            cache[indices] = replay(constructor, task, indices)
            self.replay_count += 1
        return cache[indices]

    def irreducible_success_set(
        self,
        constructor: Callable[..., Any],
        task: Task,
        full: tuple[int, ...],
        target: dict[str, Any],
        cache: dict[tuple[int, ...], dict[str, Any]],
    ) -> tuple[int, ...]:
        kept = full
        changed = True
        while changed:
            changed = False
            for index in reversed(kept):
                candidate = tuple(item for item in kept if item != index)
                result = self.run_replay(constructor, task, candidate, cache)
                if matches_target(task, result, target):
                    kept = candidate
                    changed = True
        return kept

    def process_domain(self, domain: str) -> None:
        constructor, get_tasks, get_splits = DOMAINS[domain]
        environment = constructor()
        self.operations.update(
            (row["operation_id"], row)
            for row in tool_catalog(environment, domain).values()
        )
        splits = get_splits()
        train_ids = set(splits["train"])
        test_ids = set(splits["test"])
        train_positions = {
            task_id: position for position, task_id in enumerate(splits["train"])
        }
        base_ids = train_ids | test_ids
        tasks = [task for task in get_tasks(None) if task.id in base_ids]
        for task in tasks:
            self.process_task(
                domain,
                constructor,
                task,
                "test"
                if task.id in test_ids
                else "val"
                if validation_split(train_positions[task.id])
                else "train",
            )

    def process_task(
        self,
        domain: str,
        constructor: Callable[..., Any],
        task: Task,
        split: str,
    ) -> None:
        criteria = task.evaluation_criteria
        if criteria is None:
            self.exclusions["no_evaluation_criteria"] += 1
            return
        reward_basis = set(criteria.reward_basis)
        state_basis = reward_basis.intersection(
            {RewardType.DB, RewardType.ENV_ASSERTION}
        )
        if not state_basis:
            self.exclusions["no_deterministic_state_reward"] += 1
            return
        if RewardType.ENV_ASSERTION in state_basis and not criteria.env_assertions:
            self.exclusions["missing_environment_assertions"] += 1
            return
        actions = criteria.actions or []
        if len(actions) < 2:
            self.exclusions["fewer_than_two_reference_actions"] += 1
            return

        full = tuple(range(len(actions)))
        cache: dict[tuple[int, ...], dict[str, Any]] = {}
        target = self.run_replay(constructor, task, full, cache)
        if not matches_target(task, target, target):
            self.exclusions["reference_replay_or_state_criterion_failure"] += 1
            return
        irreducible = self.irreducible_success_set(
            constructor, task, full, target, cache
        )
        if len(irreducible) < 2:
            self.exclusions["no_verified_multi_action_dependency"] += 1
            return
        for index in irreducible:
            candidate = tuple(item for item in irreducible if item != index)
            if matches_target(
                task,
                self.run_replay(constructor, task, candidate, cache),
                target,
            ):
                raise RuntimeError(
                    f"irreducibility check failed for {domain}:{task.id}:{index}"
                )

        task_key = f"{domain}:{task.id}"
        task_digest = stable_digest(task_key)
        normalized_task_id = f"tau3:{domain}:{task_digest}"
        goal_atom = f"goal:{normalized_task_id}"
        query = task_query(task)
        memory_ids = [f"mem:{normalized_task_id}:a{index}" for index in full]
        operation_ids = [
            f"tau3:{domain}:{action.requestor}:{action.name}" for action in actions
        ]
        for index, (action, memory_id, operation_id) in enumerate(
            zip(actions, memory_ids, operation_ids)
        ):
            operation = self.operations[operation_id]
            arguments = json.dumps(
                action.arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            text = compact_text(
                query,
                f"Step {index + 1}",
                action.requestor,
                action.name,
                operation["description"],
                arguments,
            )
            tokens = token_estimate(text)
            self.write(
                "memory_records.jsonl",
                {
                    "memory_id": memory_id,
                    "source": "tau3_bench_public",
                    "text": text,
                    "operation_ids": [operation_id],
                    "evidence_atoms": [goal_atom, f"step:{normalized_task_id}:{index}"],
                    "token_cost": tokens,
                    "storage_cost": tokens + 4,
                    "event_time": None,
                    "provenance": f"{self.commit}:{domain}:task:{task.id}:action:{index}",
                    "provenance_verified": True,
                    "observed_outcome": None,
                    "split": split,
                    "admissible": True,
                    "eligible_for_outcome_training": False,
                    "version_scope": [self.commit],
                },
            )

        irreducible_memories = [memory_ids[index] for index in irreducible]
        mutating = tuple(
            index
            for index, operation_id in enumerate(operation_ids)
            if self.operations[operation_id]["mutates_state"]
        )
        if len(mutating) < 2:
            raise RuntimeError(
                f"verified multi-action task lacks two declared mutations: {task_key}"
            )
        declared_memories = [memory_ids[index] for index in mutating]
        declared_operations = [operation_ids[index] for index in mutating]
        declared_edge_id = f"dep:{normalized_task_id}:declared-mutations"
        self.write(
            "dependency_hyperedges.jsonl",
            {
                "edge_id": declared_edge_id,
                "task_id": normalized_task_id,
                "source": "tau3_bench_public",
                "members": declared_memories,
                "evidence_atoms": [goal_atom],
                "relation_type": "declared_state_transition_group",
                "grounding": "official_reference_mutating_actions",
                "confidence": 1.0,
                "admissible_default": True,
                "verified": True,
                "provenance": f"tau3-reference:{self.commit}:{domain}:{task.id}",
                "signature": stable_digest("|".join(declared_memories), 32),
                "version_scope": [self.commit],
            },
        )
        oracle_edge_id = f"dep:{normalized_task_id}:oracle-irreducible"
        self.write(
            "dependency_hyperedges.jsonl",
            {
                "edge_id": oracle_edge_id,
                "task_id": normalized_task_id,
                "source": "tau3_bench_public_replay",
                "members": irreducible_memories,
                "evidence_atoms": [goal_atom],
                "relation_type": "execution_irreducible_dependency",
                "grounding": "fresh_environment_leave_one_out_replay",
                "confidence": 1.0,
                "admissible_default": False,
                "evaluation_role": "oracle_analysis_only",
                "verified": True,
                "provenance": f"tau3-state-criteria:{self.commit}:{domain}:{task.id}",
                "signature": stable_digest("|".join(irreducible_memories), 32),
                "version_scope": [self.commit],
            },
        )
        self.write(
            "retrieval_tasks.jsonl",
            {
                "task_id": normalized_task_id,
                "source_task_id": task.id,
                "source": "tau3_bench_public",
                "domain": domain,
                "query": query,
                "required_atoms": [goal_atom],
                "verified_atoms": [],
                "candidate_operation_ids": sorted(set(operation_ids)),
                "gold_operation_ids": sorted(set(declared_operations)),
                "gold_memory_ids": declared_memories,
                "dependency_edge_ids": [declared_edge_id],
                "split": split,
                "event_time": None,
                "active_versions": [self.commit],
                "supervision": "execution_verified_state_criteria",
                "is_executed_composition": True,
                "metadata": {
                    "reference_action_count": len(full),
                    "reference_mutating_action_count": len(mutating),
                    "irreducible_action_count": len(irreducible),
                    "oracle_dependency_edge_id": oracle_edge_id,
                    "official_split": "test" if split == "test" else "train",
                },
            },
        )

        candidates: dict[tuple[int, ...], set[str]] = {
            full: {"full_reference"},
            irreducible: {"irreducible_success"},
        }
        for index in full:
            candidate = tuple(item for item in full if item != index)
            candidates.setdefault(candidate, set()).add("full_leave_one_out")
        for index in irreducible:
            candidate = tuple(item for item in irreducible if item != index)
            candidates.setdefault(candidate, set()).add(
                "irreducible_leave_one_out"
            )

        for sequence, (indices, variants) in enumerate(sorted(candidates.items())):
            result = self.run_replay(constructor, task, indices, cache)
            outcome = int(matches_target(task, result, target))
            selected_memories = [memory_ids[index] for index in indices]
            selected_operations = [operation_ids[index] for index in indices]
            self.write(
                "set_outcomes.jsonl",
                {
                    "example_id": f"outcome:{normalized_task_id}:{sequence}",
                    "task_id": normalized_task_id,
                    "source_task_id": task.id,
                    "source": "tau3_bench_public_replay",
                    "domain": domain,
                    "query": query,
                    "required_atoms": [goal_atom],
                    "verified_atoms": [],
                    "gold_operation_ids": sorted(set(declared_operations)),
                    "selected_operation_ids": selected_operations,
                    "selected_memory_ids": selected_memories,
                    "outcome": outcome,
                    "split": split,
                    "event_time": None,
                    "active_versions": [self.commit],
                    "scope": LABEL_SCOPE,
                    "label_basis": sorted(item.value for item in state_basis),
                    "actually_executed": True,
                    "independently_verified": True,
                    "eligible_for_training": True,
                    "eligible_for_interaction_training": True,
                    "variants": sorted(variants),
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
                    if RewardType.ENV_ASSERTION in state_basis
                    else None,
                },
            )
            self.outcomes[outcome] += 1
            self.split_outcomes[split] += 1
        self.split_tasks[split] += 1

    def finalize(self) -> None:
        if self.temp is None:
            raise RuntimeError("builder is not active")
        with (self.temp / "operations.jsonl").open("w", encoding="utf-8") as handle:
            for operation_id in sorted(self.operations):
                json_line(handle, self.operations[operation_id])
                self.counts["operations.jsonl"] += 1
        for handle in self.handles.values():
            handle.flush()

        quality = {
            "schema_version": SCHEMA_VERSION,
            "source": {
                "url": SOURCE_URL,
                "release": SOURCE_RELEASE,
                "commit": self.commit,
                "license": "MIT",
            },
            "counts": dict(sorted(self.counts.items())),
            "task_counts_by_split": dict(sorted(self.split_tasks.items())),
            "outcome_counts_by_split": dict(sorted(self.split_outcomes.items())),
            "outcome_class_counts": {
                str(key): value for key, value in sorted(self.outcomes.items())
            },
            "excluded_task_counts": dict(sorted(self.exclusions.items())),
            "fresh_environment_replays": self.replay_count,
            "verified_invariants": {
                "all_outcomes_freshly_replayed": True,
                "all_positive_db_labels_match_agent_and_user_db_hashes": True,
                "all_positive_assertion_labels_pass_environment_assertions": True,
                "all_oracle_hyperedges_are_leave_one_out_irreducible": True,
                "default_graph_excludes_outcome_derived_oracle_edges": True,
                "official_test_split_is_preserved": True,
                "task_ids_do_not_cross_splits": True,
            },
            "label_scope": (
                "Deterministic final-state criteria only: agent/user database "
                "hashes and/or official environment assertions. Communication, "
                "action matching, and natural-language assertions are not evaluated."
            ),
        }
        (self.temp / "quality_report.json").write_text(
            json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.temp / "PUBLIC").write_text(
            "Derived from the MIT-licensed public tau3-bench repository.\n",
            encoding="utf-8",
        )
        shutil.copy2(self.source / "LICENSE", self.temp / "SOURCE_LICENSE")
        readme = f"""# tau3-CoReM Dataset

Execution-verified multi-operation examples derived from tau3-bench {SOURCE_RELEASE}
at commit `{self.commit}`. The source repository is {SOURCE_URL} and is MIT licensed.

Each included task has a successful reference trajectory and a successful subset
that is irreducible under single-action deletion. Candidate sets were replayed in
fresh official environments. Labels apply each task's official deterministic state
criteria: equality of both final database hashes, environment assertions, or both.
They do **not** evaluate communication, action matching, or natural-language criteria.

Default graph edges use only state-mutating actions declared in the official reference
trajectory. Outcome-derived irreducible edges are marked `oracle_analysis_only` and
are excluded by the loader, preventing test-label leakage in the standard experiment.

Splits preserve the official test set. Every third task in each official training
list is assigned to validation. All variants of a task stay in one split.
See `quality_report.json` for counts and exclusions.
"""
        (self.temp / "README.md").write_text(readme, encoding="utf-8")

        files: dict[str, dict[str, Any]] = {}
        for path in sorted(self.temp.iterdir()):
            if path.name in {"manifest.json", "checksums.sha256"}:
                continue
            metadata: dict[str, Any] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if path.suffix == ".jsonl":
                metadata["records"] = self.counts[path.name]
            files[path.name] = metadata
        manifest = {
            "name": "tau3_corem_execution_verified",
            "schema_version": SCHEMA_VERSION,
            "private": False,
            "redistributable": True,
            "source": quality["source"],
            "label_scope": LABEL_SCOPE,
            "files": files,
        }
        (self.temp / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checksum_paths = sorted(
            path for path in self.temp.iterdir() if path.name != "checksums.sha256"
        )
        (self.temp / "checksums.sha256").write_text(
            "".join(
                f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths
            ),
            encoding="utf-8",
        )
        self.temp.replace(self.output)
        print(json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True))
        print(f"wrote {self.output}")


def main() -> None:
    args = parse_args()
    domains = tuple(part.strip() for part in args.domains.split(",") if part.strip())
    unknown = set(domains).difference(DOMAINS)
    if unknown:
        raise ValueError(f"unknown domains: {sorted(unknown)}")
    with Builder(args.source, args.output, domains) as builder:
        for domain in domains:
            print(f"processing {domain}", flush=True)
            builder.process_domain(domain)
        builder.finalize()


if __name__ == "__main__":
    main()
