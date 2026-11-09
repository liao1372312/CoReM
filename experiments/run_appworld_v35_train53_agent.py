#!/usr/bin/env python3
"""Run two frozen, disjoint shards of a 53-task AppWorld train stress test."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_appworld_v30_full_docs_agent as v30  # noqa: E402
from experiments import run_appworld_v32_code_history_agent as v32  # noqa: E402
from experiments import run_appworld_v33_auth_contract_agent as v33  # noqa: E402
from experiments import run_appworld_v34_token_schema_guard_agent as v34  # noqa: E402

SELECTION_SEED = 2035
SHARD_SIZES = (30, 23)
SELECTION_POLICY = (
    "label_blind_generator_round_robin_excluding_seed2033_exact_tasks_shard_30_23"
)
_original_sampler = v30.base._sample_task_ids


def selected_task_ids(task_ids: list[str]) -> list[str]:
    """Select 53 tasks without reading instructions, labels, or outcomes."""
    from appworld.task import task_id_to_generator_id

    excluded = set(_original_sampler(task_ids, 2033, 9))
    grouped: dict[str, list[str]] = {}
    for task_id in task_ids:
        if task_id not in excluded:
            grouped.setdefault(task_id_to_generator_id(task_id), []).append(task_id)
    groups = sorted(grouped)
    rng = random.Random(SELECTION_SEED)
    rng.shuffle(groups)
    for group in groups:
        grouped[group].sort()
        rng.shuffle(grouped[group])
    selected: list[str] = []
    for depth in range(max(map(len, grouped.values()))):
        for group in groups:
            if depth < len(grouped[group]):
                selected.append(grouped[group][depth])
                if len(selected) == sum(SHARD_SIZES):
                    return selected
    raise ValueError("not enough eligible train tasks for the frozen 53-task set")


def sample_shard(task_ids: list[str], seed: int, count: int) -> list[str]:
    """Return one of the two declared disjoint shards."""
    if seed != SELECTION_SEED or count not in SHARD_SIZES:
        raise ValueError("v35 only permits its frozen seed and shard sizes")
    selected = selected_task_ids(task_ids)
    return selected[:30] if count == 30 else selected[30:]


def selection_digest(task_ids: list[str]) -> str:
    """Return a non-public aggregate commitment to the ordered selection."""
    selected = selected_task_ids(task_ids)
    return hashlib.sha256("\n".join(selected).encode()).hexdigest()


def main() -> int:
    v30.base._sample_task_ids = sample_shard
    v30.base._initial_messages = v33.initial_messages
    v30.context_messages = v32.context_messages
    v30.guarded_execute_and_flush = v34.guarded_execute_and_flush
    v30.run_task = v34.run_task
    status = v30.main()
    output_index = sys.argv.index("--output") + 1
    output = Path(sys.argv[output_index])
    payload = json.loads(output.read_text())
    train_ids = (Path(sys.argv[sys.argv.index("--root") + 1])
                 / "data/datasets/train.txt").read_text().splitlines()
    payload["task_sampling"] = SELECTION_POLICY
    payload["selection_seed"] = SELECTION_SEED
    payload["selection_digest"] = selection_digest(train_ids)
    payload["selection_agent_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    with output.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
