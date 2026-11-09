#!/usr/bin/env python3
"""Launch and verify the frozen two-shard AppWorld v35 train stress test."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_appworld_v28_qwen32_gate as v28  # noqa: E402
from experiments import run_appworld_v29_qwen32_guard_gate as v29  # noqa: E402
from experiments import run_appworld_v30_qwen32_full_docs_gate as v30  # noqa: E402
from experiments import run_appworld_v31_fresh_train_gate as v31  # noqa: E402
from experiments import run_appworld_v32_code_history_gate as v32  # noqa: E402
from experiments import run_appworld_v33_auth_contract_gate as v33  # noqa: E402
from experiments import run_appworld_v34_token_schema_guard_agent as v34  # noqa: E402
from experiments import run_appworld_v35_train53_agent as agent  # noqa: E402
from experiments.preflight_appworld_v23 import free_gpu_mib, sha256  # noqa: E402
from experiments.summarize_appworld_v23_paired_pilot import (  # noqa: E402
    RUNNER_SHA256,
    TRAIN_SPLIT_SHA256,
)

AGENT = ROOT / "experiments" / "run_appworld_v35_train53_agent.py"
AGENT_SHA256 = "2b34bfbbdad6c9324828c4f2dfbf5101cdb5c4b870142850dcd3996384a1b8fc"
PREFIX = "appworld_v35_train_stress_seed2035"
RESULTS = (
    v28.PRIVATE / f"{PREFIX}_n30_shard1.json",
    v28.PRIVATE / f"{PREFIX}_n23_shard2.json",
)
SUMMARY = v28.PRIVATE / f"{PREFIX}_n53_summary.json"
COST = v28.PRIVATE / f"{PREFIX}_n53_tokens.json"
SERVER_LOG = v28.PRIVATE / f"{PREFIX}_server.log"
EXPERIMENTS = (
    "v35_train_stress_seed2035_n30_shard1",
    "v35_train_stress_seed2035_n23_shard2",
)
OUTPUT_DIRS = tuple(
    v28.APPWORLD_ROOT / "experiments" / "outputs" / name for name in EXPERIMENTS
)


def runner_argv(shard: int) -> list[str]:
    command = v33.runner_argv()
    command[1] = str(AGENT)
    command[command.index("--output") + 1] = str(RESULTS[shard])
    command[command.index("--experiment-name") + 1] = EXPERIMENTS[shard]
    command[command.index("--seed") + 1] = str(agent.SELECTION_SEED)
    command[command.index("--num-tasks") + 1] = str(agent.SHARD_SIZES[shard])
    return command


def expected_hashes(shard: int) -> set[str]:
    import hashlib

    path = v28.APPWORLD_ROOT / "data" / "datasets" / "train.txt"
    selected = agent.selected_task_ids(path.read_text().splitlines())
    tasks = selected[:30] if shard == 0 else selected[30:]
    return {hashlib.sha256(task.encode()).hexdigest() for task in tasks}


def validate_result(payload: dict[str, object], shard: int) -> dict[str, int]:
    train_ids = (
        v28.APPWORLD_ROOT / "data" / "datasets" / "train.txt"
    ).read_text().splitlines()
    expected = {
        "status": "train_only_state_based_agent_pilot_not_corem_result",
        "benchmark": "AppWorld 0.1.3.post1",
        "split": "train",
        "task_sampling": agent.SELECTION_POLICY,
        "model": v28.MODEL_NAME,
        "model_endpoint": v28.ENDPOINT,
        "model_digest": v28.MODEL_DIGEST,
        "seed": agent.SELECTION_SEED,
        "appworld_random_seed": 100,
        "max_steps": 20,
        "context_tokens": 8192,
        "predict_tokens": 512,
        "prompt_variant": "checklist",
        "history_strategy": "preserve_task",
        "raise_on_failure": True,
        "num_tasks": agent.SHARD_SIZES[shard],
        "memory_condition": "none",
        "memory_view_sha256": None,
        "doc_condition": "none",
        "doc_view_sha256": None,
        "script_sha256": RUNNER_SHA256,
        "train_split_sha256": TRAIN_SPLIT_SHA256,
        "protected_text_exported": False,
        "selection_seed": agent.SELECTION_SEED,
        "selection_digest": agent.selection_digest(train_ids),
        "selection_agent_sha256": AGENT_SHA256,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"v35 shard {shard + 1} differs from frozen settings")
    rows = payload.get("private_task_outcomes")
    if not isinstance(rows, list) or len(rows) != agent.SHARD_SIZES[shard]:
        raise ValueError(f"v35 shard {shard + 1} has incomplete task rows")
    if {row.get("task_sha256") for row in rows} != expected_hashes(shard):
        raise ValueError(f"v35 shard {shard + 1} used the wrong train tasks")
    totals = {
        "steps": 0,
        "nested_guard_interventions": 0,
        "token_schema_guard_interventions": 0,
        "full_doc_observations": 0,
    }
    for row in rows:
        if (
            type(row.get("success")) is not bool
            or type(row.get("task_completed")) is not bool
            or type(row.get("steps")) is not int
            or not 0 <= row["steps"] <= 20
            or row.get("api_guard_policy")
            != "read_only_documentation_for_invalid_nested_api_paths"
            or row.get("api_guard_script_sha256") != v29.GUARD_SHA256
            or row.get("observation_script_sha256") != v30.AGENT_SHA256
            or row.get("context_script_sha256") != v32.AGENT_SHA256
            or row.get("auth_prompt_script_sha256") != v33.AGENT_SHA256
        ):
            raise ValueError(f"v35 shard {shard + 1} row provenance invalid")
        if (
            row.get("token_schema_guard_policy") != v34.TOKEN_GUARD_POLICY
            or row.get("token_schema_guard_script_sha256")
            != "54aa6ab63998e48621ae9109a84d3beb8097ec326d7f53f1f94a9e5fe5b9ceb0"
        ):
            raise ValueError(f"v35 shard {shard + 1} token guard invalid")
        for target, source in (
            ("steps", "steps"),
            ("nested_guard_interventions", "api_guard_interventions"),
            ("token_schema_guard_interventions", "token_schema_guard_interventions"),
            ("full_doc_observations", "full_doc_observations"),
        ):
            value = row.get(source)
            if type(value) is not int or value < 0:
                raise ValueError(f"v35 shard {shard + 1} row count invalid")
            totals[target] += value
    for key, row_key in (
        ("successes", "success"),
        ("completed_calls", "task_completed"),
        ("model_errors", "model_errors"),
        ("execution_errors", "execution_errors"),
        ("evaluation_errors", "evaluation_error"),
        ("execution_failure_messages", "execution_failure_messages"),
        ("api_doc_steps", "api_doc_steps"),
    ):
        if payload.get(key) != sum(int(row[row_key]) for row in rows):
            raise ValueError(f"v35 shard {shard + 1} aggregate mismatch: {key}")
        totals[key] = int(payload[key])
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    paths = (*RESULTS, SUMMARY, COST, SERVER_LOG, *OUTPUT_DIRS)
    if any(path.exists() for path in paths):
        raise ValueError("refusing to overwrite or resume partial v35 output")
    report = v28.audit(args.model_dir, free_gpu_mib())
    inherited = (
        (AGENT, AGENT_SHA256),
        (v33.AGENT, v33.AGENT_SHA256),
        (v32.AGENT, v32.AGENT_SHA256),
        (v30.AGENT, v30.AGENT_SHA256),
        (v29.GUARD, v29.GUARD_SHA256),
        (v29.V25, v29.V25_SHA256),
    )
    if any(sha256(path) != digest for path, digest in inherited):
        raise ValueError("v35 agent or inherited frozen source changed")
    v31.fresh_target_hashes()
    if len(expected_hashes(0) | expected_hashes(1)) != 53:
        raise ValueError("v35 task shards are not 53 unique tasks")
    if report["status"] != "ready":
        print(json.dumps(report, sort_keys=True))
        return 1
    v28.require_free_local_port()
    with SERVER_LOG.open("x") as server_log:
        server = subprocess.Popen(
            [
                str(v28.PYTHON_SERVER),
                str(v28.SERVER),
                "--model-dir", str(args.model_dir.resolve()),
                "--model-name", v28.MODEL_NAME,
                "--port", "11435",
                "--min-free-gib", "25",
                "--startup-headroom-gib", "19",
            ],
            cwd=ROOT,
            stdout=server_log,
            stderr=server_log,
        )
        try:
            v28.wait_for_server(server)
            before = v28.read_server_stats()
            totals: list[dict[str, int]] = []
            for shard in range(2):
                subprocess.run(runner_argv(shard), cwd=ROOT, check=True)
                totals.append(validate_result(json.loads(RESULTS[shard].read_text()), shard))
            after = v28.read_server_stats()
            merged = {
                key: sum(item[key] for item in totals)
                for key in totals[0]
            }
            counts = v28.token_cost_delta(before, after, merged["steps"])
            summary = {
                "status": "train_only_v35_executor_stress_not_corem_result",
                "benchmark": "AppWorld 0.1.3.post1",
                "split": "train",
                "selection_policy": agent.SELECTION_POLICY,
                "selection_seed": agent.SELECTION_SEED,
                "selection_digest": agent.selection_digest(
                    (v28.APPWORLD_ROOT / "data/datasets/train.txt")
                    .read_text().splitlines()
                ),
                "num_tasks": 53,
                "num_unique_tasks": 53,
                "shard_sizes": list(agent.SHARD_SIZES),
                "model": v28.MODEL_NAME,
                "model_digest": v28.MODEL_DIGEST,
                "max_steps": 20,
                "memory_condition": "none",
                **merged,
                **counts,
                "result_sha256s": [sha256(path) for path in RESULTS],
                "agent_sha256": AGENT_SHA256,
                "task_ids_or_hashes_exported": False,
                "dev_or_test_loaded": False,
            }
            with SUMMARY.open("x") as output:
                json.dump(summary, output, sort_keys=True, indent=2)
                output.write("\n")
            receipt = {
                "status": "train_only_v35_aggregate_token_cost",
                "summary_sha256": sha256(SUMMARY),
                "result_sha256s": [sha256(path) for path in RESULTS],
                "server_sha256": v28.SERVER_SHA256,
                "model_digest": v28.MODEL_DIGEST,
                "model_revision": "0499c3ac83fdef8810b907a23894ba91e95eddd8",
                "seed": agent.SELECTION_SEED,
                "thinking_enabled": False,
                **counts,
                "task_ids_or_hashes_exported": False,
            }
            with COST.open("x") as output:
                json.dump(receipt, output, sort_keys=True, indent=2)
                output.write("\n")
            print(json.dumps(summary, sort_keys=True))
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=15)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
