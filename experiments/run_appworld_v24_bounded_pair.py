#!/usr/bin/env python3
"""Run a fresh four-arm AppWorld TRAIN comparison with bounded history.

This is a post-v23-failure exploratory pilot. Never reuse or overwrite v23
outputs; an interrupted v24 arm also requires manual inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.preflight_appworld_v23 import EXPECTED, audit, free_gpu_mib, sha256  # noqa: E402
from experiments.run_appworld_v23_frozen_pair import (  # noqa: E402
    APPWORLD_ROOT, ARM_ORDER, ENDPOINT, MODEL_NAME, PRIVATE,
    read_server_stats, require_free_local_port, token_cost_delta,
    wait_for_own_server,
)
from experiments.summarize_appworld_v23_paired_pilot import (  # noqa: E402
    METHODS, MODEL_DIGEST, RUNNER_SHA256, TRAIN_SPLIT_SHA256, VIEW_SHA256,
    expected_target_hashes, paired_counts,
)


def result_path(arm: str) -> Path:
    if arm not in ARM_ORDER:
        raise ValueError("unknown v24 arm")
    return PRIVATE / f"appworld_v24_{arm}_seed2027_n9.json"


def stats_path(arm: str) -> Path:
    if arm not in ARM_ORDER:
        raise ValueError("unknown v24 arm")
    return PRIVATE / f"appworld_v24_{arm}_tokens_seed2027_n9.json"


def experiment_dir(arm: str) -> Path:
    if arm not in ARM_ORDER:
        raise ValueError("unknown v24 arm")
    return APPWORLD_ROOT / "experiments" / "outputs" / f"v24_{arm}_seed2027_n9"


def runner_argv(arm: str, python: Path) -> list[str]:
    if arm not in ARM_ORDER:
        raise ValueError("unknown v24 arm")
    command = [
        str(python), str(ROOT / "experiments" / "run_appworld_local_agent_pilot.py"),
        "--root", str(APPWORLD_ROOT), "--output", str(result_path(arm)),
        "--experiment-name", experiment_dir(arm).name,
        "--model", MODEL_NAME, "--endpoint", ENDPOINT,
        "--num-tasks", "9", "--max-steps", "20",
        "--context-tokens", "8192", "--predict-tokens", "512",
        "--prompt-variant", "checklist", "--history-strategy", "preserve_task",
        "--raise-on-failure", "--seed", "2027",
    ]
    if arm != "none":
        names = {
            "order": "appworld_train_static_order_seed2027_n9.json",
            "disjoint": "appworld_train_static_disjoint_edges_seed2027_n9.json",
            "dependency": "appworld_train_static_dependency_seed2027_n9.json",
        }
        command.extend(["--memory-view", str(PRIVATE / names[arm])])
    return command


def validated_outcomes(arm: str, payload: dict, targets: set[str]) -> dict[str, bool]:
    expected = {
        "status": "train_only_state_based_agent_pilot_not_corem_result",
        "benchmark": "AppWorld 0.1.3.post1", "split": "train",
        "task_sampling": "seeded_distinct_generators_first_variant",
        "model": MODEL_NAME, "model_endpoint": ENDPOINT, "model_digest": MODEL_DIGEST,
        "seed": 2027, "appworld_random_seed": 100, "max_steps": 20,
        "context_tokens": 8192, "predict_tokens": 512,
        "prompt_variant": "checklist", "history_strategy": "preserve_task",
        "raise_on_failure": True, "num_tasks": 9, "script_sha256": RUNNER_SHA256,
        "memory_condition": METHODS[arm], "memory_view_sha256": VIEW_SHA256[arm],
        "doc_condition": "none", "doc_view_sha256": None,
        "train_split_sha256": TRAIN_SPLIT_SHA256,
        "protected_text_exported": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{arm} differs from v24 settings")
    if any(payload.get(key) != 0 for key in
           ("evaluation_errors", "model_errors", "execution_errors")):
        raise ValueError(f"{arm} has technical errors")
    rows = payload.get("private_task_outcomes")
    if not isinstance(rows, list) or len(rows) != 9:
        raise ValueError(f"{arm} has incomplete outcomes")
    outcomes: dict[str, bool] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid task row")
        task_hash, success = row.get("task_sha256"), row.get("success")
        if (
            not isinstance(task_hash, str) or len(task_hash) != 64
            or any(c not in "0123456789abcdef" for c in task_hash)
            or type(success) is not bool or task_hash in outcomes
            or row.get("evaluation_error") or row.get("model_errors")
            or row.get("execution_errors")
        ):
            raise ValueError("invalid or technically failed task row")
        outcomes[task_hash] = success
    if set(outcomes) != targets or sum(outcomes.values()) != payload.get("successes"):
        raise ValueError("v24 target or success total mismatch")
    return outcomes


def completed_arms() -> set[str]:
    completed: set[str] = set()
    targets = expected_target_hashes()
    for arm in ARM_ORDER:
        result, receipt, directory = result_path(arm), stats_path(arm), experiment_dir(arm)
        if result.is_file():
            payload = json.loads(result.read_text())
            validated_outcomes(arm, payload, targets)
            if not directory.is_dir() or not receipt.is_file():
                raise ValueError(f"{arm} result is missing execution data or cost receipt")
            cost = json.loads(receipt.read_text())
            if (
                cost.get("status") != "train_only_v24_aggregate_token_cost"
                or cost.get("arm") != arm or cost.get("result_sha256") != sha256(result)
                or cost.get("server_sha256") != EXPECTED["server"][1]
                or cost.get("model_digest") != MODEL_DIGEST
                or cost.get("requests") != sum(r["steps"] for r in payload["private_task_outcomes"])
                or not isinstance(cost.get("prompt_tokens"), int)
                or cost["prompt_tokens"] <= 0
                or not isinstance(cost.get("completion_tokens"), int)
                or cost["completion_tokens"] <= 0
            ):
                raise ValueError(f"{arm} receipt does not match result")
            completed.add(arm)
        elif directory.exists() or receipt.exists():
            raise ValueError(f"{arm} has partial output; preserve and inspect manually")
    return completed


def write_summary() -> None:
    summary = PRIVATE / "appworld_v24_paired_seed2027_n9.json"
    if summary.exists():
        raise ValueError("refusing to overwrite v24 summary")
    targets = expected_target_hashes()
    payloads = {arm: json.loads(result_path(arm).read_text()) for arm in ARM_ORDER}
    outcomes = {arm: validated_outcomes(arm, payloads[arm], targets) for arm in ARM_ORDER}
    report = {
        "status": "train_only_post_failure_exploratory_paired_state_success_not_confirmation",
        "num_tasks": 9,
        "settings": {"context_tokens": 8192, "history_strategy": "preserve_task",
                     "max_steps": 20, "predict_tokens": 512, "seed": 2027},
        "successes": {arm: sum(outcomes[arm].values()) for arm in ARM_ORDER},
        "paired_dependency_vs": {
            arm: paired_counts(outcomes["dependency"], outcomes[arm])
            for arm in ("disjoint", "order", "none")
        },
        "totals": {
            arm: {field: sum(float(row[field]) for row in payloads[arm]["private_task_outcomes"])
                  for field in ("steps", "wall_seconds", "execution_failure_messages")}
            for arm in ARM_ORDER
        },
        "token_cost": {arm: json.loads(stats_path(arm).read_text()) for arm in ARM_ORDER},
        "input_sha256": {arm: sha256(result_path(arm)) for arm in ARM_ORDER},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task_ids_or_hashes_exported": False, "dev_or_test_task_loaded": False,
    }
    with summary.open("x") as output:
        json.dump(report, output, sort_keys=True, indent=2)
        output.write("\n")
    print(json.dumps({"successes": report["successes"],
                      "paired_dependency_vs": report["paired_dependency_vs"]}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    completed = completed_arms()
    if completed != set(ARM_ORDER):
        report = audit(args.model_dir.resolve(), free_gpu_mib())
        if report["status"] != "ready":
            print(json.dumps(report, sort_keys=True))
            return 1
        python_server = ROOT / ".venv-corem-qwen35" / "bin" / "python"
        python_agent = ROOT / ".conda-corem-appworld" / "bin" / "python"
        if not python_server.is_file() or not python_agent.is_file():
            raise ValueError("model or AppWorld environment missing")
        require_free_local_port()
        server = subprocess.Popen(
            [str(python_server), str(ROOT / "experiments" / "serve_local_qwen_agent.py"),
             "--model-dir", str(args.model_dir.resolve()), "--model-name", MODEL_NAME,
             "--disable-thinking", "--port", "11435", "--min-free-gib", "25",
             "--startup-headroom-gib", "9"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            wait_for_own_server(server)
            for arm in ARM_ORDER:
                if arm in completed:
                    continue
                print(f"running v24 train-only arm: {arm}", flush=True)
                before = read_server_stats()
                subprocess.run(runner_argv(arm, python_agent), cwd=ROOT, check=True)
                after = read_server_stats()
                payload = json.loads(result_path(arm).read_text())
                validated_outcomes(arm, payload, expected_target_hashes())
                steps = sum(r["steps"] for r in payload["private_task_outcomes"])
                cost = token_cost_delta(before, after, steps)
                receipt = {
                    "status": "train_only_v24_aggregate_token_cost", "arm": arm,
                    "result_sha256": sha256(result_path(arm)),
                    "server_sha256": EXPECTED["server"][1],
                    "model_digest": MODEL_DIGEST,
                    **cost, "task_ids_or_hashes_exported": False,
                }
                with stats_path(arm).open("x") as output:
                    json.dump(receipt, output, sort_keys=True)
                    output.write("\n")
                print(f"validated v24 arm: {arm}", flush=True)
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=15)
    if completed_arms() != set(ARM_ORDER):
        raise ValueError("all four arms must finish before paired summary")
    write_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
