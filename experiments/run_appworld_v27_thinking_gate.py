#!/usr/bin/env python3
"""Fresh nine-task TRAIN gate for Qwen3.5-9B thinking-mode composer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.preflight_appworld_v23 import EXPECTED, audit, free_gpu_mib, sha256  # noqa: E402
from experiments.run_appworld_v23_frozen_pair import (  # noqa: E402
    APPWORLD_ROOT, ENDPOINT, MODEL_NAME, PRIVATE, read_server_stats,
    require_free_local_port, token_cost_delta, wait_for_own_server,
)
from experiments.summarize_appworld_v23_paired_pilot import (  # noqa: E402
    MODEL_DIGEST, RUNNER_SHA256, TRAIN_SPLIT_SHA256, expected_target_hashes,
)

AGENT = ROOT / "experiments" / "run_appworld_v27_thinking_agent.py"
RESULT = PRIVATE / "appworld_v27_thinking_none_seed2027_n9.json"
COST = PRIVATE / "appworld_v27_thinking_none_tokens_seed2027_n9.json"
EXPERIMENT = "v27_thinking_none_seed2027_n9"
OUTPUT_DIR = APPWORLD_ROOT / "experiments" / "outputs" / EXPERIMENT


def runner_argv(python: Path) -> list[str]:
    return [
        str(python), str(AGENT), "--root", str(APPWORLD_ROOT),
        "--output", str(RESULT), "--experiment-name", EXPERIMENT,
        "--model", MODEL_NAME, "--endpoint", ENDPOINT,
        "--num-tasks", "9", "--max-steps", "20",
        "--context-tokens", "8192", "--predict-tokens", "1024",
        "--prompt-variant", "checklist", "--history-strategy", "preserve_task",
        "--raise-on-failure", "--seed", "2027",
    ]


def validate_result(payload: dict) -> int:
    expected = {
        "status": "train_only_state_based_agent_pilot_not_corem_result",
        "benchmark": "AppWorld 0.1.3.post1", "split": "train",
        "task_sampling": "seeded_distinct_generators_first_variant",
        "model": MODEL_NAME, "model_endpoint": ENDPOINT, "model_digest": MODEL_DIGEST,
        "seed": 2027, "appworld_random_seed": 100, "max_steps": 20,
        "context_tokens": 8192, "predict_tokens": 1024,
        "prompt_variant": "checklist", "history_strategy": "preserve_task",
        "raise_on_failure": True, "num_tasks": 9,
        "memory_condition": "none", "memory_view_sha256": None,
        "doc_condition": "none", "doc_view_sha256": None,
        "script_sha256": RUNNER_SHA256, "train_split_sha256": TRAIN_SPLIT_SHA256,
        "protected_text_exported": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("thinking gate differs from declared settings")
    if any(payload.get(key) != 0 for key in
           ("model_errors", "execution_errors", "evaluation_errors")):
        raise ValueError("thinking gate has technical errors or unfinished reasoning")
    rows = payload.get("private_task_outcomes")
    if not isinstance(rows, list) or len(rows) != 9:
        raise ValueError("thinking gate has incomplete outcomes")
    if {row.get("task_sha256") for row in rows} != expected_target_hashes():
        raise ValueError("thinking gate targets differ from frozen train set")
    for row in rows:
        if (
            type(row.get("success")) is not bool
            or row.get("reasoning_policy") != "execute_final_code_only_require_closed_think_tag"
            or row.get("reasoning_script_sha256") != sha256(AGENT)
            or row.get("model_errors") or row.get("execution_errors")
            or row.get("evaluation_error")
        ):
            raise ValueError("thinking gate has a malformed task row")
    if sum(row["success"] for row in rows) != payload.get("successes"):
        raise ValueError("thinking gate success total mismatch")
    return sum(row["steps"] for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    if RESULT.exists() or COST.exists() or OUTPUT_DIR.exists():
        raise ValueError("refusing to overwrite or resume a partial thinking gate")
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
         "--port", "11435", "--min-free-gib", "25",
         "--startup-headroom-gib", "9"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_own_server(server)
        before = read_server_stats()
        subprocess.run(runner_argv(python_agent), cwd=ROOT, check=True)
        after = read_server_stats()
        payload = json.loads(RESULT.read_text())
        steps = validate_result(payload)
        counts = token_cost_delta(before, after, steps)
        receipt = {
            "status": "train_only_v27_thinking_gate_aggregate_token_cost",
            "result_sha256": sha256(RESULT), "agent_sha256": sha256(AGENT),
            "server_sha256": EXPECTED["server"][1],
            "model_digest": MODEL_DIGEST,
            "thinking_enabled": True,
            **counts,
            "task_ids_or_hashes_exported": False,
        }
        with COST.open("x") as output:
            json.dump(receipt, output, sort_keys=True, indent=2)
            output.write("\n")
        print(json.dumps({"successes": payload["successes"],
                          "completed_calls": payload["completed_calls"],
                          "requests": counts["requests"]}, sort_keys=True))
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
