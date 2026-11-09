#!/usr/bin/env python3
"""Fresh-generator train stress test of the frozen v30 no-memory executor."""

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

from experiments import run_appworld_v28_qwen32_gate as v28  # noqa: E402
from experiments import run_appworld_v29_qwen32_guard_gate as v29  # noqa: E402
from experiments import run_appworld_v30_qwen32_full_docs_gate as v30  # noqa: E402
from experiments.preflight_appworld_v23 import free_gpu_mib, sha256  # noqa: E402
from experiments.summarize_appworld_v23_paired_pilot import (  # noqa: E402
    RUNNER_SHA256, TRAIN_SPLIT_SHA256, expected_target_hashes,
)

SEED = 2033
RESULT = v28.PRIVATE / "appworld_v31_fresh_train_seed2033_n9.json"
COST = v28.PRIVATE / "appworld_v31_fresh_train_tokens_seed2033_n9.json"
SERVER_LOG = v28.PRIVATE / "appworld_v31_fresh_train_server.log"
EXPERIMENT = "v31_fresh_train_seed2033_n9"
OUTPUT_DIR = v28.APPWORLD_ROOT / "experiments" / "outputs" / EXPERIMENT


def fresh_target_hashes() -> set[str]:
    from experiments.run_appworld_local_agent_pilot import _sample_task_ids

    path = v28.APPWORLD_ROOT / "data" / "datasets" / "train.txt"
    if sha256(path) != TRAIN_SPLIT_SHA256:
        raise ValueError("train split changed before fresh-generator selection")
    selected = _sample_task_ids(path.read_text().splitlines(), SEED, 9)
    hashes = {hashlib.sha256(task_id.encode()).hexdigest() for task_id in selected}
    if len(hashes) != 9 or hashes & expected_target_hashes():
        raise ValueError("fresh train targets overlap seed-2027 targets")
    return hashes


def runner_argv() -> list[str]:
    command = v30.runner_argv()
    command[command.index("--output") + 1] = str(RESULT)
    command[command.index("--experiment-name") + 1] = EXPERIMENT
    command[command.index("--seed") + 1] = str(SEED)
    return command


def validate_result(payload: dict[str, object]) -> tuple[int, int, int]:
    expected = {
        "status": "train_only_state_based_agent_pilot_not_corem_result",
        "benchmark": "AppWorld 0.1.3.post1", "split": "train",
        "task_sampling": "seeded_distinct_generators_first_variant",
        "model": v28.MODEL_NAME, "model_endpoint": v28.ENDPOINT,
        "model_digest": v28.MODEL_DIGEST, "seed": SEED,
        "appworld_random_seed": 100, "max_steps": 20,
        "context_tokens": 8192, "predict_tokens": 512,
        "prompt_variant": "checklist", "history_strategy": "preserve_task",
        "raise_on_failure": True, "num_tasks": 9,
        "memory_condition": "none", "memory_view_sha256": None,
        "doc_condition": "none", "doc_view_sha256": None,
        "script_sha256": RUNNER_SHA256,
        "train_split_sha256": TRAIN_SPLIT_SHA256,
        "protected_text_exported": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("fresh-generator gate differs from frozen v30 settings")
    if any(payload.get(key) != 0 for key in
           ("model_errors", "execution_errors", "evaluation_errors")):
        raise ValueError("fresh-generator gate has technical errors")
    rows = payload.get("private_task_outcomes")
    if not isinstance(rows, list) or len(rows) != 9:
        raise ValueError("fresh-generator gate has incomplete task rows")
    if {row.get("task_sha256") for row in rows} != fresh_target_hashes():
        raise ValueError("fresh-generator gate used the wrong train targets")
    interventions = 0
    full_docs = 0
    for row in rows:
        guard_count = row.get("api_guard_interventions")
        doc_count = row.get("full_doc_observations")
        max_chars = row.get("max_doc_observation_chars")
        if (
            type(row.get("success")) is not bool
            or row.get("model_errors") or row.get("execution_errors")
            or row.get("evaluation_error")
            or type(guard_count) is not int or guard_count < 0
            or row.get("api_guard_policy")
            != "read_only_documentation_for_invalid_nested_api_paths"
            or row.get("api_guard_script_sha256") != v29.GUARD_SHA256
            or row.get("observation_policy")
            != "latest_complete_api_doc_older_replies_2000_prefix"
            or row.get("observation_script_sha256") != v30.AGENT_SHA256
            or type(doc_count) is not int or not 0 <= doc_count <= row["steps"]
            or type(max_chars) is not int or not 0 <= max_chars <= 12_000
        ):
            raise ValueError("fresh-generator task provenance or counts invalid")
        interventions += guard_count
        full_docs += doc_count
    if sum(row["success"] for row in rows) != payload.get("successes"):
        raise ValueError("fresh-generator success total mismatch")
    return sum(int(row["steps"]) for row in rows), interventions, full_docs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    if any(path.exists() for path in (RESULT, COST, SERVER_LOG, OUTPUT_DIR)):
        raise ValueError("refusing to overwrite or resume partial v31 output")
    report = v28.audit(args.model_dir, free_gpu_mib())
    if (sha256(v30.AGENT) != v30.AGENT_SHA256
            or sha256(v29.GUARD) != v29.GUARD_SHA256
            or sha256(v29.V25) != v29.V25_SHA256):
        raise ValueError("fresh-generator agent or guard source changed")
    fresh_target_hashes()
    if report["status"] != "ready":
        print(json.dumps(report, sort_keys=True))
        return 1
    v28.require_free_local_port()
    with SERVER_LOG.open("x") as server_log:
        server = subprocess.Popen(
            [str(v28.PYTHON_SERVER), str(v28.SERVER), "--model-dir",
             str(args.model_dir.resolve()), "--model-name", v28.MODEL_NAME,
             "--port", "11435", "--min-free-gib", "25",
             "--startup-headroom-gib", "19"],
            cwd=ROOT, stdout=server_log, stderr=server_log,
        )
        try:
            v28.wait_for_server(server)
            before = v28.read_server_stats()
            subprocess.run(runner_argv(), cwd=ROOT, check=True)
            after = v28.read_server_stats()
            payload = json.loads(RESULT.read_text())
            steps, interventions, full_docs = validate_result(payload)
            counts = v28.token_cost_delta(before, after, steps)
            receipt = {
                "status": "train_only_v31_fresh_generator_aggregate_token_cost",
                "result_sha256": sha256(RESULT),
                "server_sha256": v28.SERVER_SHA256,
                "guard_sha256": v29.GUARD_SHA256,
                "agent_sha256": v30.AGENT_SHA256,
                "model_digest": v28.MODEL_DIGEST,
                "model_revision": "0499c3ac83fdef8810b907a23894ba91e95eddd8",
                "seed": SEED, "thinking_enabled": False,
                "guard_interventions": interventions,
                "full_doc_observations": full_docs,
                **counts, "task_ids_or_hashes_exported": False,
            }
            with COST.open("x") as output:
                json.dump(receipt, output, sort_keys=True, indent=2)
                output.write("\n")
            print(json.dumps({
                "successes": payload["successes"],
                "completed_calls": payload["completed_calls"],
                "guard_interventions": interventions,
                "full_doc_observations": full_docs,
                "requests": counts["requests"],
            }, sort_keys=True))
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
