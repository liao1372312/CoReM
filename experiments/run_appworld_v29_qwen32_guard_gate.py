#!/usr/bin/env python3
"""Train-only Qwen3-32B-AWQ diagnostic with the frozen nested API guard."""

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
from experiments.preflight_appworld_v23 import free_gpu_mib, sha256  # noqa: E402

GUARD = ROOT / "experiments" / "run_appworld_v26_nested_guard.py"
GUARD_SHA256 = "8d31c5aaa41baa2162534836d51aec4cb56a27db66d5f952e951d8d7dde7ae91"
V25 = ROOT / "experiments" / "run_appworld_v25_api_guard.py"
V25_SHA256 = "8b86a7866d82c9b960778efbc7c6682ee7b0f347dfce6026cc7c1e07c96cc6b3"
RESULT = v28.PRIVATE / "appworld_v29_qwen32_guard_seed2027_n9.json"
COST = v28.PRIVATE / "appworld_v29_qwen32_guard_tokens_seed2027_n9.json"
SERVER_LOG = v28.PRIVATE / "appworld_v29_qwen32_guard_server.log"
EXPERIMENT = "v29_qwen32_guard_seed2027_n9"
OUTPUT_DIR = v28.APPWORLD_ROOT / "experiments" / "outputs" / EXPERIMENT


def runner_argv() -> list[str]:
    command = v28.runner_argv()
    command[1] = str(GUARD)
    command[command.index("--output") + 1] = str(RESULT)
    command[command.index("--experiment-name") + 1] = EXPERIMENT
    return command


def validate_result(payload: dict[str, object]) -> tuple[int, int]:
    steps = v28.validate_result(payload)
    rows = payload["private_task_outcomes"]
    interventions = 0
    for row in rows:
        count = row.get("api_guard_interventions")
        if (
            type(count) is not int or count < 0
            or row.get("api_guard_policy")
            != "read_only_documentation_for_invalid_nested_api_paths"
            or row.get("api_guard_script_sha256") != GUARD_SHA256
        ):
            raise ValueError("v29 guard provenance or intervention count invalid")
        interventions += count
    return steps, interventions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    if any(path.exists() for path in (RESULT, COST, SERVER_LOG, OUTPUT_DIR)):
        raise ValueError("refusing to overwrite or resume partial v29 output")
    report = v28.audit(args.model_dir, free_gpu_mib())
    if sha256(GUARD) != GUARD_SHA256 or sha256(V25) != V25_SHA256:
        raise ValueError("nested guard source differs from frozen version")
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
            steps, interventions = validate_result(payload)
            counts = v28.token_cost_delta(before, after, steps)
            receipt = {
                "status": "train_only_v29_qwen32_nested_guard_aggregate_token_cost",
                "result_sha256": sha256(RESULT),
                "server_sha256": v28.SERVER_SHA256,
                "guard_sha256": GUARD_SHA256, "guard_parent_sha256": V25_SHA256,
                "model_digest": v28.MODEL_DIGEST,
                "model_revision": "0499c3ac83fdef8810b907a23894ba91e95eddd8",
                "thinking_enabled": False,
                "guard_interventions": interventions,
                **counts, "task_ids_or_hashes_exported": False,
            }
            with COST.open("x") as output:
                json.dump(receipt, output, sort_keys=True, indent=2)
                output.write("\n")
            print(json.dumps({
                "successes": payload["successes"],
                "completed_calls": payload["completed_calls"],
                "guard_interventions": interventions,
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
