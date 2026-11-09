#!/usr/bin/env python3
"""Train-only authentication-contract diagnostic on the v31 task group."""

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
from experiments import run_appworld_v33_auth_contract_agent as agent  # noqa: E402
from experiments.preflight_appworld_v23 import free_gpu_mib, sha256  # noqa: E402

AGENT = ROOT / "experiments" / "run_appworld_v33_auth_contract_agent.py"
AGENT_SHA256 = "611ba681e5a090abc06fac69342713c9698b3b43f8717ab951dbc13fc68c8f3d"
RESULT = v28.PRIVATE / "appworld_v33_auth_contract_seed2033_n9.json"
COST = v28.PRIVATE / "appworld_v33_auth_contract_tokens_seed2033_n9.json"
SERVER_LOG = v28.PRIVATE / "appworld_v33_auth_contract_server.log"
EXPERIMENT = "v33_auth_contract_seed2033_n9"
OUTPUT_DIR = v28.APPWORLD_ROOT / "experiments" / "outputs" / EXPERIMENT


def runner_argv() -> list[str]:
    command = v32.runner_argv()
    command[1] = str(AGENT)
    command[command.index("--output") + 1] = str(RESULT)
    command[command.index("--experiment-name") + 1] = EXPERIMENT
    return command


def validate_result(payload: dict[str, object]) -> tuple[int, int, int]:
    steps, interventions, full_docs = v32.validate_result(payload)
    for row in payload["private_task_outcomes"]:
        if (
            row.get("auth_prompt_policy") != agent.PROMPT_POLICY
            or row.get("auth_prompt_script_sha256") != AGENT_SHA256
        ):
            raise ValueError("v33 authentication prompt or provenance invalid")
    return steps, interventions, full_docs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    if any(path.exists() for path in (RESULT, COST, SERVER_LOG, OUTPUT_DIR)):
        raise ValueError("refusing to overwrite or resume partial v33 output")
    report = v28.audit(args.model_dir, free_gpu_mib())
    if (
        sha256(AGENT) != AGENT_SHA256
        or sha256(v32.AGENT) != v32.AGENT_SHA256
        or sha256(v30.AGENT) != v30.AGENT_SHA256
        or sha256(v29.GUARD) != v29.GUARD_SHA256
        or sha256(v29.V25) != v29.V25_SHA256
    ):
        raise ValueError("v33 agent or inherited source changed")
    v31.fresh_target_hashes()
    if report["status"] != "ready":
        print(json.dumps(report, sort_keys=True))
        return 1
    v28.require_free_local_port()
    with SERVER_LOG.open("x") as server_log:
        server = subprocess.Popen(
            [
                str(v28.PYTHON_SERVER), str(v28.SERVER), "--model-dir",
                str(args.model_dir.resolve()), "--model-name", v28.MODEL_NAME,
                "--port", "11435", "--min-free-gib", "25",
                "--startup-headroom-gib", "19",
            ],
            cwd=ROOT,
            stdout=server_log,
            stderr=server_log,
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
                "status": "train_only_v33_auth_contract_aggregate_token_cost",
                "result_sha256": sha256(RESULT),
                "server_sha256": v28.SERVER_SHA256,
                "guard_sha256": v29.GUARD_SHA256,
                "observation_agent_sha256": v30.AGENT_SHA256,
                "context_agent_sha256": v32.AGENT_SHA256,
                "agent_sha256": AGENT_SHA256,
                "model_digest": v28.MODEL_DIGEST,
                "model_revision": "0499c3ac83fdef8810b907a23894ba91e95eddd8",
                "seed": v31.SEED,
                "thinking_enabled": False,
                "guard_interventions": interventions,
                "full_doc_observations": full_docs,
                **counts,
                "task_ids_or_hashes_exported": False,
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
