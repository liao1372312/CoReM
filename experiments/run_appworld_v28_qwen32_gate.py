#!/usr/bin/env python3
"""Frozen, no-memory nine-task TRAIN feasibility gate for Qwen3-32B-AWQ."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.preflight_appworld_v23 import free_gpu_mib, sha256  # noqa: E402
from experiments.run_appworld_v23_frozen_pair import (  # noqa: E402
    APPWORLD_ROOT, PRIVATE, read_server_stats, require_free_local_port,
    token_cost_delta,
)
from experiments.summarize_appworld_v23_paired_pilot import (  # noqa: E402
    RUNNER_SHA256, TRAIN_SPLIT_SHA256, expected_target_hashes,
)

MODEL_NAME = "Qwen3-32B-AWQ"
MODEL_DIGEST = "7d0427c7d238f401b5d38ec08dca07cc8a921aa69d73769b2b7983520de504a1"
ENDPOINT = "http://127.0.0.1:11435/api/chat"
MANIFEST = ROOT / "experiments" / "QWEN3_32B_AWQ_SNAPSHOT_SHA256.txt"
MANIFEST_SHA256 = "668f91a4460eca0dc80acd3514a18722af6ba94c1f6ee7337d810b3cca3d8f85"
SERVER = ROOT / "experiments" / "serve_local_qwen3_32b_awq.py"
SERVER_SHA256 = "c64313db0e3197b1f152fbd83511ec6d9072c31f2439f4004fa14aa143a0a38d"
MODEL_DIR = PRIVATE / "models" / "Qwen3-32B-AWQ-0499c3a"
RESULT = PRIVATE / "appworld_v28_qwen32_none_seed2027_n9.json"
COST = PRIVATE / "appworld_v28_qwen32_none_tokens_seed2027_n9.json"
SERVER_LOG = PRIVATE / "appworld_v28_qwen32_server.log"
EXPERIMENT = "v28_qwen32_none_seed2027_n9"
OUTPUT_DIR = APPWORLD_ROOT / "experiments" / "outputs" / EXPERIMENT
PYTHON_SERVER = ROOT / ".venv-corem-qwen32-awq" / "bin" / "python"
PYTHON_AGENT = ROOT / ".conda-corem-appworld" / "bin" / "python"


def audit(model_dir: Path, gpu_free_mib: int) -> dict[str, object]:
    failures: list[str] = []
    if model_dir.resolve() != MODEL_DIR.resolve():
        failures.append("model directory differs from pinned local snapshot")
    if importlib.metadata.version("appworld") != "0.1.3.post1":
        failures.append("AppWorld version differs from frozen runner")
    if sha256(ROOT / "experiments" / "run_appworld_local_agent_pilot.py") != RUNNER_SHA256:
        failures.append("AppWorld runner differs from frozen source")
    if sha256(APPWORLD_ROOT / "data" / "datasets" / "train.txt") != TRAIN_SPLIT_SHA256:
        failures.append("train split differs from frozen source")
    if sha256(SERVER) != SERVER_SHA256:
        failures.append("AWQ server source differs from frozen gate")
    if sha256(MANIFEST) != MANIFEST_SHA256:
        failures.append("model checksum manifest differs from frozen gate")
    names: set[str] = set()
    for line in MANIFEST.read_text().splitlines():
        digest, separator, name = line.partition("  ")
        if (not separator or len(digest) != 64 or name in names
                or Path(name).name != name):
            failures.append("invalid model checksum manifest")
            break
        names.add(name)
        path = model_dir / name
        if not path.is_file() or sha256(path) != digest:
            failures.append(f"model file {name} differs from pinned checksum")
    if len(names) != 6:
        failures.append("incomplete model checksum manifest")
    if not PYTHON_SERVER.is_file() or not PYTHON_AGENT.is_file():
        failures.append("model or AppWorld environment is missing")
    if shutil.disk_usage(ROOT).free < 15 * 1024**3:
        failures.append("less than 15 GiB disk safety reserve")
    if gpu_free_mib < 44 * 1024:
        failures.append("less than 44 GiB pre-load GPU reserve")
    return {
        "status": "ready" if not failures else "not_ready",
        "scope": "train_only_v28_preflight_no_agent_score",
        "gpu_free_mib": gpu_free_mib,
        "model_files_checked": len(names),
        "failures": failures,
        "dev_or_test_loaded": False,
    }


def wait_for_server(process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("own AWQ server exited before readiness")
        try:
            with urllib.request.urlopen(ENDPOINT.rsplit("/", 1)[0] + "/tags", timeout=2) as response:
                tags = json.load(response)
            if tags.get("models") != [{"name": MODEL_NAME, "digest": MODEL_DIGEST}]:
                raise ValueError("AWQ server name or digest differs from pinned model")
            return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("own AWQ server did not become ready")


def runner_argv() -> list[str]:
    return [
        str(PYTHON_AGENT), str(ROOT / "experiments" / "run_appworld_local_agent_pilot.py"),
        "--root", str(APPWORLD_ROOT), "--output", str(RESULT),
        "--experiment-name", EXPERIMENT, "--model", MODEL_NAME,
        "--endpoint", ENDPOINT, "--num-tasks", "9", "--max-steps", "20",
        "--context-tokens", "8192", "--predict-tokens", "512",
        "--prompt-variant", "checklist", "--history-strategy", "preserve_task",
        "--raise-on-failure", "--seed", "2027",
    ]


def validate_result(payload: dict[str, object]) -> int:
    expected = {
        "status": "train_only_state_based_agent_pilot_not_corem_result",
        "benchmark": "AppWorld 0.1.3.post1", "split": "train",
        "task_sampling": "seeded_distinct_generators_first_variant",
        "model": MODEL_NAME, "model_endpoint": ENDPOINT, "model_digest": MODEL_DIGEST,
        "seed": 2027, "appworld_random_seed": 100, "max_steps": 20,
        "context_tokens": 8192, "predict_tokens": 512,
        "prompt_variant": "checklist", "history_strategy": "preserve_task",
        "raise_on_failure": True, "num_tasks": 9, "memory_condition": "none",
        "memory_view_sha256": None, "doc_condition": "none", "doc_view_sha256": None,
        "script_sha256": RUNNER_SHA256, "train_split_sha256": TRAIN_SPLIT_SHA256,
        "protected_text_exported": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("v28 gate differs from declared settings")
    if any(payload.get(key) != 0 for key in
           ("model_errors", "execution_errors", "evaluation_errors")):
        raise ValueError("v28 gate has technical errors")
    rows = payload.get("private_task_outcomes")
    if not isinstance(rows, list) or len(rows) != 9:
        raise ValueError("v28 gate has incomplete task outcomes")
    if {row.get("task_sha256") for row in rows} != expected_target_hashes():
        raise ValueError("v28 gate targets differ from frozen train set")
    for row in rows:
        if (type(row.get("success")) is not bool or row.get("model_errors")
                or row.get("execution_errors") or row.get("evaluation_error")):
            raise ValueError("v28 gate contains malformed or technically failed task")
    if sum(row["success"] for row in rows) != payload.get("successes"):
        raise ValueError("v28 gate success total mismatch")
    return sum(int(row["steps"]) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    if any(path.exists() for path in (RESULT, COST, SERVER_LOG, OUTPUT_DIR)):
        raise ValueError("refusing to overwrite or resume partial v28 output")
    report = audit(args.model_dir, free_gpu_mib())
    if report["status"] != "ready":
        print(json.dumps(report, sort_keys=True))
        return 1
    require_free_local_port()
    with SERVER_LOG.open("x") as server_log:
        server = subprocess.Popen(
            [str(PYTHON_SERVER), str(SERVER), "--model-dir", str(args.model_dir.resolve()),
             "--model-name", MODEL_NAME, "--port", "11435",
             "--min-free-gib", "25", "--startup-headroom-gib", "19"],
            cwd=ROOT, stdout=server_log, stderr=server_log,
        )
        try:
            wait_for_server(server)
            before = read_server_stats()
            subprocess.run(runner_argv(), cwd=ROOT, check=True)
            after = read_server_stats()
            payload = json.loads(RESULT.read_text())
            steps = validate_result(payload)
            counts = token_cost_delta(before, after, steps)
            receipt = {
                "status": "train_only_v28_qwen32_no_memory_gate_aggregate_token_cost",
                "result_sha256": sha256(RESULT), "server_sha256": SERVER_SHA256,
                "model_digest": MODEL_DIGEST, "model_revision":
                "0499c3ac83fdef8810b907a23894ba91e95eddd8",
                "thinking_enabled": False, **counts,
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
