#!/usr/bin/env python3
"""Run or safely resume the frozen four-arm AppWorld v23 TRAIN pilot.

No dev/test task is loaded. Existing results are validated, never overwritten;
partial experiment directories require manual diagnosis before a fresh run.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private_corem"
APPWORLD_ROOT = PRIVATE / "appworld"
ENDPOINT = "http://127.0.0.1:11435/api/chat"
MODEL_NAME = "Qwen3.5-9B-4bit"
ARM_ORDER = ("none", "order", "disjoint", "dependency")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.preflight_appworld_v23 import EXPECTED, audit, free_gpu_mib, sha256  # noqa: E402
from experiments.summarize_appworld_v23_paired_pilot import (  # noqa: E402
    MODEL_DIGEST,
    expected_target_hashes,
    validated_outcomes,
)


def result_path(arm: str) -> Path:
    if arm not in ARM_ORDER:
        raise ValueError("unknown v23 arm")
    return PRIVATE / f"appworld_v23_{arm}_seed2027_n9.json"


def stats_path(arm: str) -> Path:
    if arm not in ARM_ORDER:
        raise ValueError("unknown v23 arm")
    return PRIVATE / f"appworld_v23_{arm}_tokens_seed2027_n9.json"


def experiment_dir(arm: str) -> Path:
    if arm not in ARM_ORDER:
        raise ValueError("unknown v23 arm")
    return APPWORLD_ROOT / "experiments" / "outputs" / f"v23_{arm}_seed2027_n9"


def runner_argv(arm: str, python: Path) -> list[str]:
    """Return identical composer settings, changing only the frozen view."""
    if arm not in ARM_ORDER:
        raise ValueError("unknown v23 arm")
    command = [
        str(python), str(ROOT / "experiments" / "run_appworld_local_agent_pilot.py"),
        "--root", str(APPWORLD_ROOT),
        "--output", str(result_path(arm)),
        "--experiment-name", experiment_dir(arm).name,
        "--model", MODEL_NAME,
        "--endpoint", ENDPOINT,
        "--num-tasks", "9",
        "--max-steps", "20",
        "--context-tokens", "6144",
        "--predict-tokens", "512",
        "--prompt-variant", "checklist",
        "--history-strategy", "preserve_code",
        "--raise-on-failure",
        "--seed", "2027",
    ]
    if arm != "none":
        filenames = {
            "order": "appworld_train_static_order_seed2027_n9.json",
            "disjoint": "appworld_train_static_disjoint_edges_seed2027_n9.json",
            "dependency": "appworld_train_static_dependency_seed2027_n9.json",
        }
        command.extend(["--memory-view", str(PRIVATE / filenames[arm])])
    return command


def existing_arms() -> set[str]:
    completed: set[str] = set()
    target_hashes = expected_target_hashes()
    for arm in ARM_ORDER:
        result = result_path(arm)
        directory = experiment_dir(arm)
        if result.exists():
            payload_result = json.loads(result.read_text())
            validated_outcomes(payload_result, arm, target_hashes)
            if not directory.is_dir():
                raise ValueError(f"{arm} result lacks its AppWorld execution directory")
            receipt = stats_path(arm)
            if not receipt.is_file():
                raise ValueError(f"{arm} result lacks a token-cost receipt; inspect manually")
            payload = json.loads(receipt.read_text())
            if (
                payload.get("status") != "train_only_v23_aggregate_token_cost"
                or payload.get("arm") != arm
                or payload.get("result_sha256") != sha256(result)
                or payload.get("server_sha256") != EXPECTED["server"][1]
                or payload.get("model_digest") != MODEL_DIGEST
                or not isinstance(payload.get("prompt_tokens"), int)
                or payload["prompt_tokens"] <= 0
                or not isinstance(payload.get("completion_tokens"), int)
                or payload["completion_tokens"] <= 0
                or not isinstance(payload.get("requests"), int)
                or payload["requests"] <= 0
                or payload["requests"] != sum(
                    int(row["steps"]) for row in payload_result["private_task_outcomes"]
                )
            ):
                raise ValueError(f"{arm} token-cost receipt differs from result or server")
            completed.add(arm)
        elif directory.exists():
            raise ValueError(f"{arm} has partial AppWorld output without a result; inspect it manually")
        elif stats_path(arm).exists():
            raise ValueError(f"{arm} token-cost receipt exists without a result")
    return completed


def read_server_stats() -> dict[str, int]:
    with urllib.request.urlopen("http://127.0.0.1:11435/api/stats", timeout=5) as response:
        payload = json.load(response)
    fields = (
        "requests", "accepted_requests", "truncated_requests",
        "rejected_oversize_requests", "max_prompt_tokens",
        "prompt_tokens_sum", "completion_tokens_sum",
    )
    if any(type(payload.get(field)) is not int or payload[field] < 0 for field in fields):
        raise ValueError("model server returned invalid aggregate token statistics")
    return {field: payload[field] for field in fields}


def token_cost_delta(before: dict[str, int], after: dict[str, int], steps: int) -> dict[str, int]:
    fields = (
        "requests", "accepted_requests", "truncated_requests",
        "rejected_oversize_requests", "prompt_tokens_sum", "completion_tokens_sum",
    )
    delta = {field: after[field] - before[field] for field in fields}
    if (
        any(value < 0 for value in delta.values())
        or delta["requests"] != steps
        or delta["accepted_requests"] != steps
        or delta["truncated_requests"] != 0
        or delta["rejected_oversize_requests"] != 0
        or delta["prompt_tokens_sum"] <= 0
        or delta["completion_tokens_sum"] <= 0
        or after["max_prompt_tokens"] < before["max_prompt_tokens"]
    ):
        raise ValueError("server token counts do not match clean agent requests")
    return {
        "requests": steps,
        "prompt_tokens": delta["prompt_tokens_sum"],
        "completion_tokens": delta["completion_tokens_sum"],
        "server_cumulative_max_prompt_tokens": after["max_prompt_tokens"],
    }


def wait_for_own_server(process: subprocess.Popen[bytes], timeout_seconds: int = 240) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("own model server exited before readiness")
        try:
            with urllib.request.urlopen("http://127.0.0.1:11435/api/tags", timeout=2) as response:
                payload = json.load(response)
            models = payload.get("models", [])
            if models != [{"name": MODEL_NAME, "digest": MODEL_DIGEST}]:
                raise RuntimeError("local model service has unexpected name or digest")
            return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("own model server did not become ready")


def require_free_local_port() -> None:
    """Never attach the pilot to a pre-existing service on the fixed port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", 11435))
        except OSError as error:
            raise RuntimeError("127.0.0.1:11435 is already in use") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        parser.error("model snapshot directory is missing")
    completed = existing_arms()
    summary = PRIVATE / "appworld_v23_paired_seed2027_n9.json"
    if summary.exists():
        if completed != set(ARM_ORDER):
            parser.error("summary exists without four validated arm results")
        print("frozen v23 summary already exists; refusing to overwrite")
        return 0

    if completed != set(ARM_ORDER):
        report = audit(model_dir, free_gpu_mib())
        if report["status"] != "ready":
            print(json.dumps(report, sort_keys=True))
            return 1
        python_server = ROOT / ".venv-corem-qwen35" / "bin" / "python"
        python_agent = ROOT / ".conda-corem-appworld" / "bin" / "python"
        if not python_server.is_file() or not python_agent.is_file():
            raise ValueError("frozen server or AppWorld environment is missing")
        require_free_local_port()
        server = subprocess.Popen(
            [str(python_server), str(ROOT / "experiments" / "serve_local_qwen_agent.py"),
             "--model-dir", str(model_dir), "--model-name", MODEL_NAME,
             "--disable-thinking", "--port", "11435",
             "--min-free-gib", "25", "--startup-headroom-gib", "9"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            wait_for_own_server(server)
            for arm in ARM_ORDER:
                if arm in completed:
                    continue
                print(f"running frozen train-only arm: {arm}", flush=True)
                before = read_server_stats()
                subprocess.run(runner_argv(arm, python_agent), cwd=ROOT, check=True)
                after = read_server_stats()
                payload = json.loads(result_path(arm).read_text())
                outcomes = validated_outcomes(payload, arm, expected_target_hashes())
                task_rows = payload["private_task_outcomes"]
                cost = token_cost_delta(
                    before, after, sum(int(row["steps"]) for row in task_rows)
                )
                if len(outcomes) != 9:
                    raise ValueError("v23 result has incomplete paired tasks")
                receipt = {
                    "status": "train_only_v23_aggregate_token_cost",
                    "arm": arm,
                    "result_sha256": sha256(result_path(arm)),
                    "server_sha256": EXPECTED["server"][1],
                    "model_digest": MODEL_DIGEST,
                    **cost,
                    "task_ids_or_hashes_exported": False,
                }
                with stats_path(arm).open("x") as output:
                    json.dump(receipt, output, sort_keys=True)
                    output.write("\n")
                print(f"validated frozen train-only arm: {arm}", flush=True)
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=15)

    if existing_arms() != set(ARM_ORDER):
        raise ValueError("all four arms must be complete before summary")
    command = [str(ROOT / ".conda-corem-appworld" / "bin" / "python"),
               str(ROOT / "experiments" / "summarize_appworld_v23_paired_pilot.py")]
    for arm in ARM_ORDER:
        command.extend([f"--{arm}-result", str(result_path(arm))])
    command.extend(["--output", str(summary)])
    subprocess.run(command, cwd=ROOT, check=True)
    print("v23 train-only summary created; no independent confirmation implied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
