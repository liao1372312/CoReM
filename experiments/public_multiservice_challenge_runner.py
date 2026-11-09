"""Four-service runner for the frozen public v5 challenge."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.public_challenge_runner import PublicChallengeRunner
from experiments.public_workflows_runner import (
    PLACEHOLDER,
    REPOSITORY_ROOT,
    _capture_value,
    _free_port,
    _sha256,
)

RUNTIME_ROOT = REPOSITORY_ROOT / "experiments" / "real_tracks" / "runtime"
WIREMOCK_JARS = {
    "3.9.1": (
        RUNTIME_ROOT / "wiremock-3.9.1" / "wiremock-standalone-3.9.1.jar",
        "723a880d50d3b0a145af0df07e578c2cb85e77feb2231e6991c9a1366926912c",
    ),
    "3.13.2": (
        RUNTIME_ROOT / "wiremock-3.13.2" / "wiremock-standalone-3.13.2.jar",
        "d097b19bd483c5038479b13a5c71e9faf8f2f5106584f0c120a7770ab0bdb367",
    ),
}
MOCKSERVER_JARS = {
    "5.14.0": (
        RUNTIME_ROOT / "mockserver-5.14.0" / "mockserver-netty-5.14.0.jar",
        "2a11b20118fc2904a9408f6bcb98b63ed1f26e6db84447cbb99a6f108c2246e6",
    ),
    "5.15.0": (
        RUNTIME_ROOT / "mockserver-5.15.0" / "mockserver-netty-5.15.0.jar",
        "bf5cf50cb5bf8ef99ea372e78b5ceddcc6eb6a051f3b5b1d3e62c7b812c92aeb",
    ),
}


def _runtime(version: str, table: Mapping[str, tuple[Path, str]]) -> tuple[Path, str]:
    item = table.get(version)
    if item is None:
        raise RuntimeError(f"unsupported runtime version {version}")
    path, expected = item
    if not path.is_file():
        raise RuntimeError(f"missing pinned runtime: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise RuntimeError(f"runtime digest mismatch for {path}")
    return path, observed


def _render_json(value: Any, bindings: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        required = set(PLACEHOLDER.findall(value))
        if required.difference(bindings):
            raise ValueError("request JSON requires an unavailable binding")
        match = PLACEHOLDER.fullmatch(value)
        if match is not None:
            return bindings[match.group(1)]
        return PLACEHOLDER.sub(
            lambda match: str(bindings[match.group(1)]),
            value,
        )
    if isinstance(value, list):
        return [_render_json(item, bindings) for item in value]
    if isinstance(value, dict):
        return {key: _render_json(item, bindings) for key, item in value.items()}
    return value


class PublicMultiserviceChallengeRunner(PublicChallengeRunner):
    def __init__(self) -> None:
        super().__init__()
        self.current_service = ""

    def _wait_endpoint(self, path: str) -> bool:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    self.base_url + path, timeout=1
                ) as response:
                    if response.status < 500:
                        return True
            except urllib.error.HTTPError as error:
                if error.code < 500:
                    return True
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(0.1)
        return False

    def _start_wiremock(self, version: str, port: int) -> str:
        jar, observed = _runtime(version, WIREMOCK_JARS)
        self.tempdir = Path(tempfile.mkdtemp(prefix="corem-wiremock-v5-"))
        self.process = subprocess.Popen(
            [
                "java",
                "-jar",
                str(jar),
                "--port",
                str(port),
                "--root-dir",
                str(self.tempdir),
                "--disable-banner",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return f"maven:{version}:sha256:{observed}"

    def _start_mockserver(self, version: str, port: int) -> str:
        jar, observed = _runtime(version, MOCKSERVER_JARS)
        self.process = subprocess.Popen(
            [
                "java",
                "-jar",
                str(jar),
                "-serverPort",
                str(port),
                "-logLevel",
                "WARN",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return f"maven:{version}:sha256:{observed}"

    def reset(self, event: Mapping[str, Any]) -> Mapping[str, Any]:
        self.current_service = str(event["service_id"])
        if self.current_service not in {"wiremock", "mockserver"}:
            return super().reset(event)
        self._cleanup()
        version = str(event["version_id"])
        port = _free_port()
        self.base_url = f"http://127.0.0.1:{port}"
        if self.current_service == "wiremock":
            runtime = self._start_wiremock(version, port)
            health_path = "/__admin/health"
        else:
            runtime = self._start_mockserver(version, port)
            health_path = "/mockserver/status"
        if not self._wait_endpoint(health_path):
            self._cleanup()
            raise RuntimeError(
                f"service did not become ready: {self.current_service} {version}"
            )
        self.snapshot_id = (
            f"{self.current_service}:{version}:"
            f"{hashlib.sha256(runtime.encode()).hexdigest()}"
        )
        return {
            "reset_verified": True,
            "snapshot_id": self.snapshot_id,
            "runtime_artifact": runtime,
            "healthcheck": self.base_url + health_path,
        }

    @staticmethod
    def _challenge_plans(
        memories: Sequence[Mapping[str, Any]], operation_count: int
    ) -> list[tuple[int, str, dict[str, Any]]]:
        plans: dict[int, tuple[str, dict[str, Any]]] = {}
        allowed = set(range(1, operation_count + 1))
        for memory in sorted(memories, key=lambda row: str(row.get("memory_id", ""))):
            plan = memory.get("plan")
            step_index = memory.get("step_index")
            if (
                memory.get("role") != "workflow_step"
                or not isinstance(plan, dict)
                or step_index not in allowed
            ):
                continue
            method = plan.get("method")
            path = plan.get("path")
            if method not in {"GET", "POST", "PATCH", "PUT"} or not isinstance(
                path, str
            ):
                continue
            if not path.startswith("/") or "//" in path or "\\" in path:
                continue
            if set(plan).difference({"method", "path", "json", "capture"}):
                continue
            plans.setdefault(
                int(step_index), (str(memory.get("memory_id")), dict(plan))
            )
        return [
            (index, memory_id, plan)
            for index, (memory_id, plan) in sorted(plans.items())
        ]

    def execute(
        self,
        event: Mapping[str, Any],
        memories: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        try:
            return self._execute_selected(event, memories)
        except ValueError as error:
            if "requires an unavailable binding" not in str(error):
                raise
            return {
                "completed": False,
                "failure": "selected workflow has unsatisfied runtime bindings",
                "selected_memory_observations": self._memory_observations(memories),
                "steps": [],
                "bindings": {},
                "communication": {
                    "explicit_statuses": False,
                    "outcome_summary": "Workflow was not executed.",
                },
            }

    def _execute_selected(
        self,
        event: Mapping[str, Any],
        memories: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        observations = self._memory_observations(memories)
        operation_count = int(event.get("operation_count", 0))
        plans = self._challenge_plans(memories, operation_count)
        if operation_count < 3 or [item[0] for item in plans] != list(
            range(1, operation_count + 1)
        ):
            return {
                "completed": False,
                "failure": "selected memories do not supply every workflow step",
                "selected_memory_observations": observations,
                "steps": [],
                "bindings": {},
                "communication": {
                    "explicit_statuses": False,
                    "outcome_summary": "Workflow was not executed.",
                },
            }

        bindings: dict[str, Any] = {}
        steps: list[dict[str, Any]] = []
        for step_index, memory_id, plan in plans:
            path = self._render_path(str(plan["path"]), bindings)
            rendered_plan = dict(plan)
            if "json" in plan:
                rendered_plan["json"] = _render_json(plan["json"], bindings)
            response = self._request(rendered_plan, path)
            step = {
                "step_index": step_index,
                "plan_memory_id": memory_id,
                "request": {
                    "method": plan["method"],
                    "path_template": plan["path"],
                    "path": path,
                    "json": rendered_plan.get("json"),
                },
                "response": response,
            }
            capture = plan.get("capture")
            if capture is not None:
                if not isinstance(capture, dict) or not isinstance(
                    capture.get("name"), str
                ):
                    raise ValueError("capture must declare a scalar name")
                bindings[str(capture["name"])] = _capture_value(response, capture)
                step["captured_binding"] = str(capture["name"])
            steps.append(step)
        statuses = [int(step["response"]["status"]) for step in steps]
        outcome_summary = (
            "Executed three-step workflow with HTTP statuses "
            f"{statuses[0]}, {statuses[1]}, and {statuses[2]}."
            if len(statuses) == 3
            else (
                f"Executed {len(statuses)}-step workflow with HTTP statuses "
                + ", ".join(map(str, statuses))
                + "."
            )
        )
        return {
            "completed": True,
            "steps": steps,
            "bindings": bindings,
            "selected_memory_observations": observations,
            "communication": {
                "explicit_statuses": True,
                "outcome_summary": outcome_summary,
            },
        }

    def snapshot(self, event: Mapping[str, Any]) -> str:
        after = self.snapshot_id
        if self.current_service == "json-server" and self.tempdir is not None:
            database = self.tempdir / "db.json"
            after = f"{after}:state:{_sha256(database)}"
        self._cleanup()
        return after


def create_runner() -> PublicMultiserviceChallengeRunner:
    return PublicMultiserviceChallengeRunner()
