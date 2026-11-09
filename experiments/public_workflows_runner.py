"""Version-isolated two-step HTTP workflow runner for the public T2 track."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = REPOSITORY_ROOT / "experiments" / "real_tracks" / "runtime"
GO_IMAGES = {
    "v2.14.1": "ghcr.io/mccutchen/go-httpbin@sha256:2612f203b1be154524b0dbb65212a158776f0643e2f6e1dad45984fc93413d97",
    "v2.15.0": "ghcr.io/mccutchen/go-httpbin@sha256:24528cf5229d0b70065ac27e6c9e4d96f5452a84a3ce4433e56573c18d96827a",
}
JSON_BINARIES = {
    "0.17.4": RUNTIME_ROOT
    / "json-server-v017"
    / "node_modules"
    / ".bin"
    / "json-server",
    "1.0.0-beta.3": RUNTIME_ROOT
    / "json-server-v1"
    / "node_modules"
    / ".bin"
    / "json-server",
}
PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)\}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capture_value(response: Mapping[str, Any], rule: Mapping[str, Any]) -> Any:
    source = rule.get("source")
    if source == "status":
        return response.get("status")
    if source == "text":
        return response.get("text")
    if source != "body_path" or not isinstance(rule.get("path"), list):
        raise ValueError("unsupported capture rule")
    value: Any = response.get("body")
    for component in rule["path"]:
        if (
            isinstance(component, int)
            and isinstance(value, list)
            or isinstance(component, str)
            and isinstance(value, dict)
        ):
            value = value[component]
        else:
            raise TypeError("capture path does not match the observed response")
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise TypeError("captured values must be scalar strings or numbers")
    return value


class PublicWorkflowsRunner:
    def __init__(self) -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self.container_name: str | None = None
        self.tempdir: Path | None = None
        self.base_url = ""
        self.snapshot_id = ""

    def _cleanup(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self.process = None
        if self.container_name is not None:
            subprocess.run(
                ["docker", "stop", "--time", "2", self.container_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.container_name = None
        if self.tempdir is not None:
            shutil.rmtree(self.tempdir, ignore_errors=True)
            self.tempdir = None

    def _wait_ready(self) -> bool:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{self.base_url}/", timeout=1) as response:
                    if response.status < 500:
                        return True
            except (OSError, urllib.error.URLError):
                time.sleep(0.1)
        return False

    def _start_json_server(self, version: str, port: int) -> str:
        binary = JSON_BINARIES.get(version)
        if binary is None or not binary.is_file():
            raise RuntimeError(f"missing pinned json-server runtime for {version}")
        tempdir = Path(tempfile.mkdtemp(prefix="corem-json-workflow-"))
        self.tempdir = tempdir
        seed = json.loads((RUNTIME_ROOT / "db.seed.json").read_text(encoding="utf-8"))
        if version == "1.0.0-beta.3":
            for collection in seed.values():
                if not isinstance(collection, list):
                    continue
                for record in collection:
                    if isinstance(record, dict) and "id" in record:
                        record["id"] = str(record["id"])
                    if isinstance(record, dict) and "postId" in record:
                        record["postId"] = str(record["postId"])
        database = tempdir / "db.json"
        database.write_text(json.dumps(seed, indent=2) + "\n", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                str(binary),
                str(database),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=binary.parents[2],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        package_lock = binary.parents[2] / "package-lock.json"
        return f"npm:{version}:sha256:{_sha256(package_lock)}:seed:{_sha256(database)}"

    def _start_go_httpbin(self, version: str, port: int, event_id: str) -> str:
        image = GO_IMAGES.get(version)
        if image is None:
            raise RuntimeError(f"unsupported go-httpbin version {version}")
        safe = hashlib.sha256(event_id.encode()).hexdigest()[:12]
        name = f"corem-flow-{os.getpid()}-{safe}"
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--detach",
                "--name",
                name,
                "--publish",
                f"127.0.0.1:{port}:8080",
                image,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.container_name = name
        return image

    def reset(self, event: Mapping[str, Any]) -> Mapping[str, Any]:
        self._cleanup()
        service = str(event["service_id"])
        version = str(event["version_id"])
        port = _free_port()
        self.base_url = f"http://127.0.0.1:{port}"
        if service == "json-server":
            runtime = self._start_json_server(version, port)
        elif service == "go-httpbin":
            runtime = self._start_go_httpbin(version, port, str(event["event_id"]))
        else:
            raise RuntimeError(f"unsupported service {service}")
        if not self._wait_ready():
            self._cleanup()
            raise RuntimeError(f"service did not become ready: {service} {version}")
        self.snapshot_id = (
            f"{service}:{version}:{hashlib.sha256(runtime.encode()).hexdigest()}"
        )
        return {
            "reset_verified": True,
            "snapshot_id": self.snapshot_id,
            "runtime_artifact": runtime,
            "healthcheck": f"{self.base_url}/",
        }

    @staticmethod
    def _memory_observations(
        memories: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        fields = (
            "memory_id",
            "service_id",
            "version_scope",
            "valid_from",
            "valid_until",
            "evidence_atoms",
            "provenance_verified",
            "role",
            "step_index",
        )
        return [{field: row.get(field) for field in fields} for row in memories]

    @staticmethod
    def _plans(
        memories: Sequence[Mapping[str, Any]],
    ) -> list[tuple[int, str, dict[str, Any]]]:
        plans: dict[int, tuple[str, dict[str, Any]]] = {}
        for memory in sorted(memories, key=lambda row: str(row.get("memory_id", ""))):
            plan = memory.get("plan")
            step_index = memory.get("step_index")
            if (
                memory.get("role") != "workflow_step"
                or not isinstance(plan, dict)
                or step_index not in {1, 2}
            ):
                continue
            method = plan.get("method")
            path = plan.get("path")
            if method not in {"GET", "POST", "PATCH"} or not isinstance(path, str):
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

    @staticmethod
    def _render_path(template: str, bindings: Mapping[str, Any]) -> str:
        required = set(PLACEHOLDER.findall(template))
        if required.difference(bindings):
            raise ValueError("workflow step requires an unavailable binding")
        rendered = PLACEHOLDER.sub(
            lambda match: urllib.parse.quote(str(bindings[match.group(1)]), safe=""),
            template,
        )
        if not rendered.startswith("/") or "//" in rendered or "\\" in rendered:
            raise ValueError("rendered workflow path is unsafe")
        return rendered

    def _request(self, plan: Mapping[str, Any], path: str) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if "json" in plan:
            data = json.dumps(plan["json"]).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=str(plan["method"]),
        )
        try:
            response = urllib.request.urlopen(request, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read()
            status = int(response.status)
            response_headers = dict(response.headers.items())
        text = raw.decode("utf-8", errors="replace")
        try:
            body: Any = json.loads(text) if text else None
        except json.JSONDecodeError:
            body = text
        return {
            "status": status,
            "headers": response_headers,
            "body": body,
            "text": text,
        }

    def execute(
        self,
        event: Mapping[str, Any],
        memories: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        observations = self._memory_observations(memories)
        plans = self._plans(memories)
        if [item[0] for item in plans] != [1, 2]:
            return {
                "completed": False,
                "failure": "selected memories do not supply both workflow steps",
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
            response = self._request(plan, path)
            step = {
                "step_index": step_index,
                "plan_memory_id": memory_id,
                "request": {
                    "method": plan["method"],
                    "path_template": plan["path"],
                    "path": path,
                    "json": plan.get("json"),
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
        return {
            "completed": True,
            "steps": steps,
            "bindings": bindings,
            "selected_memory_observations": observations,
            "communication": {
                "explicit_statuses": True,
                "outcome_summary": (
                    f"Executed two-step workflow with HTTP statuses "
                    f"{statuses[0]} and {statuses[1]}."
                ),
            },
        }

    def snapshot(self, event: Mapping[str, Any]) -> str:
        after = self.snapshot_id
        if self.tempdir is not None:
            database = self.tempdir / "db.json"
            after = f"{after}:state:{_sha256(database)}"
        self._cleanup()
        return after

    def __del__(self) -> None:
        self._cleanup()


def create_runner() -> PublicWorkflowsRunner:
    return PublicWorkflowsRunner()
