"""Sandboxed HTTP runner for the pinned public version-service experiment.

The runner interprets only a small HTTP-plan schema.  It never executes shell
text supplied by dataset records.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PublicServicesRunner:
    def __init__(self) -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self.container_name: str | None = None
        self.tempdir: Path | None = None
        self.base_url = ""
        self.snapshot_id = ""
        self.active_event: Mapping[str, Any] | None = None

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
        tempdir = Path(tempfile.mkdtemp(prefix="corem-json-server-"))
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
        command = [
            str(binary),
            str(database),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        self.process = subprocess.Popen(
            command,
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
        name = f"corem-real-{os.getpid()}-{safe}"
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
        self.active_event = event
        return {
            "reset_verified": True,
            "snapshot_id": self.snapshot_id,
            "runtime_artifact": runtime,
            "healthcheck": f"{self.base_url}/",
        }

    @staticmethod
    def _safe_plan(
        memories: Sequence[Mapping[str, Any]],
    ) -> tuple[Mapping[str, Any] | None, str | None]:
        for memory in sorted(memories, key=lambda row: str(row.get("memory_id", ""))):
            plan = memory.get("plan")
            if memory.get("role") != "action" or not isinstance(plan, dict):
                continue
            method = plan.get("method")
            path = plan.get("path")
            if method not in {"GET", "POST"} or not isinstance(path, str):
                continue
            if not path.startswith("/") or "//" in path or "\\" in path:
                continue
            if set(plan).difference({"method", "path", "json"}):
                continue
            return plan, str(memory.get("memory_id"))
        return None, None

    def execute(
        self, event: Mapping[str, Any], memories: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        plan, plan_memory_id = self._safe_plan(memories)
        observed_memories = [
            {
                "memory_id": row.get("memory_id"),
                "service_id": row.get("service_id"),
                "version_scope": row.get("version_scope"),
                "valid_from": row.get("valid_from"),
                "valid_until": row.get("valid_until"),
                "evidence_atoms": row.get("evidence_atoms"),
                "provenance_verified": row.get("provenance_verified"),
                "role": row.get("role"),
            }
            for row in memories
        ]
        if plan is None:
            return {
                "completed": False,
                "failure": "no admissible HTTP action plan was selected",
                "selected_memory_observations": observed_memories,
                "communication": {
                    "explicit_status": False,
                    "outcome_summary": "No request executed.",
                },
            }
        data = None
        headers = {"Accept": "application/json"}
        if "json" in plan:
            data = json.dumps(plan["json"]).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + str(plan["path"]),
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
            "completed": True,
            "request": {
                "method": plan["method"],
                "path": plan["path"],
                "json": plan.get("json"),
            },
            "response": {
                "status": status,
                "headers": response_headers,
                "body": body,
                "text": text,
            },
            "plan_memory_id": plan_memory_id,
            "selected_memory_observations": observed_memories,
            "communication": {
                "explicit_status": True,
                "outcome_summary": f"{plan['method']} {plan['path']} returned HTTP {status}.",
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


def create_runner() -> PublicServicesRunner:
    return PublicServicesRunner()
