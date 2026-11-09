"""Resettable Gitea/Meilisearch runner for prospective v7 API workflows.

This module only supplies execution. It does not generate tasks, labels, or
claims of independent confirmation. Every reset uses an isolated service state.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.public_multiservice_challenge_runner import (
    PublicMultiserviceChallengeRunner,
)
from experiments.public_workflows_runner import REPOSITORY_ROOT, _free_port, _sha256

RUNTIME = REPOSITORY_ROOT / "experiments" / "real_tracks" / "runtime" / "independent_v7"
GITEA_BINARY = RUNTIME / "gitea-28.1.0-linux-amd64"
GITEA_SHA256 = "ea93cfa4a300564ac457d3903469fea49ec5a4da37d0cebaa435f8e329ea64d0"
MEILI_BINARY = RUNTIME / "meilisearch-v1.55.0-linux-amd64"
MEILI_SHA256 = "7e33f5889235f14290074a19a6d3574bd9b10fb2ee8731eabb7dafd0714b0513"
CONTAINER_IMAGE = "postgres:16"
CONTAINER_IMAGE_ID = (
    "sha256:539fe39e8a63b510fe8fd0e65d9a21f36e80c20f797bd7fc00ff25f4957ab167"
)


def _checked_binary(path: Path, expected: str) -> str:
    if not path.is_file() or _sha256(path) != expected:
        raise RuntimeError(f"missing or changed pinned runtime: {path}")
    return expected


class IndependentV7Runner(PublicMultiserviceChallengeRunner):
    """Run a selected workflow bundle against a fresh real service."""

    def __init__(self) -> None:
        super().__init__()
        self._gitea_auth = ""

    def _wait_endpoint(self, path: str) -> bool:
        """Allow up to 40 seconds for isolated binary/container startup."""
        return super()._wait_endpoint(path) or super()._wait_endpoint(path)

    @staticmethod
    def _challenge_plans(
        memories: Sequence[Mapping[str, Any]], operation_count: int
    ) -> list[tuple[int, str, dict[str, Any]]]:
        """Accept destructive HTTP verbs only inside resettable v7 snapshots."""
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
            if plan.get("method") not in {"GET", "POST", "PATCH", "PUT", "DELETE"}:
                continue
            path = plan.get("path")
            if (
                not isinstance(path, str)
                or not path.startswith("/")
                or "//" in path
                or "\\" in path
            ):
                continue
            if set(plan).difference({"method", "path", "json", "capture", "content_type"}):
                continue
            plans.setdefault(
                int(step_index), (str(memory.get("memory_id")), dict(plan))
            )
        return [
            (index, memory_id, plan)
            for index, (memory_id, plan) in sorted(plans.items())
        ]

    def _start_gitea(self, port: int) -> str:
        digest = _checked_binary(GITEA_BINARY, GITEA_SHA256)
        assert self.tempdir is not None
        password = secrets.token_urlsafe(24)
        config = self.tempdir / "app.ini"
        config.write_text(
            "APP_NAME = CoReM isolated evaluation\n"
            "RUN_MODE = prod\n"
            f"APP_DATA_PATH = {self.tempdir / 'data'}\n"
            "[server]\n"
            "DOMAIN = 127.0.0.1\n"
            "HTTP_ADDR = 127.0.0.1\n"
            f"HTTP_PORT = {port}\n"
            f"ROOT_URL = http://127.0.0.1:{port}/\n"
            "DISABLE_SSH = true\n"
            "OFFLINE_MODE = true\n"
            "[database]\n"
            "DB_TYPE = sqlite3\n"
            f"PATH = {self.tempdir / 'gitea.db'}\n"
            "[repository]\n"
            f"ROOT = {self.tempdir / 'repos'}\n"
            "[security]\n"
            "INSTALL_LOCK = true\n"
            f"SECRET_KEY = {secrets.token_urlsafe(32)}\n"
            f"INTERNAL_TOKEN = {secrets.token_urlsafe(32)}\n"
            "[service]\n"
            "DISABLE_REGISTRATION = true\n"
            "REQUIRE_SIGNIN_VIEW = true\n"
            "[log]\n"
            "MODE = console\n"
            "LEVEL = Error\n",
            encoding="utf-8",
        )
        command = [
            str(GITEA_BINARY),
            "--work-path",
            str(self.tempdir),
            "--config",
            str(config),
        ]
        self.process = subprocess.Popen(
            [str(GITEA_BINARY), "web", *command[1:]],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        if not self._wait_endpoint("/api/v1/version"):
            raise RuntimeError("Gitea did not become ready")
        subprocess.run(
            [
                str(GITEA_BINARY),
                "admin",
                "user",
                "create",
                "--work-path",
                str(self.tempdir),
                "--config",
                str(config),
                "--username",
                "corem_eval",
                "--password",
                password,
                "--email",
                "corem-eval@localhost.invalid",
                "--admin",
                "--must-change-password=false",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        self._gitea_auth = base64.b64encode(f"corem_eval:{password}".encode()).decode()
        return f"gitea:28.1.0:sha256:{digest}"

    def _start_meilisearch(self, port: int) -> str:
        digest = _checked_binary(MEILI_BINARY, MEILI_SHA256)
        inspected = subprocess.run(
            ["docker", "image", "inspect", CONTAINER_IMAGE, "--format", "{{.Id}}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        if inspected != CONTAINER_IMAGE_ID:
            raise RuntimeError("Meilisearch runtime container image changed")
        self.container_name = f"corem-v7-meili-{uuid.uuid4().hex[:12]}"
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-d",
                "--name",
                self.container_name,
                "-p",
                f"127.0.0.1:{port}:7700",
                "--mount",
                f"type=bind,src={RUNTIME},dst=/runtime,readonly",
                "--entrypoint",
                "/runtime/meilisearch-v1.55.0-linux-amd64",
                CONTAINER_IMAGE,
                "--http-addr",
                "0.0.0.0:7700",
                "--env",
                "development",
                "--db-path",
                "/tmp/corem-v7-evaluation",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        if not self._wait_endpoint("/health"):
            observed = subprocess.run(
                [
                    "docker",
                    "inspect",
                    self.container_name,
                    "--format",
                    "{{.State.Status}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            state = observed.stdout.strip() or observed.stderr.strip()[:160]
            logs = subprocess.run(
                ["docker", "logs", "--tail", "12", self.container_name],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            diagnostic = (logs.stderr + logs.stdout).strip()[-500:]
            raise RuntimeError(
                "Meilisearch did not become ready; "
                f"container_state={state}; log_tail={diagnostic}"
            )
        return f"meilisearch:1.55.0:sha256:{digest}:container:{CONTAINER_IMAGE_ID}"

    def reset(self, event: Mapping[str, Any]) -> Mapping[str, Any]:
        self._cleanup()
        self._gitea_auth = ""
        self.current_service = str(event["service_id"])
        if self.current_service not in {"gitea", "meilisearch"}:
            raise ValueError(f"unsupported v7 service: {self.current_service}")
        expected_version = {"gitea": "28.1.0", "meilisearch": "1.55.0"}[
            self.current_service
        ]
        if str(event["version_id"]) != expected_version:
            raise ValueError("event references an unpinned service version")
        port = _free_port()
        self.base_url = f"http://127.0.0.1:{port}"
        try:
            if self.current_service == "gitea":
                self.tempdir = Path(tempfile.mkdtemp(prefix="corem-v7-gitea-"))
                runtime = self._start_gitea(port)
                health_path = "/api/v1/version"
            else:
                runtime = self._start_meilisearch(port)
                health_path = "/health"
        except Exception:
            self._cleanup()
            raise
        self.snapshot_id = (
            f"{self.current_service}:{expected_version}:"
            f"{hashlib.sha256(runtime.encode()).hexdigest()}"
        )
        return {
            "reset_verified": True,
            "snapshot_id": self.snapshot_id,
            "runtime_artifact": runtime,
            "healthcheck": self.base_url + health_path,
        }

    def _request(self, plan: Mapping[str, Any], path: str) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        data = None
        if "json" in plan:
            data = json.dumps(plan["json"]).encode()
            headers["Content-Type"] = "application/json"
        if self.current_service == "gitea":
            headers["Authorization"] = f"Basic {self._gitea_auth}"
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=str(plan["method"]),
        )
        for attempt in range(20):
            try:
                response = urllib.request.urlopen(request, timeout=10)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                raw = response.read()
                status = int(response.status)
                response_headers = dict(response.headers.items())
            response_text = raw.decode("utf-8", errors="replace")
            try:
                body: Any = json.loads(response_text) if response_text else None
            except json.JSONDecodeError:
                body = response_text
            result = {
                "status": status,
                "headers": response_headers,
                "body": body,
                "text": response_text,
                "attempts": attempt + 1,
            }
            if (
                self.current_service != "meilisearch"
                or not path.startswith("/tasks/")
                or not isinstance(body, dict)
                or body.get("status") not in {"enqueued", "processing"}
            ):
                return result
            time.sleep(0.05)
        raise TimeoutError("Meilisearch task did not finish within 20 polls")

    def snapshot(self, event: Mapping[str, Any]) -> str:
        try:
            return f"{self.snapshot_id}:event:{event['event_id']}"
        finally:
            self._cleanup()


def create_runner() -> IndependentV7Runner:
    return IndependentV7Runner()
