"""Isolated Qdrant/OpenBao workflow runner for prospective new-service tests.

This runner supplies execution only. No task set has been frozen or scored.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.independent_v7_runner import IndependentV7Runner
from experiments.public_workflows_runner import REPOSITORY_ROOT, _free_port, _sha256

RUNTIME = REPOSITORY_ROOT / "experiments" / "real_tracks" / "runtime" / "redesign_v8"
BINARIES = {
    "qdrant": (
        "1.15.5",
        RUNTIME / "qdrant-1.15.5-linux-amd64",
        "8f8ecf5066dc7eee8b8b7e457ae0758eb7328b0aed58b910bac0d7a23f3d184a",
    ),
    "openbao": (
        "2.7.0",
        RUNTIME / "openbao-2.7.0-linux-amd64",
        "9403c2b121e13fe79b3182051320d2096d10519b597ee587e322dab5e359c51e",
    ),
}


class IndependentV8Runner(IndependentV7Runner):
    """Execute a complete selected bundle against a fresh local service."""

    def __init__(self) -> None:
        super().__init__()
        self._openbao_token = ""

    def reset(self, event: Mapping[str, Any]) -> Mapping[str, Any]:
        self._cleanup()
        self._openbao_token = ""
        self.current_service = str(event["service_id"])
        if self.current_service not in BINARIES:
            raise ValueError(f"unsupported v8 service: {self.current_service}")
        version, binary, expected_digest = BINARIES[self.current_service]
        if str(event["version_id"]) != version:
            raise ValueError("event references an unpinned service version")
        if not binary.is_file() or _sha256(binary) != expected_digest:
            raise RuntimeError(f"missing or changed pinned runtime: {binary}")
        port = _free_port()
        self.base_url = f"http://127.0.0.1:{port}"
        self.tempdir = Path(tempfile.mkdtemp(prefix=f"corem-v8-{self.current_service}-"))
        try:
            if self.current_service == "qdrant":
                environment = os.environ.copy()
                environment.update(
                    {
                        "QDRANT__SERVICE__HOST": "127.0.0.1",
                        "QDRANT__SERVICE__HTTP_PORT": str(port),
                        "QDRANT__SERVICE__GRPC_PORT": str(_free_port()),
                    }
                )
                command = [str(binary), "--disable-telemetry"]
                health_path = "/"
            else:
                self._openbao_token = secrets.token_urlsafe(32)
                environment = os.environ.copy()
                command = [
                    str(binary),
                    "server",
                    "-dev",
                    f"-dev-listen-address=127.0.0.1:{port}",
                    f"-dev-root-token-id={self._openbao_token}",
                ]
                health_path = "/v1/sys/health"
            self.process = subprocess.Popen(
                command,
                cwd=self.tempdir,
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            if not self._wait_endpoint(health_path):
                raise RuntimeError(f"{self.current_service} did not become ready")
        except Exception:
            self._cleanup()
            raise
        runtime = f"{self.current_service}:{version}:sha256:{expected_digest}"
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

    def _request(self, plan: Mapping[str, Any], path: str) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if self.current_service == "openbao":
            headers["X-Vault-Token"] = self._openbao_token
        data = None
        if "json" in plan:
            data = json.dumps(plan["json"]).encode()
            content_type = plan.get("content_type", "application/json")
            if content_type not in {"application/json", "application/merge-patch+json"}:
                raise ValueError("unsupported request content type")
            headers["Content-Type"] = str(content_type)
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=str(plan["method"]),
        )
        try:
            response = urllib.request.urlopen(request, timeout=15)
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
        return {
            "status": status,
            "headers": response_headers,
            "body": body,
            "text": response_text,
            "attempts": 1,
        }

    def checker_context(self) -> dict[str, str]:
        """Give a separate checker live-service access before snapshot teardown."""
        if self.process is None or not self.base_url.startswith("http://127.0.0.1:"):
            raise RuntimeError("no live isolated service for independent checking")
        return {
            "base_url": self.base_url,
            "service_id": self.current_service,
            "token": self._openbao_token if self.current_service == "openbao" else "",
        }

    def snapshot(self, event: Mapping[str, Any]) -> str:
        try:
            return f"{self.snapshot_id}:event:{event['event_id']}"
        finally:
            self._cleanup()


def create_runner() -> IndependentV8Runner:
    return IndependentV8Runner()
