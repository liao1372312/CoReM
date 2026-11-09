"""Fresh isolated Typesense/Solr workflow runner for prospective v9 tasks.

No v9 task set or model comparison has been frozen or scored.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.independent_v7_runner import IndependentV7Runner
from experiments.public_workflows_runner import REPOSITORY_ROOT, _free_port

RUNTIME_ROOT = REPOSITORY_ROOT / "experiments/real_tracks/runtime/redesign_v9"
TYPESENSE_BINARY = RUNTIME_ROOT / "typesense-server"
TYPESENSE_SHA256 = "1178111344c96bf91567f579a30e0153e346f716b6f3e10f484ba099f21450b5"
SOLR_ROOT = RUNTIME_ROOT / "solr-9.10.1-slim"
SOLR_ARCHIVE = RUNTIME_ROOT / "solr-9.10.1-slim.tgz"
SOLR_SHA512 = "8720f813f1679360f11c753ad516d4680db31afc28065626d2558fb078bd163b79107326733bee3ba6702ca2fa7ef86bd608d594a740b7dcc5475e7c8650cae1"
VERSIONS = {"typesense": "30.2", "solr": "9.10.1"}


def _digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class IndependentV9Runner(IndependentV7Runner):
    """Run a complete candidate bundle against a new localhost-only service."""

    def __init__(self) -> None:
        super().__init__()
        self._api_key = ""

    def _cleanup(self) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=10)
            self.process = None
        if self.tempdir is not None:
            shutil.rmtree(self.tempdir, ignore_errors=True)
            self.tempdir = None
        self.base_url = ""
        self._api_key = ""

    def reset(self, event: Mapping[str, Any]) -> Mapping[str, Any]:
        self._cleanup()
        self.current_service = str(event["service_id"])
        if self.current_service not in VERSIONS:
            raise ValueError(f"unsupported v9 service: {self.current_service}")
        version = VERSIONS[self.current_service]
        if str(event["version_id"]) != version:
            raise ValueError("event version does not match pinned runtime")
        if self.current_service == "typesense":
            if not TYPESENSE_BINARY.is_file() or _digest(TYPESENSE_BINARY, "sha256") != TYPESENSE_SHA256:
                raise RuntimeError("missing or changed Typesense binary")
            runtime_hash = TYPESENSE_SHA256
        else:
            if not SOLR_ARCHIVE.is_file() or _digest(SOLR_ARCHIVE, "sha512") != SOLR_SHA512:
                raise RuntimeError("missing or changed Apache Solr archive")
            if not (SOLR_ROOT / "bin/solr").is_file():
                raise RuntimeError("missing extracted Apache Solr runtime")
            runtime_hash = SOLR_SHA512
        port = _free_port()
        self.base_url = f"http://127.0.0.1:{port}"
        self.tempdir = Path(tempfile.mkdtemp(prefix=f"corem-v9-{self.current_service}-"))
        try:
            if self.current_service == "typesense":
                self._api_key = secrets.token_urlsafe(32)
                command = [
                    str(TYPESENSE_BINARY),
                    "--data-dir", str(self.tempdir),
                    "--api-key", self._api_key,
                    "--api-address", "127.0.0.1",
                    "--api-port", str(port),
                    "--peering-address", "127.0.0.1",
                    "--peering-port", str(_free_port()),
                ]
                environment = os.environ.copy()
                cwd = self.tempdir
                health_path = "/health"
            else:
                home = self.tempdir / "solrhome"
                home.mkdir()
                shutil.copy2(SOLR_ROOT / "server/solr/solr.xml", home / "solr.xml")
                shutil.copytree(SOLR_ROOT / "server/solr/configsets/_default", home / "configsets/_default")
                data = self.tempdir / "data"
                data.mkdir()
                environment = os.environ.copy()
                environment["SOLR_JETTY_HOST"] = "127.0.0.1"
                environment["SOLR_ULIMIT_CHECKS"] = "false"
                command = [
                    str(SOLR_ROOT / "bin/solr"), "start", "-f", "--host", "127.0.0.1",
                    "-p", str(port), "--solr-home", str(home), "--data-home", str(data),
                    "-m", "512m",
                ]
                cwd = SOLR_ROOT
                health_path = "/solr/admin/info/system?wt=json"
            self.process = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            if not self._wait_endpoint(health_path):
                raise RuntimeError(f"{self.current_service} did not become ready")
            health = self._request({"method": "GET"}, health_path)
            if health["status"] != 200:
                raise RuntimeError(f"{self.current_service} health probe failed")
            if self.current_service == "typesense":
                debug = self._request({"method": "GET"}, "/debug")
                if debug["status"] != 200 or debug["body"].get("version") != version:
                    raise RuntimeError("wrong Typesense runtime version")
            elif health["body"].get("lucene", {}).get("solr-spec-version") != version:
                raise RuntimeError("wrong Apache Solr runtime version")
        except Exception:
            self._cleanup()
            raise
        self.snapshot_id = f"{self.current_service}:{version}:sha:{runtime_hash}"
        return {
            "reset_verified": True,
            "snapshot_id": self.snapshot_id,
            "runtime_artifact": self.snapshot_id,
            "healthcheck": self.base_url + health_path,
        }

    def _request(self, plan: Mapping[str, Any], path: str) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if self.current_service == "typesense":
            headers["X-TYPESENSE-API-KEY"] = self._api_key
        data = None
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
        if self.process is None or not self.base_url.startswith("http://127.0.0.1:"):
            raise RuntimeError("no live isolated v9 service")
        return {
            "base_url": self.base_url,
            "service_id": self.current_service,
            "api_key": self._api_key if self.current_service == "typesense" else "",
        }

    def snapshot(self, event: Mapping[str, Any]) -> str:
        try:
            return f"{self.snapshot_id}:event:{event['event_id']}"
        finally:
            self._cleanup()


def create_runner() -> IndependentV9Runner:
    return IndependentV9Runner()
