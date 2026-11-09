#!/usr/bin/env python3
"""Isolated Typesense 30.2 runtime/API pilot; exclude its task pattern from v9."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import write_json  # noqa: E402

RUNTIME = ROOT / "experiments/real_tracks/runtime/redesign_v9/typesense-server"
RUNTIME_SHA256 = "1178111344c96bf91567f579a30e0153e346f716b6f3e10f484ba099f21450b5"
SOURCE_URI = "https://typesense.org/docs/guide/install-typesense.html"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(base: str, method: str, path: str, key: str, body: Any = None) -> tuple[int, Any]:
    headers = {"Accept": "application/json", "X-TYPESENSE-API-KEY": key}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        response = urllib.request.urlopen(request, timeout=5)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        payload = response.read()
        status = int(response.status)
    return status, json.loads(payload) if payload else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a prior Typesense pilot")
    if not RUNTIME.is_file() or hashlib.sha256(RUNTIME.read_bytes()).hexdigest() != RUNTIME_SHA256:
        raise RuntimeError("Typesense runtime missing or digest changed")
    port = _free_port()
    peering_port = _free_port()
    token = secrets.token_urlsafe(24)
    base = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="corem-v9-typesense-pilot-") as directory:
        process = subprocess.Popen(
            [
                str(RUNTIME),
                "--data-dir", directory,
                "--api-key", token,
                "--api-address", "127.0.0.1",
                "--api-port", str(port),
                "--peering-address", "127.0.0.1",
                "--peering-port", str(peering_port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Typesense exited during startup")
                try:
                    health_status, health = _request(base, "GET", "/health", token)
                    if health_status == 200 and health == {"ok": True}:
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("Typesense health endpoint did not become ready")
                time.sleep(0.1)
            debug_status, debug = _request(base, "GET", "/debug", token)
            if debug_status != 200 or debug.get("version") != "30.2":
                raise RuntimeError("Typesense version probe failed")
            collection = "corem_v9_smoke"
            create_status, created = _request(base, "POST", "/collections", token, {
                "name": collection,
                "fields": [{"name": "title", "type": "string"}],
            })
            write_status, written = _request(
                base, "POST", f"/collections/{collection}/documents", token,
                {"id": "1", "title": "isolated-pilot"},
            )
            read_status, read = _request(base, "GET", f"/collections/{collection}/documents/1", token)
            if (
                create_status != 201
                or created.get("name") != collection
                or write_status != 201
                or written.get("id") != "1"
                or read_status != 200
                or read.get("title") != "isolated-pilot"
            ):
                raise RuntimeError("Typesense create/write/read pilot failed")
            result = {
                "status": "development_runtime_pilot_passed",
                "claim_boundary": "Collection/document create-read is an inspected pilot pattern, excluded from future untouched v9 task families.",
                "service": "typesense",
                "version": "30.2",
                "runtime_sha256": RUNTIME_SHA256,
                "source_uri": SOURCE_URI,
                "health_status": health_status,
                "version_status": debug_status,
                "create_status": create_status,
                "write_status": write_status,
                "read_status": read_status,
                "state_verified": True,
            }
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print("Typesense 30.2 isolated runtime/API pilot passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
