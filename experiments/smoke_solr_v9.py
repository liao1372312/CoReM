#!/usr/bin/env python3
"""Isolated Apache Solr 9.10.1 runtime/API pilot; exclude this task pattern."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
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

from corem.real_eval import sha256_file, write_json  # noqa: E402

RUNTIME_ROOT = ROOT / "experiments/real_tracks/runtime/redesign_v9"
SOLR = RUNTIME_ROOT / "solr-9.10.1-slim"
ARCHIVE = RUNTIME_ROOT / "solr-9.10.1-slim.tgz"
SHA512_FILE = RUNTIME_ROOT / "solr-9.10.1-slim.tgz.sha512"
SOURCE_URI = "https://solr.apache.org/guide/solr/9_10/deployment-guide/solr-control-script-reference.html"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(base: str, method: str, path: str, body: Any = None) -> tuple[int, Any]:
    headers = {"Accept": "application/json"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        response = urllib.request.urlopen(request, timeout=10)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        raw = response.read()
        status = int(response.status)
    try:
        value = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        value = raw.decode(errors="replace")
    return status, value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a prior Solr pilot")
    expected_sha512 = SHA512_FILE.read_text().split()[0]
    import hashlib
    if hashlib.sha512(ARCHIVE.read_bytes()).hexdigest() != expected_sha512:
        raise RuntimeError("Apache Solr release checksum changed")
    if not (SOLR / "bin/solr").is_file():
        raise RuntimeError("Apache Solr runtime missing")
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="corem-v9-solr-pilot-") as directory:
        temp = Path(directory)
        home = temp / "solrhome"
        home.mkdir()
        shutil.copy2(SOLR / "server/solr/solr.xml", home / "solr.xml")
        shutil.copytree(SOLR / "server/solr/configsets/_default", home / "configsets/_default")
        data = temp / "data"
        data.mkdir()
        environment = os.environ.copy()
        environment["SOLR_JETTY_HOST"] = "127.0.0.1"
        environment["SOLR_ULIMIT_CHECKS"] = "false"
        process = subprocess.Popen(
            [
                str(SOLR / "bin/solr"), "start", "-f", "--host", "127.0.0.1",
                "-p", str(port), "--solr-home", str(home), "--data-home", str(data),
                "-m", "512m",
            ],
            cwd=SOLR,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 45
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Solr exited during startup")
                try:
                    health_status, health = _request(base, "GET", "/solr/admin/info/system?wt=json")
                    if health_status == 200 and health.get("responseHeader", {}).get("status") == 0:
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("Solr did not become ready")
                time.sleep(0.2)
            version = health.get("lucene", {}).get("solr-spec-version")
            if version != "9.10.1":
                raise RuntimeError(f"wrong Solr runtime version: {version}")
            create_status, created = _request(
                base, "GET",
                "/solr/admin/cores?action=CREATE&name=corem_pilot&instanceDir=corem_pilot&configSet=_default&wt=json",
            )
            if create_status != 200 or created.get("responseHeader", {}).get("status") != 0:
                raise RuntimeError(f"Solr core creation failed: {create_status} {created}")
            write_status, written = _request(
                base, "POST", "/solr/corem_pilot/update/json/docs?commit=true",
                {"id": "pilot-1", "title": "isolated-pilot"},
            )
            if write_status != 200 or written.get("responseHeader", {}).get("status") != 0:
                raise RuntimeError(f"Solr document write failed: {write_status} {written}")
            read_status, read = _request(base, "GET", "/solr/corem_pilot/select?q=id%3Apilot-1&wt=json")
            if read_status != 200 or read.get("response", {}).get("numFound") != 1:
                raise RuntimeError(f"Solr document read failed: {read_status} {read}")
            result = {
                "status": "development_runtime_pilot_passed",
                "claim_boundary": "Core creation and document index/read are inspected pilot operations excluded from future untouched v9 families.",
                "service": "solr",
                "version": version,
                "archive_sha256": sha256_file(ARCHIVE),
                "archive_sha512": expected_sha512,
                "source_uri": SOURCE_URI,
                "health_status": health_status,
                "create_status": create_status,
                "write_status": write_status,
                "read_status": read_status,
                "state_verified": True,
            }
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print("Apache Solr 9.10.1 isolated runtime/API pilot passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
