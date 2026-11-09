#!/usr/bin/env python3
"""Download and verify the pinned WireMock and MockServer v5 runtimes."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_ROOT = REPOSITORY_ROOT / "experiments" / "real_tracks" / "runtime"
ARTIFACTS = (
    (
        "wiremock-3.9.1/wiremock-standalone-3.9.1.jar",
        "https://repo.maven.apache.org/maven2/org/wiremock/wiremock-standalone/3.9.1/wiremock-standalone-3.9.1.jar",
        "723a880d50d3b0a145af0df07e578c2cb85e77feb2231e6991c9a1366926912c",
    ),
    (
        "wiremock-3.13.2/wiremock-standalone-3.13.2.jar",
        "https://repo.maven.apache.org/maven2/org/wiremock/wiremock-standalone/3.13.2/wiremock-standalone-3.13.2.jar",
        "d097b19bd483c5038479b13a5c71e9faf8f2f5106584f0c120a7770ab0bdb367",
    ),
    (
        "mockserver-5.14.0/mockserver-netty-5.14.0.jar",
        "https://repo.maven.apache.org/maven2/org/mock-server/mockserver-netty/5.14.0/mockserver-netty-5.14.0-jar-with-dependencies.jar",
        "2a11b20118fc2904a9408f6bcb98b63ed1f26e6db84447cbb99a6f108c2246e6",
    ),
    (
        "mockserver-5.15.0/mockserver-netty-5.15.0.jar",
        "https://repo.maven.apache.org/maven2/org/mock-server/mockserver-netty/5.15.0/mockserver-netty-5.15.0-jar-with-dependencies.jar",
        "bf5cf50cb5bf8ef99ea372e78b5ceddcc6eb6a051f3b5b1d3e62c7b812c92aeb",
    ),
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    args = parser.parse_args()
    for relative, url, expected in ARTIFACTS:
        target = args.runtime_root / relative
        if target.is_file():
            if digest(target) != expected:
                raise ValueError(f"existing runtime has the wrong digest: {target}")
            print(f"verified {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "corem-v5"})
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            partial.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
        if digest(partial) != expected:
            partial.unlink(missing_ok=True)
            raise ValueError(f"downloaded runtime has the wrong digest: {url}")
        partial.replace(target)
        print(f"installed {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
