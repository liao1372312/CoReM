#!/usr/bin/env python3
"""Read-only preflight for the frozen, train-only AppWorld v23 pilot.

This checks exact local inputs and the GPU startup reserve. It neither loads
the model nor reads target ground truth or dev/test task content.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private_corem"
EXPECTED = {
    "runner": (ROOT / "experiments" / "run_appworld_local_agent_pilot.py", "4aa2e921bc8677703e80c5a91c3a4566357fe43f8c85581c8fcda1f8e96b4e5c"),
    "server": (ROOT / "experiments" / "serve_local_qwen_agent.py", "f43af6a9c9e9f9cd807d57e8c45e65f86870c2d862b6f7d76b709f97e9440cfc"),
    "train_split": (PRIVATE / "appworld" / "data" / "datasets" / "train.txt", "93d9fe71e7a2e3b7529803d4a20b604f4ebf5ae806f321081140238068189d37"),
    "order_view": (PRIVATE / "appworld_train_static_order_seed2027_n9.json", "0e324719baab4fc8117cce72ec2bd04de7043ab09c75018fa06a8750504e38ca"),
    "disjoint_view": (PRIVATE / "appworld_train_static_disjoint_edges_seed2027_n9.json", "d10599c4330d97e2464edbe6117b11e49a549e940b0eb017f313503167a170f5"),
    "dependency_view": (PRIVATE / "appworld_train_static_dependency_seed2027_n9.json", "3432c233921fa2487e7379dccbfd6cf78fbde0e40c7bad6e02930ca371818e3d"),
}
MODEL_INDEX_SHA256 = "26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5"
MODEL_MANIFEST_SHA256 = "decf46de5ac60f4cf88ab722c587f2a335136915a473491bfab43b985f71f002"
MODEL_FILES = {
    *(f"model.safetensors-{index:05d}-of-00004.safetensors" for index in range(1, 5)),
    "model.safetensors.index.json",
    "tokenizer.json",
}
STARTUP_MIN_FREE_MIB = 34 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_manifest(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in text.splitlines():
        digest, separator, name = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or not name
            or name in rows
            or Path(name).name != name
        ):
            raise ValueError("invalid frozen model checksum manifest")
        rows[name] = digest
    if set(rows) != MODEL_FILES or rows["model.safetensors.index.json"] != MODEL_INDEX_SHA256:
        raise ValueError("frozen model checksum manifest changed")
    return rows


def free_gpu_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if len(values) != 1:
        raise ValueError("v23 requires exactly one visible GPU")
    return values[0]


def audit(model_dir: Path, free_mib: int) -> dict[str, object]:
    failures: list[str] = []
    if importlib.metadata.version("appworld") != "0.1.3.post1":
        failures.append("AppWorld version differs from 0.1.3.post1")
    for name, (path, expected) in EXPECTED.items():
        if not path.is_file() or sha256(path) != expected:
            failures.append(f"{name} does not match frozen SHA-256")
    manifest_path = ROOT / "experiments" / "QWEN35_9B_SNAPSHOT_SHA256.txt"
    if sha256(manifest_path) != MODEL_MANIFEST_SHA256:
        failures.append("model checksum manifest does not match frozen SHA-256")
    manifest = parse_manifest(manifest_path.read_text())
    for name, expected in manifest.items():
        path = model_dir / name
        if not path.is_file() or sha256(path) != expected:
            failures.append(f"model file {name} does not match frozen SHA-256")
    if free_mib < STARTUP_MIN_FREE_MIB:
        failures.append("GPU has less than the frozen 34-GiB startup reserve")
    return {
        "status": "ready" if not failures else "not_ready",
        "scope": "train_only_v23_preflight_no_agent_score",
        "gpu_free_mib": free_mib,
        "gpu_required_mib": STARTUP_MIN_FREE_MIB,
        "frozen_files_checked": len(EXPECTED) + len(manifest),
        "failures": failures,
        "task_content_read": False,
        "dev_or_test_loaded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.model_dir.resolve(), free_gpu_mib())
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
