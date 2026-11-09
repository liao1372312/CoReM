#!/usr/bin/env python3
"""Build a hash-verified, privacy-allowlisted local v7 score-replay package."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_json, write_json  # noqa: E402
from experiments.summarize_independent_v7_confirmation import (  # noqa: E402
    relocated_path,
    sha256,
)

PREFIX = "results_independent_v7_candidate_v4_"
PRIVATE_PARTS = {"private_corem", "assignment_salt.txt"}
EVALUATOR_LABEL_FILE = (
    "experiments/results_independent_v7_candidate_v4_selections/selections.jsonl"
)


def pinned_entries(node: Any) -> list[dict[str, str]]:
    entries = []
    if isinstance(node, dict):
        if isinstance(node.get("path"), str) and isinstance(node.get("sha256"), str):
            entries.append({"path": node["path"], "sha256": node["sha256"]})
        for value in node.values():
            entries.extend(pinned_entries(value))
    elif isinstance(node, list):
        for value in node:
            entries.extend(pinned_entries(value))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")

    experiments = REPOSITORY_ROOT / "experiments"
    commitment_path = experiments / f"{PREFIX}all_method_selection_commitment.json"
    commitment = load_json(commitment_path)
    if commitment["status"] != "all_methods_committed_before_outcome_mapping":
        raise ValueError("missing pre-outcome selection commitment")
    frozen_root = Path(commitment["preselection_freeze"]["path"]).parents[1]
    freeze_path = relocated_path(commitment["preselection_freeze"]["path"], frozen_root)
    if sha256(freeze_path) != commitment["preselection_freeze"]["sha256"]:
        raise ValueError("preselection freeze changed")
    freeze = load_json(freeze_path)

    files: dict[str, Path] = {}

    def add_file(path: Path, expected_hash: str | None = None) -> None:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact source missing or symlinked: {path}")
        try:
            relative = path.relative_to(REPOSITORY_ROOT)
        except ValueError as error:
            raise ValueError(f"artifact source outside repository: {path}") from error
        if PRIVATE_PARTS.intersection(relative.parts):
            raise ValueError(f"private file entered artifact allowlist: {relative}")
        if relative.as_posix() == EVALUATOR_LABEL_FILE:
            raise ValueError("evaluator labels entered artifact allowlist")
        if expected_hash is not None and sha256(path) != expected_hash:
            raise ValueError(f"pinned artifact source changed: {relative}")
        files[relative.as_posix()] = path

    add_file(commitment_path)
    add_file(freeze_path, commitment["preselection_freeze"]["sha256"])
    for entry in pinned_entries(freeze):
        add_file(relocated_path(entry["path"], frozen_root), entry["sha256"])
    for entry in pinned_entries(commitment):
        add_file(relocated_path(entry["path"], frozen_root), entry["sha256"])

    for path in (REPOSITORY_ROOT / "corem").glob("*.py"):
        add_file(path)
    for relative in (
        "experiments/__init__.py",
        "experiments/summarize_independent_v7_confirmation.py",
        "experiments/INDEPENDENT_V7_REPRODUCTION.md",
        "experiments/real_tracks/runtime/independent_v7/SOURCE_MANIFEST.md",
        "tests/test_independent_v7_summary.py",
    ):
        add_file(REPOSITORY_ROOT / relative)

    for method in sorted(commitment["methods"]):
        analysis_dir = experiments / f"{PREFIX}{method}_analysis_v2"
        cache_dir = experiments / f"{PREFIX}{method}_cached_execution_v2"
        execution_dir = (
            cache_dir
            if cache_dir.is_dir()
            else experiments / f"{PREFIX}{method}_execution_v2"
        )
        for directory in (analysis_dir, execution_dir):
            if not directory.is_dir():
                raise ValueError(f"missing required method artifact: {directory}")
            for path in directory.rglob("*"):
                if path.is_file():
                    add_file(path)

    args.output.mkdir(parents=True)
    manifest_files = {}
    for relative, source in sorted(files.items()):
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied_hash = sha256(target)
        if copied_hash != sha256(source):
            raise ValueError(f"artifact copy changed: {relative}")
        manifest_files[relative] = copied_hash
    write_json(
        args.output / "PACKAGE_MANIFEST.json",
        {
            "schema_version": "corem-v7-score-replay-package-v1",
            "status": "local_score_replay_package_not_public_release",
            "files": manifest_files,
            "excluded": [
                "data/private_corem/",
                EVALUATOR_LABEL_FILE,
                "experiments/results_independent_v7_candidate_v4_selections/assignment_salt.txt",
            ],
            "physical_replay_limit": (
                "The package includes pinned service binaries but not the local "
                "Postgres container image; physical replay requires a compatible "
                "container installation and a separately verified image digest."
            ),
        },
    )
    print(
        json.dumps(
            {
                "status": "packaged",
                "files": len(manifest_files),
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
