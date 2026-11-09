#!/usr/bin/env python3
"""Audit AppWorld split files without printing protected task identifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "data" / "private_corem"
SPLITS = ("train", "dev", "test_normal", "test_challenge")


def audit_splits(appworld_root: Path) -> dict[str, object]:
    split_root = appworld_root / "data" / "datasets"
    seen: set[str] = set()
    summaries: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        path = split_root / f"{split}.txt"
        raw = path.read_bytes()
        task_ids = [line.strip() for line in raw.decode().splitlines()]
        if not task_ids or any(not task_id for task_id in task_ids):
            raise ValueError("empty split or task identifier")
        task_set = set(task_ids)
        if len(task_set) != len(task_ids):
            raise ValueError("duplicate task identifier within a split")
        if task_set & seen:
            raise ValueError("task identifier appears in multiple splits")
        seen.update(task_set)
        summaries[split] = {
            "count": len(task_ids),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return {
        "splits": summaries,
        "total_unique_task_ids": len(seen),
        "task_ids_exported": False,
        "task_content_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error("AppWorld root must stay under data/private_corem")
    print(json.dumps(audit_splits(root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
