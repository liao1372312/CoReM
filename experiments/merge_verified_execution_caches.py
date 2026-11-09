#!/usr/bin/env python3
"""Merge independently checked exact event/selection execution caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXECUTION_FIELDS = (
    "execution_latency_ms",
    "reset_verified",
    "snapshot_before",
    "snapshot_after",
    "execution_error",
    "checker_error",
    "execution",
    "checker",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")
    cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    sources = []
    duplicates = 0
    for path in args.inputs:
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        for line in path.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            selected = tuple(str(value) for value in row["selected_memory_ids"])
            if selected != tuple(sorted(set(selected))):
                raise ValueError(f"non-canonical selection in {path}")
            if (
                row.get("checker_independent") is not True
                or row.get("reset_verified") is not True
                or row.get("execution_error") is not None
                or row.get("checker_error") is not None
            ):
                raise ValueError(f"unverified execution in {path}")
            key = (str(row["event_id"]), selected)
            canonical = {
                "event_id": key[0],
                "selected_memory_ids": list(selected),
                "checker_independent": True,
                **{field: row[field] for field in EXECUTION_FIELDS},
            }
            previous = cache.get(key)
            if previous is not None:
                duplicates += 1
                # Wall-clock latency and snapshots may differ across valid fresh
                # replays. Outcome payloads and independent checks may not.
                for field in ("execution", "checker"):
                    if previous[field] != canonical[field]:
                        raise ValueError(f"conflicting exact execution for {key}")
                continue
            cache[key] = canonical
    args.output.mkdir(parents=True)
    merged = args.output / "execution_results.jsonl"
    with merged.open("w") as handle:
        for key in sorted(cache):
            handle.write(json.dumps(cache[key], sort_keys=True) + "\n")
    manifest = {
        "status": "complete",
        "contract": "Exact event ID plus sorted complete selected-memory tuple.",
        "unique_verified_executions": len(cache),
        "duplicate_rows_deduplicated": duplicates,
        "sources": sources,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "checksums.sha256").write_text(
        f"{hashlib.sha256(merged.read_bytes()).hexdigest()}  execution_results.jsonl\n"
        f"{hashlib.sha256((args.output / 'manifest.json').read_bytes()).hexdigest()}  manifest.json\n"
    )
    print(f"wrote {len(cache)} unique verified executions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
