#!/usr/bin/env python3
"""Merge independently checked executions by exact event/selection key."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_jsonl, write_checksums, write_json, write_jsonl  # noqa: E402


def cache_key(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    raw = tuple(map(str, row["selected_memory_ids"]))
    if len(raw) != len(set(raw)):
        raise ValueError("selected_memory_ids must be unique")
    return str(row["event_id"]), tuple(sorted(raw))


def verified(row: dict[str, Any]) -> bool:
    return (
        row.get("checker_independent") is True
        and row.get("reset_verified") is True
        and row.get("execution_error") is None
        and row.get("checker_error") is None
        and isinstance(row.get("execution"), dict)
        and isinstance(row.get("checker"), dict)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    merged: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    source_counts = {}
    rejected = 0
    duplicates = 0
    for source in sorted(set(args.sources)):
        accepted_here = 0
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        for row in load_jsonl(source):
            if not verified(row):
                rejected += 1
                continue
            key = cache_key(row)
            if key in merged:
                duplicates += 1
                continue
            merged[key] = {
                **row,
                "selected_memory_ids": list(key[1]),
                "cache_source": str(source.resolve()),
                "cache_source_sha256": source_digest,
            }
            accepted_here += 1
        source_counts[str(source.resolve())] = accepted_here
    if not merged:
        raise ValueError("no verified executions were found")
    args.output.mkdir(parents=True)
    rows = [merged[key] for key in sorted(merged)]
    write_jsonl(args.output / "execution_results.jsonl", rows)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": "corem-verified-execution-cache-v1",
            "status": "complete",
            "unique_event_selection_keys": len(rows),
            "duplicate_rows_ignored": duplicates,
            "unverified_rows_rejected": rejected,
            "source_counts": source_counts,
            "reuse_contract": (
                "Rows are reusable only under exact event_id and complete sorted "
                "selected_memory_ids equality. Every retained row passed reset, "
                "independent-checker, and infrastructure-error gates."
            ),
        },
    )
    write_checksums(args.output, ("execution_results.jsonl", "manifest.json"))
    print(f"merged {len(rows)} exact verified executions from {len(args.sources)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
