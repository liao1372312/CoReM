#!/usr/bin/env python3
"""Materialize selections from exact event/bundle executions already on disk."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import (  # noqa: E402
    load_jsonl,
    write_checksums,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--execution-sources", type=Path, nargs="+", required=True)
    parser.add_argument("--target-splits", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def key(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return (
        str(row["event_id"]),
        tuple(sorted(map(str, row["selected_memory_ids"]))),
    )


def main() -> int:
    args = parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    target_ids = None
    if args.target_splits is not None:
        target_ids = {
            str(row["event_id"])
            for row in load_jsonl(args.target_splits)
            if row.get("split") == "test"
        }
        if not target_ids:
            raise ValueError("target split contains no test events")
    selections = [
        row
        for row in load_jsonl(args.selections)
        if target_ids is None or str(row["event_id"]) in target_ids
    ]
    if not selections:
        raise ValueError("no target selections")
    cached: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    duplicate_cache_rows = 0
    for source in args.execution_sources:
        for row in load_jsonl(source):
            cache_key = key(row)
            if cache_key in cached:
                duplicate_cache_rows += 1
                continue
            cached[cache_key] = row
    output = []
    missing = []
    for selection in selections:
        cache_key = key(selection)
        source = cached.get(cache_key)
        if source is None:
            missing.append(cache_key)
            continue
        row = dict(source)
        row.update(
            {
                "run_id": selection["run_id"],
                "method": selection["method"],
                "seed": selection["seed"],
                "selected_memory_ids": selection["selected_memory_ids"],
                "token_cost": selection["token_cost"],
                "selection_latency_ms": selection.get("latency_ms", 0.0),
                "latency_ms": float(selection.get("latency_ms", 0.0))
                + float(source.get("execution_latency_ms", 0.0)),
                "physical_execution_reused": True,
                "physical_execution_source_run_id": source.get("run_id"),
            }
        )
        output.append(row)
    if missing:
        sample = [(event_id, list(ids)) for event_id, ids in missing[:3]]
        raise ValueError(f"{len(missing)} selections lack exact cached execution: {sample}")
    expected = {
        (str(row["method"]), int(row["seed"]), str(row["event_id"]))
        for row in selections
    }
    observed = {
        (str(row["method"]), int(row["seed"]), str(row["event_id"]))
        for row in output
    }
    if expected != observed or len(output) != len(observed):
        raise ValueError("materialized executions are incomplete or duplicated")

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "execution_results.jsonl", output)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": "corem-exact-execution-cache-v1",
            "status": "complete_cached_real_executions",
            "selections": str(args.selections.resolve()),
            "selection_sha256": hashlib.sha256(args.selections.read_bytes()).hexdigest(),
            "execution_sources": [str(path.resolve()) for path in args.execution_sources],
            "target_splits": str(args.target_splits.resolve()) if args.target_splits else None,
            "rows": len(output),
            "duplicate_cache_rows_ignored": duplicate_cache_rows,
            "reuse_contract": (
                "Every row reuses an independently checked reset-environment execution "
                "only when event_id and the complete selected-memory tuple match exactly."
            ),
        },
    )
    write_checksums(args.output, ("execution_results.jsonl", "manifest.json"))
    print(f"materialized {len(output)} exact cached executions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
