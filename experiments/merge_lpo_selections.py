#!/usr/bin/env python3
"""Merge disjoint leave-provider-out folds into one complete test export."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_jsonl, write_checksums, write_json, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="hgt_leave_provider_out")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    target_ids = {
        str(row["event_id"])
        for row in load_jsonl(args.splits)
        if row.get("split") == "test"
    }
    rows: list[dict[str, Any]] = []
    folds = []
    for directory in args.fold_dirs:
        result_path = directory / "result.json"
        manifest_path = directory / "manifest.json"
        if result_path.is_file():
            metadata_path = result_path
            metadata = json.loads(result_path.read_text(encoding="utf-8"))
            provider = metadata["configuration"].get("test_provider")
        elif manifest_path.is_file():
            metadata_path = manifest_path
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
            provider = metadata.get("provider")
        else:
            raise ValueError(f"{directory} has no result or manifest")
        if not provider:
            raise ValueError(f"{directory} is not a leave-provider-out fold")
        fold_rows = load_jsonl(directory / "selections.jsonl")
        for source in fold_rows:
            row = dict(source)
            row["method"] = args.method
            row["run_id"] = f"{args.method}:{row['seed']}:{row['event_id']}"
            row["heldout_provider_id"] = provider
            rows.append(row)
        folds.append(
            {
                "provider_id": provider,
                "directory": str(directory.resolve()),
                "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                "rows": len(fold_rows),
            }
        )
    keys = [
        (str(row["method"]), int(row["seed"]), str(row["event_id"])) for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("leave-provider-out folds overlap")
    observed_ids = {event_id for _, _, event_id in keys}
    seeds = {seed for _, seed, _ in keys}
    expected = {(args.method, seed, event_id) for seed in seeds for event_id in target_ids}
    if set(keys) != expected:
        raise ValueError(
            f"LPO export is incomplete: observed {len(keys)}, expected {len(expected)}"
        )
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "selections.jsonl", rows)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": "corem-leave-provider-out-selections-v1",
            "method": args.method,
            "providers": folds,
            "seeds": sorted(seeds),
            "events": len(target_ids),
            "selection_rows": len(rows),
            "label_boundary": (
                "Each provider's rows are selected by models for which that provider "
                "contributed no train, tune, or validation outcomes. Public provider "
                "specifications and memory records remain visible at inference."
            ),
        },
    )
    write_checksums(args.output, ("manifest.json", "selections.jsonl"))
    print(f"merged {len(rows)} LPO selections to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
