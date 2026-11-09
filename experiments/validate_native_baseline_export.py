#!/usr/bin/env python3
"""Validate a native baseline export for budget parity and label blindness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import (
    audit_real_dataset,
    load_json,
    load_jsonl,
    validate_native_export,
    validate_split_assignments,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = audit_real_dataset(args.dataset)
    if not audit.eligible:
        result = {
            "status": "blocked",
            "valid": False,
            "errors": ["source dataset failed the real-track gate"],
            "dataset_audit": audit.to_dict(),
        }
    else:
        events = load_jsonl(args.dataset / "events.jsonl")
        memories = load_jsonl(args.dataset / "memories.jsonl")
        assignments = load_jsonl(args.splits)
        validate_split_assignments(events, assignments)
        split_event_ids = {
            str(row["event_id"]) for row in assignments if row["split"] == args.split
        }
        result = validate_native_export(
            events,
            load_json(args.manifest),
            load_jsonl(args.selections),
            split_event_ids=split_event_ids,
            known_memory_ids={str(row["memory_id"]) for row in memories},
        )
        result["dataset_audit"] = audit.to_dict()
        result["target_split"] = args.split
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

