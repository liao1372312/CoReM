#!/usr/bin/env python3
"""Create grouped chronological and provider-holdout real-track splits."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import (
    SPLIT_RATIOS,
    audit_real_dataset,
    grouped_chronological_split,
    load_jsonl,
    stratified_grouped_chronological_split,
    validate_split_assignments,
    write_checksums,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--provider-holdout-fraction", type=float, default=0.20)
    parser.add_argument(
        "--stratify-by",
        action="append",
        default=[],
        help="repeatable event field; preserves provider/service coverage per split",
    )
    return parser.parse_args()


def provider_holdout(
    events: list[dict[str, object]], fraction: float
) -> list[dict[str, object]]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("provider holdout fraction must lie strictly between 0 and 1")
    providers = sorted({str(row["provider_id"]) for row in events})
    count = max(1, math.ceil(len(providers) * fraction))
    ordered = sorted(
        providers,
        key=lambda value: hashlib.sha256(
            f"corem-provider-holdout-v1:{value}".encode()
        ).hexdigest(),
    )
    held_out = set(ordered[:count])
    return [
        {
            "provider_id": provider,
            "provider_split": "heldout" if provider in held_out else "development",
        }
        for provider in providers
    ]


def main() -> int:
    args = parse_args()
    report = audit_real_dataset(args.dataset)
    if not report.eligible:
        print("dataset is not eligible; run audit_real_track_dataset.py for details")
        return 2
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    events = load_jsonl(args.dataset / "events.jsonl")
    assignments = (
        stratified_grouped_chronological_split(events, fields=args.stratify_by)
        if args.stratify_by
        else grouped_chronological_split(events)
    )
    validate_split_assignments(events, assignments)
    providers = provider_holdout(events, args.provider_holdout_fraction)
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "chronological_splits.jsonl", assignments)
    write_jsonl(args.output / "provider_splits.jsonl", providers)
    split_counts = Counter(str(row["split"]) for row in assignments)
    provider_counts = Counter(str(row["provider_split"]) for row in providers)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": "corem-real-splits-v1",
            "source_dataset": str(args.dataset.resolve()),
            "source_audit": report.to_dict(),
            "chronological_ratios": list(SPLIT_RATIOS),
            "stratification_fields": args.stratify_by,
            "chronological_counts": dict(sorted(split_counts.items())),
            "provider_holdout_fraction": args.provider_holdout_fraction,
            "provider_counts": dict(sorted(provider_counts.items())),
            "policy": (
                "Transitive incident/duplicate closure; chronological boundaries "
                "within declared strata (global when empty); deterministic SHA-256 "
                "provider holdout."
            ),
        },
    )
    write_checksums(
        args.output,
        ("chronological_splits.jsonl", "provider_splits.jsonl", "manifest.json"),
    )
    print(f"wrote {len(assignments)} assignments to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
