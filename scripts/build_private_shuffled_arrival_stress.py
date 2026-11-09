#!/usr/bin/env python3
"""Build a controlled shuffled-arrival stress copy of the v5 dataset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import (  # noqa: E402
    sha256_file,
    write_checksums,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20271024)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")
    source_events = load_jsonl(args.source / "events.jsonl")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in source_events:
        groups[(str(event["service_id"]), str(event["version_id"]))].append(event)

    rng = random.Random(args.seed)
    events: list[dict[str, Any]] = []
    permutations: dict[str, list[str]] = {}
    for group, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: (row["event_time"], row["event_id"]))
        slots = [str(row["event_time"]) for row in ordered]
        shuffled = list(ordered)
        rng.shuffle(shuffled)
        key = f"{group[0]}::{group[1]}"
        permutations[key] = [str(row["event_id"]) for row in shuffled]
        for slot, row in zip(slots, shuffled):
            updated = dict(row)
            updated["event_time"] = slot
            updated["arrival_stress"] = "seeded_within_service_version_shuffle"
            events.append(updated)
    events.sort(key=lambda row: (row["event_time"], row["event_id"]))

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "events.jsonl", events)
    copied = [
        "memories.jsonl",
        "dependency_hyperedges.jsonl",
        "version_transitions.jsonl",
        "source_releases.json",
    ]
    for name in copied:
        shutil.copy2(args.source / name, args.output / name)
    write_json(args.output / "arrival_permutations.json", permutations)
    parent_manifest = json.loads((args.source / "manifest.json").read_text())
    manifest = dict(parent_manifest)
    manifest.update(
        {
            "dataset_name": "CoReM v5 Controlled Shuffled-Arrival Stress",
            "parent_dataset": str(args.source.resolve()),
            "parent_manifest_sha256": sha256_file(args.source / "manifest.json"),
            "arrival_stress": {
                "kind": "within_service_version_seeded_shuffle",
                "seed": args.seed,
                "label_blind": True,
                "preserves": (
                    "events, queries, memories, services, versions, release "
                    "boundaries, runner, and checker"
                ),
            },
            "claim_boundary": (
                "Derived controlled arrival-order stress; not independent data, "
                "natural traffic, or a new execution benchmark."
            ),
        }
    )
    files = ["events.jsonl", *copied, "arrival_permutations.json"]
    manifest["files"] = {
        name: {
            "sha256": sha256_file(args.output / name),
            "records": (
                len(load_jsonl(args.output / name))
                if name.endswith(".jsonl")
                else None
            ),
        }
        for name in files
    }
    write_json(args.output / "manifest.json", manifest)
    write_checksums(
        args.output,
        [*files, "manifest.json"],
    )
    print(f"wrote {len(events)} shuffled-arrival events to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
