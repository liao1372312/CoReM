#!/usr/bin/env python3
"""Generate gold-bundle selections used only for benchmark feasibility checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_jsonl, write_checksums, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    events = load_jsonl(args.dataset / "events.jsonl")
    memories = load_jsonl(args.dataset / "memories.jsonl")
    rows = []
    for event in events:
        selected = sorted(
            str(memory["memory_id"])
            for memory in memories
            if memory["service_id"] == event["service_id"]
            and event["version_id"] in memory["version_scope"]
            and event["task_key"] in memory.get("operation_ids", [])
        )
        if len(selected) != 4:
            raise ValueError(
                f"{event['event_id']} expected four oracle records, found {len(selected)}"
            )
        rows.append(
            {
                "run_id": f"oracle_feasibility:0:{event['event_id']}",
                "event_id": event["event_id"],
                "seed": 0,
                "method": "oracle_feasibility",
                "selected_memory_ids": selected,
                "token_cost": 160,
                "latency_ms": 0.0,
                "llm_calls": 0,
                "active_bank_cost": 0,
                "retrieval_status": "oracle_preflight_only",
                "evaluations": 0,
            }
        )
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "selections.jsonl", rows)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": "corem-public-challenge-oracle-v1",
            "dataset": str(args.dataset.resolve()),
            "method": "oracle_feasibility",
            "selection_rows": len(rows),
            "report_as_comparative_method": False,
            "purpose": "Reject benchmark tasks whose known-correct bundle cannot satisfy the independent checker.",
        },
    )
    write_checksums(args.output, ("manifest.json", "selections.jsonl"))
    print(f"wrote {len(rows)} oracle preflight selections to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
