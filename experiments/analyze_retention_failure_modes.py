#!/usr/bin/env python3
"""Compare matched online-retention trajectories and localize failures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bundle-run", type=Path, required=True)
    parser.add_argument("--corem-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def by_event(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["event_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate event id")
    return result


def operation_signature(
    selected_ids: list[str], memories: dict[str, dict[str, Any]]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(operation)
                for memory_id in selected_ids
                for operation in memories[memory_id]["operation_ids"]
            }
        )
    )


def summarize_method(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "events": len(rows),
        "vsr": mean(bool(row["checker"]["goal_satisfied"]) for row in rows),
        "retrieval_ok": mean(row["retrieval_status"] == "ok" for row in rows),
        "mean_selected_records": mean(len(row["selected_memory_ids"]) for row in rows),
        "mean_update_ms": mean(float(row["post_update_latency_ms"]) for row in rows),
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")
    memories = {
        str(row["memory_id"]): row
        for row in load_jsonl(args.dataset / "memories.jsonl")
    }
    events = by_event(load_jsonl(args.dataset / "events.jsonl"))
    runs = {
        "bundle": by_event(load_jsonl(args.bundle_run / "execution_results.jsonl")),
        "corem": by_event(load_jsonl(args.corem_run / "execution_results.jsonl")),
    }
    event_ids = list(runs["bundle"])
    if set(event_ids) != set(runs["corem"]) or set(event_ids) != set(events):
        raise ValueError("runs and dataset do not contain identical events")

    raw: list[dict[str, Any]] = []
    service_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    block_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, event_id in enumerate(event_ids, start=1):
        event = events[event_id]
        bundle = runs["bundle"][event_id]
        corem = runs["corem"][event_id]
        bundle_success = int(bundle["checker"]["goal_satisfied"] is True)
        corem_success = int(corem["checker"]["goal_satisfied"] is True)
        bundle_signature = operation_signature(bundle["selected_memory_ids"], memories)
        corem_signature = operation_signature(corem["selected_memory_ids"], memories)
        task_key = str(event["task_key"])
        row = {
            "event_index": index,
            "event_id": event_id,
            "service_id": event["service_id"],
            "version_id": event["version_id"],
            "task_key": task_key,
            "bundle_success": bundle_success,
            "corem_success": corem_success,
            "success_delta": corem_success - bundle_success,
            "bundle_retrieval_ok": int(bundle["retrieval_status"] == "ok"),
            "corem_retrieval_ok": int(corem["retrieval_status"] == "ok"),
            "bundle_selected_records": len(bundle["selected_memory_ids"]),
            "corem_selected_records": len(corem["selected_memory_ids"]),
            "bundle_operation_count": len(bundle_signature),
            "corem_operation_count": len(corem_signature),
            "bundle_task_match": int(bundle_signature == (task_key,)),
            "corem_task_match": int(corem_signature == (task_key,)),
            "same_selected_set": int(
                set(bundle["selected_memory_ids"])
                == set(corem["selected_memory_ids"])
            ),
            "bundle_update_ms": float(bundle["post_update_latency_ms"]),
            "corem_update_ms": float(corem["post_update_latency_ms"]),
        }
        raw.append(row)
        service_rows[str(event["service_id"])].append(row)
        block_rows[(index - 1) // 20 + 1].append(row)

    def grouped_summary(key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "group": key,
            "events": len(rows),
            "bundle_vsr": mean(row["bundle_success"] for row in rows),
            "corem_vsr": mean(row["corem_success"] for row in rows),
            "delta_vsr": mean(row["success_delta"] for row in rows),
            "bundle_retrieval_ok": mean(row["bundle_retrieval_ok"] for row in rows),
            "corem_retrieval_ok": mean(row["corem_retrieval_ok"] for row in rows),
            "bundle_task_match": mean(row["bundle_task_match"] for row in rows),
            "corem_task_match": mean(row["corem_task_match"] for row in rows),
        }

    service_summary = [
        grouped_summary(key, rows) for key, rows in sorted(service_rows.items())
    ]
    block_summary = [
        grouped_summary(str(key), rows) for key, rows in sorted(block_rows.items())
    ]
    disagreement = {
        "both_success": sum(
            row["bundle_success"] and row["corem_success"] for row in raw
        ),
        "bundle_only_success": sum(
            row["bundle_success"] and not row["corem_success"] for row in raw
        ),
        "corem_only_success": sum(
            row["corem_success"] and not row["bundle_success"] for row in raw
        ),
        "neither_success": sum(
            not row["bundle_success"] and not row["corem_success"] for row in raw
        ),
    }
    result = {
        "status": "complete",
        "bundle": summarize_method(list(runs["bundle"].values())),
        "corem": summarize_method(list(runs["corem"].values())),
        "disagreement": disagreement,
        "selected_set_agreement": mean(row["same_selected_set"] for row in raw),
        "task_match_rate": {
            "bundle": mean(row["bundle_task_match"] for row in raw),
            "corem": mean(row["corem_task_match"] for row in raw),
        },
        "mixed_operation_selection_rate": {
            "bundle": mean(row["bundle_operation_count"] > 1 for row in raw),
            "corem": mean(row["corem_operation_count"] > 1 for row in raw),
        },
        "interpretation_boundary": (
            "Post-hoc diagnostic on the already inspected 320-event stream; "
            "it may motivate but cannot confirm an optimized method."
        ),
    }

    args.output.mkdir(parents=True)
    for name, rows in (
        ("event_comparison.csv", raw),
        ("service_comparison.csv", service_summary),
        ("block_comparison.csv", block_summary),
    ):
        with (args.output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
