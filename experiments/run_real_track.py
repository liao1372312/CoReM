#!/usr/bin/env python3
"""Execute precomputed selections against a resettable real service snapshot."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import (  # noqa: E402
    RealTrackError,
    STATIC_SCHEMA_VERSION,
    audit_real_dataset,
    load_jsonl,
    load_plugin,
    parse_utc,
    validate_split_assignments,
    write_checksums,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--runner-plugin", required=True, help="module:factory")
    parser.add_argument("--checker-plugin", required=True, help="module:factory")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--event-id",
        help="Optionally execute only one event from the selected split.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _failure_execution(stage: str, error: Exception) -> dict[str, Any]:
    return {
        "completed": False,
        "error_stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def main() -> int:
    args = parse_args()
    audit = audit_real_dataset(args.dataset)
    if not audit.eligible:
        print("dataset failed the real-track gate; no service calls were made")
        return 2
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2

    events = load_jsonl(args.dataset / "events.jsonl")
    memories = load_jsonl(args.dataset / "memories.jsonl")
    assignments = load_jsonl(args.splits)
    selections = load_jsonl(args.selections)
    validate_split_assignments(
        events,
        assignments,
        allow_test_only=audit.schema_version == STATIC_SCHEMA_VERSION,
    )
    events_by_id = {str(row["event_id"]): row for row in events}
    memories_by_id = {str(row["memory_id"]): row for row in memories}
    target_ids = (
        {str(row["event_id"]) for row in assignments}
        if args.split == "all"
        else {str(row["event_id"]) for row in assignments if row["split"] == args.split}
    )
    if args.event_id is not None:
        if args.event_id not in target_ids:
            print(f"event {args.event_id!r} is not in split {args.split!r}")
            return 2
        target_ids = {args.event_id}
    if not target_ids:
        print(f"split {args.split!r} has no events")
        return 2
    selected_rows = [
        row for row in selections if str(row.get("event_id")) in target_ids
    ]
    if not selected_rows:
        print(f"no selections found for split {args.split!r}")
        return 2
    selected_rows.sort(
        key=lambda row: (
            str(row.get("method", "")),
            int(row.get("seed", 0)),
            parse_utc(events_by_id[str(row["event_id"])]["event_time"], "event_time"),
            str(row["event_id"]),
        )
    )

    try:
        runner = load_plugin(args.runner_plugin)
        checker = load_plugin(args.checker_plugin)
    except (ImportError, AttributeError, RealTrackError) as error:
        print(f"could not load plugins: {error}")
        return 2
    for method_name in ("reset", "execute", "snapshot"):
        if not callable(getattr(runner, method_name, None)):
            print(f"runner plugin lacks {method_name}()")
            return 2
    if not callable(getattr(checker, "check", None)):
        print("checker plugin lacks check()")
        return 2

    checker_independent = args.runner_plugin != args.checker_plugin
    outputs: list[dict[str, Any]] = []
    for selection in selected_rows:
        event_id = str(selection.get("event_id"))
        event = events_by_id[event_id]
        memory_ids = selection.get("selected_memory_ids")
        if not isinstance(memory_ids, list) or any(
            not isinstance(memory_id, str) for memory_id in memory_ids
        ):
            raise RealTrackError(
                f"selection for {event_id} has invalid selected_memory_ids"
            )
        unknown = set(memory_ids).difference(memories_by_id)
        if unknown:
            raise RealTrackError(
                f"selection for {event_id} references unknown memories: {sorted(unknown)}"
            )
        selected_memories = [memories_by_id[memory_id] for memory_id in memory_ids]
        started = time.perf_counter()
        execution_error = None
        reset_verified = False
        snapshot_before = None
        snapshot_after = None
        try:
            reset = runner.reset(event)
            if not isinstance(reset, dict):
                raise RealTrackError("runner.reset() must return an object")
            reset_verified = reset.get("reset_verified") is True
            snapshot_before = reset.get("snapshot_id")
            if not reset_verified or not isinstance(snapshot_before, str):
                raise RealTrackError(
                    "runner.reset() must verify reset and return snapshot_id"
                )
            execution = runner.execute(event, selected_memories)
            if not isinstance(execution, dict):
                raise RealTrackError("runner.execute() must return an object")
            snapshot_after = runner.snapshot(event)
            if not isinstance(snapshot_after, str) or not snapshot_after:
                raise RealTrackError("runner.snapshot() must return a snapshot id")
        except Exception as error:  # noqa: BLE001 -- plugin boundary
            execution_error = f"{type(error).__name__}: {error}"
            execution = _failure_execution("runner", error)
        try:
            checker_output = checker.check(event, execution)
            if not isinstance(checker_output, dict):
                raise RealTrackError("checker.check() must return an object")
            checker_error = None
        except Exception as error:  # noqa: BLE001 -- plugin boundary
            checker_output = None
            checker_error = f"{type(error).__name__}: {error}"
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        outputs.append(
            {
                "run_id": selection.get(
                    "run_id",
                    f"{selection.get('method', 'unknown')}:{selection.get('seed', 0)}:{event_id}",
                ),
                "event_id": event_id,
                "method": selection.get("method"),
                "seed": selection.get("seed"),
                "selected_memory_ids": memory_ids,
                "token_cost": selection.get("token_cost"),
                "selection_latency_ms": selection.get("latency_ms"),
                "execution_latency_ms": elapsed_ms,
                "latency_ms": float(selection.get("latency_ms", 0.0)) + elapsed_ms,
                "reset_verified": reset_verified,
                "snapshot_before": snapshot_before,
                "snapshot_after": snapshot_after,
                "execution_error": execution_error,
                "checker_error": checker_error,
                "checker_independent": checker_independent,
                "execution": execution,
                "checker": checker_output,
            }
        )

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "execution_results.jsonl", outputs)
    write_json(
        args.output / "run_manifest.json",
        {
            "schema_version": "corem-real-execution-v1",
            "status": "executed_pending_analysis",
            "dataset": str(args.dataset.resolve()),
            "dataset_audit": audit.to_dict(),
            "splits": str(args.splits.resolve()),
            "selections": str(args.selections.resolve()),
            "target_split": args.split,
            "target_event_id": args.event_id,
            "runner_plugin": args.runner_plugin,
            "checker_plugin": args.checker_plugin,
            "checker_independent": checker_independent,
            "rows": len(outputs),
            "runner_errors": sum(row["execution_error"] is not None for row in outputs),
            "checker_errors": sum(row["checker_error"] is not None for row in outputs),
        },
    )
    write_checksums(args.output, ("execution_results.jsonl", "run_manifest.json"))
    print(f"wrote {len(outputs)} execution records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
