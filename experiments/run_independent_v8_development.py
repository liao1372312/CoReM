#!/usr/bin/env python3
"""Execute prospective v8 choices with live checking before service teardown.

This harness is development-only until a new task set and methods are frozen.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_jsonl, write_checksums, write_json, write_jsonl  # noqa: E402
from corem.v8_candidates import validate_candidate  # noqa: E402
from experiments.independent_v8_checker import IndependentV8Checker  # noqa: E402
from experiments.independent_v8_runner import IndependentV8Runner  # noqa: E402
from experiments.summarize_independent_v7_confirmation import sha256  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-id", help="Execute only one development incident.")
    parser.add_argument("--retrieval-budget", type=int, default=160)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.retrieval_budget < 1:
        parser.error("retrieval budget must be positive")

    events_rows = load_jsonl(args.dataset / "events.jsonl")
    events = {str(row["event_id"]): row for row in events_rows}
    memories_rows = load_jsonl(args.dataset / "memories.jsonl")
    memories = {str(row["memory_id"]): row for row in memories_rows}
    edges_rows = load_jsonl(args.dataset / "dependency_hyperedges.jsonl")
    edges = {str(row["edge_id"]): row for row in edges_rows}
    if (
        len(events) != len(events_rows)
        or len(memories) != len(memories_rows)
        or len(edges) != len(edges_rows)
    ):
        raise ValueError("duplicate event, memory, or dependency edge")
    edges_by_members: dict[frozenset[str], str] = {}
    for edge in edges_rows:
        key = frozenset(map(str, edge["members"]))
        if key in edges_by_members:
            raise ValueError("multiple edges with the same selected memory set")
        edges_by_members[key] = str(edge["edge_id"])
    selection_rows = load_jsonl(args.selections)
    if args.event_id:
        selection_rows = [
            row for row in selection_rows if row["event_id"] == args.event_id
        ]
    if not selection_rows:
        raise ValueError("no selected development events")
    runner = IndependentV8Runner()
    checker = IndependentV8Checker()
    outputs = []
    for selection in selection_rows:
        event_id = str(selection["event_id"])
        event = events.get(event_id)
        if event is None:
            raise ValueError(f"selection references unknown event: {event_id}")
        selected = selection.get("selected_memory_ids")
        if not isinstance(selected, list) or any(
            not isinstance(memory_id, str) for memory_id in selected
        ):
            raise ValueError(f"invalid selected memories: {event_id}")
        key = frozenset(selected)
        if len(key) != len(selected) or key not in edges_by_members:
            raise ValueError(f"selection does not match one complete edge: {event_id}")
        edge_id = edges_by_members[key]
        validate_candidate(
            {
                "event_id": event_id,
                "candidate_edge_id": edge_id,
                "selected_memory_ids": selected,
            },
            event,
            memories,
            edges,
        )
        cost = sum(int(memories[memory_id]["token_cost"]) for memory_id in selected)
        if cost > args.retrieval_budget:
            raise ValueError(f"selection exceeds retrieval budget: {event_id}")

        started = time.perf_counter()
        reset_verified = False
        execution_error = None
        checker_error = None
        snapshot_before = None
        snapshot_after = None
        execution: dict[str, Any] | None = None
        checker_output: dict[str, Any] | None = None
        try:
            reset = runner.reset(event)
            reset_verified = reset["reset_verified"] is True
            snapshot_before = reset["snapshot_id"]
            if not reset_verified:
                raise RuntimeError("runner reset was not verified")
            execution = dict(runner.execute(event, [memories[memory_id] for memory_id in selected]))
            try:
                checker_output = checker.check(event, execution, runner.checker_context())
            except Exception as error:  # noqa: BLE001 -- checker boundary
                checker_error = f"{type(error).__name__}: {error}"
        except Exception as error:  # noqa: BLE001 -- runner boundary
            execution_error = f"{type(error).__name__}: {error}"
        finally:
            if reset_verified:
                try:
                    snapshot_after = runner.snapshot(event)
                except Exception as error:  # noqa: BLE001 -- teardown boundary
                    execution_error = f"snapshot {type(error).__name__}: {error}"
            else:
                runner._cleanup()
        outputs.append(
            {
                "run_id": selection.get("run_id", f"v8-dev:{event_id}"),
                "event_id": event_id,
                "method": selection.get("method", "development_choice"),
                "selected_memory_ids": sorted(selected),
                "token_cost": cost,
                "reset_verified": reset_verified,
                "snapshot_before": snapshot_before,
                "snapshot_after": snapshot_after,
                "checker_independent": True,
                "checker_before_teardown": True,
                "execution_error": execution_error,
                "checker_error": checker_error,
                "execution": execution,
                "checker": checker_output,
                "wall_latency_ms": (time.perf_counter() - started) * 1000,
            }
        )
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "execution_results.jsonl", outputs)
    write_json(
        args.output / "run_manifest.json",
        {
            "status": "development_only_execution",
            "claim_boundary": "No new v8 task set or method selections are frozen.",
            "dataset": str(args.dataset.resolve()),
            "dataset_files_sha256": {
                name: sha256(args.dataset / name)
                for name in (
                    "events.jsonl",
                    "memories.jsonl",
                    "dependency_hyperedges.jsonl",
                )
            },
            "selections_sha256": sha256(args.selections),
            "retrieval_budget": args.retrieval_budget,
            "rows": len(outputs),
            "runner_errors": sum(row["execution_error"] is not None for row in outputs),
            "checker_errors": sum(row["checker_error"] is not None for row in outputs),
            "verified_successes": sum(
                bool(
                    row["reset_verified"]
                    and row["execution_error"] is None
                    and row["checker_error"] is None
                    and row["checker"]
                    and row["checker"].get("goal_satisfied")
                )
                for row in outputs
            ),
        },
    )
    write_checksums(args.output, ("execution_results.jsonl", "run_manifest.json"))
    print(f"executed {len(outputs)} development choices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
