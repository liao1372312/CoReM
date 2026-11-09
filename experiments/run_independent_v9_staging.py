#!/usr/bin/env python3
"""Physically check prospective v9 selections before any model scoring.

Staging output is a task-validity gate, not an independent method result.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import load_jsonl, sha256_file, write_checksums, write_json, write_jsonl  # noqa: E402
from corem.v8_candidates import validate_candidate  # noqa: E402
from experiments.independent_v9_checker import IndependentV9Checker  # noqa: E402
from experiments.independent_v9_runner import IndependentV9Runner, VERSIONS  # noqa: E402


def _rows_by_id(path: Path, id_name: str) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    indexed = {str(row[id_name]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate {id_name} in {path}")
    return indexed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-id")
    parser.add_argument("--retrieval-budget", type=int, default=192)
    args = parser.parse_args()
    if args.output.exists() or args.retrieval_budget < 1:
        parser.error("output must be new and retrieval budget positive")
    events = _rows_by_id(args.dataset / "events.jsonl", "event_id")
    memories = _rows_by_id(args.dataset / "memories.jsonl", "memory_id")
    edges = _rows_by_id(args.dataset / "dependency_hyperedges.jsonl", "edge_id")
    selection_rows = load_jsonl(args.selections)
    if args.event_id:
        selection_rows = [row for row in selection_rows if row["event_id"] == args.event_id]
    if not selection_rows:
        raise ValueError("no staging selections")
    outputs = []
    runner = IndependentV9Runner()
    checker = IndependentV9Checker()
    try:
        for selection in selection_rows:
            event_id = str(selection["event_id"])
            event = events.get(event_id)
            if event is None or VERSIONS.get(str(event["service_id"])) != event["version_id"]:
                raise ValueError(f"unknown or unpinned v9 event: {event_id}")
            selected = selection.get("selected_memory_ids")
            if not isinstance(selected, list) or any(not isinstance(item, str) for item in selected):
                raise ValueError(f"invalid selected IDs: {event_id}")
            candidate = {
                "event_id": event_id,
                "candidate_edge_id": str(selection["candidate_edge_id"]),
                "selected_memory_ids": selected,
            }
            validate_candidate(candidate, event, memories, edges)
            cost = sum(int(memories[item]["token_cost"]) for item in selected)
            if cost > args.retrieval_budget:
                raise ValueError(f"selection over budget: {event_id}")
            started = time.perf_counter()
            reset_ok = False
            execution = None
            checked = None
            execution_error = None
            checker_error = None
            before = None
            after = None
            try:
                reset = runner.reset(event)
                reset_ok = reset["reset_verified"] is True
                before = reset["snapshot_id"]
                if not reset_ok:
                    raise RuntimeError("isolated reset not verified")
                execution = dict(runner.execute(event, [memories[item] for item in selected]))
                try:
                    checked = checker.check(event, execution, runner.checker_context())
                except Exception as error:  # noqa: BLE001 -- independent checker boundary
                    checker_error = f"{type(error).__name__}: {error}"
            except Exception as error:  # noqa: BLE001 -- service execution boundary
                execution_error = f"{type(error).__name__}: {error}"
            finally:
                if reset_ok:
                    try:
                        after = runner.snapshot(event)
                    except Exception as error:  # noqa: BLE001 -- teardown boundary
                        execution_error = f"snapshot {type(error).__name__}: {error}"
                else:
                    runner._cleanup()
            outputs.append({
                "run_id": selection.get("run_id", f"v9-staging:{event_id}"),
                "event_id": event_id,
                "method": selection.get("method", "staging_choice"),
                "candidate_edge_id": candidate["candidate_edge_id"],
                "selected_memory_ids": sorted(selected),
                "token_cost": cost,
                "reset_verified": reset_ok,
                "snapshot_before": before,
                "snapshot_after": after,
                "checker_independent": True,
                "checker_before_teardown": True,
                "execution_error": execution_error,
                "checker_error": checker_error,
                "execution": execution,
                "checker": checked,
                "wall_latency_ms": (time.perf_counter() - started) * 1000,
            })
    finally:
        runner._cleanup()
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "execution_results.jsonl", outputs)
    write_json(args.output / "run_manifest.json", {
        "status": "prospective_v9_physical_staging_only",
        "claim_boundary": "No model or baseline may use these outcomes to revise a sealed task set; this is not a method result.",
        "dataset": str(args.dataset.resolve()),
        "dataset_files_sha256": {
            name: sha256_file(args.dataset / name)
            for name in ("events.jsonl", "memories.jsonl", "dependency_hyperedges.jsonl")
        },
        "selections_sha256": sha256_file(args.selections),
        "runner_sha256": sha256_file(ROOT / "experiments/independent_v9_runner.py"),
        "checker_sha256": sha256_file(ROOT / "experiments/independent_v9_checker.py"),
        "retrieval_budget": args.retrieval_budget,
        "rows": len(outputs),
        "runner_errors": sum(row["execution_error"] is not None for row in outputs),
        "checker_errors": sum(row["checker_error"] is not None for row in outputs),
        "verified_successes": sum(bool(row["checker"] and row["checker"].get("goal_satisfied")) for row in outputs),
    })
    write_checksums(args.output, ("execution_results.jsonl", "run_manifest.json"))
    print(f"checked {len(outputs)} prospective v9 selections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
