#!/usr/bin/env python3
"""Map already-inspected v7 choices to physically checked development outcomes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_jsonl, write_checksums, write_json  # noqa: E402
from experiments.analyze_online_group_controls import verify_checksums  # noqa: E402
from experiments.summarize_independent_v7_confirmation import sha256  # noqa: E402


def key(row: dict) -> tuple[str, tuple[str, ...]]:
    return str(row["event_id"]), tuple(sorted(map(str, row["selected_memory_ids"])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executions", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    verify_checksums(args.executions)
    outcomes = {}
    for row in load_jsonl(args.executions / "execution_results.jsonl"):
        if (
            row["reset_verified"] is not True
            or row["checker_independent"] is not True
            or row["execution_error"] is not None
            or row["checker_error"] is not None
        ):
            raise ValueError(f"unverified execution: {row['event_id']}")
        candidate = key(row)
        if candidate in outcomes:
            raise ValueError(f"duplicate checked outcome: {candidate[0]}")
        outcomes[candidate] = bool(row["checker"]["goal_satisfied"])
    selections = load_jsonl(args.selections)
    if len({str(row["event_id"]) for row in selections}) != len(selections):
        raise ValueError("duplicate method selection")
    if len(selections) != 128 or len(outcomes) != 256:
        raise ValueError("expected 128 complete v7 choices and 256 checked candidates")
    missing = [key(row) for row in selections if key(row) not in outcomes]
    if missing:
        raise ValueError(f"unmatched checked selections: {missing[:3]}")
    successes = sum(outcomes[key(row)] for row in selections)
    args.output.mkdir(parents=True)
    write_json(
        args.output / "result.json",
        {
            "status": "already_inspected_v7_development_only",
            "claim_boundary": "This is not untouched confirmation of a redesigned method.",
            "events": len(selections),
            "verified_successes": successes,
            "verified_success_rate": successes / len(selections),
            "selection_sha256": sha256(args.selections),
            "checked_executions_sha256": sha256(args.executions / "execution_results.jsonl"),
        },
    )
    write_checksums(args.output, ("result.json",))
    print(f"inspected v7 development choices: {successes}/{len(selections)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
