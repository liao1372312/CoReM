#!/usr/bin/env python3
"""Report known-pilot score attribution; never treat it as a sealed test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_jsonl, sha256_file, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--executions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing pilot audit")
    scores = load_jsonl(args.scores)
    executions = load_jsonl(args.executions)
    outcomes = {
        (str(row["event_id"]), tuple(sorted(row["selected_memory_ids"]))): bool(
            row["checker"]["goal_satisfied"] is True
            and all(item.get("passed") is True for item in row["checker"]["binding_checks"])
            and all(
                item.get("passed") is True
                for item in row["checker"]["communication_checks"]
            )
            and not row.get("execution_error")
            and not row.get("checker_error")
        )
        for row in executions
    }
    grouped: dict[str, list[dict]] = {}
    for row in scores:
        grouped.setdefault(str(row["event_id"]), []).append(row)
    rows = []
    for event_id, candidates in sorted(grouped.items()):
        if len(candidates) != 2:
            raise ValueError(
                "pilot audit expects exactly two complete candidates per event"
            )
        if any(
            (event_id, tuple(sorted(row["selected_memory_ids"]))) not in outcomes
            for row in candidates
        ):
            raise ValueError(f"missing exact execution for {event_id}")
        winners = {}
        lexical_field = (
            "bm25_raw" if all("bm25_raw" in row for row in candidates) else "bm25_max"
        )
        for name, field in (
            ("frozen_ensemble", "score"),
            ("neural_only", "ensemble_probability"),
            ("bm25_only", lexical_field),
        ):
            ordered = sorted(
                candidates, key=lambda row: float(row[field]), reverse=True
            )
            if float(ordered[0][field]) == float(ordered[1][field]):
                winners[name] = {"success": None, "margin": 0.0}
            else:
                key = (event_id, tuple(sorted(ordered[0]["selected_memory_ids"])))
                winners[name] = {
                    "success": outcomes[key],
                    "margin": float(ordered[0][field]) - float(ordered[1][field]),
                }
        seed_scores = [row.get("seed_probabilities", []) for row in candidates]
        if all(isinstance(item, list) and item for item in seed_scores):
            seed_ids = {int(item["seed"]) for item in seed_scores[0]}
            if any(
                {int(item["seed"]) for item in item_list} != seed_ids
                for item_list in seed_scores
            ):
                raise ValueError(f"unequal model seeds for {event_id}")
            for seed in sorted(seed_ids):
                values = [
                    next(
                        float(item["probability"])
                        for item in item_list
                        if int(item["seed"]) == seed
                    )
                    for item_list in seed_scores
                ]
                name = f"neural_seed_{seed}"
                if values[0] == values[1]:
                    winners[name] = {"success": None, "margin": 0.0}
                else:
                    winning_index = 0 if values[0] > values[1] else 1
                    key = (
                        event_id,
                        tuple(sorted(candidates[winning_index]["selected_memory_ids"])),
                    )
                    winners[name] = {
                        "success": outcomes[key],
                        "margin": abs(values[0] - values[1]),
                    }
        rows.append({"event_id": event_id, "winners": winners})
    if not rows:
        raise ValueError("score audit has no events")
    summary = {
        name: {
            "successes": sum(row["winners"][name]["success"] is True for row in rows),
            "ties": sum(row["winners"][name]["success"] is None for row in rows),
            "events": len(rows),
        }
        for name in rows[0]["winners"]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output,
        {
            "status": (
                f"inspected {len(rows)}-event engineering/development diagnostic; "
                "not confirmatory"
            ),
            "source_sha256": {
                "scores": sha256_file(args.scores),
                "executions": sha256_file(args.executions),
            },
            "summary": summary,
            "events": rows,
        },
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
