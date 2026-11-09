#!/usr/bin/env python3
"""Post-hoc exact-value/role lexical diagnostic on inspected v8 tasks.

This baseline was designed after seeing v8 outcomes. Its score is only a
benchmark-shortcut audit and is not a precommitted confirmation control.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import load_jsonl, sha256_file, write_checksums, write_json, write_jsonl  # noqa: E402
from experiments.develop_v8_delta_pairwise_model import _selected  # noqa: E402


def _scalar_differences(left: Any, right: Any) -> list[tuple[Any, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        return [
            pair
            for key in sorted(set(left) | set(right))
            for pair in _scalar_differences(left.get(key), right.get(key))
        ]
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [
            pair
            for a, b in zip(left, right)
            for pair in _scalar_differences(a, b)
        ]
    if left == right:
        return []
    return [(left, right)]


def _path_differences(first: str, second: str) -> list[tuple[str, str]]:
    left = first.split("/")
    right = second.split("/")
    if len(left) != len(right):
        return [(first, second)]
    return [(a, b) for a, b in zip(left, right) if a != b]


def _positions(query: str, value: Any) -> list[int]:
    if value is None or isinstance(value, bool):
        return []
    literal = str(value).casefold()
    if not literal:
        return []
    pattern = rf"(?<![a-z0-9_]){re.escape(literal)}(?![a-z0-9_])"
    return [match.start() for match in re.finditer(pattern, query)]


def _role_score(query: str, plan: dict[str, Any], value: Any) -> float:
    hits = _positions(query, value)
    if not hits:
        return 0.0
    score = 1.0 + math.log1p(len(str(value)))
    resource = str(plan.get("path", "")).split("/")[-1]
    resource_hits = _positions(query, resource)
    if resource_hits and resource != str(value):
        gap = min(abs(a - b) for a in hits for b in resource_hits)
        score += 6.0 / (1.0 + gap)
    return score


def role_literal_scores(
    query: str,
    first: dict[str, Any],
    second: dict[str, Any],
    memories: dict[str, dict[str, Any]],
) -> tuple[float, float]:
    """Score only exact literals in the pair's changed plan slots."""
    query = query.casefold()

    def plans(candidate: dict[str, Any]) -> list[dict[str, Any]]:
        rows = [memories[memory_id] for memory_id in candidate["selected_memory_ids"]]
        steps = [row for row in rows if row.get("role") == "workflow_step"]
        return [row["plan"] for row in sorted(steps, key=lambda row: int(row["step_index"]))]

    left = plans(first)
    right = plans(second)
    if len(left) != len(right):
        raise ValueError("candidate workflow lengths differ")
    scores = [0.0, 0.0]
    for a, b in zip(left, right):
        if a == b:
            continue
        differences = _path_differences(str(a.get("path", "")), str(b.get("path", "")))
        differences += _scalar_differences(a.get("json", {}), b.get("json", {}))
        for left_value, right_value in differences:
            scores[0] += _role_score(query, a, left_value)
            scores[1] += _role_score(query, b, right_value)
    return scores[0], scores[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--oracle-selections", type=Path, required=True)
    parser.add_argument("--bm25-selections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a previous development result")
    events = {str(row["event_id"]): row for row in load_jsonl(args.dataset / "query_only_events.jsonl")}
    memories = {str(row["memory_id"]): row for row in load_jsonl(args.dataset / "memories.jsonl")}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(args.dataset / "candidate_selections.jsonl"):
        grouped[str(row["event_id"])].append(row)
    oracle_rows = load_jsonl(args.oracle_selections)
    oracle = {str(row["event_id"]): _selected(row) for row in oracle_rows}
    bm25_rows = load_jsonl(args.bm25_selections)
    bm25 = {str(row["event_id"]): _selected(row) for row in bm25_rows}
    if (
        len(events) != 128 or len(oracle_rows) != 128 or len(bm25_rows) != 128
        or set(grouped) != set(events) or set(oracle) != set(events) or set(bm25) != set(events)
        or any(len(rows) != 2 for rows in grouped.values())
    ):
        raise ValueError("incomplete v8 development candidate pool")
    selections = []
    by_family = Counter()
    for event_id in sorted(events):
        rows = grouped[event_id]
        values = role_literal_scores(str(events[event_id]["query"]), rows[0], rows[1], memories)
        ranked = []
        for row, score in zip(rows, values):
            selected = _selected(row)
            digest = hashlib.sha256(f"{args.seed}|{event_id}|{'|'.join(selected)}".encode()).hexdigest()
            ranked.append((score, digest, selected))
        chosen = max(ranked)
        by_family[event_id.split(":")[1]] += chosen[2] == oracle[event_id]
        selections.append({
            "event_id": event_id,
            "selected_memory_ids": list(chosen[2]),
            "score_margin": abs(values[0] - values[1]),
            "verified_success": int(chosen[2] == oracle[event_id]),
            "bm25_verified_success": int(bm25[event_id] == oracle[event_id]),
        })
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "selections.jsonl", selections)
    write_json(args.output / "result.json", {
        "status": "posthoc_v8_literal_role_diagnostic_only",
        "claim_boundary": "This control was designed after the v8 outcomes were inspected; it diagnoses template shortcuts but is not precommitted evidence.",
        "successes": sum(row["verified_success"] for row in selections),
        "bm25_successes": sum(row["bm25_verified_success"] for row in selections),
        "by_family_successes": dict(sorted(by_family.items())),
        "seed": args.seed,
        "script_sha256": sha256_file(Path(__file__)),
        "oracle_selections_sha256": sha256_file(args.oracle_selections),
        "bm25_selections_sha256": sha256_file(args.bm25_selections),
    })
    write_checksums(args.output, ("selections.jsonl", "result.json"))
    print(f"post-hoc role/literal diagnostic: {sum(row['verified_success'] for row in selections)}/128")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
