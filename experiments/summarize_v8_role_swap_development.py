#!/usr/bin/env python3
"""Summarize post-inspection role-swap development choices, never as confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import load_jsonl, sha256_file, write_json  # noqa: E402


def _selected(row: dict) -> tuple[str, ...]:
    return tuple(sorted(map(str, row["selected_memory_ids"])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--method", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a prior development summary")
    dataset_manifest = json.loads((args.dataset / "manifest.json").read_text())
    if dataset_manifest.get("status") != "post_v8_inspection_development_only":
        raise ValueError("role-swap dataset is not marked as development")
    events = {str(row["event_id"]): row for row in load_jsonl(args.dataset / "events.jsonl")}
    if len(events) != 24:
        raise ValueError("expected 24 development events")
    gate_files = {}
    oracle = {}
    for label, target_successes in (("oracle", 24), ("negative", 0)):
        directory = args.private / f"{label}_gate_v1"
        manifest = json.loads((directory / "run_manifest.json").read_text())
        rows = load_jsonl(directory / "execution_results.jsonl")
        if (
            manifest.get("rows") != 24
            or manifest.get("verified_successes") != target_successes
            or manifest.get("runner_errors") != 0
            or manifest.get("checker_errors") != 0
            or len(rows) != 24
            or {str(row["event_id"]) for row in rows} != set(events)
        ):
            raise ValueError(f"invalid live {label} gate")
        gate_files[label] = sha256_file(directory / "execution_results.jsonl")
        if label == "oracle":
            oracle = {str(row["event_id"]): _selected(row) for row in rows}
    allowed: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for row in load_jsonl(args.dataset / "candidate_selections.jsonl"):
        allowed[str(row["event_id"])].add(_selected(row))
    if any(len(allowed.get(event_id, ())) != 2 for event_id in events):
        raise ValueError("candidate pool is incomplete")
    methods = {}
    directories = {}
    for item in args.method:
        name, sep, directory = item.partition("=")
        if not sep or name in directories:
            parser.error(f"expected unique NAME=DIR, got {item}")
        directories[name] = Path(directory)
    for name, directory in directories.items():
        manifest = json.loads((directory / "manifest.json").read_text())
        selections = load_jsonl(directory / "selections.jsonl")
        scores = load_jsonl(directory / "candidate_scores.jsonl")
        if manifest.get("outcomes_loaded") is not False or len(selections) != 24:
            raise ValueError(f"{name}: selections may contain outcomes or be incomplete")
        by_family = Counter()
        success = 0
        for row in selections:
            event_id = str(row["event_id"])
            selected = _selected(row)
            if selected not in allowed[event_id]:
                raise ValueError(f"{name}: selection left candidate pool")
            correct = selected == oracle[event_id]
            success += correct
            by_family[str(events[event_id]["task_key"])] += correct
        score_groups: dict[str, list[float]] = defaultdict(list)
        for row in scores:
            score_groups[str(row["event_id"])].append(float(
                row.get("score", row.get("bm25_score", row.get("literal_score")))
            ))
        if set(score_groups) != set(events) or any(len(values) != 2 for values in score_groups.values()):
            raise ValueError(f"{name}: incomplete candidate scores")
        ties = sum(values[0] == values[1] for values in score_groups.values())
        methods[name] = {
            "successes": success,
            "events": 24,
            "exact_score_ties": ties,
            "by_family_successes": dict(sorted(by_family.items())),
            "selection_sha256": sha256_file(directory / "selections.jsonl"),
            "manifest_sha256": sha256_file(directory / "manifest.json"),
        }
    write_json(args.output, {
        "status": "post_v8_inspection_role_swap_development_only",
        "claim_boundary": "Tasks were designed after the v8 result and share the same two providers; this is not independent confirmation.",
        "dataset_manifest_sha256": sha256_file(args.dataset / "manifest.json"),
        "physical_gate_sha256": gate_files,
        "methods": methods,
    })
    print(json.dumps({name: {"successes": row["successes"], "exact_score_ties": row["exact_score_ties"]} for name, row in methods.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
