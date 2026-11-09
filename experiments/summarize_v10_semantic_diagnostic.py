#!/usr/bin/env python3
"""Audit post-outcome semantic-selector diagnostics on v10 E2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import (  # noqa: E402
    load_jsonl,
    sha256_file,
    write_checksums,
    write_json,
)


def key(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return str(row["event_id"]), tuple(sorted(map(str, row["selected_memory_ids"])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--bm25", type=Path, required=True)
    parser.add_argument("--trained-nli", type=Path, required=True)
    parser.add_argument("--base-nli", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite prior diagnostic")

    labels = {}
    gate_sources = {}
    for name, directory, expected in (
        ("oracle", args.oracle, 1),
        ("complete_negative", args.negative, 0),
    ):
        path = directory / "execution_results.jsonl"
        rows = load_jsonl(path)
        if len(rows) != 128:
            raise ValueError(f"{name}: incomplete physical gate")
        for row in rows:
            if (
                row.get("execution_error") is not None
                or row.get("checker_error") is not None
                or row.get("checker", {}).get("goal_satisfied") is not bool(expected)
                or key(row) in labels
            ):
                raise ValueError(f"{name}: invalid physical outcome")
            labels[key(row)] = expected
        gate_sources[name] = sha256_file(path)
    if len(labels) != 256:
        raise ValueError("diagnostic requires all 256 physical outcomes")

    methods = {}
    for name, directory, filename in (
        ("bm25_sum_text_and_plan", args.bm25, "selections.jsonl"),
        ("trained_difference_path_nli", args.trained_nli, "neural_only_selections.jsonl"),
        ("base_difference_path_nli", args.base_nli, "neural_only_selections.jsonl"),
    ):
        path = directory / filename
        rows = load_jsonl(path)
        if len(rows) != 128 or len({str(row["event_id"]) for row in rows}) != 128:
            raise ValueError(f"{name}: incomplete choices")
        outcomes = [labels[key(row)] for row in rows]
        methods[name] = {
            "verified_successes": sum(outcomes),
            "events": len(outcomes),
            "vsr": sum(outcomes) / len(outcomes),
            "selections_sha256": sha256_file(path),
            "manifest_sha256": sha256_file(directory / "manifest.json"),
        }

    args.output.mkdir(parents=True)
    write_json(
        args.output / "result.json",
        {
            "schema_version": "corem-v10-semantic-diagnostic-v1",
            "status": "complete_post_outcome_diagnostic",
            "claim_boundary": (
                "v10 outcomes were inspected before this diagnostic was run. "
                "The result may motivate a future frozen method but is not v10 "
                "confirmation evidence."
            ),
            "candidate_physical_executions": len(labels),
            "methods": methods,
            "physical_gate_sha256": gate_sources,
        },
    )
    write_checksums(args.output, ("result.json",))
    print({name: row["verified_successes"] for name, row in methods.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
