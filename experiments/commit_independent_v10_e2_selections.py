#!/usr/bin/env python3
"""Commit all eight v10 E2 selections before outcome execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import load_jsonl, sha256_file, write_json  # noqa: E402

CELLS = {
    "graph_complement": ("hgt_corem", 0.9),
    "graph_additive": ("hgt_additive", 0.9),
    "text_complement": ("text_complement", 0.9),
    "text_additive": ("text_additive", 0.9),
    "graph_complement_no_lexical": ("hgt_corem", 1.0),
    "graph_additive_no_lexical": ("hgt_additive", 1.0),
    "text_complement_no_lexical": ("text_complement", 1.0),
    "text_additive_no_lexical": ("text_additive", 1.0),
}


def selected(row: dict) -> tuple[str, ...]:
    return tuple(sorted(map(str, row["selected_memory_ids"])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selections-root", type=Path, required=True)
    parser.add_argument("--structural-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a prior commitment")

    dataset_manifest = json.loads(
        (args.dataset / "manifest.json").read_text(encoding="utf-8")
    )
    if dataset_manifest.get("status") != "prospective_outcome_hidden_confirmation":
        raise ValueError("dataset is not an outcome-hidden confirmation set")
    audit = json.loads(args.structural_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "structurally_ready_for_physical_gate" or audit.get(
        "issues"
    ):
        raise ValueError("structural audit is not clean")

    query_path = args.dataset / "query_only_events.jsonl"
    candidate_path = args.dataset / "candidate_selections.jsonl"
    event_ids = {str(row["event_id"]) for row in load_jsonl(query_path)}
    if len(event_ids) != 128:
        raise ValueError("expected 128 unique query-only events")
    allowed: dict[str, set[tuple[str, ...]]] = {event_id: set() for event_id in event_ids}
    for row in load_jsonl(candidate_path):
        event_id = str(row["event_id"])
        if event_id not in allowed:
            raise ValueError(f"candidate has unknown event: {event_id}")
        allowed[event_id].add(selected(row))
    if any(len(candidates) != 2 for candidates in allowed.values()):
        raise ValueError("every event must expose exactly two candidates")

    commitments = {}
    for method, (state_variant, neural_weight) in CELLS.items():
        directory = args.selections_root / f"results_independent_v10_e2_{method}_selections"
        selections_path = directory / "selections.jsonl"
        scores_path = directory / "candidate_scores.jsonl"
        manifest_path = directory / "manifest.json"
        rows = load_jsonl(selections_path)
        scores = load_jsonl(scores_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        observed_ids = {str(row["event_id"]) for row in rows}
        if len(rows) != 128 or observed_ids != event_ids or len(scores) != 256:
            raise ValueError(f"{method}: incomplete selections or scores")
        if any(
            selected(row) not in allowed[str(row["event_id"])] for row in rows
        ):
            raise ValueError(f"{method}: selection left the candidate pool")
        if (
            manifest.get("outcomes_loaded") is not False
            or manifest.get("label_blind") is not True
            or manifest.get("label_free_candidate_input_enforced") is not True
            or manifest.get("state_variant") != state_variant
            or float(manifest.get("neural_weight", -1.0)) != neural_weight
            or manifest.get("relation_vocabulary_policy") != "collapse_to_training"
            or manifest.get("tie_break_policy") != "seeded_hash"
            or manifest.get("event_view", {}).get("sha256") != sha256_file(query_path)
        ):
            raise ValueError(f"{method}: selector deviates from frozen contract")
        commitments[method] = {
            "state_variant": state_variant,
            "neural_weight": neural_weight,
            "selection_rows": len(rows),
            "candidate_score_rows": len(scores),
            "manifest_sha256": sha256_file(manifest_path),
            "selections_sha256": sha256_file(selections_path),
            "candidate_scores_sha256": sha256_file(scores_path),
            "model_states": manifest["model_states"],
        }

    output = {
        "schema_version": "corem-independent-v10-e2-commitment-v1",
        "status": "all_factorial_cells_committed_before_outcome_execution",
        "claim_boundary": (
            "Selections are frozen on a query-only, label-free view. The tasks reuse "
            "known families but contain new incidents and hidden candidate-side assignments."
        ),
        "dataset_files_sha256": {
            path.name: sha256_file(path)
            for path in sorted(args.dataset.iterdir())
            if path.is_file()
        },
        "structural_audit_sha256": sha256_file(args.structural_audit),
        "selector_source_sha256": sha256_file(
            ROOT / "experiments/generate_public_multiservice_hgt_ensemble_selections.py"
        ),
        "graph_adapter_source_sha256": sha256_file(
            ROOT / "experiments/generate_public_workflow_selections.py"
        ),
        "events": len(event_ids),
        "cells": commitments,
        "outcomes_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, output)
    print("committed eight v10 E2 cells before outcome execution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
