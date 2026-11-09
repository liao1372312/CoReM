#!/usr/bin/env python3
"""Commit all label-blind v7 selections before opening outcome mappings."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import (  # noqa: E402
    load_jsonl,
    sha256_file,
    write_json,
)

METHODS = {
    "graph_complement": "graph_complement",
    "graph_additive": "graph_additive",
    "text_complement": "text_complement",
    "text_additive": "text_additive",
    "candidate_bm25_max": "candidate_bm25_max",
    "version_bm25_closure": "version_bm25_bundle",
    "pure_neural": "pure_neural",
    "candidate_bm25_sum": "candidate_bm25_sum",
}
FACTORIAL = {
    "graph_complement": "hgt_corem",
    "graph_additive": "hgt_additive",
    "text_complement": "text_complement",
    "text_additive": "text_additive",
}


def _verify_entry(entry: dict[str, str]) -> None:
    path = Path(entry["path"])
    if not path.is_file() or sha256_file(path) != entry["sha256"]:
        raise ValueError(f"frozen input changed: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    for name in METHODS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a previous selection commitment")
    frozen = json.loads(args.freeze.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_method_selection":
        parser.error("invalid preselection freeze")
    for group in ("dataset", "selector_inputs", "source_modules", "runtime_assets"):
        for entry in frozen[group].values():
            _verify_entry(entry)
    for group in ("model_states", "model_training_manifests"):
        for entry in frozen[group]:
            _verify_entry(entry)
    _verify_entry(frozen["structural_audit"])
    _verify_entry(frozen["oracle_analysis"])
    _verify_entry(frozen["oracle_executions"])
    query_path = Path(frozen["selector_inputs"]["query_only_events.jsonl"]["path"])
    expected_ids = {str(row["event_id"]) for row in load_jsonl(query_path)}
    candidate_path = Path(
        frozen["selector_inputs"]["candidate_selections.jsonl"]["path"]
    )
    allowed_candidates: dict[str, set[tuple[str, ...]]] = {}
    for row in load_jsonl(candidate_path):
        allowed_candidates.setdefault(str(row["event_id"]), set()).add(
            tuple(sorted(map(str, row["selected_memory_ids"])))
        )
    if len(expected_ids) < 128:
        parser.error("frozen query view is incomplete")
    commitments = {}
    for name, expected_method in METHODS.items():
        directory = getattr(args, name)
        selection_path = directory / "selections.jsonl"
        manifest_path = directory / "manifest.json"
        rows = load_jsonl(selection_path)
        observed_ids = {str(row.get("event_id")) for row in rows}
        if (
            len(rows) != len(expected_ids)
            or observed_ids != expected_ids
            or any(row.get("method") != expected_method for row in rows)
        ):
            parser.error(f"{name}: incomplete, duplicate, or mislabeled selections")
        if name != "version_bm25_closure" and any(
            tuple(sorted(map(str, row.get("selected_memory_ids", []))))
            not in allowed_candidates[str(row["event_id"])]
            for row in rows
        ):
            parser.error(f"{name}: selection left the frozen candidate pool")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("outcomes_loaded") is not False:
            parser.error(f"{name}: selector manifest does not certify label blindness")
        if name in FACTORIAL and (
            manifest.get("state_variant") != FACTORIAL[name]
            or manifest.get("neural_weight") != 0.9
            or manifest.get("lexical_aggregation") != "max"
            or manifest.get("relation_vocabulary_policy") != "collapse_to_training"
            or manifest.get("tie_break_policy") != "seeded_hash"
            or manifest.get("event_view", {}).get("sha256")
            != frozen["selector_inputs"]["query_only_events.jsonl"]["sha256"]
        ):
            parser.error(f"{name}: deviates from frozen factorial contract")
        if name == "pure_neural" and (
            manifest.get("state_variant") != "hgt_corem"
            or manifest.get("neural_weight") != 1.0
            or manifest.get("relation_vocabulary_policy") != "collapse_to_training"
            or manifest.get("event_view", {}).get("sha256")
            != frozen["selector_inputs"]["query_only_events.jsonl"]["sha256"]
        ):
            parser.error("pure-neural diagnostic differs from frozen contract")
        if name.startswith("candidate_bm25_") and (
            manifest.get("aggregation") != name.removeprefix("candidate_bm25_")
            or manifest.get("token_budget") != 256
            or manifest.get("query_view_sha256")
            != frozen["selector_inputs"]["query_only_events.jsonl"]["sha256"]
            or manifest.get("candidate_input_sha256")
            != frozen["selector_inputs"]["candidate_selections.jsonl"]["sha256"]
        ):
            parser.error(f"{name}: wrong lexical aggregation or budget")
        if name == "version_bm25_closure" and (
            manifest.get("token_budget") != 256
            or manifest.get("event_view") is None
            or manifest["event_view"].get("sha256")
            != frozen["selector_inputs"]["query_only_events.jsonl"]["sha256"]
        ):
            parser.error("closure baseline lacks frozen budget or query-only input")
        commitments[name] = {
            "selections": {
                "path": str(selection_path.resolve()),
                "sha256": sha256_file(selection_path),
            },
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
            },
            "rows": len(rows),
        }
    write_json(
        args.output,
        {
            "schema_version": "corem-v7-selection-commitment-v1",
            "status": "all_methods_committed_before_outcome_mapping",
            "created_utc": datetime.now(UTC).isoformat(),
            "preselection_freeze": {
                "path": str(args.freeze.resolve()),
                "sha256": sha256_file(args.freeze),
            },
            "methods": commitments,
            "test_incidents": len(expected_ids),
            "outcomes_loaded": False,
        },
    )
    print(
        f"committed {len(commitments)} complete methods over {len(expected_ids)} events"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
