#!/usr/bin/env python3
"""Hash-commit all label-blind v9 method choices before outcome mapping."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import load_jsonl, sha256_file, write_json  # noqa: E402
from experiments.export_v9_selector_view import SELECTOR_FILES  # noqa: E402

METHODS = (
    "hybrid", "bm25_text_and_plan", "bm25_text", "literal_overlap",
    "neural_only", "base_nli_hybrid", "base_nli_only", "role_literal",
)
INPUT_KEYS = {
    "query_only_events.jsonl": "events", "candidate_selections.jsonl": "candidates",
    "memories.jsonl": "memories", "dependency_hyperedges.jsonl": "hyperedges",
}


def _selected(row: dict) -> tuple[str, ...]:
    return tuple(sorted(map(str, row["selected_memory_ids"])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bm25-plan-dir", type=Path, required=True)
    parser.add_argument("--bm25-text-dir", type=Path, required=True)
    parser.add_argument("--literal-dir", type=Path, required=True)
    parser.add_argument("--role-literal-dir", type=Path, required=True)
    parser.add_argument("--hybrid-dir", type=Path, required=True)
    parser.add_argument("--base-nli-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an earlier commitment")
    freeze = json.loads(args.freeze.read_text())
    if freeze.get("status") != "frozen_unscored_v9_candidate":
        raise ValueError("missing prospective v9 freeze")
    if freeze["dataset_files_sha256"] != {
        name: sha256_file(args.dataset / name) for name in freeze["dataset_files_sha256"]
    }:
        raise ValueError("public dataset changed after freeze")
    selector_view = Path(freeze["selector_view_path"])
    if (
        sha256_file(selector_view / "manifest.json") != freeze["selector_view_manifest_sha256"]
        or {path.name for path in selector_view.iterdir()}
        != {*SELECTOR_FILES, "manifest.json", "checksums.sha256"}
        or any(sha256_file(selector_view / name) != freeze["dataset_files_sha256"][name]
               for name in SELECTOR_FILES)
    ):
        raise ValueError("frozen label-blind selector view changed")
    for name, digest in freeze["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"source changed after freeze: {name}")
    query_rows = load_jsonl(args.dataset / "query_only_events.jsonl")
    ids = {str(row["event_id"]) for row in query_rows}
    allowed: dict[str, dict[tuple[str, ...], str]] = {event_id: {} for event_id in ids}
    for row in load_jsonl(args.dataset / "candidate_selections.jsonl"):
        allowed[str(row["event_id"])][_selected(row)] = str(row["candidate_edge_id"])
    if len(query_rows) != 128 or len(ids) != 128 or any(len(allowed[event_id]) != 2 for event_id in ids):
        raise ValueError("frozen candidate pool is incomplete")
    sources = {
        "hybrid": (args.hybrid_dir, "selections.jsonl"),
        "neural_only": (args.hybrid_dir, "neural_only_selections.jsonl"),
        "base_nli_hybrid": (args.base_nli_dir, "selections.jsonl"),
        "base_nli_only": (args.base_nli_dir, "neural_only_selections.jsonl"),
        "bm25_text_and_plan": (args.bm25_plan_dir, "selections.jsonl"),
        "bm25_text": (args.bm25_text_dir, "selections.jsonl"),
        "literal_overlap": (args.literal_dir, "selections.jsonl"),
        "role_literal": (args.role_literal_dir, "selections.jsonl"),
    }
    bm25_score_sha = sha256_file(args.bm25_plan_dir / "candidate_scores.jsonl")
    committed = {}
    for method in METHODS:
        directory, filename = sources[method]
        selection_path = directory / filename
        manifest_path = directory / "manifest.json"
        score_path = directory / "candidate_scores.jsonl"
        selections = load_jsonl(selection_path)
        observed_ids = [str(row["event_id"]) for row in selections]
        if len(selections) != 128 or len(set(observed_ids)) != 128 or set(observed_ids) != ids:
            raise ValueError(f"{method}: selections are incomplete")
        for row in selections:
            candidate = _selected(row)
            if candidate not in allowed[str(row["event_id"])]:
                raise ValueError(f"{method}: choice outside frozen candidate pool")
            if method in {"hybrid", "neural_only", "base_nli_hybrid", "base_nli_only", "role_literal"} and row.get("candidate_edge_id") != allowed[str(row["event_id"])][candidate]:
                raise ValueError(f"{method}: candidate edge does not match memory bundle")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("outcomes_loaded") is not False or manifest.get("events") != 128:
            raise ValueError(f"{method}: manifest is not label-blind and complete")
        if method in {"hybrid", "neural_only", "base_nli_hybrid", "base_nli_only"}:
            expected_model_sha = (
                freeze["base_nli_model_files_sha256"]["model.safetensors"]
                if method.startswith("base_nli_") else freeze["model_files_sha256"]["model.safetensors"]
            )
            if (
                manifest.get("status") != "label_blind_frozen_rule_selection"
                or manifest.get("model_weights_sha256") != expected_model_sha
                or manifest.get("bm25_scores_sha256") != bm25_score_sha
                or manifest.get("tie_tolerance") != freeze["rule"]["tie_tolerance"]
                or manifest.get("max_length") != freeze["rule"]["max_length"]
                or manifest.get("seed") != freeze["rule"]["seed"]
                or manifest.get("token_budget") != freeze["rule"]["token_budget"]
                or manifest.get("neural_tie_events", -1) + manifest.get("bm25_non_tie_events", -1) != 128
            ):
                raise ValueError(f"{method}: neural hybrid rule differs from freeze")
            for filename_input in INPUT_KEYS:
                if manifest.get("input_sha256", {}).get(filename_input) != freeze["dataset_files_sha256"][filename_input]:
                    raise ValueError(f"{method}: selector input changed: {filename_input}")
        elif method == "role_literal":
            if (
                manifest.get("status") != "label_blind_role_literal_control"
                or manifest.get("bm25_scores_sha256") != bm25_score_sha
                or manifest.get("seed") != freeze["rule"]["seed"]
                or manifest.get("token_budget") != freeze["rule"]["token_budget"]
            ):
                raise ValueError("role/literal control differs from freeze")
            for filename_input in INPUT_KEYS:
                if manifest.get("input_sha256", {}).get(filename_input) != freeze["dataset_files_sha256"][filename_input]:
                    raise ValueError(f"role/literal selector input changed: {filename_input}")
        else:
            if method.startswith("bm25_") and (
                manifest.get("aggregation") != "sum"
                or manifest.get("document_view") != ("text_and_plan" if method == "bm25_text_and_plan" else "text")
            ):
                raise ValueError(f"{method}: BM25 setting differs from freeze")
            if manifest.get("seed") != freeze["rule"]["seed"] or manifest.get("token_budget") != freeze["rule"]["token_budget"]:
                raise ValueError(f"{method}: budget or seed differs from freeze")
            for filename_input, manifest_key in INPUT_KEYS.items():
                if manifest.get("input_sha256", {}).get(manifest_key) != freeze["dataset_files_sha256"][filename_input]:
                    raise ValueError(f"{method}: selector input changed: {filename_input}")
        committed[method] = {
            "selection_path": str(selection_path.resolve()),
            "selections_sha256": sha256_file(selection_path),
            "candidate_scores_sha256": sha256_file(score_path),
            "manifest_sha256": sha256_file(manifest_path),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {
        "status": "all_v9_methods_committed_before_outcome_mapping",
        "created_utc": datetime.now(UTC).isoformat(),
        "freeze_sha256": sha256_file(args.freeze), "outcomes_loaded": False,
        "methods": committed,
    })
    print("committed eight label-blind v9 choices across 128 events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
