#!/usr/bin/env python3
"""Hash-commit all label-blind v8 choices before mapping them to outcomes."""

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

METHODS = (
    "path_reranker",
    "bm25_text_and_plan",
    "bm25_text",
    "literal_overlap",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--method", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a prior selection commitment")
    freeze = json.loads(args.freeze.read_text())
    if freeze.get("status") != "frozen_unscored_v8_candidate":
        raise ValueError("not a v8 unscored candidate freeze")
    if freeze["dataset_files_sha256"] != {
        name: sha256_file(args.dataset / name) for name in freeze["dataset_files_sha256"]
    }:
        raise ValueError("frozen dataset changed")
    for name, digest in freeze["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"frozen selector or checker source changed: {name}")
    expected_ids = {str(row["event_id"]) for row in load_jsonl(args.dataset / "query_only_events.jsonl")}
    allowed: dict[str, set[tuple[str, ...]]] = {}
    for row in load_jsonl(args.dataset / "candidate_selections.jsonl"):
        allowed.setdefault(str(row["event_id"]), set()).add(
            tuple(sorted(map(str, row["selected_memory_ids"])))
        )
    if len(expected_ids) != 128 or any(len(allowed.get(event_id, ())) != 2 for event_id in expected_ids):
        raise ValueError("frozen candidate pool is incomplete")
    directories: dict[str, Path] = {}
    for item in args.method:
        name, sep, path = item.partition("=")
        if not sep or name in directories:
            parser.error(f"expected unique NAME=DIR, got {item}")
        directories[name] = Path(path)
    if set(directories) != set(METHODS):
        parser.error(f"required four methods: {METHODS}")
    committed = {}
    for name in METHODS:
        directory = directories[name]
        selections_path = directory / "selections.jsonl"
        manifest_path = directory / "manifest.json"
        rows = load_jsonl(selections_path)
        ids = [str(row.get("event_id")) for row in rows]
        if len(rows) != 128 or len(set(ids)) != 128 or set(ids) != expected_ids:
            raise ValueError(f"{name}: incomplete or duplicated selections")
        for row in rows:
            candidate = tuple(sorted(map(str, row.get("selected_memory_ids", []))))
            if candidate not in allowed[str(row["event_id"])]:
                raise ValueError(f"{name}: choice outside frozen pool")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("outcomes_loaded") is not False:
            raise ValueError(f"{name}: outcomes may have been loaded")
        if name == "path_reranker":
            if (
                manifest.get("serialization") != "body_v1"
                or manifest.get("max_length") != 384
                or manifest.get("model_weights_sha256")
                != freeze["model_files_sha256"]["model.safetensors"]
            ):
                raise ValueError("path model differs from freeze")
        elif name.startswith("bm25_"):
            if (
                manifest.get("aggregation") != "sum"
                or manifest.get("document_view")
                != ("text_and_plan" if name == "bm25_text_and_plan" else "text")
            ):
                raise ValueError(f"{name}: lexical method differs from freeze")
        for input_name, manifest_key in (
            ("query_only_events.jsonl", "events_sha256"),
            ("candidate_selections.jsonl", "candidates_sha256"),
            ("memories.jsonl", "memories_sha256"),
            ("dependency_hyperedges.jsonl", "hyperedges_sha256"),
        ):
            observed = manifest.get(manifest_key, manifest.get("input_sha256", {}).get(
                {"query_only_events.jsonl": "events", "candidate_selections.jsonl": "candidates", "memories.jsonl": "memories", "dependency_hyperedges.jsonl": "hyperedges"}[input_name]
            ))
            if observed != freeze["dataset_files_sha256"][input_name]:
                raise ValueError(f"{name}: selector input hash mismatch: {input_name}")
        committed[name] = {
            "selections_sha256": sha256_file(selections_path),
            "manifest_sha256": sha256_file(manifest_path),
            "selected_events": 128,
            "selection_path": str(selections_path.resolve()),
        }
    write_json(args.output, {
        "status": "all_v8_methods_committed_before_outcome_mapping",
        "created_utc": datetime.now(UTC).isoformat(),
        "freeze_sha256": sha256_file(args.freeze),
        "outcomes_loaded": False,
        "methods": committed,
    })
    print("committed four label-blind methods across 128 events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
