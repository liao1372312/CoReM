#!/usr/bin/env python3
"""Audit the frozen complement head's influence on matched candidate rankings."""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_json, load_jsonl, write_checksums, write_json  # noqa: E402


def candidate_key(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return str(row["event_id"]), tuple(sorted(map(str, row["selected_memory_ids"])))


def rank(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (float(row[field]), candidate_key(row)[1]),
        reverse=True,
    )


def verify_frozen_pair(
    full_manifest: dict[str, Any], off_manifest: dict[str, Any]
) -> None:
    for field in (
        "dataset",
        "target_split",
        "model_seeds",
        "model_states",
        "neural_weight",
        "bm25_weight",
        "selection_rows",
    ):
        if full_manifest[field] != off_manifest[field]:
            raise ValueError(f"frozen manifests differ on {field}")
    if off_manifest.get("state_variant") != "hgt_corem":
        raise ValueError("head-off control must load full-model weights")
    if off_manifest.get("model_variant") != "hgt_additive":
        raise ValueError("head-off control must disable the complement head")
    if full_manifest.get("method") != "hgt_rank_ensemble":
        raise ValueError("full source is not the frozen HGT ensemble")


def audit(
    full_rows: list[dict[str, Any]],
    off_rows: list[dict[str, Any]],
    events: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    full = {candidate_key(row): row for row in full_rows}
    off = {candidate_key(row): row for row in off_rows}
    if len(full) != len(full_rows) or len(off) != len(off_rows):
        raise ValueError("duplicate event/candidate scores")
    if not full or set(full) != set(off):
        raise ValueError("full and head-off candidate pools differ")

    grouped: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    nonzero_probability_deltas = []
    probability_deltas = []
    for event_id, ids in sorted(full):
        if event_id not in events:
            raise ValueError(f"missing event metadata: {event_id}")
        on_row, off_row = full[(event_id, ids)], off[(event_id, ids)]
        if (
            abs(float(on_row["bm25_normalized"]) - float(off_row["bm25_normalized"]))
            > 1e-10
        ):
            raise ValueError("lexical control differs between matched candidates")
        delta = float(on_row["ensemble_probability"]) - float(
            off_row["ensemble_probability"]
        )
        probability_deltas.append(delta)
        if abs(delta) > 1e-8:
            nonzero_probability_deltas.append(delta)
        grouped[event_id].append(ids)

    per_event = []
    for event_id, candidates in sorted(grouped.items()):
        full_rank = rank([full[(event_id, ids)] for ids in candidates], "score")
        off_rank = rank([off[(event_id, ids)] for ids in candidates], "score")
        bm25_rank = rank(
            [full[(event_id, ids)] for ids in candidates], "bm25_normalized"
        )
        on_ids = candidate_key(full_rank[0])[1]
        off_ids = candidate_key(off_rank[0])[1]
        bm25_ids = candidate_key(bm25_rank[0])[1]
        on_margin = (
            float(full_rank[0]["score"]) - float(full_rank[1]["score"])
            if len(full_rank) > 1
            else None
        )
        off_margin = (
            float(off_rank[0]["score"]) - float(off_rank[1]["score"])
            if len(off_rank) > 1
            else None
        )
        on_order = [candidate_key(row)[1] for row in full_rank]
        off_order = [candidate_key(row)[1] for row in off_rank]
        per_event.append(
            {
                "event_id": event_id,
                "service_id": str(events[event_id]["service_id"]),
                "task_key": str(events[event_id]["task_key"]),
                "candidates": len(candidates),
                "full_winner": list(on_ids),
                "head_off_winner": list(off_ids),
                "bm25_winner": list(bm25_ids),
                "winner_changed": on_ids != off_ids,
                "ranking_changed": on_order != off_order,
                "full_matches_bm25": on_ids == bm25_ids,
                "full_top_margin": on_margin,
                "head_off_top_margin": off_margin,
                "maximum_probability_delta": max(
                    abs(
                        float(full[(event_id, ids)]["ensemble_probability"])
                        - float(off[(event_id, ids)]["ensemble_probability"])
                    )
                    for ids in candidates
                ),
            }
        )

    return {
        "status": "complete_exploratory_score_diagnostic",
        "claim_boundary": (
            "Inspected v6 test scores; this measures frozen-head score influence, "
            "not a retrained causal ablation or independent task/provider confirmation."
        ),
        "events": len(per_event),
        "candidates": len(full),
        "candidate_probabilities_changed": len(nonzero_probability_deltas),
        "mean_probability_delta": statistics.mean(probability_deltas),
        "mean_absolute_probability_delta": statistics.mean(
            abs(value) for value in probability_deltas
        ),
        "maximum_absolute_probability_delta": max(
            abs(value) for value in probability_deltas
        ),
        "events_with_any_ranking_change": sum(
            row["ranking_changed"] for row in per_event
        ),
        "events_with_winner_change": sum(row["winner_changed"] for row in per_event),
        "full_winner_matches_bm25": sum(row["full_matches_bm25"] for row in per_event),
        "mean_full_top_margin": statistics.mean(
            row["full_top_margin"]
            for row in per_event
            if row["full_top_margin"] is not None
        )
        if any(row["full_top_margin"] is not None for row in per_event)
        else None,
        "mean_head_off_top_margin": statistics.mean(
            row["head_off_top_margin"]
            for row in per_event
            if row["head_off_top_margin"] is not None
        )
        if any(row["head_off_top_margin"] is not None for row in per_event)
        else None,
        "per_event": per_event,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--head-off", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")

    full_manifest_path = args.full.parent / "manifest.json"
    off_manifest_path = args.head_off.parent / "manifest.json"
    verify_frozen_pair(load_json(full_manifest_path), load_json(off_manifest_path))

    events = {
        str(row["event_id"]): row for row in load_jsonl(args.dataset / "events.jsonl")
    }
    result = audit(load_jsonl(args.full), load_jsonl(args.head_off), events)
    result["sources"] = {
        name: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in (
            ("full", args.full),
            ("head_off", args.head_off),
            ("full_manifest", full_manifest_path),
            ("head_off_manifest", off_manifest_path),
        )
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "result.json", result)
    write_checksums(args.output, ("result.json",))
    print(
        f"audited {result['events']} events; winner changed on "
        f"{result['events_with_winner_change']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
