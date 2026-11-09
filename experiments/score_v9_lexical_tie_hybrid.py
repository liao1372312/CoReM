#!/usr/bin/env python3
"""Label-blind, predeclared BM25/non-tie and neural/tie candidate selector.

Read only query-only events, complete candidate graphs, BM25 candidate scores,
and a frozen model. Never read workflow contracts or physical outcomes here.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import load_jsonl, sha256_file, write_checksums, write_json, write_jsonl  # noqa: E402
from corem.v8_candidates import validate_candidate  # noqa: E402
from experiments.develop_v8_contrastive_path_input import difference_focused_paths  # noqa: E402
from experiments.score_complete_candidate_bm25 import QUERY_FIELDS  # noqa: E402


def _selected(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(map(str, row["selected_memory_ids"])))


def _digest(seed: int, event_id: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{seed}|{event_id}|{'|'.join(_selected(row))}".encode()).hexdigest()


def _inputs(
    dataset: Path, bm25_scores: Path, budget: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[tuple[str, tuple[str, ...]], float]]:
    query_rows = load_jsonl(dataset / "query_only_events.jsonl")
    if any(set(row) != QUERY_FIELDS for row in query_rows):
        raise ValueError("selector received fields beyond the query-only view")
    events = {str(row["event_id"]): row for row in query_rows}
    memory_rows = load_jsonl(dataset / "memories.jsonl")
    memories = {str(row["memory_id"]): row for row in memory_rows}
    edge_rows = load_jsonl(dataset / "dependency_hyperedges.jsonl")
    edges = {str(row["edge_id"]): row for row in edge_rows}
    if len(events) != 128 or len(memory_rows) != len(memories) or len(edge_rows) != len(edges):
        raise ValueError("incomplete or duplicate selector inputs")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(dataset / "candidate_selections.jsonl"):
        event_id = str(row["event_id"])
        if event_id not in events:
            raise ValueError(f"unknown candidate event: {event_id}")
        validate_candidate(row, events[event_id], memories, edges)
        if sum(int(memories[item]["token_cost"]) for item in row["selected_memory_ids"]) > budget:
            raise ValueError(f"candidate exceeds budget: {event_id}")
        grouped[event_id].append(row)
    if set(grouped) != set(events) or any(
        len(rows) != 2 or _selected(rows[0]) == _selected(rows[1]) for rows in grouped.values()
    ):
        raise ValueError("expected exactly two complete candidates per event")
    scores = {}
    for row in load_jsonl(bm25_scores):
        key = str(row["event_id"]), _selected(row)
        value = float(row["bm25_score"])
        if key in scores or not math.isfinite(value):
            raise ValueError(f"duplicate or invalid BM25 score: {key}")
        scores[key] = value
    expected = {(event_id, _selected(row)) for event_id, rows in grouped.items() for row in rows}
    if set(scores) != expected:
        raise ValueError("BM25 scores do not match the candidate set")
    return events, memories, grouped, scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bm25-candidate-scores", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists() or args.token_budget < 1 or args.max_length < 32 or args.batch_size < 1:
        parser.error("output must be new and numeric parameters valid")
    if args.tie_tolerance < 0 or not math.isfinite(args.tie_tolerance):
        parser.error("tie tolerance must be finite and nonnegative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    started = time.perf_counter()
    events, memories, grouped, bm25 = _inputs(args.dataset, args.bm25_candidate_scores, args.token_budget)
    neural_inputs = []
    non_tie = {}
    for event_id in sorted(events):
        rows = grouped[event_id]
        margin = abs(bm25[event_id, _selected(rows[0])] - bm25[event_id, _selected(rows[1])])
        left, right, changed = difference_focused_paths(rows[0], rows[1], memories)
        hypothesis = "This workflow fulfills the request: " + str(events[event_id]["query"])
        neural_inputs.extend((
            (event_id, rows[0], left, hypothesis, changed),
            (event_id, rows[1], right, hypothesis, changed),
        ))
        if margin > args.tie_tolerance:
            non_tie[event_id] = max(rows, key=lambda row: (bm25[event_id, _selected(row)], _digest(args.seed, event_id, row)))
    neural_scores: dict[tuple[str, tuple[str, ...]], float] = {}
    model_load_ms = 0.0
    model_scoring_ms = 0.0
    if neural_inputs:
        torch.manual_seed(args.seed)
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed)
        model_load_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True).to(args.device)
        model_load_ms = (time.perf_counter() - model_load_started) * 1000
        labels = {value.lower(): int(key) for key, value in model.config.id2label.items()}
        if "entailment" not in labels or "contradiction" not in labels:
            raise ValueError("checkpoint lacks entailment and contradiction labels")
        model.eval()
        for start in range(0, len(neural_inputs), args.batch_size):
            batch_started = time.perf_counter()
            batch = neural_inputs[start:start + args.batch_size]
            encoded = tokenizer(
                [item[2] for item in batch], [item[3] for item in batch],
                padding=True, truncation=True, max_length=args.max_length, return_tensors="pt",
            ).to(args.device)
            with torch.inference_mode():
                logits = model(**encoded).logits.float().cpu()
            for item, values in zip(batch, logits):
                score = float(
                    values[labels["entailment"]] - values[labels["contradiction"]]
                )
                if not math.isfinite(score):
                    raise ValueError(f"non-finite neural score: {item[0]}")
                neural_scores[item[0], _selected(item[1])] = score
            model_scoring_ms += (time.perf_counter() - batch_started) * 1000
    selected, neural_only, scores = [], [], []
    for event_id in sorted(events):
        rows = grouped[event_id]
        if event_id in non_tie:
            winner = non_tie[event_id]
            decision = "bm25_non_tie"
        else:
            winner = max(rows, key=lambda row: (neural_scores[event_id, _selected(row)], _digest(args.seed, event_id, row)))
            decision = "neural_lexical_tie"
        ids = _selected(winner)
        neural_winner = max(rows, key=lambda row: (neural_scores[event_id, _selected(row)], _digest(args.seed, event_id, row)))
        neural_ids = _selected(neural_winner)
        selected.append({
            "run_id": f"v9_hybrid:{args.seed}:{event_id}", "event_id": event_id,
            "method": "v9_lexical_tie_hybrid", "candidate_edge_id": winner["candidate_edge_id"],
            "selected_memory_ids": list(ids), "token_cost": sum(int(memories[item]["token_cost"]) for item in ids),
            "decision": decision, "retrieval_status": "ok",
        })
        neural_only.append({
            "run_id": f"v9_neural_only:{args.seed}:{event_id}", "event_id": event_id,
            "method": "v9_neural_only", "candidate_edge_id": neural_winner["candidate_edge_id"],
            "selected_memory_ids": list(neural_ids),
            "token_cost": sum(int(memories[item]["token_cost"]) for item in neural_ids),
            "retrieval_status": "ok",
        })
        for row in rows:
            key = event_id, _selected(row)
            scores.append({
                "event_id": event_id, "candidate_edge_id": row["candidate_edge_id"],
                "selected_memory_ids": list(key[1]), "bm25_score": bm25[key],
                "neural_score": neural_scores.get(key), "decision": decision,
            })
    prewrite_wall_ms = (time.perf_counter() - started) * 1000
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "selections.jsonl", selected)
    write_jsonl(args.output / "neural_only_selections.jsonl", neural_only)
    write_jsonl(args.output / "candidate_scores.jsonl", scores)
    write_json(args.output / "manifest.json", {
        "status": "label_blind_frozen_rule_selection", "outcomes_loaded": False,
        "claim_boundary": "Selections only; compare with physical outcomes after choices are hash-committed.",
        "events": len(selected), "neural_tie_events": len(events) - len(non_tie),
        "bm25_non_tie_events": len(non_tie), "token_budget": args.token_budget,
        "tie_tolerance": args.tie_tolerance, "max_length": args.max_length,
        "batch_size": args.batch_size, "seed": args.seed,
        "model_load_ms": model_load_ms, "model_scoring_ms": model_scoring_ms,
        "prewrite_wall_ms": prewrite_wall_ms,
        "model_weights_sha256": sha256_file(args.model_dir / "model.safetensors"),
        "bm25_scores_sha256": sha256_file(args.bm25_candidate_scores),
        "input_sha256": {
            name: sha256_file(args.dataset / name) for name in (
                "query_only_events.jsonl", "memories.jsonl", "dependency_hyperedges.jsonl", "candidate_selections.jsonl",
            )
        },
        "script_sha256": sha256_file(Path(__file__)),
    })
    write_checksums(args.output, (
        "selections.jsonl", "neural_only_selections.jsonl", "candidate_scores.jsonl", "manifest.json",
    ))
    print(f"selected {len(selected)} events; neural tie-breaker on {len(events) - len(non_tie)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
