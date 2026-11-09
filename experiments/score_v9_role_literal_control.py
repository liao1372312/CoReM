#!/usr/bin/env python3
"""Label-blind role/literal shortcut control over identical v9 candidates."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import sha256_file, write_checksums, write_json, write_jsonl  # noqa: E402
from experiments.develop_v8_role_literal_control import role_literal_scores  # noqa: E402
from experiments.score_v9_lexical_tie_hybrid import _digest, _inputs, _selected  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bm25-candidate-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if args.output.exists() or args.token_budget < 1:
        parser.error("output must be new and budget positive")
    events, memories, grouped, _bm25 = _inputs(args.dataset, args.bm25_candidate_scores, args.token_budget)
    selections = []
    scores = []
    for event_id in sorted(events):
        started = time.perf_counter()
        pair = grouped[event_id]
        values = role_literal_scores(str(events[event_id]["query"]), pair[0], pair[1], memories)
        if len(values) != 2 or any(not math.isfinite(value) for value in values):
            raise ValueError(f"invalid role-literal score: {event_id}")
        ranked = sorted(zip(pair, values), key=lambda item: (item[1], _digest(args.seed, event_id, item[0])), reverse=True)
        winner = ranked[0][0]
        ids = _selected(winner)
        selections.append({
            "run_id": f"v9_role_literal:{args.seed}:{event_id}",
            "event_id": event_id, "method": "v9_role_literal_control",
            "candidate_edge_id": winner["candidate_edge_id"],
            "selected_memory_ids": list(ids),
            "token_cost": sum(int(memories[item]["token_cost"]) for item in ids),
            "retrieval_status": "ok", "latency_ms": (time.perf_counter() - started) * 1000,
        })
        for row, score in zip(pair, values):
            scores.append({"event_id": event_id, "candidate_edge_id": row["candidate_edge_id"],
                           "selected_memory_ids": list(_selected(row)), "role_literal_score": score})
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "selections.jsonl", selections)
    write_jsonl(args.output / "candidate_scores.jsonl", scores)
    write_json(args.output / "manifest.json", {
        "status": "label_blind_role_literal_control", "outcomes_loaded": False,
        "claim_boundary": "Precommitted v9 control adapted from a post-inspection v8 shortcut diagnostic.",
        "events": len(selections), "seed": args.seed, "token_budget": args.token_budget,
        "bm25_scores_sha256": sha256_file(args.bm25_candidate_scores),
        "input_sha256": {name: sha256_file(args.dataset / name) for name in (
            "query_only_events.jsonl", "memories.jsonl", "dependency_hyperedges.jsonl", "candidate_selections.jsonl",
        )},
        "script_sha256": sha256_file(Path(__file__)),
    })
    write_checksums(args.output, ("selections.jsonl", "candidate_scores.jsonl", "manifest.json"))
    print(f"wrote {len(selections)} label-blind role/literal control choices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
