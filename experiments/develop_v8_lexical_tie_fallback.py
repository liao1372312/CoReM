#!/usr/bin/env python3
"""Post-inspection diagnostic of a generic lexical-tie neural fallback.

This rule was conceived after v8 outcomes were inspected. Its v8 score is
development evidence only, even though the neural input was provider-held-out.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import load_jsonl, sha256_file, write_checksums, write_json, write_jsonl  # noqa: E402


def _selected(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(map(str, row["selected_memory_ids"])))


def _unique(path: Path, key: str) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    mapped = {str(row[key]): row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError(f"duplicate {key}: {path}")
    return mapped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-predictions", type=Path, required=True)
    parser.add_argument("--bm25-selections", type=Path, required=True)
    parser.add_argument("--bm25-candidate-scores", type=Path, required=True)
    parser.add_argument("--oracle-selections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    if args.output.exists() or args.tie_tolerance < 0 or not math.isfinite(args.tie_tolerance):
        parser.error("output must be new and tie tolerance nonnegative")
    neural = _unique(args.neural_predictions, "event_id")
    bm25 = _unique(args.bm25_selections, "event_id")
    oracle = _unique(args.oracle_selections, "event_id")
    scored: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(args.bm25_candidate_scores):
        scored[str(row["event_id"])].append(row)
    ids = set(neural)
    if len(ids) != 128 or set(bm25) != ids or set(oracle) != ids or set(scored) != ids:
        raise ValueError("v8 development inputs do not have matching 128 events")
    selections = []
    counts = Counter()
    for event_id in sorted(ids):
        pair = scored[event_id]
        if len(pair) != 2 or not all(math.isfinite(float(row["bm25_score"])) for row in pair):
            raise ValueError(f"invalid BM25 candidate scores: {event_id}")
        candidate_sets = {_selected(row) for row in pair}
        if len(candidate_sets) != 2 or {_selected(neural[event_id]), _selected(bm25[event_id])}.difference(candidate_sets):
            raise ValueError(f"inconsistent candidate pool: {event_id}")
        margin = abs(float(pair[0]["bm25_score"]) - float(pair[1]["bm25_score"]))
        use_neural = margin <= args.tie_tolerance
        choice = _selected(neural[event_id] if use_neural else bm25[event_id])
        success = int(choice == _selected(oracle[event_id]))
        counts[("tie_neural" if use_neural else "non_tie_bm25", "events")] += 1
        counts[("tie_neural" if use_neural else "non_tie_bm25", "successes")] += success
        selections.append({
            "event_id": event_id, "selected_memory_ids": list(choice),
            "decision": "neural_on_lexical_tie" if use_neural else "bm25_on_lexical_difference",
            "bm25_margin": margin, "verified_success": success,
            "bm25_verified_success": int(_selected(bm25[event_id]) == _selected(oracle[event_id])),
            "neural_verified_success": int(_selected(neural[event_id]) == _selected(oracle[event_id])),
        })
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "selections.jsonl", selections)
    write_json(args.output / "result.json", {
        "status": "post_inspection_v8_tie_fallback_development_only",
        "claim_boundary": "The rule was designed after observing v8 outcomes; this is not an independent result or a WWW-ready performance claim.",
        "events": len(selections),
        "successes": sum(row["verified_success"] for row in selections),
        "bm25_successes": sum(row["bm25_verified_success"] for row in selections),
        "neural_successes": sum(row["neural_verified_success"] for row in selections),
        "tie_neural_events": counts[("tie_neural", "events")],
        "tie_neural_successes": counts[("tie_neural", "successes")],
        "non_tie_bm25_events": counts[("non_tie_bm25", "events")],
        "non_tie_bm25_successes": counts[("non_tie_bm25", "successes")],
        "tie_tolerance": args.tie_tolerance,
        "neural_predictions_sha256": sha256_file(args.neural_predictions),
        "bm25_selections_sha256": sha256_file(args.bm25_selections),
        "bm25_scores_sha256": sha256_file(args.bm25_candidate_scores),
        "oracle_selections_sha256": sha256_file(args.oracle_selections),
        "script_sha256": sha256_file(Path(__file__)),
    })
    write_checksums(args.output, ("selections.jsonl", "result.json"))
    print(f"post-inspection v8 tie fallback: {sum(row['verified_success'] for row in selections)}/128")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
