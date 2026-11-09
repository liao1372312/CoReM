#!/usr/bin/env python3
"""Exploratory leave-family-out gate between frozen v7 graph and BM25 choices.

This reads already inspected outcomes and is never an independent test.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_jsonl, write_checksums, write_json  # noqa: E402
from experiments.analyze_online_group_controls import verify_checksums  # noqa: E402
from experiments.summarize_independent_v7_confirmation import sha256  # noqa: E402


def selected_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(map(str, row["selected_memory_ids"])))


def load_choices(directory: Path) -> dict[str, tuple[str, ...]]:
    verify_checksums(directory)
    rows = load_jsonl(directory / "selections.jsonl")
    choices = {str(row["event_id"]): selected_key(row) for row in rows}
    if len(choices) != len(rows):
        raise ValueError(f"duplicate selection event in {directory}")
    return choices


def load_scores(directory: Path) -> dict[str, dict[tuple[str, ...], dict[str, Any]]]:
    rows: dict[str, dict[tuple[str, ...], dict[str, Any]]] = defaultdict(dict)
    for row in load_jsonl(directory / "candidate_scores.jsonl"):
        event_id = str(row["event_id"])
        key = selected_key(row)
        if key in rows[event_id]:
            raise ValueError(f"duplicate candidate: {event_id} {key}")
        rows[event_id][key] = row
    return dict(rows)


def features(
    graph: dict[tuple[str, ...], dict[str, Any]],
    lexical: dict[tuple[str, ...], dict[str, Any]],
    graph_choice: tuple[str, ...],
    bm25_choice: tuple[str, ...],
) -> list[float]:
    graph_row = graph[graph_choice]
    bm25_row = graph[bm25_choice]
    return [
        float(graph_row["score"]) - float(bm25_row["score"]),
        float(graph_row["ensemble_probability"])
        - float(bm25_row["ensemble_probability"]),
        float(lexical[bm25_choice]["bm25_score"])
        - float(lexical[graph_choice]["bm25_score"]),
        float(graph_row["seed_probability_std"]),
        float(bm25_row["seed_probability_std"]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--bm25", type=Path, required=True)
    parser.add_argument("--executions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")

    graph_choices = load_choices(args.graph)
    bm25_choices = load_choices(args.bm25)
    graph_scores = load_scores(args.graph)
    bm25_scores = load_scores(args.bm25)
    verify_checksums(args.executions)
    executions = load_jsonl(args.executions / "execution_results.jsonl")
    outcomes: dict[str, dict[tuple[str, ...], int]] = defaultdict(dict)
    for row in executions:
        if (
            row["reset_verified"] is not True
            or row["checker_independent"] is not True
            or row["execution_error"] is not None
            or row["checker_error"] is not None
        ):
            raise ValueError(f"unverified physical outcome: {row['event_id']}")
        event_id = str(row["event_id"])
        key = selected_key(row)
        if key in outcomes[event_id]:
            raise ValueError(f"duplicate physical execution: {event_id} {key}")
        outcomes[event_id][key] = int(row["checker"]["goal_satisfied"])
    event_ids = set(graph_choices)
    if not (
        event_ids == set(bm25_choices) == set(graph_scores) == set(bm25_scores)
        == set(outcomes)
    ) or len(event_ids) != 128:
        raise ValueError("expected identical complete 128-event v7 coverage")

    disagreements = []
    for event_id in sorted(event_ids):
        keys = set(graph_scores[event_id])
        if (
            keys != set(bm25_scores[event_id])
            or keys != set(outcomes[event_id])
            or len(keys) != 2
            or sum(outcomes[event_id].values()) != 1
            or graph_choices[event_id] not in keys
            or bm25_choices[event_id] not in keys
        ):
            raise ValueError(f"invalid candidate/outcome coverage: {event_id}")
        if graph_choices[event_id] == bm25_choices[event_id]:
            continue
        disagreements.append(
            {
                "event_id": event_id,
                "family": event_id.split(":")[1],
                "features": features(
                    graph_scores[event_id],
                    bm25_scores[event_id],
                    graph_choices[event_id],
                    bm25_choices[event_id],
                ),
                "graph_success": outcomes[event_id][graph_choices[event_id]],
                "bm25_success": outcomes[event_id][bm25_choices[event_id]],
            }
        )

    predictions = {}
    for held_family in sorted({row["family"] for row in disagreements}):
        train = [row for row in disagreements if row["family"] != held_family]
        held = [row for row in disagreements if row["family"] == held_family]
        labels = [row["graph_success"] for row in train]
        if len(set(labels)) < 2:
            probabilities = [0.0] * len(held)
        else:
            model = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000, random_state=2027)
            )
            model.fit(np.asarray([row["features"] for row in train]), labels)
            probabilities = model.predict_proba(
                np.asarray([row["features"] for row in held])
            )[:, 1].tolist()
        for row, probability in zip(held, probabilities):
            predictions[row["event_id"]] = float(probability)

    gate_choices = {
        event_id: (
            graph_choices[event_id]
            if predictions.get(event_id, 0.0) > 0.5
            else bm25_choices[event_id]
        )
        for event_id in event_ids
    }
    successes = {
        "graph_complement": sum(
            outcomes[event_id][graph_choices[event_id]] for event_id in event_ids
        ),
        "candidate_bm25_sum": sum(
            outcomes[event_id][bm25_choices[event_id]] for event_id in event_ids
        ),
        "family_held_out_gate": sum(
            outcomes[event_id][gate_choices[event_id]] for event_id in event_ids
        ),
    }
    result = {
        "status": "exploratory_old_v7_development_only",
        "claim_boundary": (
            "All v7 outcomes were inspected before this gate was designed. "
            "Family-held-out fitting is a diagnostic, not independent confirmation."
        ),
        "events": len(event_ids),
        "families": len({event_id.split(":")[1] for event_id in event_ids}),
        "discordant_events": len(disagreements),
        "graph_chosen_by_gate_on_disagreements": sum(
            probability > 0.5 for probability in predictions.values()
        ),
        "successes": successes,
        "feature_names": [
            "graph_score_advantage",
            "neural_probability_advantage",
            "bm25_sum_advantage",
            "graph_choice_seed_std",
            "bm25_choice_seed_std",
        ],
        "source_sha256": {
            "graph_selections": sha256(args.graph / "selections.jsonl"),
            "graph_scores": sha256(args.graph / "candidate_scores.jsonl"),
            "bm25_selections": sha256(args.bm25 / "selections.jsonl"),
            "bm25_scores": sha256(args.bm25 / "candidate_scores.jsonl"),
            "physical_executions": sha256(args.executions / "execution_results.jsonl"),
        },
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "result.json", result)
    write_checksums(args.output, ("result.json",))
    print(result["successes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
