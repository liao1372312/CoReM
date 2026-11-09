"""Fit and freeze the E3/E4 online scorer from verified development executions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.online_predictor import OnlineLogisticPredictor
from corem.real_eval import load_jsonl, parse_utc, score_execution_row
from corem.types import OutcomeExample
from experiments.run_online_retention import make_graph, request_for


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dataset", type=Path, required=True)
    parser.add_argument("--development-executions", type=Path, required=True)
    parser.add_argument("--test-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    development_events = {
        str(event["event_id"]): event
        for event in load_jsonl(args.development_dataset / "events.jsonl")
    }
    test_rows = load_jsonl(args.test_dataset / "events.jsonl")
    test_events = {str(event["event_id"]) for event in test_rows}
    if development_events.keys() & test_events:
        raise ValueError("development and test event IDs overlap")
    if not development_events or not test_rows:
        raise ValueError("development and test event streams must be nonempty")
    if max(parse_utc(event["event_time"], "event_time") for event in development_events.values()) >= min(
        parse_utc(event["event_time"], "event_time") for event in test_rows
    ):
        raise ValueError("development executions must precede the test stream")
    graph = make_graph(
        load_jsonl(args.development_dataset / "memories.jsonl"),
        load_jsonl(args.development_dataset / "dependency_hyperedges.jsonl"),
    )
    examples = []
    for row in load_jsonl(args.development_executions):
        event_id = str(row["event_id"])
        if event_id not in development_events:
            raise ValueError(f"execution outside development split: {event_id}")
        if row.get("retrieval_status") == "fallback":
            continue
        if (
            row.get("checker_independent") is not True
            or row.get("reset_verified") is not True
            or row.get("execution_error") is not None
            or row.get("checker_error") is not None
        ):
            raise ValueError(f"unverified development execution: {event_id}")
        selected = frozenset(str(key) for key in row["selected_memory_ids"])
        graph._require_records(selected)
        counts = score_execution_row(row, development_events[event_id])
        examples.append(OutcomeExample(
            request_for(development_events[event_id]),
            selected,
            counts["verified_success"],
        ))
    model = OnlineLogisticPredictor()
    model.fit_development(
        examples, graph,
        l2=args.l2,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    provenance = {
        "source_kind": "verified_development_fit",
        "feature_version": model.feature_version,
        "dimension": model.dimension,
        "model_sha256": _sha256(args.output),
        "development_dataset_manifest_sha256": _sha256(args.development_dataset / "manifest.json"),
        "development_executions_sha256": _sha256(args.development_executions),
        "test_dataset_manifest_sha256": _sha256(args.test_dataset / "manifest.json"),
        "verified_development_examples": len(examples),
        "test_event_count": len(test_events),
        "l2": args.l2,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
    }
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
