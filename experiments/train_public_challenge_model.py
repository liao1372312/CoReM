#!/usr/bin/env python3
"""Fit the transparent CoReM set-utility head on verified train executions."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.model import SetUtilityModel, TrainingConfig
from corem.real_eval import (
    load_jsonl,
    parse_utc,
    score_execution_row,
    write_checksums,
    write_json,
)
from corem.types import OutcomeExample, Request
from experiments.generate_public_challenge_selections import GENERIC_REQUIRED_ATOMS
from experiments.generate_public_workflow_selections import make_graph


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_for(event: dict[str, object]) -> Request:
    return Request(
        request_id=str(event["event_id"]),
        text=str(event["query"]),
        required_atoms=GENERIC_REQUIRED_ATOMS,
        operation_ids=frozenset(),
        timestamp=parse_utc(str(event["event_time"]), "event_time").timestamp(),
        allowed_scopes=frozenset((str(event["service_id"]),)),
        active_versions=frozenset((str(event["version_id"]),)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--executions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2

    events = {
        str(row["event_id"]): row for row in load_jsonl(args.dataset / "events.jsonl")
    }
    train_ids = {
        str(row["event_id"])
        for row in load_jsonl(args.splits)
        if row["split"] == "train"
    }
    memories = load_jsonl(args.dataset / "memories.jsonl")
    graph = make_graph(
        memories, load_jsonl(args.dataset / "dependency_hyperedges.jsonl")
    )
    rows = load_jsonl(args.executions)
    examples: list[OutcomeExample] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    exclusions = {
        "non_train": 0,
        "not_completed": 0,
        "inadmissible": 0,
        "generic_incomplete": 0,
        "duplicate": 0,
    }
    class_counts = {"failure": 0, "success": 0}
    for row in rows:
        event_id = str(row["event_id"])
        if event_id not in train_ids:
            exclusions["non_train"] += 1
            continue
        if (
            row.get("checker_error") is not None
            or row.get("execution_error") is not None
        ):
            exclusions["not_completed"] += 1
            continue
        execution = row.get("execution")
        if not isinstance(execution, dict) or execution.get("completed") is not True:
            exclusions["not_completed"] += 1
            continue
        selected = frozenset(map(str, row["selected_memory_ids"]))
        event = events[event_id]
        request = request_for(event)
        if any(not graph.is_admissible(memory_id, request) for memory_id in selected):
            exclusions["inadmissible"] += 1
            continue
        if not graph.is_evidence_complete(request, selected):
            exclusions["generic_incomplete"] += 1
            continue
        key = (event_id, tuple(sorted(selected)))
        if key in seen:
            exclusions["duplicate"] += 1
            continue
        seen.add(key)
        outcome = int(score_execution_row(row, event)["verified_success"])
        class_counts["success" if outcome else "failure"] += 1
        examples.append(
            OutcomeExample(
                request=request,
                selected_ids=selected,
                outcome=outcome,
                actually_executed=True,
                independently_verified=True,
                eligible_for_training=True,
            )
        )

    if not class_counts["success"] or not class_counts["failure"]:
        print(f"training requires both classes, observed {class_counts}")
        return 2
    model = SetUtilityModel()
    config = TrainingConfig(
        learning_rate=0.03,
        epochs=400,
        patience=20,
        class_balance_power=1.0,
    )
    fit = model.fit(examples, graph, config)
    args.output.mkdir(parents=True)
    write_json(
        args.output / "model.json",
        {
            "schema_version": "corem-public-challenge-model-v1",
            "training_data": {
                "execution_path": str(args.executions.resolve()),
                "execution_sha256": sha256(args.executions),
                "split_path": str(args.splits.resolve()),
                "split_sha256": sha256(args.splits),
                "eligible_unique_examples": len(examples),
                "class_counts": class_counts,
                "exclusions": exclusions,
            },
            "retrieval_inputs": {
                "event_task_key": False,
                "exact_required_evidence_atoms": False,
                "generic_role_atoms": sorted(GENERIC_REQUIRED_ATOMS),
            },
            "training_config": {
                "learning_rate": config.learning_rate,
                "epochs": config.epochs,
                "patience": config.patience,
                "class_balance_power": config.class_balance_power,
            },
            "fit": {
                "converged": fit.converged,
                "epochs": fit.epochs,
                "initial_loss": fit.losses[0],
                "final_loss": fit.losses[-1],
                "losses": list(fit.losses),
            },
            "weights": {
                "base": model.base_weights,
                "unary": model.unary_weights,
                "complement": model.complement_weights,
                "redundancy": model.redundancy_weights,
                "relation_heads": model.relation_heads,
            },
        },
    )
    write_checksums(args.output, ("model.json",))
    print(
        f"fit {len(examples)} verified unique examples "
        f"({class_counts['success']} success, {class_counts['failure']} failure)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
