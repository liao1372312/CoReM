#!/usr/bin/env python3
"""Validate and aggregate four frozen v23 TRAIN state-based pilot arms.

No task-level outcome or task hash is printed or written to the summary.
Technical failures invalidate the comparison instead of becoming zeros.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "data" / "private_corem"
MODEL_DIGEST = "26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5"
RUNNER_SHA256 = "4aa2e921bc8677703e80c5a91c3a4566357fe43f8c85581c8fcda1f8e96b4e5c"
VIEW_SHA256 = {
    "none": None,
    "order": "0e324719baab4fc8117cce72ec2bd04de7043ab09c75018fa06a8750504e38ca",
    "disjoint": "d10599c4330d97e2464edbe6117b11e49a549e940b0eb017f313503167a170f5",
    "dependency": "3432c233921fa2487e7379dccbfd6cf78fbde0e40c7bad6e02930ca371818e3d",
}
METHODS = {
    "none": "none",
    "order": "bm25_strict_source_static_api_order_only",
    "disjoint": "bm25_strict_source_static_api_order_plus_disjoint_index_edges",
    "dependency": "bm25_strict_source_static_api_order_plus_candidate_defuse",
}
TRAIN_SPLIT_SHA256 = "93d9fe71e7a2e3b7529803d4a20b604f4ebf5ae806f321081140238068189d37"


def expected_target_hashes() -> set[str]:
    """Reconstruct the fixed train targets from split IDs, not task content."""
    from experiments.run_appworld_local_agent_pilot import _sample_task_ids

    path = PRIVATE_ROOT / "appworld" / "data" / "datasets" / "train.txt"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TRAIN_SPLIT_SHA256:
        raise ValueError("train split differs from frozen v23 split")
    task_ids = [line.strip() for line in raw.decode().splitlines()]
    if len(task_ids) != 90 or any(not task_id for task_id in task_ids):
        raise ValueError("frozen train split has unexpected task identifiers")
    selected = _sample_task_ids(task_ids, 2027, 9)
    return {hashlib.sha256(task_id.encode()).hexdigest() for task_id in selected}


def validated_outcomes(
    payload: dict[str, object], arm: str, target_hashes: set[str] | None = None
) -> dict[str, bool]:
    """Reject partial, mismatched, or technically failed pilot results."""
    expected = {
        "status": "train_only_state_based_agent_pilot_not_corem_result",
        "benchmark": "AppWorld 0.1.3.post1",
        "split": "train",
        "model": "Qwen3.5-9B-4bit",
        "model_endpoint": "http://127.0.0.1:11435/api/chat",
        "model_digest": MODEL_DIGEST,
        "task_sampling": "seeded_distinct_generators_first_variant",
        "seed": 2027,
        "appworld_random_seed": 100,
        "max_steps": 20,
        "context_tokens": 6144,
        "predict_tokens": 512,
        "prompt_variant": "checklist",
        "history_strategy": "preserve_code",
        "raise_on_failure": True,
        "num_tasks": 9,
        "script_sha256": RUNNER_SHA256,
        "memory_condition": METHODS[arm],
        "memory_view_sha256": VIEW_SHA256[arm],
        "doc_condition": "none",
        "doc_view_sha256": None,
        "train_split_sha256": TRAIN_SPLIT_SHA256,
        "protected_text_exported": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{arm} pilot differs from frozen comparison")
    if any(payload.get(key) != 0 for key in ("evaluation_errors", "model_errors", "execution_errors")):
        raise ValueError(f"{arm} pilot has technical failures")
    rows = payload.get("private_task_outcomes")
    if not isinstance(rows, list) or len(rows) != 9:
        raise ValueError(f"{arm} pilot has incomplete task outcomes")
    outcomes: dict[str, bool] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("malformed task outcome")
        task_hash = row.get("task_sha256")
        success = row.get("success")
        if (
            not isinstance(task_hash, str)
            or len(task_hash) != 64
            or any(char not in "0123456789abcdef" for char in task_hash)
            or type(success) is not bool
            or task_hash in outcomes
        ):
            raise ValueError("invalid or duplicate task outcome")
        if row.get("evaluation_error") or row.get("model_errors") or row.get("execution_errors"):
            raise ValueError("technical failure in task outcome")
        outcomes[task_hash] = success
    if sum(outcomes.values()) != payload.get("successes"):
        raise ValueError("aggregate success count differs from task outcomes")
    if target_hashes is not None and set(outcomes) != target_hashes:
        raise ValueError("pilot task hashes differ from frozen train targets")
    return outcomes


def paired_counts(actual: dict[str, bool], baseline: dict[str, bool]) -> dict[str, int]:
    if set(actual) != set(baseline):
        raise ValueError("paired task sets do not match")
    return {
        "actual_wins": sum(actual[key] and not baseline[key] for key in actual),
        "baseline_wins": sum(baseline[key] and not actual[key] for key in actual),
        "ties": sum(actual[key] == baseline[key] for key in actual),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in VIEW_SHA256:
        parser.add_argument(f"--{arm}-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = {arm: getattr(args, f"{arm}_result").resolve() for arm in VIEW_SHA256}
    output = args.output.resolve()
    if any(not path.is_relative_to(PRIVATE_ROOT.resolve()) for path in (*paths.values(), output)):
        parser.error("pilot results and output must stay private")
    if output.exists() or len(set((*paths.values(), output))) != 5:
        parser.error("refusing to overwrite or alias an artifact")
    payloads = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
    target_hashes = expected_target_hashes()
    outcomes = {
        arm: validated_outcomes(payload, arm, target_hashes)
        for arm, payload in payloads.items()
    }
    actual = outcomes["dependency"]
    for arm, baseline in outcomes.items():
        if set(actual) != set(baseline):
            raise ValueError(f"{arm} task set differs from dependency arm")
    result = {
        "status": "train_only_exploratory_paired_state_success_not_confirmation",
        "num_tasks": len(actual),
        "successes": {arm: sum(rows.values()) for arm, rows in outcomes.items()},
        "paired_dependency_vs": {
            arm: paired_counts(actual, outcomes[arm])
            for arm in ("disjoint", "order", "none")
        },
        "totals": {
            arm: {
                field: sum(float(row[field]) for row in payload["private_task_outcomes"])
                for field in ("steps", "wall_seconds", "execution_failure_messages")
            }
            for arm, payload in payloads.items()
        },
        "input_sha256": {
            arm: hashlib.sha256(path.read_bytes()).hexdigest()
            for arm, path in paths.items()
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task_ids_or_hashes_exported": False,
        "dev_or_test_task_loaded": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"successes": result["successes"], "paired_dependency_vs": result["paired_dependency_vs"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
