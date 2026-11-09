#!/usr/bin/env python3
"""Fresh TRAIN-only gate for the nested-path API guard refinement."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_appworld_v25_guard_gate as gate  # noqa: E402
from experiments.summarize_appworld_v23_paired_pilot import expected_target_hashes  # noqa: E402

PRIVATE = ROOT / "data" / "private_corem"
GUARD = ROOT / "experiments" / "run_appworld_v26_nested_guard.py"
RESULT = PRIVATE / "appworld_v26_guard_none_seed2027_n9.json"
COST = PRIVATE / "appworld_v26_guard_none_tokens_seed2027_n9.json"
EXPERIMENT = "v26_guard_none_seed2027_n9"


def validate_result(payload: dict) -> int:
    expected = {
        "status": "train_only_state_based_agent_pilot_not_corem_result",
        "benchmark": "AppWorld 0.1.3.post1", "split": "train",
        "task_sampling": "seeded_distinct_generators_first_variant",
        "model": gate.MODEL_NAME, "model_endpoint": gate.ENDPOINT,
        "model_digest": gate.MODEL_DIGEST, "seed": 2027,
        "appworld_random_seed": 100, "max_steps": 20,
        "context_tokens": 8192, "predict_tokens": 512,
        "prompt_variant": "checklist", "history_strategy": "preserve_task",
        "raise_on_failure": True, "num_tasks": 9,
        "memory_condition": "none", "memory_view_sha256": None,
        "doc_condition": "none", "doc_view_sha256": None,
        "script_sha256": gate.RUNNER_SHA256,
        "train_split_sha256": gate.TRAIN_SPLIT_SHA256,
        "protected_text_exported": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("nested guard gate differs from declared train settings")
    if any(payload.get(key) != 0 for key in
           ("model_errors", "execution_errors", "evaluation_errors")):
        raise ValueError("nested guard gate has a technical error")
    rows = payload.get("private_task_outcomes")
    if not isinstance(rows, list) or len(rows) != 9:
        raise ValueError("nested guard gate has incomplete task outcomes")
    if {row.get("task_sha256") for row in rows} != expected_target_hashes():
        raise ValueError("nested guard gate targets differ from frozen train set")
    for row in rows:
        if (
            type(row.get("success")) is not bool
            or type(row.get("api_guard_interventions")) is not int
            or row["api_guard_interventions"] < 0
            or row.get("api_guard_policy")
            != "read_only_documentation_for_invalid_nested_api_paths"
            or row.get("api_guard_script_sha256") != gate.sha256(GUARD)
            or row.get("model_errors") or row.get("execution_errors")
            or row.get("evaluation_error")
        ):
            raise ValueError("nested guard gate has a malformed task row")
    if sum(row["success"] for row in rows) != payload.get("successes"):
        raise ValueError("nested guard gate success total mismatch")
    return sum(row["steps"] for row in rows)


def main() -> int:
    gate.GUARD = GUARD
    gate.RESULT = RESULT
    gate.COST = COST
    gate.EXPERIMENT = EXPERIMENT
    gate.OUTPUT_DIR = gate.APPWORLD_ROOT / "experiments" / "outputs" / EXPERIMENT
    gate.validate_result = validate_result
    return gate.main()


if __name__ == "__main__":
    raise SystemExit(main())
