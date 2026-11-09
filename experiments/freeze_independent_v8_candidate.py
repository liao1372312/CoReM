#!/usr/bin/env python3
"""Freeze a structurally valid v8 candidate set after full live validity gates.

This records inputs and rules, not method outcomes. No selector is run here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import load_jsonl, sha256_file, write_json  # noqa: E402

DATA_FILES = (
    "events.jsonl",
    "query_only_events.jsonl",
    "memories.jsonl",
    "dependency_hyperedges.jsonl",
    "candidate_selections.jsonl",
    "manifest.json",
)
SOURCES = (
    "experiments/build_independent_v8_candidate_staging.py",
    "experiments/audit_independent_v8_structure.py",
    "experiments/independent_v8_runner.py",
    "experiments/independent_v8_checker.py",
    "experiments/public_multiservice_challenge_runner.py",
    "experiments/independent_v7_runner.py",
    "experiments/score_dependency_path_reranker.py",
    "experiments/path_serialization.py",
    "experiments/score_v7_path_nli_development.py",
    "experiments/score_complete_candidate_bm25.py",
    "experiments/score_complete_candidate_literals.py",
    "experiments/commit_independent_v8_selections.py",
    "experiments/summarize_independent_v8_candidate.py",
    "experiments/generate_public_workflow_selections.py",
    "corem/v8_candidates.py",
    "corem/real_eval.py",
)


def _validated_gate(directory: Path, dataset: Path, expected: int) -> dict[str, str | int]:
    manifest_path = directory / "run_manifest.json"
    results_path = directory / "execution_results.jsonl"
    manifest = json.loads(manifest_path.read_text())
    results = load_jsonl(results_path)
    events = load_jsonl(dataset / "events.jsonl")
    expected_ids = {str(row["event_id"]) for row in events}
    observed_ids = [str(row["event_id"]) for row in results]
    if (
        len(events) != 128
        or len(results) != 128
        or len(observed_ids) != len(set(observed_ids))
        or set(observed_ids) != expected_ids
        or manifest.get("rows") != 128
        or manifest.get("verified_successes") != expected
        or manifest.get("runner_errors") != 0
        or manifest.get("checker_errors") != 0
        or manifest.get("dataset_files_sha256") != {
            name: sha256_file(dataset / name)
            for name in ("events.jsonl", "memories.jsonl", "dependency_hyperedges.jsonl")
        }
    ):
        raise ValueError(f"invalid or incomplete live gate: {directory}")
    for row in results:
        checker = row.get("checker")
        passed = bool(checker.get("goal_satisfied")) if isinstance(checker, dict) else False
        if (
            row.get("reset_verified") is not True
            or row.get("checker_independent") is not True
            or row.get("checker_before_teardown") is not True
            or row.get("execution_error") is not None
            or row.get("checker_error") is not None
            or passed is not (expected == 128)
        ):
            raise ValueError(f"nonconforming gate row: {row.get('event_id')}")
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "execution_results_sha256": sha256_file(results_path),
        "verified_successes": expected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--oracle-gate", type=Path, required=True)
    parser.add_argument("--negative-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a prior freeze")
    structural = args.private / "structural_audit_v1.json"
    audit = json.loads(structural.read_text())
    if audit.get("status") != "structurally_ready_for_oracle_gate" or audit.get("issues"):
        raise ValueError("structural gate is not clean")
    if audit.get("input_sha256") != {
        name: sha256_file(args.dataset / name) for name in DATA_FILES[:-1]
    }:
        raise ValueError("structural audit input hashes changed")
    oracle = _validated_gate(args.oracle_gate, args.dataset, 128)
    negative = _validated_gate(args.negative_gate, args.dataset, 0)
    model_files = sorted(path for path in args.model_dir.iterdir() if path.is_file())
    if not model_files or not (args.model_dir / "model.safetensors").is_file():
        raise ValueError("missing complete model checkpoint")
    result = {
        "status": "frozen_unscored_v8_candidate",
        "claim_boundary": "Live oracle/negative gates establish task validity only; no model or baseline outcome has been measured.",
        "primary_contrast": "body_v1 path reranker versus same-candidate BM25-sum with text_and_plan view",
        "secondary_controls": ["BM25-sum text-only", "literal-value overlap"],
        "retrieval_budget": 160,
        "model_inference": {"serialization": "body_v1", "max_length": 384, "seed": 2027},
        "dataset_files_sha256": {name: sha256_file(args.dataset / name) for name in DATA_FILES},
        "structural_audit_sha256": sha256_file(structural),
        "validity_gates": {"oracle": oracle, "complete_negative": negative},
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCES},
        "model_files_sha256": {path.name: sha256_file(path) for path in model_files},
        "training_manifest_sha256": sha256_file(args.model_dir.parent / "result.json"),
        "private_assignment_commitment_sha256": json.loads((args.dataset / "manifest.json").read_text())["assignment_commitment_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print("frozen 128 unscored v8 candidate incidents and three selector rules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
