#!/usr/bin/env python3
"""Seal task, model, and comparator before any v9 candidate score is computed."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import load_jsonl, sha256_file, write_json  # noqa: E402
from experiments.audit_independent_v9_structure import DATA_FILES  # noqa: E402
from experiments.export_v9_selector_view import SELECTOR_FILES  # noqa: E402
from experiments.independent_v9_runner import SOLR_ARCHIVE, SOLR_SHA512, TYPESENSE_BINARY, TYPESENSE_SHA256, _digest  # noqa: E402

SOURCES = (
    "experiments/build_independent_v9_staging.py",
    "experiments/build_independent_v9_staging_v2.py",
    "experiments/build_independent_v9_staging_v4.py",
    "experiments/build_independent_v9_staging_v5.py",
    "experiments/build_independent_v9_staging_v7.py",
    "experiments/audit_independent_v9_structure.py",
    "experiments/export_v9_selector_view.py",
    "experiments/freeze_independent_v9_candidate.py",
    "experiments/run_independent_v9_staging.py",
    "experiments/independent_v9_runner.py",
    "experiments/independent_v9_checker.py",
    "experiments/independent_v8_checker.py",
    "experiments/independent_v7_runner.py",
    "experiments/public_multiservice_challenge_runner.py",
    "experiments/public_workflows_runner.py",
    "experiments/score_complete_candidate_bm25.py",
    "experiments/score_complete_candidate_literals.py",
    "experiments/score_v9_lexical_tie_hybrid.py",
    "experiments/score_v9_role_literal_control.py",
    "experiments/develop_v8_role_literal_control.py",
    "experiments/develop_v8_delta_pairwise_model.py",
    "experiments/commit_independent_v9_selections.py",
    "experiments/summarize_independent_v9_candidate.py",
    "experiments/summarize_independent_v7_confirmation.py",
    "experiments/develop_v8_contrastive_path_input.py",
    "experiments/train_v9_hybrid_neural_candidate.py",
    "experiments/generate_public_workflow_selections.py",
    "corem/v8_candidates.py",
    "corem/real_eval.py",
)


def _selected(row: dict) -> tuple[str, ...]:
    return tuple(sorted(map(str, row["selected_memory_ids"])))


def _gate(directory: Path, dataset: Path, selections: Path, successes: int) -> dict[str, str | int]:
    manifest_path = directory / "run_manifest.json"
    outcomes_path = directory / "execution_results.jsonl"
    manifest = json.loads(manifest_path.read_text())
    outcomes = load_jsonl(outcomes_path)
    labels = load_jsonl(selections)
    expected = {str(row["event_id"]): row for row in labels}
    observed = {str(row["event_id"]): row for row in outcomes}
    if (
        len(labels) != 128 or len(expected) != 128 or len(outcomes) != 128 or len(observed) != 128
        or set(expected) != set(observed)
        or manifest.get("rows") != 128 or manifest.get("verified_successes") != successes
        or manifest.get("runner_errors") != 0 or manifest.get("checker_errors") != 0
        or manifest.get("retrieval_budget") != 256
        or manifest.get("selections_sha256") != sha256_file(selections)
        or manifest.get("dataset_files_sha256") != {
            name: sha256_file(dataset / name)
            for name in ("events.jsonl", "memories.jsonl", "dependency_hyperedges.jsonl")
        }
        or manifest.get("runner_sha256") != sha256_file(ROOT / "experiments/independent_v9_runner.py")
        or manifest.get("checker_sha256") != sha256_file(ROOT / "experiments/independent_v9_checker.py")
    ):
        raise ValueError(f"incomplete or changed physical gate: {directory}")
    for event_id, row in observed.items():
        if (
            _selected(row) != _selected(expected[event_id])
            or row.get("candidate_edge_id") != expected[event_id].get("candidate_edge_id")
            or row.get("reset_verified") is not True
            or row.get("checker_independent") is not True
            or row.get("checker_before_teardown") is not True
            or row.get("execution_error") is not None or row.get("checker_error") is not None
            or not isinstance(row.get("execution"), dict)
            or row["execution"].get("completed") is not True
            or not isinstance(row.get("checker"), dict)
            or row["checker"].get("goal_satisfied") is not (successes == 128)
            or row["checker"].get("trace_statuses_verified") is not True
            or not isinstance(row["checker"].get("state_checks"), list)
            or not row["checker"]["state_checks"]
            or any(not isinstance(check, dict) for check in row["checker"]["state_checks"])
            or all(check.get("passed") is True for check in row["checker"]["state_checks"]) is not (successes == 128)
        ):
            raise ValueError(f"nonconforming physical gate row: {event_id}")
    return {"manifest_sha256": sha256_file(manifest_path),
            "execution_results_sha256": sha256_file(outcomes_path), "verified_successes": successes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--selector-view", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--base-nli-model-dir", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a prior freeze")
    audit_path = args.private / "structural_audit_v1.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "structurally_ready_for_physical_gate" or audit.get("issues"):
        raise ValueError("v9 structural audit is not clean")
    if audit.get("input_sha256") != {name: sha256_file(args.dataset / name) for name in DATA_FILES}:
        raise ValueError("public candidate changed since structural audit")
    selector_manifest = json.loads((args.selector_view / "manifest.json").read_text())
    if (
        {path.name for path in args.selector_view.iterdir()}
        != {*SELECTOR_FILES, "manifest.json", "checksums.sha256"}
        or selector_manifest.get("status") != "label_blind_selector_view"
        or selector_manifest.get("source_dataset_manifest_sha256") != sha256_file(args.dataset / "manifest.json")
        or selector_manifest.get("structural_audit_sha256") != sha256_file(audit_path)
        or selector_manifest.get("files_sha256") != {
            name: sha256_file(args.dataset / name) for name in SELECTOR_FILES
        }
        or any(sha256_file(args.selector_view / name) != sha256_file(args.dataset / name)
               for name in SELECTOR_FILES)
    ):
        raise ValueError("selector view is not a current query-only projection")
    oracle = _gate(args.private / "oracle_gate_v1", args.dataset,
                   args.private / "oracle_selections.jsonl", 128)
    negative = _gate(args.private / "negative_gate_v1", args.dataset,
                     args.private / "complete_negative_selections.jsonl", 0)
    events = {str(row["event_id"]): row for row in load_jsonl(args.dataset / "events.jsonl")}
    positives = {str(row["event_id"]): row for row in load_jsonl(args.private / "oracle_selections.jsonl")}
    negatives = {str(row["event_id"]): row for row in load_jsonl(args.private / "complete_negative_selections.jsonl")}
    if set(events) != set(positives) or set(events) != set(negatives):
        raise ValueError("private assignments do not cover every event")
    sides = Counter()
    pool = {str(row["event_id"]): [] for row in load_jsonl(args.dataset / "query_only_events.jsonl")}
    for row in load_jsonl(args.dataset / "candidate_selections.jsonl"):
        pool[str(row["event_id"])].append(row)
    for event_id, event in events.items():
        candidates = pool[event_id]
        if len(candidates) != 2 or _selected(positives[event_id]) == _selected(negatives[event_id]):
            raise ValueError(f"invalid positive/negative pairing: {event_id}")
        side = next((index for index, row in enumerate(candidates)
                     if _selected(row) == _selected(positives[event_id])), None)
        if side is None or _selected(candidates[1 - side]) != _selected(negatives[event_id]):
            raise ValueError(f"private pair differs from public candidates: {event_id}")
        sides[str(event["task_key"]), side] += 1
    if len(sides) != 32 or any(count != 4 for count in sides.values()):
        raise ValueError("candidate-side assignments are not balanced 4/4 within each family")
    training = json.loads(args.training_result.read_text())
    if (
        training.get("status") != "trained_on_inspected_development_data_for_prospective_v9_only"
        or training.get("source_event_counts") != {"role_swap": 24, "v7": 128, "v8": 128}
        or training.get("epochs") != 2 or training.get("role_repeat") != 4
        or training.get("batch_size") != 2 or training.get("max_length") != 384
        or training.get("learning_rate") != 2e-5 or training.get("seed") != 2027
        or training.get("script_sha256") != sha256_file(ROOT / "experiments/train_v9_hybrid_neural_candidate.py")
    ):
        raise ValueError("unexpected model-training provenance")
    model_files = sorted(path for path in args.model_dir.iterdir() if path.is_file())
    if not model_files or not (args.model_dir / "model.safetensors").is_file():
        raise ValueError("missing complete model checkpoint")
    model_hashes = {path.name: sha256_file(path) for path in model_files}
    if training.get("model_files_sha256") != model_hashes:
        raise ValueError("model files changed since training")
    base_nli_files = sorted(path for path in args.base_nli_model_dir.iterdir() if path.is_file())
    if not base_nli_files or not (args.base_nli_model_dir / "model.safetensors").is_file():
        raise ValueError("missing off-the-shelf NLI control checkpoint")
    base_nli_hashes = {path.name: sha256_file(path) for path in base_nli_files}
    if training.get("base_model_sha256") != base_nli_hashes["model.safetensors"]:
        raise ValueError("off-the-shelf NLI control does not match training initialization")
    if _digest(TYPESENSE_BINARY, "sha256") != TYPESENSE_SHA256 or _digest(SOLR_ARCHIVE, "sha512") != SOLR_SHA512:
        raise ValueError("pinned real-service runtimes changed")
    result = {
        "status": "frozen_unscored_v9_candidate",
        "claim_boundary": "Only task validity and a prospective rule are frozen; no method or baseline score has been measured.",
        "primary_contrast": "BM25-sum text_and_plan versus BM25 on non-ties plus contrastive NLI on exact lexical ties",
        "primary_method": "v9_lexical_tie_hybrid",
        "primary_baseline": "same-candidate BM25-sum text_and_plan",
        "decision_gate": (
            "Advance the method only if the primary paired difference is positive, "
            "its 16-family cluster-bootstrap 95% interval has a lower bound above zero, "
            "and neither new provider has a negative method-minus-BM25 difference. "
            "A claim for development training additionally requires the trained hybrid "
            "to outperform the off-the-shelf NLI hybrid with a positive clustered interval. "
            "This controlled benchmark alone does not establish WWW submission readiness."
        ),
        "secondary_controls": ["BM25-sum text-only", "literal-value overlap", "role-literal shortcut", "trained neural-only",
                               "off-the-shelf NLI hybrid", "off-the-shelf NLI-only", "seeded-hash tie break"],
        "rule": {"tie_tolerance": 1e-12, "seed": 2027, "max_length": 384,
                 "neural_serializer": "difference_focused_paths", "token_budget": 256},
        "dataset_path": str(args.dataset.resolve()),
        "selector_view_path": str(args.selector_view.resolve()),
        "dataset_files_sha256": {name: sha256_file(args.dataset / name) for name in DATA_FILES},
        "selector_view_manifest_sha256": sha256_file(args.selector_view / "manifest.json"),
        "structural_audit_sha256": sha256_file(audit_path),
        "validity_gates": {"oracle": oracle, "complete_negative": negative},
        "runtime_sha256": {"typesense": TYPESENSE_SHA256, "solr_archive_sha512": SOLR_SHA512},
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCES},
        "model_files_sha256": model_hashes,
        "base_nli_model_files_sha256": base_nli_hashes,
        "training_result_sha256": sha256_file(args.training_result),
        "private_assignment_commitment_sha256": json.loads((args.dataset / "manifest.json").read_text())["assignment_salt_commitment_sha256"],
        "balanced_private_sides": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print("frozen 128 unscored v9 candidate incidents and primary comparator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
