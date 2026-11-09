#!/usr/bin/env python3
"""Hash the exact v7 inputs and prespecified comparisons before method scoring."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import (  # noqa: E402
    load_jsonl,
    parse_utc,
    sha256_file,
    write_json,
)

SOURCE_MODULES = (
    "corem/real_eval.py",
    "corem/types.py",
    "corem/torch_encoding.py",
    "corem/torch_model.py",
    "experiments/build_independent_v7_staging.py",
    "experiments/independent_v7_extra_specs.py",
    "experiments/independent_v7_diverse_specs.py",
    "experiments/independent_v7_runner.py",
    "experiments/public_multiservice_challenge_runner.py",
    "experiments/independent_v7_checker.py",
    "experiments/independent_v7_checker_v2.py",
    "experiments/independent_v7_checker_v3.py",
    "experiments/independent_v7_checker_v4.py",
    "experiments/independent_v7_checker_v5.py",
    "experiments/independent_v7_checker_v6.py",
    "experiments/independent_v7_checker_v7.py",
    "experiments/audit_independent_v7_confirmation_gate.py",
    "experiments/generate_public_multiservice_hgt_ensemble_selections.py",
    "experiments/generate_public_multiservice_matched_baseline.py",
    "experiments/generate_independent_v7_candidate_bm25.py",
    "experiments/generate_public_workflow_selections.py",
    "experiments/evaluate_public_multiservice_hgt_ensemble.py",
    "experiments/run_real_track.py",
    "experiments/analyze_real_track.py",
    "experiments/freeze_independent_v7_candidate.py",
    "experiments/commit_independent_v7_selections.py",
)
DATA_FILES = (
    "events.jsonl",
    "memories.jsonl",
    "dependency_hyperedges.jsonl",
    "manifest.json",
    "checksums.sha256",
)
SELECTOR_FILES = (
    "splits.jsonl",
    "candidate_selections.jsonl",
    "query_only_events.jsonl",
)
RUNTIME_FILES = (
    "gitea-28.1.0-linux-amd64",
    "gitea-28.1.0-linux-amd64.xz",
    "meilisearch-v1.55.0-linux-amd64",
)
EXPECTED_RUNTIME_SHA256 = {
    "gitea-28.1.0-linux-amd64": "ea93cfa4a300564ac457d3903469fea49ec5a4da37d0cebaa435f8e329ea64d0",
    "gitea-28.1.0-linux-amd64.xz": "d998643be28c10bbc94a3ab5a5705230fcb54e92f04b9aef3f694dd6d20cb97d",
    "meilisearch-v1.55.0-linux-amd64": "7e33f5889235f14290074a19a6d3574bd9b10fb2ee8731eabb7dafd0714b0513",
}


def _entry(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--structural-audit", type=Path, required=True)
    parser.add_argument("--oracle-analysis", type=Path, required=True)
    parser.add_argument("--oracle-executions", type=Path, required=True)
    parser.add_argument("--model-dirs", type=Path, nargs=5, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a prior freeze manifest")
    audit = json.loads(args.structural_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "structurally_ready_for_oracle_gate":
        parser.error("structural gate must pass before freezing")
    validity = json.loads(args.oracle_analysis.read_text(encoding="utf-8"))
    methods = {row.get("method"): row for row in validity.get("methods", [])}
    oracle = methods.get("oracle", {})
    negative = methods.get("complete_negative", {})
    if (
        validity.get("status") != "complete_static_real_track"
        or oracle.get("n", 0) < 128
        or oracle.get("vsr", 0) < 0.95
        or negative.get("n", 0) < 128
        or negative.get("vsr") != 0
    ):
        parser.error("independent oracle/negative validity gate has not passed")
    events = load_jsonl(args.dataset / "events.jsonl")
    dataset_manifest = json.loads(
        (args.dataset / "manifest.json").read_text(encoding="utf-8")
    )
    collection_end = parse_utc(
        dataset_manifest["provenance"]["collection_end_utc"], "collection_end_utc"
    )
    if any(
        parse_utc(row["event_time"], "event_time") > collection_end for row in events
    ):
        parser.error("event chronology exceeds source-collection cutoff")
    if not dataset_manifest.get("assignment_commitment_sha256"):
        parser.error("candidate requires randomized, opaque assignment")
    candidates = load_jsonl(args.selections / "candidate_selections.jsonl")
    executions = load_jsonl(args.oracle_executions)
    if len(events) < 128 or len(candidates) < 256:
        parser.error("candidate count below the prespecified gate")
    runtime_root = ROOT / "experiments/real_tracks/runtime/independent_v7"
    if any(
        sha256_file(runtime_root / name) != EXPECTED_RUNTIME_SHA256[name]
        for name in RUNTIME_FILES
    ):
        parser.error("runtime binary checksum differs from pinned release")
    expected_execution_keys = {
        (str(event["event_id"]), method)
        for event in events
        for method in ("oracle", "complete_negative")
    }
    observed_execution_keys = {
        (str(row.get("event_id")), str(row.get("method"))) for row in executions
    }
    if (
        len(executions) != 2 * len(events)
        or any(
            row.get("reset_verified") is not True
            or row.get("runner_error") is not None
            or row.get("execution_error") is not None
            or row.get("checker_error") is not None
            or not isinstance(row.get("execution"), dict)
            or row["execution"].get("completed") is not True
            for row in executions
        )
        or observed_execution_keys != expected_execution_keys
    ):
        parser.error("complete error-free fresh-service validity executions required")
    states = []
    training_manifests = []
    for seed, directory in zip((17, 29, 41, 53, 67), args.model_dirs):
        result_path = directory / "result.json"
        training_result = json.loads(result_path.read_text(encoding="utf-8"))
        config = training_result.get("configuration", {})
        if (
            config.get("seeds") != [seed]
            or config.get("epochs") != 16
            or config.get("dimension") != 32
            or config.get("text_hash_dimension") != 64
            or config.get("message_layers") != 2
            or config.get("lexical_features") is not True
            or config.get("relation_weight") != 0
            or config.get("exclude_provider") is not None
            or set(config.get("variants", []))
            != {"hgt_corem", "hgt_additive", "text_complement", "text_additive"}
        ):
            parser.error(
                f"model directory has an unexpected training contract: {directory}"
            )
        training_manifests.append({"seed": seed, **_entry(result_path)})
        for variant in (
            "hgt_corem",
            "hgt_additive",
            "text_complement",
            "text_additive",
        ):
            path = directory / f"{variant}_seed_{seed}.pt"
            states.append({"seed": seed, "variant": variant, **_entry(path)})
    write_json(
        args.output,
        {
            "schema_version": "corem-v7-preselection-freeze-v1",
            "status": "frozen_before_method_selection",
            "created_utc": datetime.now(UTC).isoformat(),
            "dataset": {name: _entry(args.dataset / name) for name in DATA_FILES},
            "selector_inputs": {
                name: _entry(args.selections / name) for name in SELECTOR_FILES
            },
            "structural_audit": _entry(args.structural_audit),
            "oracle_analysis": _entry(args.oracle_analysis),
            "oracle_executions": _entry(args.oracle_executions),
            "source_modules": {name: _entry(ROOT / name) for name in SOURCE_MODULES},
            "runtime_assets": {
                name: _entry(runtime_root / name) for name in RUNTIME_FILES
            },
            "model_states": states,
            "model_training_manifests": training_manifests,
            "method_contract": {
                "same_complete_candidates": True,
                "label_free_query_view": True,
                "primary_methods": [
                    "graph_complement",
                    "graph_additive",
                    "text_complement",
                    "text_additive",
                    "candidate_bm25_max",
                    "version_bm25_closure",
                ],
                "diagnostics": ["pure_neural", "candidate_bm25_sum"],
                "neural_weight": 0.9,
                "bm25_weight": 0.1,
                "lexical_aggregation": "max",
                "relation_vocabulary_policy": "collapse_to_training",
                "training_relation_type": "three-step-runtime-handoff",
                "architecture_transfer_note": (
                    "The v5 checkpoint has one HGT edge relation, while the new "
                    "seven-/five-/four-/three-step contracts introduce four names. "
                    "Before opening outcomes, project all handoff-length names "
                    "to the frozen training relation for neural inference only; "
                    "keep native hyperedges for candidate and closure checks."
                ),
                "tie_break_policy": "seeded_hash",
                "tie_seed": 2027,
                "token_budget": 256,
                "version_bm25_closure_scope": (
                    "Same active memory bank and verified closure, but not restricted "
                    "to the two enumerated candidates; report separately from "
                    "candidate-matched factorial contrasts."
                ),
                "model_seeds": [17, 29, 41, 53, 67],
                "model_dimension": 32,
                "text_hash_dimension": 64,
                "message_layers": 2,
                "lexical_features": True,
                "selection_split": "test",
                "outcomes_used_for_method_choice": False,
            },
            "important_limitations": [
                "A structural pass is not an oracle-validity pass.",
                "The same team generated items and evaluates methods; cryptographic IDs reduce but do not eliminate human-design bias.",
                "Only two new providers are represented; no provider-population interval is claimed.",
            ],
        },
    )
    print(
        f"froze {len(events)} events, {len(candidates)} candidates, and {len(states)} model states"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
