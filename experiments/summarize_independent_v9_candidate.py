#!/usr/bin/env python3
"""Score only hash-committed v9 choices against both complete live gates."""

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
from experiments.commit_independent_v9_selections import METHODS  # noqa: E402
from experiments.export_v9_selector_view import SELECTOR_FILES  # noqa: E402
from experiments.summarize_independent_v7_confirmation import paired_contrast  # noqa: E402


def _selected(row: dict) -> tuple[str, ...]:
    return tuple(sorted(map(str, row["selected_memory_ids"])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument("--oracle-gate", type=Path, required=True)
    parser.add_argument("--negative-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a prior analysis")
    freeze = json.loads(args.freeze.read_text())
    commitment = json.loads(args.commitment.read_text())
    if (
        freeze.get("status") != "frozen_unscored_v9_candidate"
        or commitment.get("status") != "all_v9_methods_committed_before_outcome_mapping"
        or commitment.get("freeze_sha256") != sha256_file(args.freeze)
        or commitment.get("outcomes_loaded") is not False
        or set(commitment.get("methods", {})) != set(METHODS)
    ):
        raise ValueError("missing or changed pre-outcome v9 commitment")
    if freeze["dataset_files_sha256"] != {
        name: sha256_file(args.dataset / name) for name in freeze["dataset_files_sha256"]
    }:
        raise ValueError("frozen public dataset changed")
    selector_view = Path(freeze["selector_view_path"])
    if (
        sha256_file(selector_view / "manifest.json") != freeze["selector_view_manifest_sha256"]
        or {path.name for path in selector_view.iterdir()}
        != {*SELECTOR_FILES, "manifest.json", "checksums.sha256"}
        or any(sha256_file(selector_view / name) != freeze["dataset_files_sha256"][name]
               for name in SELECTOR_FILES)
    ):
        raise ValueError("frozen selector-only view changed")
    for name, digest in freeze["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"frozen source changed: {name}")
    event_rows = load_jsonl(args.dataset / "events.jsonl")
    events = {str(row["event_id"]): row for row in event_rows}
    if len(event_rows) != 128 or len(events) != 128:
        raise ValueError("incomplete event set")
    candidate_sets: dict[str, set[tuple[str, ...]]] = {event_id: set() for event_id in events}
    for row in load_jsonl(args.dataset / "candidate_selections.jsonl"):
        candidate_sets[str(row["event_id"])].add(_selected(row))
    if any(len(pool) != 2 for pool in candidate_sets.values()):
        raise ValueError("incomplete two-candidate pool")
    gate_outcomes: dict[tuple[str, tuple[str, ...]], int] = {}
    for label, directory, expected in (
        ("oracle", args.oracle_gate, 1), ("complete_negative", args.negative_gate, 0),
    ):
        gate = freeze["validity_gates"][label]
        results_path = directory / "execution_results.jsonl"
        manifest_path = directory / "run_manifest.json"
        if (
            sha256_file(results_path) != gate["execution_results_sha256"]
            or sha256_file(manifest_path) != gate["manifest_sha256"]
        ):
            raise ValueError(f"{label}: physical gate changed since freeze")
        rows = load_jsonl(results_path)
        if len(rows) != 128 or {str(row["event_id"]) for row in rows} != set(events):
            raise ValueError(f"{label}: incomplete physical gate")
        for row in rows:
            event_id = str(row["event_id"])
            selected = _selected(row)
            key = event_id, selected
            if (
                selected not in candidate_sets[event_id] or key in gate_outcomes
                or row.get("checker", {}).get("goal_satisfied") is not bool(expected)
                or row.get("execution_error") is not None or row.get("checker_error") is not None
            ):
                raise ValueError(f"{label}: invalid physical outcome: {event_id}")
            gate_outcomes[key] = expected
    if len(gate_outcomes) != 256:
        raise ValueError("not all candidates have physical outcomes")
    outcomes: dict[str, dict[str, int]] = {}
    selections: dict[str, dict[str, dict]] = {}
    method_summary = {}
    families = {event_id: str(row["task_key"]) for event_id, row in events.items()}
    family_keys = sorted(set(families.values()))
    providers = sorted({str(row["provider_id"]) for row in events.values()})
    for method in METHODS:
        entry = commitment["methods"][method]
        selection_path = Path(entry["selection_path"])
        if (
            sha256_file(selection_path) != entry["selections_sha256"]
            or sha256_file(selection_path.parent / "manifest.json") != entry["manifest_sha256"]
            or sha256_file(selection_path.parent / "candidate_scores.jsonl") != entry["candidate_scores_sha256"]
        ):
            raise ValueError(f"{method}: selection artifact changed after commitment")
        rows = load_jsonl(selection_path)
        by_event = {str(row["event_id"]): row for row in rows}
        if len(rows) != 128 or len(by_event) != 128 or set(by_event) != set(events):
            raise ValueError(f"{method}: incomplete selections")
        selections[method] = by_event
        outcomes[method] = {
            event_id: gate_outcomes[event_id, _selected(row)] for event_id, row in by_event.items()
        }
        method_summary[method] = {
            "verified_successes": sum(outcomes[method].values()),
            "events": 128, "vsr": sum(outcomes[method].values()) / 128,
            "by_provider_successes": {
                provider: sum(outcomes[method][event_id] for event_id, event in events.items()
                              if event["provider_id"] == provider) for provider in providers
            },
            "by_family_successes": {
                family: sum(outcomes[method][event_id] for event_id in events if families[event_id] == family)
                for family in family_keys
            },
            "mean_token_cost": sum(int(row["token_cost"]) for row in rows) / 128,
        }
    primary = paired_contrast(outcomes, families, {"hybrid": 1, "bm25_text_and_plan": -1})
    training_contrast = paired_contrast(outcomes, families, {"hybrid": 1, "base_nli_hybrid": -1})
    secondary = {
        comparator: paired_contrast(outcomes, families, {"hybrid": 1, comparator: -1})
        for comparator in METHODS if comparator not in {"hybrid", "bm25_text_and_plan"}
    }
    by_decision: dict[str, Counter[str]] = {}
    for event_id, row in selections["hybrid"].items():
        decision = str(row.get("decision"))
        counter = by_decision.setdefault(decision, Counter())
        counter["events"] += 1
        counter["hybrid_successes"] += outcomes["hybrid"][event_id]
        counter["bm25_successes"] += outcomes["bm25_text_and_plan"][event_id]
    expected_strata = {"bm25_non_tie", "neural_lexical_tie"}
    if not set(by_decision).issubset(expected_strata) or sum(
        count["events"] for count in by_decision.values()
    ) != 128:
        raise ValueError("hybrid decision strata are invalid")
    result = {
        "status": "frozen_v9_candidate_scored_from_complete_real_service_gates",
        "claim_boundary": "Controlled generated operation templates, not natural production drift or third-party replay; interpret method advantage with clustered uncertainty.",
        "freeze_sha256": sha256_file(args.freeze),
        "commitment_sha256": sha256_file(args.commitment),
        "candidate_physical_executions": 256, "family_clusters": 16,
        "methods": method_summary, "primary_paired_contrast": primary,
        "training_paired_contrast": training_contrast,
        "secondary_paired_contrasts": secondary,
        "hybrid_decision_strata": {
            key: {metric: by_decision.get(key, Counter())[metric] for metric in (
                "events", "hybrid_successes", "bm25_successes",
            )} for key in sorted(expected_strata)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({"methods": method_summary, "primary": primary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
