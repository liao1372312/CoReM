#!/usr/bin/env python3
"""Join committed v8 choices to previously measured real-service candidate gates."""

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
from experiments.summarize_independent_v7_confirmation import paired_contrast  # noqa: E402


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
        freeze.get("status") != "frozen_unscored_v8_candidate"
        or commitment.get("status") != "all_v8_methods_committed_before_outcome_mapping"
        or commitment.get("freeze_sha256") != sha256_file(args.freeze)
        or commitment.get("outcomes_loaded") is not False
    ):
        raise ValueError("missing or changed pre-outcome method commitment")
    if freeze["dataset_files_sha256"] != {
        name: sha256_file(args.dataset / name) for name in freeze["dataset_files_sha256"]
    }:
        raise ValueError("frozen dataset changed")
    for name, digest in freeze["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"frozen source changed: {name}")
    events = {str(row["event_id"]): row for row in load_jsonl(args.dataset / "events.jsonl")}
    if len(events) != 128:
        raise ValueError("expected 128 unique events")
    candidates: dict[str, set[tuple[str, ...]]] = {}
    for row in load_jsonl(args.dataset / "candidate_selections.jsonl"):
        candidates.setdefault(str(row["event_id"]), set()).add(
            tuple(sorted(map(str, row["selected_memory_ids"])))
        )
    if any(len(candidates.get(event_id, ())) != 2 for event_id in events):
        raise ValueError("each event needs two frozen candidates")
    gate_outcomes: dict[tuple[str, tuple[str, ...]], int] = {}
    for label, gate_dir, expected in (
        ("oracle", args.oracle_gate, 1),
        ("complete_negative", args.negative_gate, 0),
    ):
        gate = freeze["validity_gates"][label]
        manifest_path = gate_dir / "run_manifest.json"
        execution_path = gate_dir / "execution_results.jsonl"
        if (
            sha256_file(manifest_path) != gate["manifest_sha256"]
            or sha256_file(execution_path) != gate["execution_results_sha256"]
        ):
            raise ValueError(f"{label} gate differs from the freeze")
        rows = load_jsonl(execution_path)
        if len(rows) != 128 or {str(row["event_id"]) for row in rows} != set(events):
            raise ValueError(f"{label} gate is incomplete")
        for row in rows:
            event_id = str(row["event_id"])
            selected = tuple(sorted(map(str, row["selected_memory_ids"])))
            if selected not in candidates[event_id]:
                raise ValueError(f"{label} gate selected an unknown candidate")
            outcome = int(bool(row["checker"]["goal_satisfied"]))
            if outcome != expected or (event_id, selected) in gate_outcomes:
                raise ValueError(f"{label} gate has wrong/duplicate outcome")
            gate_outcomes[(event_id, selected)] = outcome
    if len(gate_outcomes) != 256:
        raise ValueError("not every candidate received a physical outcome")
    outcomes: dict[str, dict[str, int]] = {}
    methods = {}
    for method, entry in commitment["methods"].items():
        path = Path(entry["selection_path"])
        if (
            sha256_file(path) != entry["selections_sha256"]
            or sha256_file(path.parent / "manifest.json") != entry["manifest_sha256"]
        ):
            raise ValueError(f"{method} selections changed after commitment")
        rows = load_jsonl(path)
        if len(rows) != 128 or {str(row["event_id"]) for row in rows} != set(events):
            raise ValueError(f"{method} selections are incomplete")
        selected_outcomes = {}
        for row in rows:
            event_id = str(row["event_id"])
            candidate = tuple(sorted(map(str, row["selected_memory_ids"])))
            selected_outcomes[event_id] = gate_outcomes[(event_id, candidate)]
        outcomes[method] = selected_outcomes
        by_provider = Counter()
        by_family = Counter()
        for event_id, outcome in selected_outcomes.items():
            if outcome:
                event = events[event_id]
                by_provider[str(event["provider_id"])] += 1
                by_family[str(event["task_key"])] += 1
        methods[method] = {
            "verified_successes": sum(selected_outcomes.values()),
            "events": 128,
            "vsr": sum(selected_outcomes.values()) / 128,
            "by_provider_successes": dict(sorted(by_provider.items())),
            "by_family_successes": dict(sorted(by_family.items())),
            "mean_token_cost": sum(int(row["token_cost"]) for row in rows) / 128,
        }
    families = {event_id: str(event["task_key"]) for event_id, event in events.items()}
    primary = paired_contrast(outcomes, families, {"path_reranker": 1, "bm25_text_and_plan": -1})
    comparisons = {
        comparator: paired_contrast(outcomes, families, {"path_reranker": 1, comparator: -1})
        for comparator in ("bm25_text", "literal_overlap")
    }
    write_json(args.output, {
        "status": "frozen_v8_candidate_scored_from_complete_real_service_gates",
        "claim_boundary": "Controlled synthetic operation-family benchmark; not natural drift, not an external independent replay, and not alone a WWW-ready method claim.",
        "freeze_sha256": sha256_file(args.freeze),
        "commitment_sha256": sha256_file(args.commitment),
        "methods": methods,
        "primary_paired_contrast": primary,
        "secondary_paired_contrasts": comparisons,
        "candidate_physical_executions": 256,
        "family_clusters": 16,
    })
    print(json.dumps({"methods": methods, "primary": primary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
