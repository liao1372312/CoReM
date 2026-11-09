#!/usr/bin/env python3
"""Summarize the hash-committed v10 E2 outcome-hidden factorial."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import load_jsonl, sha256_file, write_json  # noqa: E402
from experiments.commit_independent_v10_e2_selections import CELLS  # noqa: E402

FACTORS = {
    "graph_complement": ("On", "On", "On"),
    "graph_additive": ("On", "Off", "On"),
    "text_complement": ("Off", "On", "On"),
    "text_additive": ("Off", "Off", "On"),
    "graph_complement_no_lexical": ("On", "On", "Off"),
    "graph_additive_no_lexical": ("On", "Off", "Off"),
    "text_complement_no_lexical": ("Off", "On", "Off"),
    "text_additive_no_lexical": ("Off", "Off", "Off"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument("--oracle-gate", type=Path, required=True)
    parser.add_argument("--negative-gate", type=Path, required=True)
    parser.add_argument("--selections-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def bundle_key(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return (
        str(row["event_id"]),
        tuple(sorted(map(str, row["selected_memory_ids"]))),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_sign_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def cluster_interval(
    deltas: dict[str, int], families: dict[str, str]
) -> tuple[float, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for event_id, delta in deltas.items():
        grouped[families[event_id]].append(delta)
    names = sorted(grouped)
    generator = random.Random(2027)
    draws = []
    for _ in range(20_000):
        sampled = [generator.choice(names) for _ in names]
        values = [value for name in sampled for value in grouped[name]]
        draws.append(sum(values) / len(values))
    draws.sort()
    return 100.0 * draws[500], 100.0 * draws[19_499]


def validate_gate_row(row: dict[str, Any], expected: int) -> None:
    checker = row.get("checker")
    if (
        row.get("reset_verified") is not True
        or row.get("checker_independent") is not True
        or row.get("checker_before_teardown") is not True
        or row.get("execution_error") is not None
        or row.get("checker_error") is not None
        or not isinstance(checker, dict)
        or checker.get("goal_satisfied") is not bool(expected)
    ):
        raise ValueError(f"invalid physical gate row: {row.get('event_id')}")
    if expected:
        execution = row.get("execution")
        checks = checker.get("state_checks")
        if (
            not isinstance(execution, dict)
            or execution.get("completed") is not True
            or checker.get("trace_statuses_verified") is not True
            or not isinstance(checks, list)
            or not checks
            or any(check.get("passed") is not True for check in checks)
        ):
            raise ValueError(f"positive gate lacks complete evidence: {row['event_id']}")


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    commitment = json.loads(args.commitment.read_text(encoding="utf-8"))
    if (
        commitment.get("status")
        != "all_factorial_cells_committed_before_outcome_execution"
        or commitment.get("outcomes_loaded") is not False
        or set(commitment.get("cells", {})) != set(CELLS)
    ):
        raise ValueError("missing complete pre-outcome factorial commitment")
    for name, digest in commitment["dataset_files_sha256"].items():
        if sha256_file(args.dataset / name) != digest:
            raise ValueError(f"public dataset changed after commitment: {name}")
    events = {
        str(row["event_id"]): row for row in load_jsonl(args.dataset / "events.jsonl")
    }
    if len(events) != 128:
        raise ValueError("v10 E2 requires 128 unique events")
    candidates: dict[str, set[tuple[str, ...]]] = {key: set() for key in events}
    for row in load_jsonl(args.dataset / "candidate_selections.jsonl"):
        candidates[str(row["event_id"])].add(bundle_key(row)[1])
    if any(len(pool) != 2 for pool in candidates.values()):
        raise ValueError("each event must contain two candidate bundles")

    labels: dict[tuple[str, tuple[str, ...]], int] = {}
    gate_sources = {}
    for label, gate, expected in (
        ("oracle", args.oracle_gate, 1),
        ("complete_negative", args.negative_gate, 0),
    ):
        results_path = gate / "execution_results.jsonl"
        manifest_path = gate / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = load_jsonl(results_path)
        if (
            len(rows) != 128
            or manifest.get("rows") != 128
            or manifest.get("runner_errors") != 0
            or manifest.get("checker_errors") != 0
            or manifest.get("verified_successes") != 128 * expected
            or {str(row["event_id"]) for row in rows} != set(events)
        ):
            raise ValueError(f"{label}: incomplete or failed physical gate")
        for row in rows:
            validate_gate_row(row, expected)
            key = bundle_key(row)
            if key in labels or key[1] not in candidates[key[0]]:
                raise ValueError(f"{label}: duplicate or unknown candidate execution")
            labels[key] = expected
        gate_sources[label] = {
            "execution_results_sha256": sha256_file(results_path),
            "manifest_sha256": sha256_file(manifest_path),
        }
    if len(labels) != 256 or any(
        sorted(labels[(event_id, bundle)] for bundle in pool) != [0, 1]
        for event_id, pool in candidates.items()
    ):
        raise ValueError("physical gates do not establish one positive and one negative")

    metrics = []
    outcomes: dict[str, dict[str, int]] = {}
    sources = {}
    for method in FACTORS:
        directory = args.selections_root / f"results_independent_v10_e2_{method}_selections"
        selection_path = directory / "selections.jsonl"
        scores_path = directory / "candidate_scores.jsonl"
        manifest_path = directory / "manifest.json"
        committed = commitment["cells"][method]
        observed = {
            "selections_sha256": sha256_file(selection_path),
            "candidate_scores_sha256": sha256_file(scores_path),
            "manifest_sha256": sha256_file(manifest_path),
        }
        if any(observed[name] != committed[name] for name in observed):
            raise ValueError(f"{method}: selection artifact changed after commitment")
        selections = load_jsonl(selection_path)
        scores = load_jsonl(scores_path)
        if len(selections) != 128 or len(scores) != 256:
            raise ValueError(f"{method}: incomplete selections or scores")
        by_event = {str(row["event_id"]): row for row in selections}
        if len(by_event) != 128 or set(by_event) != set(events):
            raise ValueError(f"{method}: duplicate or missing event")
        if {bundle_key(row) for row in scores} != set(labels):
            raise ValueError(f"{method}: candidate inventory differs from gates")
        outcomes[method] = {
            event_id: labels[bundle_key(row)] for event_id, row in by_event.items()
        }
        successes = sum(outcomes[method].values())
        squared_errors = []
        for row in scores:
            probability = float(row["score"])
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{method}: score is not a probability")
            squared_errors.append((probability - labels[bundle_key(row)]) ** 2)
        graph, set_term, lexical = FACTORS[method]
        metrics.append(
            {
                "graph": graph,
                "set_term": set_term,
                "lexical": lexical,
                "method": method,
                "events": 128,
                "candidate_sets": 256,
                "verified_successes": successes,
                "vsr": successes / 128,
                "top1": successes / 128,
                "brier": sum(squared_errors) / len(squared_errors),
            }
        )
        sources[method] = observed

    def mean_vsr(field: str, value: str) -> float:
        values = [float(row["vsr"]) for row in metrics if row[field] == value]
        return sum(values) / len(values)

    main_effects = {
        "graph_on_minus_off": 100.0 * (mean_vsr("graph", "On") - mean_vsr("graph", "Off")),
        "set_on_minus_off": 100.0 * (mean_vsr("set_term", "On") - mean_vsr("set_term", "Off")),
        "lexical_on_minus_off": 100.0 * (mean_vsr("lexical", "On") - mean_vsr("lexical", "Off")),
    }
    families = {event_id: str(event["task_key"]) for event_id, event in events.items()}
    full = outcomes["graph_complement"]
    comparisons = []
    for comparator in FACTORS:
        if comparator == "graph_complement":
            continue
        deltas = {
            event_id: full[event_id] - outcomes[comparator][event_id]
            for event_id in events
        }
        wins = sum(value > 0 for value in deltas.values())
        losses = sum(value < 0 for value in deltas.values())
        family_values = {
            family: sum(deltas[event_id] for event_id in events if families[event_id] == family) / 8
            for family in sorted(set(families.values()))
        }
        low, high = cluster_interval(deltas, families)
        comparisons.append(
            {
                "contrast": f"graph_complement_minus_{comparator}",
                "delta_vsr_pp": 100.0 * sum(deltas.values()) / 128,
                "event_wins": wins,
                "event_losses": losses,
                "event_ties": 128 - wins - losses,
                "event_exact_sign_p": exact_sign_p(wins, losses),
                "family_wins": sum(value > 0 for value in family_values.values()),
                "family_losses": sum(value < 0 for value in family_values.values()),
                "family_ties": sum(value == 0 for value in family_values.values()),
                "family_bootstrap_ci_low_pp": low,
                "family_bootstrap_ci_high_pp": high,
            }
        )

    args.output.mkdir(parents=True)
    write_csv(args.output / "method_metrics.csv", metrics)
    write_csv(args.output / "paired_comparisons.csv", comparisons)
    table = [
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Graph & Set term & Lexical & VSR & Top-1 & Brier\\",
        r"\midrule",
    ]
    for row in metrics:
        table.append(
            f'{row["graph"]} & {row["set_term"]} & {row["lexical"]} & '
            f'{100.0 * float(row["vsr"]):.1f} & '
            f'{100.0 * float(row["top1"]):.1f} & {float(row["brier"]):.3f}\\\\'
        )
    table.extend((r"\bottomrule", r"\end{tabular}"))
    (args.output / "table.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    write_json(
        args.output / "result.json",
        {
            "schema_version": "corem-independent-v10-e2-factorial-v1",
            "status": "complete_outcome_hidden_event_confirmation",
            "claim_boundary": (
                "Hash-committed selections on 128 previously unexecuted incidents from "
                "sixteen known Solr/Typesense task families. This supports untouched-event "
                "replication, not unseen-family or natural-production generalization."
            ),
            "events": 128,
            "candidate_sets": 256,
            "family_clusters": 16,
            "rows": metrics,
            "main_effects_vsr_pp": main_effects,
            "paired_comparisons": comparisons,
            "commitment_sha256": sha256_file(args.commitment),
            "physical_gate_sources": gate_sources,
            "sources": sources,
        },
    )
    names = ("method_metrics.csv", "paired_comparisons.csv", "result.json", "table.tex")
    (args.output / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256((args.output / name).read_bytes()).hexdigest()}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    print(json.dumps({row["method"]: row["verified_successes"] for row in metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
