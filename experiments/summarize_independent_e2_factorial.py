#!/usr/bin/env python3
"""Summarize the post-hoc 2x2x2 new-task factorial for paper E2."""

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

from corem.real_eval import load_jsonl, score_execution_row, write_json  # noqa: E402


CELLS = (
    ("On", "On", "On", "graph_complement", "results_independent_v7_candidate_v4_graph_complement_selections_v2"),
    ("On", "Off", "On", "graph_additive", "results_independent_v7_candidate_v4_graph_additive_selections_v2"),
    ("Off", "On", "On", "text_complement", "results_independent_v7_candidate_v4_text_complement_selections_v2"),
    ("Off", "Off", "On", "text_additive", "results_independent_v7_candidate_v4_text_additive_selections_v2"),
    ("On", "On", "Off", "pure_neural", "results_independent_v7_candidate_v4_pure_neural_selections_v2"),
    ("On", "Off", "Off", "graph_additive_no_lexical", "results_independent_v7_candidate_v4_graph_additive_no_lexical_selections_v1"),
    ("Off", "On", "Off", "text_complement_no_lexical", "results_independent_v7_candidate_v4_text_complement_no_lexical_selections_v1"),
    ("Off", "Off", "Off", "text_additive_no_lexical", "results_independent_v7_candidate_v4_text_additive_no_lexical_selections_v1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/public_independent_v7_candidate_v4",
    )
    parser.add_argument(
        "--execution-cache",
        type=Path,
        default=(
            ROOT
            / "experiments/results_independent_v7_candidate_v4_full_oracle_negative_clean/execution_results.jsonl"
        ),
    )
    parser.add_argument(
        "--experiments-root", type=Path, default=ROOT / "experiments"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/results_independent_e2_factorial_v2",
    )
    return parser.parse_args()


def bundle_key(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return (
        str(row["event_id"]),
        tuple(sorted(map(str, row["selected_memory_ids"]))),
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    events = {
        str(row["event_id"]): row for row in load_jsonl(args.dataset / "events.jsonl")
    }
    labels: dict[tuple[str, tuple[str, ...]], int] = {}
    labels_by_event: dict[str, list[int]] = defaultdict(list)
    for row in load_jsonl(args.execution_cache):
        event_id = str(row["event_id"])
        scored = score_execution_row(row, events[event_id])
        key = bundle_key(row)
        if key in labels:
            raise ValueError(f"duplicate execution label: {key}")
        labels[key] = int(scored["verified_success"])
        labels_by_event[event_id].append(labels[key])
    if len(labels_by_event) != 128 or len(labels) != 256:
        raise ValueError("E2 requires 128 events and two executed candidates per event")
    if any(sorted(values) != [0, 1] for values in labels_by_event.values()):
        raise ValueError("each E2 event must have one verified positive and one negative")

    rows: list[dict[str, Any]] = []
    selected_outcomes: dict[str, dict[str, int]] = {}
    canonical_inventory: set[tuple[str, tuple[str, ...]]] | None = None
    sources: dict[str, dict[str, str]] = {}
    for graph, set_term, lexical, method, directory_name in CELLS:
        directory = args.experiments_root / directory_name
        selections_path = directory / "selections.jsonl"
        scores_path = directory / "candidate_scores.jsonl"
        manifest_path = directory / "manifest.json"
        selections = load_jsonl(selections_path)
        candidate_scores = load_jsonl(scores_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("outcomes_loaded") is not False:
            raise ValueError(f"{method}: selection was not certified label-blind")
        expected_weight = 0.9 if lexical == "On" else 1.0
        if float(manifest.get("neural_weight", -1.0)) != expected_weight:
            raise ValueError(f"{method}: wrong lexical residual setting")
        if len(selections) != 128 or len(candidate_scores) != 256:
            raise ValueError(f"{method}: incomplete selections or candidate scores")
        inventory = {bundle_key(row) for row in candidate_scores}
        if inventory != set(labels):
            raise ValueError(f"{method}: candidate inventory differs from execution cache")
        if canonical_inventory is None:
            canonical_inventory = inventory
        elif inventory != canonical_inventory:
            raise ValueError(f"{method}: factorial cells do not share candidates")

        outcomes_by_event = {
            str(row["event_id"]): labels[bundle_key(row)] for row in selections
        }
        if len(outcomes_by_event) != 128:
            raise ValueError(f"{method}: duplicate or missing selected events")
        selected_outcomes[method] = outcomes_by_event
        selected_labels = list(outcomes_by_event.values())
        vsr = sum(selected_labels) / len(selected_labels)
        # With one verified positive and one hard negative per event, selecting the
        # positive candidate is both top-1 accuracy and verified workflow success.
        top1 = vsr
        squared_errors = []
        for score_row in candidate_scores:
            probability = float(score_row["score"])
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{method}: final score is outside [0, 1]")
            label = labels[bundle_key(score_row)]
            squared_errors.append((probability - label) ** 2)
        brier = sum(squared_errors) / len(squared_errors)
        rows.append(
            {
                "graph": graph,
                "set_term": set_term,
                "lexical": lexical,
                "method": method,
                "events": len(selections),
                "candidate_sets": len(candidate_scores),
                "verified_successes": sum(selected_labels),
                "vsr": vsr,
                "top1": top1,
                "brier": brier,
            }
        )
        sources[method] = {
            "manifest_sha256": sha256(manifest_path),
            "selections_sha256": sha256(selections_path),
            "candidate_scores_sha256": sha256(scores_path),
        }

    def mean_vsr(field: str, value: str) -> float:
        members = [float(row["vsr"]) for row in rows if row[field] == value]
        return sum(members) / len(members)

    main_effects_pp = {
        "graph_on_minus_off": 100.0
        * (mean_vsr("graph", "On") - mean_vsr("graph", "Off")),
        "set_on_minus_off": 100.0
        * (mean_vsr("set_term", "On") - mean_vsr("set_term", "Off")),
        "lexical_on_minus_off": 100.0
        * (mean_vsr("lexical", "On") - mean_vsr("lexical", "Off")),
    }

    full = selected_outcomes["graph_complement"]
    double_off = selected_outcomes["text_additive"]
    event_ids = sorted(full)
    wins = sum(double_off[event_id] > full[event_id] for event_id in event_ids)
    losses = sum(double_off[event_id] < full[event_id] for event_id in event_ids)
    ties = len(event_ids) - wins - losses
    discordant = wins + losses
    exact_p = min(
        1.0,
        2.0
        * sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
        / (2**discordant),
    )
    families: dict[str, list[str]] = defaultdict(list)
    for event_id in event_ids:
        parts = event_id.split(":")
        if len(parts) < 3:
            raise ValueError(f"cannot recover task family from {event_id}")
        families[parts[1]].append(event_id)
    family_deltas = {
        family: sum(double_off[event_id] - full[event_id] for event_id in members)
        / len(members)
        for family, members in families.items()
    }
    generator = random.Random(2027)
    family_names = sorted(families)
    bootstrap_deltas = []
    for _ in range(20_000):
        sampled = [generator.choice(family_names) for _ in family_names]
        numerator = sum(
            sum(double_off[event_id] - full[event_id] for event_id in families[name])
            for name in sampled
        )
        denominator = sum(len(families[name]) for name in sampled)
        bootstrap_deltas.append(numerator / denominator)
    bootstrap_deltas.sort()
    paired_comparison = {
        "contrast": "text_additive_minus_graph_complement",
        "delta_vsr_pp": 100.0
        * (sum(double_off.values()) - sum(full.values()))
        / len(event_ids),
        "event_wins": wins,
        "event_losses": losses,
        "event_ties": ties,
        "event_exact_sign_p": exact_p,
        "family_wins": sum(value > 0 for value in family_deltas.values()),
        "family_losses": sum(value < 0 for value in family_deltas.values()),
        "family_ties": sum(value == 0 for value in family_deltas.values()),
        "family_bootstrap_ci_low_pp": 100.0 * bootstrap_deltas[500],
        "family_bootstrap_ci_high_pp": 100.0 * bootstrap_deltas[19_499],
        "bootstrap_draws": len(bootstrap_deltas),
    }

    args.output.mkdir(parents=True)
    write_csv(args.output / "method_metrics.csv", rows)
    write_csv(args.output / "paired_comparisons.csv", [paired_comparison])
    table_lines = [
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Graph & Set term & Lexical & VSR & Top-1 & Brier\\",
        r"\midrule",
    ]
    for row in rows:
        table_lines.append(
            f'{row["graph"]} & {row["set_term"]} & {row["lexical"]} & '
            f'{100.0 * float(row["vsr"]):.1f} & '
            f'{100.0 * float(row["top1"]):.1f} & {float(row["brier"]):.3f}\\\\'
        )
    table_lines.extend((r"\bottomrule", r"\end{tabular}"))
    (args.output / "table.tex").write_text(
        "\n".join(table_lines) + "\n", encoding="utf-8"
    )
    write_json(
        args.output / "result.json",
        {
            "schema_version": "corem-independent-e2-factorial-v2",
            "status": "complete_post_hoc_factorial",
            "claim_boundary": (
                "The 128-event test outcomes had been inspected before the three "
                "missing lexical-off cells were generated. Treat the full factorial "
                "as post-hoc mechanism analysis, not a precommitted confirmation."
            ),
            "metric_definitions": {
                "vsr": "Verified successes among the 128 selected workflows.",
                "top1": "Events whose top-ranked candidate is the verified positive.",
                "brier": (
                    "Mean squared error of the final neural/BM25 score against "
                    "execution labels over all 256 candidate sets."
                ),
            },
            "events": 128,
            "candidate_sets": 256,
            "rows": rows,
            "main_effects_vsr_pp": main_effects_pp,
            "paired_comparison": paired_comparison,
            "family_deltas": family_deltas,
            "execution_cache_sha256": sha256(args.execution_cache),
            "sources": sources,
        },
    )
    checksum_files = (
        "method_metrics.csv",
        "paired_comparisons.csv",
        "result.json",
        "table.tex",
    )
    (args.output / "checksums.sha256").write_text(
        "".join(f"{sha256(args.output / name)}  {name}\n" for name in checksum_files),
        encoding="utf-8",
    )
    print(f"wrote complete 2x2x2 E2 factorial to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
