#!/usr/bin/env python3
"""Generate the panelled ablation table from three frozen result summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "v5": ROOT / "experiments/results_public_multiservice_challenge_v5_extended_summary_v2/result.json",
    "v6": ROOT / "experiments/results_public_multiservice_confirmation_v6_optimized_ablation_summary_v2/result.json",
    "retention": ROOT / "experiments/results_tau3_paper_suite/result.json",
}
OUTPUT = ROOT / "experiments/generated/ablation_table.tex"
SOURCE_LABELS = {
    "v5": "Controlled-challenge",
    "v6": "Fresh-query",
    "retention": "Retention",
}
ROWS = (
    ("v5", "corem_query_only", r"\corm\ (query-only)", "Controlled challenge (64)", "corem_query_only", "(a) Query-time retrieval mechanisms"),
    ("v5", "corem_additive_utility", r"\quad w/o non-additive set utility", "Controlled challenge (64)", "corem_query_only", "(a) Query-time retrieval mechanisms"),
    ("v5", "corem_no_hyperedges", r"\quad w/o hypergraph bundle exposure", "Controlled challenge (64)", "corem_query_only", "(a) Query-time retrieval mechanisms"),
    ("v5", "corem_item_only", r"\quad Item-wise retrieval", "Controlled challenge (64)", "corem_query_only", "(a) Query-time retrieval mechanisms"),
    ("v5", "corem_no_closure", r"\quad w/o post-selection closure", "Controlled challenge (64)", "corem_query_only", "(a) Query-time retrieval mechanisms"),
    ("v5", "hgt_corem", r"\corm\ (trained selector)", "Controlled challenge (64)", "hgt_corem", "(b) Learned selector components"),
    ("v5", "hgt_additive", r"\quad w/o complement/redundancy terms", "Controlled challenge (64)", "hgt_corem", "(b) Learned selector components"),
    ("v5", "hgt_no_hyperedges", r"\quad w/o hyperedge message passing", "Controlled challenge (64)", "hgt_corem", "(b) Learned selector components"),
    ("v6", "hgt_rank_ensemble", r"\corm", "Fresh-query test (128)", "hgt_rank_ensemble", "(c) Frozen selector components"),
    ("v6", "hgt_rank_ensemble_additive", r"\quad w/o complement/redundancy terms", "Fresh-query test (128)", "hgt_rank_ensemble", "(c) Frozen selector components"),
    ("v6", "hgt_rank_ensemble_no_hyperedges", r"\quad w/o hyperedge message passing", "Fresh-query test (128)", "hgt_rank_ensemble", "(c) Frozen selector components"),
    ("v6", "hgt_rank_ensemble_head_off", r"\quad w/o complement head at inference", "Fresh-query test (128)", "hgt_rank_ensemble", "(c) Frozen selector components"),
    ("v6", "hgt_rank_ensemble_ranking_off", r"\quad w/o ranking loss", "Fresh-query test (128)", "hgt_rank_ensemble", "(c) Frozen selector components"),
    ("v6", "hgt_rank_ensemble_pure", r"\quad w/o lexical residual", "Fresh-query test (128)", "hgt_rank_ensemble", "(c) Frozen selector components"),
    ("retention", "group_aware_adaptive", "Full group-aware retention", "Past-only storage sweep", "group_aware_adaptive", "(d) Retention policy"),
    ("retention", "a6_independent_priority", r"\quad Independent item retention", "Past-only storage sweep", "group_aware_adaptive", "(d) Retention policy"),
    ("retention", "a7_singleton_swaps", r"\quad w/o group-level swaps", "Past-only storage sweep", "group_aware_adaptive", "(d) Retention policy"),
    ("retention", "a8_uniform_forecast", r"\quad Uniform workload forecast", "Past-only storage sweep", "group_aware_adaptive", "(d) Retention policy"),
)


def render_table(sources: dict[str, dict[str, object]], hashes: dict[str, str]) -> str:
    v5, v6, retention = (sources[key] for key in SOURCES)
    if (
        v5.get("status") != "complete_posthoc_v5_extended_analysis"
        or v5.get("test_events") != 64
        or v5.get("task_version_families") != 16
        or v6.get("status") != "complete_optimized_ablation_confirmation"
        or v6.get("events") != 128
        or v6.get("task_version_families") != 16
        or retention.get("status") != "complete_with_explicitly_blocked_external_and_natural_tracks"
    ):
        raise ValueError("unexpected ablation source")
    raw = {
        "v5": v5["method_metrics"],
        "v6": v6["method_metrics"],
        "retention": retention["controlled_retention_default_budget"],
    }
    metrics = {source: {row["method"]: row for row in rows} for source, rows in raw.items()}
    if any(len(metrics[source]) != len(raw[source]) for source in raw):
        raise ValueError("duplicate ablation method")
    if len(raw["v5"]) != 13 or len(raw["v6"]) != 8 or len(raw["retention"]) != 5:
        raise ValueError("unexpected ablation method coverage")

    def score(source: str, method: str) -> float:
        row = metrics[source][method]
        field = "controlled_logged_replay_success" if source == "retention" else "vsr"
        if source == "v5":
            strict_single = method in {
                "corem_additive_utility",
                "corem_no_hyperedges",
                "corem_item_only",
                "corem_no_closure",
            }
            expected_seeds = 1 if strict_single else 5
            if (
                row.get("events") != 64
                or row.get("seeds") != expected_seeds
                or row.get("executions") != 64 * expected_seeds
            ):
                raise ValueError(f"invalid v5 coverage: {method}")
        if source == "v6" and (row.get("events") != 128 or row.get("verified_successes") != round(row[field] * 128)):
            raise ValueError(f"invalid v6 coverage: {method}")
        if source == "retention" and (row.get("storage_budget") != 32000 or row.get("seeds") != 5):
            raise ValueError(f"invalid retention coverage: {method}")
        value = row.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid ablation score: {method}")
        return value

    lines = [
        "% Generated from frozen controlled-challenge, fresh-query, and retention summaries.",
        *[f"% {SOURCE_LABELS[key]} SHA-256: {hashes[key]}" for key in SOURCES],
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Model variant & Evaluation setting & VSR (\%) & $\Delta$VSR (pp)\\",
        r"\midrule",
    ]
    previous_panel = None
    for source, method, label, track, baseline, panel in ROWS:
        if panel != previous_panel:
            if previous_panel is not None:
                lines.append(r"\addlinespace[2pt]")
            lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{panel}}}}}\\")
            previous_panel = panel
        value = score(source, method)
        difference = 100 * (value - score(source, baseline))
        delta = "---" if method == baseline else (
            "0.0" if abs(difference) < 0.05 else "$" + f"{difference:+.1f}" + "$"
        )
        lines.append(f"{label} & {track} & {100 * value:.1f} & {delta}" + r"\\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    raw = {key: path.read_bytes() for key, path in SOURCES.items()}
    rendered = render_table(
        {key: json.loads(value) for key, value in raw.items()},
        {key: hashlib.sha256(value).hexdigest() for key, value in raw.items()},
    )
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != rendered:
            raise SystemExit("generated ablation table is absent or stale")
        print("ablation paper table matches frozen summaries")
        return 0
    if OUTPUT.exists() and not args.force:
        raise SystemExit("refusing to overwrite existing generated table; use --check")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered)
    print("generated ablation paper table from frozen summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
