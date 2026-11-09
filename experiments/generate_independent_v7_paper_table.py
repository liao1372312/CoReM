#!/usr/bin/env python3
"""Generate the independent-task paper rows from a frozen analysis result.

The committed TeX fragment is checked against the source on every rerun;
an unexpected difference is reported rather than silently overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/results_independent_v7_candidate_v4_confirmatory_summary_v5_venv/result.json"
OUTPUT = ROOT / "experiments/generated/independent_v7_table_rows.tex"
METHODS = (
    ("graph_complement", r"\corm"),
    ("graph_additive", r"\quad w/o set interactions"),
    ("text_complement", r"\quad w/o graph message passing"),
    ("text_additive", r"\quad w/o graph and set interactions"),
    ("pure_neural", r"\quad w/o lexical residual"),
    ("candidate_bm25_max", "Candidate BM25-max"),
    ("candidate_bm25_sum", "Candidate BM25-sum"),
    ("version_bm25_closure", "Version BM25 + closure"),
)


def render_rows(result: dict[str, object], source_sha256: str) -> str:
    if (
        result.get("status") != "complete_independent_v7_confirmatory_analysis"
        or result.get("events") != 128
        or result.get("families") != 16
    ):
        raise ValueError("unexpected independent-v7 result identity")
    methods = result.get("methods")
    if not isinstance(methods, dict) or set(methods) != {key for key, _ in METHODS}:
        raise ValueError("independent-v7 method set changed")
    highest = max(int(methods[key]["successes"]) for key, _ in METHODS)
    fastest = min(float(methods[key]["p95_selection_latency_ms"]) for key, _ in METHODS)
    rows = [
        "% Generated from frozen independent-task summary; do not edit rows manually.",
        f"% Source SHA-256: {source_sha256}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & All & Gitea & Meili. & Sel. p95\\",
        r"\midrule",
    ]
    for key, label in METHODS:
        row = methods[key]
        successes = row["successes"]
        latency = row["p95_selection_latency_ms"]
        providers = row["by_provider"]
        if (
            type(successes) is not int
            or not 0 <= successes <= 128
            or not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or latency < 0
            or row.get("events") != 128
            or row.get("mean_tokens") != 170
            or set(providers) != {"gitea", "meilisearch"}
            or any(providers[name].get("events") != 64 for name in providers)
        ):
            raise ValueError(f"invalid method aggregate: {key}")
        gitea = providers["gitea"]["successes"]
        meili = providers["meilisearch"]["successes"]
        if (
            type(gitea) is not int
            or type(meili) is not int
            or not 0 <= gitea <= 64
            or not 0 <= meili <= 64
            or gitea + meili != successes
        ):
            raise ValueError(f"provider totals do not reconcile: {key}")
        score_text = str(successes)
        latency_text = f"{float(latency):.1f}"
        if successes == highest:
            score_text = f"\\textbf{{{score_text}}}"
        if float(latency) == fastest:
            latency_text = f"\\textbf{{{latency_text}}}"
        rows.append(
            f"{label} & {score_text} & {gitea} & {meili} & {latency_text} \\\\"
        )
    rows.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if committed rows differ")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    source_bytes = SOURCE.read_bytes()
    result = json.loads(source_bytes)
    rendered = render_rows(result, hashlib.sha256(source_bytes).hexdigest())
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != rendered:
            raise SystemExit("generated independent-v7 table is absent or stale")
        print("independent-task paper table matches frozen result")
        return 0
    if OUTPUT.exists() and not args.force:
        raise SystemExit("refusing to overwrite existing generated table; use --check")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered)
    print("generated independent-task paper table from frozen result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
