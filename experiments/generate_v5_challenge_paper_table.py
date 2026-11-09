#!/usr/bin/env python3
"""Generate the controlled-challenge table from its frozen extended summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/results_public_multiservice_challenge_v5_extended_summary_v2/result.json"
OUTPUT = ROOT / "experiments/generated/v5_challenge_table.tex"
ORDER = (
    ("no_memory", "No memory"),
    ("service_bm25", "Service BM25"),
    ("recency_bm25", "Recency BM25"),
    ("version_bm25", "Version BM25"),
    ("version_bm25_bundle", "Version BM25 + closure"),
    ("corem_query_only", r"\corm\ (query-only)"),
    ("hgt_corem", r"\corm\ (trained HGT)"),
    ("hgt_additive", r"\quad w/o set interactions"),
)


def percent(value: float) -> str:
    return str((Decimal(str(value)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def render_table(summary: dict[str, object], source_hash: str) -> str:
    if (
        summary.get("status") != "complete_posthoc_v5_extended_analysis"
        or summary.get("test_events") != 64
        or summary.get("task_version_families") != 16
        or summary.get("providers") != 4
        or summary.get("bootstrap_draws") != 5000
    ):
        raise ValueError("unexpected v5 challenge summary")
    metrics = summary.get("method_metrics")
    if not isinstance(metrics, list) or len(metrics) != 13:
        raise ValueError("unexpected v5 method count")
    by_method = {row["method"]: row for row in metrics}
    if len(by_method) != 13 or not set(dict(ORDER)).issubset(by_method):
        raise ValueError("v5 method identities are missing or duplicated")
    for key, _ in ORDER:
        row = by_method[key]
        if row.get("events") != 64 or row.get("executions") != 320 or row.get("seeds") != 5:
            raise ValueError(f"invalid v5 coverage: {key}")
        for field in ("vsr", "bs", "ecr", "vsr_ci_low", "vsr_ci_high"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"invalid {field}: {key}")
        if not row["vsr_ci_low"] <= row["vsr"] <= row["vsr_ci_high"]:
            raise ValueError(f"invalid VSR interval: {key}")
        sur = row.get("sur")
        if key == "no_memory":
            if sur is not None:
                raise ValueError("no-memory SUR must be inapplicable")
        elif not isinstance(sur, (int, float)) or not math.isfinite(sur) or not 0 <= sur <= 1:
            raise ValueError(f"invalid SUR: {key}")
    maxima = {
        field: max(by_method[key][field] for key, _ in ORDER)
        for field in ("vsr", "bs", "ecr")
    }
    lines = [
        "% Generated from frozen post-hoc controlled-challenge summary.",
        f"% Source SHA-256: {source_hash}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & VSR & BS & ECR & SUR & VSR 95\% CI\\",
        r"\midrule",
    ]
    for key, label in ORDER:
        row = by_method[key]
        cells = []
        for field in ("vsr", "bs", "ecr"):
            cell = percent(row[field])
            if row[field] == maxima[field]:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        sur = r"\notapplicable" if row["sur"] is None else percent(row["sur"])
        interval = f"[{percent(row['vsr_ci_low'])},{percent(row['vsr_ci_high'])}]"
        lines.append(
            f"{label} & {cells[0]} & {cells[1]} & {cells[2]} & {sur} & $"
            + interval + r"$\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    raw = SOURCE.read_bytes()
    rendered = render_table(json.loads(raw), hashlib.sha256(raw).hexdigest())
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != rendered:
            raise SystemExit("generated v5 challenge table is absent or stale")
        print("controlled-challenge paper table matches frozen extended summary")
        return 0
    if OUTPUT.exists() and not args.force:
        raise SystemExit("refusing to overwrite existing generated table; use --check")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered)
    print("generated controlled-challenge paper table from frozen extended summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
