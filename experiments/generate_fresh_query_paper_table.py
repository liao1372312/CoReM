#!/usr/bin/env python3
"""Generate the fresh-query paper table from frozen, paired analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "experiments/results_public_multiservice_confirmation_v6_summary/result.json"
ANALYSES = {
    "hgt_rank_ensemble": ROOT / "experiments/results_public_multiservice_confirmation_v6_hgt_ensemble_test_analysis/result.json",
    "version_bm25_bundle": ROOT / "experiments/results_public_multiservice_confirmation_v6_matched_test_analysis/result.json",
    "corem_query_only": ROOT / "experiments/results_public_multiservice_confirmation_v6_query_only_test_analysis/result.json",
}
ORDER = (
    ("hgt_rank_ensemble", r"\corm"),
    ("version_bm25_bundle", "Version BM25 + closure"),
    ("corem_query_only", r"\corm\ (query-only)"),
)
OUTPUT = ROOT / "experiments/generated/fresh_query_table.tex"


def render_table(
    summary: dict[str, object],
    analyses: dict[str, dict[str, object]],
    hashes: dict[str, str],
) -> str:
    if (
        summary.get("status") != "complete_fresh_query_confirmation"
        or summary.get("events") != 128
        or summary.get("task_version_families") != 16
    ):
        raise ValueError("unexpected fresh-query summary")
    metrics = summary.get("method_metrics")
    if not isinstance(metrics, list) or len(metrics) != 3:
        raise ValueError("unexpected method count")
    summarized = {row["method"]: row for row in metrics}
    if set(summarized) != set(ANALYSES) or set(analyses) != set(ANALYSES):
        raise ValueError("method set differs from frozen comparison")
    sources = summary.get("sources")
    if not isinstance(sources, list) or len(sources) != 3:
        raise ValueError("unexpected source count")
    for key in summarized:
        result = analyses[key]
        if (
            result.get("status") != "complete_real_track"
            or result.get("target_split") != "test"
            or not isinstance(result.get("methods"), list)
            or len(result["methods"]) != 1
        ):
            raise ValueError(f"invalid analysis: {key}")
        row, score = result["methods"][0], summarized[key]
        source_rows = [
            source for source in sources
            if Path(source["analysis"]).name == ANALYSES[key].parent.name
        ]
        if (
            len(source_rows) != 1
            or source_rows[0]["execution_sha256"] != result["execution_sha256"]
            or row.get("method") != key
            or row.get("n") != 128
            or score.get("events") != 128
            or score.get("task_version_families") != 16
            or score.get("verified_successes") != round(row["vsr"] * 128)
            or abs(score["vsr"] - row["vsr"]) > 1e-12
        ):
            raise ValueError(f"summary and analysis do not reconcile: {key}")
        for field in ("vsr", "bs", "ecr", "sur"):
            value = row.get(field)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= value <= 1
            ):
                raise ValueError(f"invalid {field}: {key}")
        if not (
            0 <= score["family_cluster_ci_low"]
            <= score["vsr"]
            <= score["family_cluster_ci_high"]
            <= 1
        ):
            raise ValueError(f"invalid family interval: {key}")
    best = {
        field: max(analyses[key]["methods"][0][field] for key in analyses)
        for field in ("vsr", "bs", "ecr")
    }
    lines = [
        "% Generated from frozen fresh-query summary and method analyses.",
        f"% Summary SHA-256: {hashes['summary']}",
        *[f"% {key} SHA-256: {hashes[key]}" for key, _ in ORDER],
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & VSR & BS & ECR & SUR & VSR 95\% CI\\",
        r"\midrule",
    ]
    for key, label in ORDER:
        row, score = analyses[key]["methods"][0], summarized[key]
        cells = []
        for field in ("vsr", "bs", "ecr"):
            value = f"{100 * row[field]:.1f}"
            if row[field] == best[field]:
                value = rf"\textbf{{{value}}}"
            cells.append(value)
        ci = (
            f"[{100 * score['family_cluster_ci_low']:.1f},"
            f"{100 * score['family_cluster_ci_high']:.1f}]"
        )
        lines.append(
            f"{label} & {cells[0]} & {cells[1]} & {cells[2]} & "
            + f"{100 * row['sur']:.1f} & $" + ci + r"$ \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    paths = {"summary": SUMMARY, **ANALYSES}
    payloads = {key: json.loads(path.read_bytes()) for key, path in paths.items()}
    hashes = {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in paths.items()
    }
    rendered = render_table(
        payloads["summary"], {key: payloads[key] for key in ANALYSES}, hashes
    )
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != rendered:
            raise SystemExit("generated fresh-query table is absent or stale")
        print("fresh-query paper table matches frozen analyses")
        return 0
    if OUTPUT.exists() and not args.force:
        raise SystemExit("refusing to overwrite existing generated table; use --check")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered)
    print("generated fresh-query paper table from frozen analyses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
