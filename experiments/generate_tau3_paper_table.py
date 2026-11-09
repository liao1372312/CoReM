#!/usr/bin/env python3
"""Generate the tau3 effectiveness table from verified fresh-replay summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/results_tau3_metric_extension_optimized/result.json"
OUTPUT = ROOT / "experiments/generated/tau3_effectiveness_table.tex"
ORDER = (
    ("no_memory", "No memory (diagnostic only)"),
    ("bm25_topk", r"BM25 Top-$k$"),
    ("dense_hash_topk", r"Dense-hash Top-$k$"),
    ("hybrid_rrf_topk", "Hybrid RRF"),
    ("graph_traversal", "Graph traversal"),
    ("magma_adapted", "MAGMA (adapted)"),
    ("acgm_adapted", "ACGM (adapted)"),
    ("hypermem_adapted", "HyperMem (adapted)"),
    ("hgmem_adapted", "HGMem (adapted)"),
    ("agemem_adapted", "AgeMem (adapted)"),
    ("full_corem", r"\textbf{\corm}"),
)


def render_table(summary: dict[str, object], source_hash: str) -> str:
    if summary.get("status") != "complete_adapted_external_baselines":
        raise ValueError("unexpected tau3 summary")
    counts = summary.get("counts")
    if not isinstance(counts, dict) or (
        counts.get("methods") != 11
        or counts.get("test_tasks") != 43
        or counts.get("seeds") != 5
    ):
        raise ValueError("unexpected tau3 coverage")
    rows = summary.get("summaries")
    if not isinstance(rows, list) or len(rows) != 11:
        raise ValueError("unexpected tau3 method count")
    by_method = {row["method"]: row for row in rows}
    if set(by_method) != set(dict(ORDER)):
        raise ValueError("unexpected or duplicated tau3 methods")
    for key, _ in ORDER:
        row = by_method[key]
        if row.get("tasks") != 43 or row.get("seeds") != 5 or row.get("task_seed_n") != 215:
            raise ValueError(f"invalid tau3 coverage: {key}")
        for field in ("vsr", "p95_latency_ms", "mean_tokens"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {field}: {key}")
        for field in ("bs", "ecr"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"invalid {field}: {key}")
        sur = row.get("sur")
        if key == "no_memory":
            if sur is not None:
                raise ValueError("SUR must be undefined when no reuse is attempted")
        elif not isinstance(sur, (int, float)) or not 0 <= sur <= 1:
            raise ValueError(f"invalid sur: {key}")
        if row["vsr"] > 1:
            raise ValueError(f"invalid tau3 VSR: {key}")
        if key.endswith("_adapted") and row.get("adapter_level") != "adapted_not_native":
            raise ValueError(f"native/adapted boundary changed: {key}")
    best_vsr = max(row["vsr"] for row in rows)
    best_tokens = min(by_method[key]["mean_tokens"] for key, _ in ORDER if key != "no_memory")
    lines = [
        "% Generated from checksum-verified tau3 external/adapted replay.",
        f"% Source SHA-256: {source_hash}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc}",
        r"\toprule",
        r"Method & VSR $\uparrow$ & BS $\uparrow$ & ECR $\uparrow$ & SUR $\downarrow$ & Sel. p95 (ms) $\downarrow$ & Tokens $\downarrow$\\",
        r"\midrule",
    ]
    for key, label in ORDER:
        row = by_method[key]
        vsr = f"{100 * row['vsr']:.1f}"
        if row["vsr"] == best_vsr:
            vsr = rf"\textbf{{{vsr}}}"
        latency = "$<0.1$" if row["p95_latency_ms"] < 0.1 else f"{row['p95_latency_ms']:,.1f}"
        if key == "full_corem":
            latency = "$" + latency.replace(",", r"{,}") + "$"
        tokens = f"{row['mean_tokens']:.1f}"
        if row["mean_tokens"] == best_tokens:
            tokens = rf"\textbf{{{tokens}}}"
        bs = f"{100 * row['bs']:.1f}"
        ecr = f"{100 * row['ecr']:.1f}"
        sur = r"\notapplicable" if row["sur"] is None else f"{100 * row['sur']:.1f}"
        lines.append(
            f"{label} & {vsr} & {bs} & {ecr} & {sur}"
            + f" & {latency} & {tokens}" + r"\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular*}"))
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
            raise SystemExit("generated tau3 paper table is absent or stale")
        print("tau3 paper table matches frozen replay")
        return 0
    if OUTPUT.exists() and not args.force:
        raise SystemExit("refusing to overwrite existing generated table; use --check")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered)
    print("generated tau3 paper table from frozen replay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
