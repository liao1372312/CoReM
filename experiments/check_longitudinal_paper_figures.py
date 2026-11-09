#!/usr/bin/env python3
"""Replay the strict longitudinal analysis and verify its three paper figures."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "experiments/results_longitudinal_extensions_v5_strict_semantic_cache"
PAPER = ROOT / "CoReM_WWW2027_Main.tex"
FIGURES = (
    "corem_online_retention.pdf",
    "corem_rolling_lpo.pdf",
    "corem_public_release_drift.pdf",
)
TABLES = (
    "online_retention_metrics.csv",
    "online_retention_contrasts.csv",
    "rolling_window_metrics.csv",
    "rolling_window_contrasts.csv",
    "leave_provider_out_metrics.csv",
    "leave_provider_out_contrasts.csv",
    "public_release_drift.csv",
    "public_release_recovery.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raster(pdf: Path, prefix: Path) -> bytes:
    subprocess.run(
        [
            "pdftoppm",
            "-f", "1", "-l", "1",
            "-scale-to", "850",
            "-png", "-singlefile",
            str(pdf), str(prefix),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return Path(str(prefix) + ".png").read_bytes()


def main() -> int:
    if shutil.which("pdftoppm") is None:
        raise SystemExit("pdftoppm is required for figure-render verification")
    subprocess.run(
        ["sha256sum", "-c", "--quiet", "checksums.sha256"],
        cwd=ANALYSIS,
        check=True,
    )
    source = json.loads((ANALYSIS / "result.json").read_text())
    stamp = source["reproducibility_stamp"]
    if (
        source.get("status") != "complete_longitudinal_extensions"
        or stamp.get("bootstrap_draws") != 2000
        or source.get("natural_traffic_track", {}).get("status")
        != "not_measured_missing_eligible_source"
    ):
        raise SystemExit("strict longitudinal contract changed")
    for relative, expected in stamp["measurement_input_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"measurement input differs from freeze: {relative}")
    paper_text = PAPER.read_text()
    for name in FIGURES:
        relative = f"experiments/results_longitudinal_extensions_v5_strict_semantic_cache/figures/{name}"
        if paper_text.count(relative) != 1:
            raise SystemExit(f"paper does not input exactly one frozen figure: {name}")

    with tempfile.TemporaryDirectory(prefix="corem-longitudinal-figure-check-") as temporary:
        scratch = Path(temporary)
        command = [
            sys.executable,
            str(ROOT / "experiments/analyze_longitudinal_extensions.py"),
            "--dataset", str(ROOT / "data/public_multiservice_challenge_v5"),
            "--online", str(ROOT / "experiments/results_online_retention_v5_b640_strict_semantic_cache"),
            "--lpo-executions", str(ROOT / "experiments/results_public_multiservice_challenge_v5_lpo_ensemble_cached/execution_results.jsonl"),
            "--matched-test-executions", str(ROOT / "experiments/results_public_multiservice_challenge_v5_matched_baseline_test/execution_results.jsonl"),
            "--rolling-hgt",
            *[
                str(ROOT / f"experiments/results_public_multiservice_challenge_v5_rolling_window{index}_cached/execution_results.jsonl")
                for index in (1, 2, 3)
            ],
            "--rolling-matched",
            *[
                str(ROOT / f"experiments/results_public_multiservice_challenge_v5_rolling_window{index}_matched_cached/execution_results.jsonl")
                for index in (1, 2)
            ],
            str(ROOT / "experiments/results_public_multiservice_challenge_v5_matched_baseline_test/execution_results.jsonl"),
            "--output", str(scratch / "analysis"),
            "--figures", str(scratch / "figures"),
            "--bootstrap-draws", "2000",
        ]
        subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        if (scratch / "analysis/result.json").read_bytes() != (ANALYSIS / "result.json").read_bytes():
            raise SystemExit("longitudinal replay result differs")
        for name in TABLES:
            if (scratch / "analysis" / name).read_bytes() != (ANALYSIS / name).read_bytes():
                raise SystemExit(f"longitudinal replay table differs: {name}")
        for name in FIGURES:
            if raster(scratch / "figures" / name, scratch / (name + "-replay")) != raster(
                ANALYSIS / "figures" / name, scratch / (name + "-frozen")
            ):
                raise SystemExit(f"rendered paper figure differs: {name}")
    print("strict longitudinal result, eight tables, and three paper figure rasters replay exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
