#!/usr/bin/env python3
"""Replay controlled appendix analyses and verify the four paper figures."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "CoReM_WWW2027_Main.tex"
FIGURES = ROOT / "experiments/generated/paper_figures"
RUNS = (
    (
        "budget",
        "experiments/summarize_public_challenge_budget_sweep.py",
        "experiments/results_public_challenge_v4_budget_summary",
        ("corem_challenge_budget.pdf", "corem_challenge_efficiency.pdf"),
        (
            "--splits", "experiments/public_challenge_v4_splits/chronological_splits.jsonl",
            "--sweep-root", "experiments/results_public_challenge_v4_budget_sweep_v2",
            "--primary-execution", "experiments/results_public_challenge_v4_test/execution_results.jsonl",
            "--profile", "experiments/results_public_challenge_v4_retrieval_profile",
        ),
    ),
    (
        "retention",
        "experiments/summarize_public_challenge_retention.py",
        "experiments/results_public_challenge_v4_retention_summary",
        ("corem_challenge_storage_budget.pdf",),
        (
            "--dataset", "data/public_versioned_challenge_v4",
            "--splits", "experiments/public_challenge_v4_splits/chronological_splits.jsonl",
            "--sweep-root", "experiments/results_public_challenge_v4_retention",
            "--primary-execution", "experiments/results_public_challenge_v4_test/execution_results.jsonl",
        ),
    ),
    (
        "v5",
        "experiments/summarize_public_multiservice_challenge_v5_extended.py",
        "experiments/results_public_multiservice_challenge_v5_extended_summary_v2",
        ("corem_multiservice_v5.pdf",),
        (
            "--dataset", "data/public_multiservice_challenge_v5",
            "--main-analysis", "experiments/results_public_multiservice_challenge_v5_test_analysis",
            "--matched-analysis", "experiments/results_public_multiservice_challenge_v5_matched_baseline_test_analysis",
            "--ablation-analysis", "experiments/results_public_multiservice_challenge_v5_ablation_test_analysis",
            "--hgt-analysis", "experiments/results_public_multiservice_challenge_v5_hgt_final_test_analysis",
            "--bootstrap-draws", "5000",
        ),
    ),
)


def raster(pdf: Path, prefix: Path) -> bytes:
    subprocess.run(
        [
            "pdftoppm", "-f", "1", "-l", "1", "-scale-to", "850",
            "-png", "-singlefile", str(pdf), str(prefix),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return Path(str(prefix) + ".png").read_bytes()


def main() -> int:
    if shutil.which("pdftoppm") is None:
        raise SystemExit("pdftoppm is required for figure-render verification")
    paper_text = PAPER.read_text()
    with tempfile.TemporaryDirectory(prefix="corem-appendix-figure-check-") as temporary:
        scratch = Path(temporary)
        for label, script, source, names, args in RUNS:
            frozen = ROOT / source
            subprocess.run(
                ["sha256sum", "-c", "--quiet", "checksums.sha256"],
                cwd=frozen,
                check=True,
            )
            command = [
                sys.executable,
                str(ROOT / script),
                *[
                    str(ROOT / arg) if index > 0 and args[index - 1] in {
                        "--dataset", "--splits", "--sweep-root",
                        "--primary-execution", "--profile", "--main-analysis",
                        "--matched-analysis", "--ablation-analysis", "--hgt-analysis",
                    } else arg
                    for index, arg in enumerate(args)
                ],
                "--output", str(scratch / label),
                "--figures", str(scratch / "figures"),
            ]
            subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            for name in ("result.json", *(path.name for path in frozen.glob("*.csv"))):
                if (frozen / name).read_bytes() != (scratch / label / name).read_bytes():
                    raise SystemExit(f"{label} replay differs: {name}")
            for name in names:
                relative = f"experiments/generated/paper_figures/{name}"
                if paper_text.count(relative) != 1:
                    raise SystemExit(f"paper does not input exactly one generated figure: {name}")
                if raster(FIGURES / name, scratch / (name + "-frozen")) != raster(
                    scratch / "figures" / name, scratch / (name + "-replay")
                ):
                    raise SystemExit(f"rendered paper figure differs: {name}")
    print("three controlled summaries and four appendix figure rasters replay exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
