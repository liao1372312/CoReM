#!/usr/bin/env python3
"""Create a claim-scoped analysis of the final tau3 multi-step experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("experiments/results_tau3_multistep"),
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/tau3_corem"))
    parser.add_argument(
        "--affine-audit",
        type=Path,
        default=Path("experiments/results_tau3_multistep_affine_audit"),
    )
    return parser.parse_args()


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, metric in enumerate(ordered):
        running = max(running, min(1.0, pvalues[metric] * (count - rank)))
        adjusted[metric] = running
    return adjusted


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    result_path = args.results / "tau3_multistep_multiseed.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    quality = json.loads(
        (args.dataset / "quality_report.json").read_text(encoding="utf-8")
    )
    comparison = result["paired_seed_comparisons"][
        "hgt_additive_encoder_matched"
    ]
    no_hyperedge_comparison = result["paired_seed_comparisons"][
        "hgt_no_hyperedges"
    ]
    redundancy_comparison = result["paired_seed_comparisons"].get(
        "hgt_no_redundancy"
    )
    primary = ("roc_auc", "brier", "log_loss")
    raw_pvalues = {
        metric: float(comparison[metric]["exact_sign_flip_pvalue"])
        for metric in primary
    }
    adjusted = holm_adjust(raw_pvalues)
    bootstrap = result[
        "task_clustered_bootstrap_95ci_hgt_minus_reference"
    ]["hgt_additive_encoder_matched"]
    direction_ok = {
        "roc_auc": bootstrap["roc_auc"][0] > 0,
        "brier": bootstrap["brier"][1] < 0,
        "log_loss": bootstrap["log_loss"][1] < 0,
    }
    affine_comparison: dict[str, Any] | None = None
    affine_path = args.affine_audit / "tau3_multistep_multiseed.json"
    if affine_path.is_file():
        affine = json.loads(affine_path.read_text(encoding="utf-8"))
        if (
            affine.get("dataset") == result.get("dataset")
            and affine.get("configuration", {}).get(
                "gold_operation_ids_used_as_request_features"
            ) is False
        ):
            affine_comparison = {
                metric: {
                    "affine_mean": affine["aggregate"]["hgt_corem"][metric][
                        "mean"
                    ],
                    "identity_mean": result["aggregate"]["hgt_corem"][metric][
                        "mean"
                    ],
                    "identity_minus_affine": (
                        result["aggregate"]["hgt_corem"][metric]["mean"]
                        - affine["aggregate"]["hgt_corem"][metric]["mean"]
                    ),
                }
                for metric in ("brier", "ece_10", "log_loss")
            }

    analysis = {
        "status": "complete",
        "claim_scope": (
            "Execution-verified multi-operation deterministic final state on "
            "strict airline, retail, and telecom tau3-bench derivatives."
        ),
        "dataset": {
            "tasks": quality["task_counts_by_split"],
            "outcome_examples": quality["outcome_counts_by_split"],
            "class_counts": quality["outcome_class_counts"],
            "fresh_environment_replays": quality["fresh_environment_replays"],
            "source": quality["source"],
        },
        "primary_hgt_corem_vs_additive_ablation": {
            metric: {
                "mean_difference": comparison[metric][
                    "hgt_corem_minus_reference"
                ]["mean"],
                "exact_sign_flip_pvalue": raw_pvalues[metric],
                "holm_adjusted_pvalue": adjusted[metric],
                "task_clustered_bootstrap_95ci": bootstrap[metric],
                "favorable_ci_excludes_zero": direction_ok[metric],
            }
            for metric in primary
        },
        "secondary_hgt_corem_vs_no_hyperedges": no_hyperedge_comparison,
        "secondary_hgt_corem_vs_no_redundancy": redundancy_comparison,
        "method_metrics": result["aggregate"],
        "ensemble_test": result["ensemble_test"],
        "runtime": result.get("runtime"),
        "calibration_audit": affine_comparison,
        "claim_checks": {
            "primary_ablation_supported": all(direction_ok.values())
            and all(adjusted[metric] < 0.05 for metric in primary),
            "ece_improvement_supported": (
                float(
                    comparison["ece_10"]["hgt_corem_minus_reference"]["mean"]
                )
                < 0
                and float(comparison["ece_10"]["exact_sign_flip_pvalue"])
                < 0.05
            ),
            "single_hgt_superior_to_stable_hash_supported": False,
            "redundancy_ablation_supported": (
                redundancy_comparison is not None
                and float(
                    redundancy_comparison["roc_auc"][
                        "hgt_corem_minus_reference"
                    ]["mean"]
                )
                > 0
                and float(
                    redundancy_comparison["brier"][
                        "hgt_corem_minus_reference"
                    ]["mean"]
                )
                < 0
                and float(
                    redundancy_comparison["roc_auc"][
                        "exact_sign_flip_pvalue"
                    ]
                )
                < 0.05
                and float(
                    redundancy_comparison["brier"][
                        "exact_sign_flip_pvalue"
                    ]
                )
                < 0.05
            ),
            "oracle_edges_excluded": not result["configuration"][
                "outcome_derived_oracle_edges_used"
            ],
        },
        "interpretation": [
            "The encoder-matched A1 isolates the non-additive set head; the no-hyperedge A2 separately removes graph structure.",
            "The full HGT is not uniformly superior to the non-neural stable-hash set-utility model: it has lower mean AUC but lower mean log loss.",
            (
                f"The {len(result['per_seed'])}-model HGT ensemble reaches "
                f"{result['ensemble_test']['hgt_corem']['accuracy']:.4f} accuracy "
                f"and {result['ensemble_test']['hgt_corem']['roc_auc']:.4f} AUC "
                "on the expanded held-out set."
            ),
            "HGT ECE improves over the additive ablation under identity calibration, but calibration remains imperfect.",
            "Declared mutation-group completion remains a mandatory no-training baseline; the augmented dataset additionally contains execution-verified complete-but-failing sets.",
        ],
        "limitations": result["limitations"],
        "input_sha256": {
            str(result_path): sha256_file(result_path),
            str(args.dataset / "manifest.json"): sha256_file(
                args.dataset / "manifest.json"
            ),
            str(args.dataset / "quality_report.json"): sha256_file(
                args.dataset / "quality_report.json"
            ),
        },
    }
    analysis_path = args.results / "final_analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    test_examples = result["dataset"]["examples"]["test"]
    test_tasks = result["dataset"]["tasks"]["test"]
    readme = f"""# tau3 Multi-step Results

This directory contains the frozen {len(result['per_seed'])}-seed evaluation on {test_examples}
execution-verified multi-operation test sets from {test_tasks} held-out
tau3-bench tasks. Training uses 20
epochs selected exclusively on validation data. Outcome-derived oracle edges are
excluded; only official reference-trajectory mutation groups enter the graph.

The neural models ran on {result['runtime']['gpu_name']} with PyTorch
{result['runtime']['torch_version']} and CUDA {result['runtime']['cuda_runtime']}.

`tau3_multistep_multiseed.json` contains per-seed probabilities and metrics.
`final_analysis.json` adds Holm correction, task-clustered bootstrap intervals,
claim checks, and dataset limitations. Checksums cover every result artifact.
"""
    (args.results / "README.md").write_text(readme, encoding="utf-8")
    checksum_paths = sorted(
        path
        for path in args.results.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    (args.results / "checksums.sha256").write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths
        ),
        encoding="utf-8",
    )
    print(json.dumps(analysis["claim_checks"], ensure_ascii=False, indent=2))
    print(f"wrote {analysis_path}")


if __name__ == "__main__":
    main()
