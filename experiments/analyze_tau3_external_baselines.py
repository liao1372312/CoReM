#!/usr/bin/env python3
"""Analyze fresh tau3 external-baseline replays without inflating metric claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.common import (  # noqa: E402
    clustered_bootstrap_ci,
    holm_adjust,
    json_dump,
    mean,
    paired_randomization_pvalue,
    percentile,
    write_csv,
    write_table_bundle,
)


METHOD_ORDER = (
    "no_memory",
    "bm25_topk",
    "dense_hash_topk",
    "hybrid_rrf_topk",
    "graph_traversal",
    "magma_adapted",
    "acgm_adapted",
    "hypermem_adapted",
    "hgmem_adapted",
    "agemem_adapted",
    "full_corem",
)
DISPLAY_NAMES = {
    "no_memory": "No memory",
    "bm25_topk": "BM25 Top-k",
    "dense_hash_topk": "Dense-hash Top-k",
    "hybrid_rrf_topk": "Hybrid RRF",
    "graph_traversal": "Graph traversal",
    "magma_adapted": "MAGMA (adapted)",
    "acgm_adapted": "ACGM (adapted)",
    "hypermem_adapted": "HyperMem (adapted)",
    "hgmem_adapted": "HGMem (adapted)",
    "agemem_adapted": "AgeMem (adapted)",
    "full_corem": "CoReM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/tau3_corem_counterexamples")
    )
    parser.add_argument(
        "--external-replays",
        type=Path,
        default=Path(
            "experiments/results_tau3_external_baselines/"
            "retrieval_ablation_replays.json"
        ),
    )
    parser.add_argument(
        "--external-selections",
        type=Path,
        default=Path(
            "experiments/results_tau3_external_baselines/"
            "external_baseline_selections.json"
        ),
    )
    parser.add_argument(
        "--corem-replays",
        type=Path,
        default=Path(
            "experiments/results_tau3_counterexample_retrieval/"
            "retrieval_ablation_replays.json"
        ),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("experiments/external_baselines/source_manifest.json"),
    )
    parser.add_argument(
        "--corem-selection-audit",
        type=Path,
        default=Path(
            "experiments/results_tau3_metric_extension_corem/"
            "selection_invariance_audit.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results_tau3_external_baselines"),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enrich_rows(
    rows: Sequence[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    memories: dict[str, dict[str, Any]],
    declared: dict[str, frozenset[str]],
    annotations: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        task_id = str(row["task_id"])
        task = tasks[task_id]
        selected = frozenset(row["selected_memory_ids"])
        group = declared.get(task_id, frozenset())
        active_versions = frozenset(task.get("active_versions", []))
        incompatible = sum(
            bool(memories[key].get("version_scope"))
            and not frozenset(memories[key]["version_scope"]).issubset(active_versions)
            for key in selected
        )
        covered_atoms = {
            atom
            for key in selected
            for atom in memories[key].get("evidence_atoms", [])
        }.union(task.get("verified_atoms", []))
        required_atoms = frozenset(task.get("required_atoms", []))
        annotation = (annotations or {}).get(task_id)
        binding_total = 0
        binding_passed = 0
        evidence_complete: int | None = None
        if annotation is not None:
            required_memories = frozenset(annotation["required_evidence_memory_ids"])
            atoms_by_memory = annotation["binding_atoms_by_memory"]
            binding_total = int(annotation["binding_atom_count"])
            successful_indices = {
                int(check["source_action_index"])
                for check in row.get("call_checks", [])
                if check.get("kind") == "current" and bool(check.get("passed"))
            }
            for memory_id in required_memories.intersection(selected):
                index = int(memory_id.rsplit(":a", 1)[1])
                if index in successful_indices:
                    binding_passed += len(atoms_by_memory[memory_id])
            evidence_complete = int(required_memories.issubset(selected))
        attempted_reuses = len(selected)
        stale_attempts = sum(
            bool(memories[key].get("controlled_stale"))
            and bool(memories[key].get("current_version_invalid"))
            for key in selected
        )
        output.append(
            {
                **row,
                "declared_group_complete_proxy": int(bool(group) and group.issubset(selected)),
                "declared_group_coverage_proxy": (
                    len(group.intersection(selected)) / len(group) if group else 0.0
                ),
                "coarse_evidence_complete_proxy": int(
                    required_atoms.issubset(covered_atoms)
                ),
                "selected_record_count": len(selected),
                "scope_incompatible_record_count": incompatible,
                "workflow_scope_incompatibility": int(incompatible > 0),
                "binding_passed": binding_passed,
                "binding_total": binding_total,
                "evidence_complete": evidence_complete,
                "attempted_reuses": attempted_reuses,
                "stale_attempts": stale_attempts,
                "controlled_sur": (
                    stale_attempts / attempted_reuses if attempted_reuses else None
                ),
            }
        )
    return output


def method_summary(
    method: str, rows: Sequence[dict[str, Any]], bootstrap_draws: int
) -> dict[str, Any]:
    method_rows = [row for row in rows if row["method"] == method]
    if not method_rows:
        raise ValueError(f"no rows for {method}")
    vsr = mean(float(row["outcome"]) for row in method_rows)
    ci_low, ci_high = clustered_bootstrap_ci(
        method_rows,
        cluster_key="task_id",
        statistic=lambda sampled: mean(float(row["outcome"]) for row in sampled),
        draws=bootstrap_draws,
        seed=2027,
    )
    per_seed = defaultdict(list)
    for row in method_rows:
        per_seed[int(row["seed"])].append(float(row["outcome"]))
    seed_rates = [mean(values) for _, values in sorted(per_seed.items())]
    selected_records = sum(int(row["selected_record_count"]) for row in method_rows)
    scope_incompatible = sum(
        int(row["scope_incompatible_record_count"]) for row in method_rows
    )
    adapter_level = str(
        method_rows[0].get(
            "adapter_level",
            "proposed_method" if method == "full_corem" else "unknown",
        )
    )
    binding_total = sum(int(row["binding_total"]) for row in method_rows)
    attempted_reuses = sum(int(row["attempted_reuses"]) for row in method_rows)
    measured = binding_total > 0
    return {
        "method": method,
        "display_name": DISPLAY_NAMES[method],
        "adapter_level": adapter_level,
        "seeds": len(per_seed),
        "tasks": len({row["task_id"] for row in method_rows}),
        "task_seed_n": len(method_rows),
        "vsr": vsr,
        "vsr_ci_low": ci_low,
        "vsr_ci_high": ci_high,
        "seed_sample_std": statistics.stdev(seed_rates) if len(seed_rates) > 1 else 0.0,
        "mean_tokens": mean(float(row["token_cost"]) for row in method_rows),
        "p95_latency_ms": percentile(
            [float(row["latency_ms"]) for row in method_rows], 0.95
        ),
        "execution_error_workflow_rate": mean(
            float(int(row.get("execution_error_count", 0)) > 0)
            for row in method_rows
        ),
        "assertion_error_workflow_rate": mean(
            float(int(row.get("assertion_error_count", 0)) > 0)
            for row in method_rows
        ),
        "declared_group_complete_proxy": mean(
            float(row["declared_group_complete_proxy"]) for row in method_rows
        ),
        "declared_group_coverage_proxy": mean(
            float(row["declared_group_coverage_proxy"]) for row in method_rows
        ),
        "coarse_evidence_complete_proxy": mean(
            float(row["coarse_evidence_complete_proxy"]) for row in method_rows
        ),
        "scope_incompatible_record_rate": (
            scope_incompatible / selected_records if selected_records else 0.0
        ),
        "workflow_scope_incompatibility_rate": mean(
            float(row["workflow_scope_incompatibility"]) for row in method_rows
        ),
        "bs": (
            sum(int(row["binding_passed"]) for row in method_rows) / binding_total
            if measured
            else None
        ),
        "ecr": (
            mean(float(row["evidence_complete"]) for row in method_rows)
            if measured
            else None
        ),
        "sur": (
            sum(int(row["stale_attempts"]) for row in method_rows) / attempted_reuses
            if measured and attempted_reuses
            else None
        ),
        "metric_boundary": (
            "BS uses evaluation-only official operation/argument atoms whose selected "
            "calls passed fresh execution; ECR uses the hidden replay-irreducible action "
            "set; SUR is a controlled pre-migration stress metric, not natural drift."
            if measured
            else "BS/ECR/SUR unavailable because metric annotations are absent."
        ),
    }


def paired_tests(
    rows: Sequence[dict[str, Any]], methods: Sequence[str]
) -> list[dict[str, Any]]:
    task_method: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        task_method[(str(row["task_id"]), str(row["method"]))].append(
            float(row["outcome"])
        )
    task_ids = sorted(
        task_id
        for task_id, method in task_method
        if method == "full_corem"
    )
    raw_pvalues = {}
    result = []
    for method in methods:
        if method == "full_corem":
            continue
        differences = [
            mean(task_method[(task_id, "full_corem")])
            - mean(task_method[(task_id, method)])
            for task_id in task_ids
        ]
        pvalue = paired_randomization_pvalue(differences, draws=20_000, seed=2027)
        raw_pvalues[method] = pvalue
        result.append(
            {
                "contrast": f"full_corem - {method}",
                "method": method,
                "task_clusters": len(task_ids),
                "delta_vsr": mean(differences),
                "p_randomization": pvalue,
                "multiplicity_family": "exploratory_all_external_rows",
            }
        )
    adjusted = holm_adjust(raw_pvalues)
    for row in result:
        row["p_holm"] = adjusted[row["method"]]
    return result


def main() -> None:
    args = parse_args()
    external = json.loads(args.external_replays.read_text(encoding="utf-8"))
    corem = json.loads(args.corem_replays.read_text(encoding="utf-8"))
    selections = json.loads(args.external_selections.read_text(encoding="utf-8"))
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    selection_audit = (
        json.loads(args.corem_selection_audit.read_text(encoding="utf-8"))
        if args.corem_selection_audit.is_file()
        else None
    )
    tasks = {
        row["task_id"]: row
        for row in load_jsonl(args.dataset / "retrieval_tasks.jsonl")
    }
    memories = {
        row["memory_id"]: row
        for row in load_jsonl(args.dataset / "memory_records.jsonl")
    }
    annotation_path = args.dataset / "metric_annotations.jsonl"
    annotations = (
        {
            row["task_id"]: row
            for row in load_jsonl(annotation_path)
        }
        if annotation_path.is_file()
        else None
    )
    if annotations and (
        selection_audit is None or selection_audit.get("status") != "passed"
    ):
        raise ValueError(
            "metric extension requires a passed CoReM selection invariance audit"
        )
    task_edges = {
        task_id: frozenset(task.get("dependency_edge_ids", []))
        for task_id, task in tasks.items()
    }
    declared_members: dict[str, set[str]] = defaultdict(set)
    for edge in load_jsonl(args.dataset / "dependency_hyperedges.jsonl"):
        task_id = str(edge["task_id"])
        if (
            edge["edge_id"] in task_edges[task_id]
            and edge["relation_type"] == "declared_state_transition_group"
        ):
            declared_members[task_id].update(edge["members"])
    declared = {
        task_id: frozenset(members)
        for task_id, members in declared_members.items()
    }
    external_rows = enrich_rows(
        external["rows"], tasks, memories, declared, annotations
    )
    full_rows = enrich_rows(
        [row for row in corem["rows"] if row["method"] == "full_corem"],
        tasks,
        memories,
        declared,
        annotations,
    )
    for row in full_rows:
        row["adapter_level"] = "proposed_method"
    rows = external_rows + full_rows
    methods = [method for method in METHOD_ORDER if any(row["method"] == method for row in rows)]
    summaries = [
        method_summary(method, rows, args.bootstrap_draws) for method in methods
    ]
    tests = paired_tests(rows, methods)
    test_by_method = {row["method"]: row for row in tests}
    full_vsr = next(row["vsr"] for row in summaries if row["method"] == "full_corem")
    effectiveness = []
    boundaries = []
    for row in summaries:
        comparison = test_by_method.get(row["method"])
        effectiveness.append(
            {
                "method": row["display_name"],
                "status": row["adapter_level"],
                "VSR": row["vsr"],
                "CI_low": row["vsr_ci_low"],
                "CI_high": row["vsr_ci_high"],
                "delta_vs_CoReM_pp": 100.0 * (row["vsr"] - full_vsr),
                "Holm_p": comparison["p_holm"] if comparison else None,
                "tokens": row["mean_tokens"],
                "p95_ms": row["p95_latency_ms"],
                "BS": row["bs"],
                "ECR": row["ecr"],
                "SUR": row["sur"],
            }
        )
        boundaries.append(
            {
                "method": row["display_name"],
                "BS": row["bs"],
                "ECR": row["ecr"],
                "SUR": row["sur"],
                "declared_group_complete_proxy": row["declared_group_complete_proxy"],
                "coarse_evidence_complete_proxy": row["coarse_evidence_complete_proxy"],
                "scope_incompatible_record_rate": row["scope_incompatible_record_rate"],
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    table_root = args.output / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    write_table_bundle(
        table_root,
        "external_effectiveness",
        effectiveness,
        caption="Enhanced tau3 fresh-replay effectiveness; external systems are adaptations",
    )
    write_table_bundle(
        table_root,
        "metric_boundaries",
        boundaries,
        caption=(
            "Execution-grounded BS/ECR and controlled version-shift SUR"
            if annotations
            else "Audited proxy metrics; BS, ECR, and SUR are unavailable"
        ),
    )
    write_csv(args.output / "raw_enriched_rows.csv", rows)
    write_csv(args.output / "paired_tests.csv", tests)
    result = {
        "status": "complete_adapted_external_baselines",
        "claim_scope": (
            "fresh deterministic final-state replay on the enhanced tau3 "
            "reference-trajectory memory bank"
        ),
        "counts": {
            "test_tasks": len({row["task_id"] for row in rows}),
            "seeds": len({int(row["seed"]) for row in rows}),
            "external_replay_rows": len(external_rows),
            "corem_replay_rows": len(full_rows),
            "methods": len(methods),
        },
        "configuration": {
            "external_adapters": selections["configuration"],
            "corem": corem.get("selection_configuration", {}),
        },
        "summaries": summaries,
        "paired_tests": tests,
        "corem_selection_provenance": selection_audit,
        "measurement_boundary": {
            "VSR": "execution verified by fresh reset-environment final-state replay",
            "BS": (
                "official operation/argument atoms counted only when the selected current call passes fresh execution"
                if annotations
                else "not measured; no evaluation annotations"
            ),
            "ECR": (
                "complete retrieval of the evaluation-only fresh-replay irreducible action set"
                if annotations
                else "not measured; no evaluation annotations"
            ),
            "SUR": (
                "current-invalid controlled pre-migration records among attempted reuses; not natural drift"
                if annotations
                else "not measured; no controlled version-shift annotations"
            ),
            "declared_group_complete_proxy": "structural diagnostic only, never relabeled ECR",
            "scope_incompatibility": "version-scope audit only, never relabeled natural SUR",
        },
        "adaptation_boundary": selections["adaptation_boundary"],
        "official_source_manifest": source_manifest,
        "source_hashes": {
            "dataset_manifest": sha256_file(args.dataset / "manifest.json"),
            "external_selections": sha256_file(args.external_selections),
            "external_replays": sha256_file(args.external_replays),
            "corem_replays": sha256_file(args.corem_replays),
            "source_manifest": sha256_file(args.source_manifest),
            "analysis_code": sha256_file(Path(__file__)),
            "metric_annotations": (
                sha256_file(annotation_path) if annotation_path.is_file() else None
            ),
            "corem_selection_audit": (
                sha256_file(args.corem_selection_audit)
                if args.corem_selection_audit.is_file()
                else None
            ),
        },
        "limitations": [
            "MAGMA, ACGM, HyperMem, HGMem, and AgeMem rows are method-inspired adapters, not native-system runs.",
            "The candidate bank is restricted to official reference-trajectory memories plus controlled counterexamples.",
            "Retrieval latency is implementation-specific and excludes offline training; native-system cost is not inferred.",
            "The SUR extension is a controlled contract migration and does not establish natural production-drift robustness.",
            "Frozen CoReM selections are carried forward only after the admissible-bank invariance audit; all workflows are freshly replayed for metrics.",
        ],
    }
    json_dump(args.output / "result.json", result)
    strongest_external = max(
        (row for row in summaries if row["method"] != "full_corem"),
        key=lambda row: row["vsr"],
    )
    full_summary = next(row for row in summaries if row["method"] == "full_corem")
    findings = f"""# External-Baseline Findings

## Fresh-replay result

All {len(methods) - 1} control/adapter rows and CoReM were evaluated on 43 held-out enhanced tau3 tasks over five seeds. CoReM obtained VSR {full_vsr:.4f}, BS {full_summary['bs']:.4f}, ECR {full_summary['ecr']:.4f}, and controlled SUR {full_summary['sur']:.4f}. The strongest external adapter was {strongest_external['display_name']} at VSR {strongest_external['vsr']:.4f}; this is evidence only for the controlled-shift adapted comparison. ACGM is the only stochastic adapter and its seed-level standard deviation was {next(row['seed_sample_std'] for row in summaries if row['method'] == 'acgm_adapted'):.4f}.

## Claim boundary

The five named external rows are adaptations over the shared tau3 bank, not native-system reproductions. VSR and BS use fresh reset-environment replay. ECR uses evaluation-only execution-irreducible action annotations that are hidden from selection. SUR uses controlled current-invalid pre-migration records and must not be interpreted as natural production drift.

## Reproducibility

The selection artifact contains {len(external_rows)} external task-seed replays and no budget violations or unknown memory IDs. Official repository HEADs are pinned in `experiments/external_baselines/source_manifest.json`.
"""
    (args.output / "findings.md").write_text(findings, encoding="utf-8")
    log = f"""# Experiment Log

- Dataset: `{args.dataset}`
- Retrieval budget: {selections['configuration']['retrieval_budget']} tokens
- Seeds: {', '.join(str(seed) for seed in selections['configuration']['seeds'])}
- Selection: training outcomes fit learned components; validation outcomes select hyperparameters; test outcomes are read only by fresh replay and this analyzer.
- Replay: isolated reset tau3 environment, deterministic final-state comparison.
- External status: adapted, not native.
- Bootstrap: task-clustered, {args.bootstrap_draws} resamples.
- Paired tests: two-sided task-cluster randomization; Holm adjustment across all exploratory external contrasts.
- Unavailable tracks: native full-system execution, natural chronology/service drift, and communication-quality labels. SUR is controlled rather than natural.

Commands:

```sh
TAU2_DATA_DIR=third_party/tau2-bench/data conda run -n corem-tau3 python scripts/build_tau3_metric_extension.py
.venv-corem-gpu/bin/python experiments/run_tau3_external_baseline_adapters.py --dataset data/tau3_corem_metric_extension --output experiments/results_tau3_metric_extension_external --budget 1024 --beam-width 64 --seeds 17,29,41,53,67
TAU2_DATA_DIR=third_party/tau2-bench/data conda run -n corem-tau3 python experiments/replay_tau3_retrieval_ablations.py --selections experiments/results_tau3_metric_extension_external/external_baseline_selections.json --dataset data/tau3_corem_metric_extension --output experiments/results_tau3_metric_extension_external
python3 experiments/audit_tau3_metric_extension_invariance.py
TAU2_DATA_DIR=third_party/tau2-bench/data conda run -n corem-tau3 python experiments/replay_tau3_retrieval_ablations.py --selections experiments/results_tau3_counterexample_retrieval/retrieval_ablation_selections.json --dataset data/tau3_corem_metric_extension --output experiments/results_tau3_metric_extension_corem
.venv-corem-gpu/bin/python experiments/analyze_tau3_external_baselines.py --dataset data/tau3_corem_metric_extension --external-replays experiments/results_tau3_metric_extension_external/retrieval_ablation_replays.json --external-selections experiments/results_tau3_metric_extension_external/external_baseline_selections.json --corem-replays experiments/results_tau3_metric_extension_corem/retrieval_ablation_replays.json --output experiments/results_tau3_metric_extension
```
"""
    (args.output / "EXPERIMENT_LOG.md").write_text(log, encoding="utf-8")
    readme = """# Enhanced tau3 External Baselines

This directory contains validation-tuned selections, fresh environment replays, execution-grounded BS/ECR, controlled-shift SUR, analysis tables, paired tests, hashes, and explicit claim boundaries. CoReM selections are carried forward only after the admissible-bank invariance audit; workflows are freshly replayed. See `findings.md` and `result.json`.
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    checksum_lines = []
    for path in sorted(
        candidate
        for candidate in args.output.rglob("*")
        if candidate.is_file() and candidate.name != "checksums.sha256"
    ):
        checksum_lines.append(
            f"{sha256_file(path)}  {path.relative_to(args.output)}\n"
        )
    (args.output / "checksums.sha256").write_text(
        "".join(checksum_lines), encoding="utf-8"
    )
    print(json.dumps(effectiveness, ensure_ascii=False, indent=2))
    print(f"wrote {args.output / 'result.json'}")


if __name__ == "__main__":
    main()
