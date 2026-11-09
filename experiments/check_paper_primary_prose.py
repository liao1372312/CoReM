#!/usr/bin/env python3
"""Check headline numerical prose against checksum-verified frozen analyses."""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "CoReM_WWW2027_Main.tex"
SOURCES = {
    "tau3": "results_tau3_metric_extension",
    "tau3_predictor": "results_tau3_counterexample_suite",
    "v5": "results_public_multiservice_challenge_v5_extended_summary_v2",
    "v6": "results_public_multiservice_confirmation_v6_summary",
    "v6_ablation": "results_public_multiservice_confirmation_v6_optimized_ablation_summary_v2",
    "v7": "results_independent_v7_candidate_v4_confirmatory_summary_v5_venv",
    "online": "results_longitudinal_extensions_v5_strict_semantic_cache",
    "aligned": "results_online_retention_hgt_aligned_analysis_v1",
    "budget": "results_public_challenge_v4_budget_summary",
    "storage": "results_public_challenge_v4_retention_summary",
}


def pct(rate: float) -> str:
    return f"{100 * rate:.1f}"


def pct_compact(rate: float) -> str:
    value = f"{100 * rate:.2f}".rstrip("0").rstrip(".")
    return value if "." in value else value + ".0"


def p_fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".lstrip("0")


def prose_count(value: int) -> str:
    return {3: "three", 18: "eighteen"}.get(value, str(value))


def normalized(value: str) -> str:
    return " ".join(value.split())


def excerpt(text: str, start: str, end: str) -> str:
    if text.count(start) != 1:
        raise ValueError(f"ambiguous paper anchor: {start}")
    section = text.split(start, 1)[1]
    if end not in section:
        raise ValueError(f"missing paper end anchor: {end}")
    return normalized(section.split(end, 1)[0])


def assert_text(section: str, expected: str, claim: str) -> None:
    if normalized(expected) not in section:
        raise ValueError(f"{claim} differs from frozen analysis: {expected}")


def main() -> int:
    data = {}
    for key, name in SOURCES.items():
        source = ROOT / "experiments" / name
        subprocess.run(
            ["sha256sum", "-c", "--quiet", "checksums.sha256"],
            cwd=source, check=True,
        )
        data[key] = json.loads((source / "result.json").read_text())
    paper = PAPER.read_text()
    abstract = excerpt(paper, r"\begin{abstract}", r"\end{abstract}")
    tau3 = {row["method"]: row for row in data["tau3"]["summaries"]}
    tau3_pairs = {row["method"]: row for row in data["tau3"]["paired_tests"]}
    predictors = {row["method"]: row for row in data["tau3_predictor"]["predictor_summary"]}
    v5 = {row["method"]: row for row in data["v5"]["method_metrics"]}
    v5_pairs = {row["contrast"]: row for row in data["v5"]["paired_comparisons"]}
    v6 = {row["method"]: row for row in data["v6"]["method_metrics"]}
    v6_pairs = {row["contrast"]: row for row in data["v6"]["paired_comparisons"]}
    v7 = data["v7"]
    online = {row["method"]: row for row in data["aligned"]["summaries"]}
    online_pairs = data["aligned"]["paired_contrasts"]
    if (
        data["tau3"]["counts"]["test_tasks"] != 43
        or data["v5"]["test_events"] != 64
        or data["v6"]["events"] != 128
        or v7["events"] != 128
        or v7["families"] != 16
        or online["group_aware_online"]["n"] != 320
        or online["lru_online"]["vsr"] != online["lfu_online"]["vsr"]
        or tau3["full_corem"]["vsr"] <= tau3["hgmem_adapted"]["vsr"]
    ):
        raise ValueError("primary experiment coverage or tied controls changed")

    assert_text(
        abstract,
        rf"\corm\ {pct(tau3['full_corem']['vsr'])}\% verified success and {pct(tau3['full_corem']['sur'])}\% controlled stale use, versus {pct(tau3['hgmem_adapted']['vsr'])}\% and {pct(tau3['hgmem_adapted']['sur'])}\% for an adapted HGMem control",
        "tau3 abstract",
    )
    assert_text(
        abstract,
        rf"query-only \corm\ reference {pct(v5['corem_query_only']['vsr'])}\% success; a candidate-matched version-BM25 control reaches {pct(v5['version_bm25_bundle']['vsr'])}\%",
        "v5 abstract",
    )
    assert_text(
        abstract,
        rf"selected \corm\ configuration reaches {pct(v6['hgt_rank_ensemble']['vsr'])}\% on 128 new phrasings of those tasks, versus {pct(v6['version_bm25_bundle']['vsr'])}\% for matched BM25",
        "v6 abstract",
    )
    if v7["methods"]["graph_complement"]["successes"] - v7["methods"]["graph_additive"]["successes"] != 1:
        raise ValueError("v7 complement-ablation count changed")
    assert_text(
        abstract,
        rf"\corm\ again reaches {pct(v7['methods']['graph_complement']['vsr'])}\%, below same-candidate BM25-sum at {pct(v7['methods']['candidate_bm25_sum']['vsr'])}\%; its complement ablation differs by one event",
        "v7 abstract",
    )
    assert_text(
        abstract,
        rf"group-aware retention reaches {pct(online['group_aware_online']['vsr'])}\% success versus {pct(online['lru_online']['vsr'])}\% for item LRU/LFU and {pct(online['group_bm25_residual_online']['vsr'])}\% for whole-group BM25",
        "retention abstract",
    )
    assert_text(
        abstract,
        f"{online['group_aware_online']['mean_update_latency_ms'] / 1000:.2f} seconds versus {online['group_bm25_residual_online']['mean_update_latency_ms']:.1f} milliseconds",
        "retention abstract latency",
    )

    tau3_text = excerpt(paper, "The task-clustered 95", r"\noindent\textbf{Two-step harness pilot.}")
    acgm = tau3_pairs["acgm_adapted"]
    assert_text(
        tau3_text,
        rf"bootstrap interval is $[{pct(tau3['full_corem']['vsr_ci_low'])},{pct(tau3['full_corem']['vsr_ci_high'])}]$ for \corm, $[{pct(tau3['hgmem_adapted']['vsr_ci_low'])},{pct(tau3['hgmem_adapted']['vsr_ci_high'])}]$ for adapted HGMem, and $[{pct(tau3['acgm_adapted']['vsr_ci_low'])},{pct(tau3['acgm_adapted']['vsr_ci_high'])}]$ for adapted ACGM",
        "tau3 interval",
    )
    assert_text(
        tau3_text,
        rf"the {pct(acgm['delta_vsr'])}-point \corm--ACGM difference ($p<.001$)",
        "tau3 adjusted comparison",
    )
    assert_text(
        tau3_text,
        f"ROC-AUC {p_fmt(predictors['hgt_corem']['roc_auc'], 4)} and Brier score {p_fmt(predictors['hgt_corem']['brier'], 4)}",
        "tau3 predictor",
    )

    v5_text = excerpt(paper, r"Table~\ref{tab:t2_challenge} combines", "We also enumerate")
    v5_pair = v5_pairs["version_bm25_bundle_minus_corem_query_only"]
    assert_text(
        v5_text,
        rf"reaches {pct(v5['version_bm25_bundle']['vsr'])}\%, exceeding query-only \corm\ by {pct(v5_pair['delta_vsr'])} points; the 16-family cluster-bootstrap interval $[{pct(v5_pair['family_cluster_ci_low'])},{pct(v5_pair['family_cluster_ci_high'])}]$ crosses zero and the family sign test is $p={p_fmt(v5_pair['family_sign_p'], 3)}$",
        "v5 matched control",
    )
    v6_text = excerpt(paper, "We then construct 640 query events", r"\begin{table}[t]")
    v6_pair = v6_pairs["hgt_rank_ensemble_minus_version_bm25_bundle"]
    assert_text(
        v6_text,
        rf"\corm\ succeeds on {v6['hgt_rank_ensemble']['verified_successes']}/128 incidents ({pct(v6['hgt_rank_ensemble']['vsr'])}\%), compared with {v6['version_bm25_bundle']['verified_successes']}/128 ({pct(v6['version_bm25_bundle']['vsr'])}\%) for candidate-matched BM25",
        "fresh-query counts",
    )
    assert_text(
        v6_text,
        "difference is only $" + f"{100 * v6_pair['delta_vsr']:+.1f}" +
        rf"$ points (family-cluster CI $[{pct(v6_pair['family_cluster_ci_low'])},{pct(v6_pair['family_cluster_ci_high'])}]$; three discordant wins, one loss; exact $p={str(v6_pair['event_exact_sign_p']).lstrip('0')}$)",
        "fresh-query paired inference",
    )

    v7_text = excerpt(paper, r"Table~\ref{tab:independent_test} gives", "On the earlier two-provider")
    v7_pair = v7["paired_contrasts"]["graph_complement_minus_candidate_bm25_sum"]
    assert_text(
        v7_text,
        rf"\corm\ {v7['methods']['graph_complement']['successes']}/128 ({pct(v7['methods']['graph_complement']['vsr'])}\%) versus candidate BM25-sum {v7['methods']['candidate_bm25_sum']['successes']}/128 ({pct(v7['methods']['candidate_bm25_sum']['vsr'])}\%)",
        "independent-test counts",
    )
    assert_text(
        v7_text,
        "paired difference is $" + pct(v7_pair["difference"]) +
        rf"$ points; its 16-family bootstrap 95\% CI is $[{pct(v7_pair['family_cluster_95ci'][0])},{pct(v7_pair['family_cluster_95ci'][1])}]$",
        "independent-test paired inference",
    )
    assert_text(
        v7_text,
        f"({prose_count(v7_pair['discordant_wins'])} discordant wins, {prose_count(v7_pair['discordant_losses'])} losses)",
        "independent-test discordance",
    )

    budgets = data["budget"]
    storage = data["storage"]
    if (
        budgets["status"] != "passed"
        or storage["status"] != "passed"
        or budgets["budgets"] != [80, 120, 160, 200, 240]
        or storage["storage_budgets"] != [640, 1280, 1920, 2560, 3840, 7680]
        or sum(row["execution_rows"] for row in budgets["gate_rows"]) != 2000
        or sum(row["execution_rows"] for row in storage["gate_rows"]) != 1920
        or storage["highest_budget_parity_mismatches"] != 0
        or storage["highest_budget_parity_rows"] != 80
    ):
        raise ValueError("controlled budget/storage coverage changed")
    budget_text = excerpt(
        paper, "On the earlier two-provider controlled challenge",
        r"\subsection{Component Ablations and Interaction Tests}",
    )
    budget_rows = {
        (row["budget"], row["method"]): row["vsr"]
        for row in budgets["budget_metrics"]
    }
    for method, label in (
        ("corem_query_only", "\\corm\\"),
        ("version_bm25", "version BM25"),
    ):
        sequence = "/".join(
            pct_compact(budget_rows[budget, method]) for budget in (160, 200, 240)
        )
        assert_text(
            budget_text,
            f"{label} reaches {sequence}",
            "retrieval-budget VSR",
        )
    storage_rows = {
        (row["storage_budget"], row["method"]): row["vsr"]
        for row in storage["storage_metrics"]
    }
    for method, label in (
        ("group_aware_retention", "group-aware retention"),
        ("independent_relevance", "Independent relevance"),
        ("item_recency", "item recency"),
    ):
        sequence = "/".join(
            pct_compact(storage_rows[budget, method])
            for budget in storage["storage_budgets"]
        )
        assert_text(
            budget_text,
            f"{label} obtains {sequence}",
            "storage-budget VSR",
        )

    ablation_text = excerpt(
        paper, "The query-only interventions on the controlled challenge are strict mechanism ablations",
        r"\begin{table}[t]",
    )
    strict = [
        v5_pairs["corem_query_only_minus_" + key]
        for key in ("corem_additive_utility", "corem_no_hyperedges", "corem_item_only")
    ]
    if any(row["family_losses"] != 0 for row in strict):
        raise ValueError("strict ablation family-win interpretation changed")
    assert_text(
        ablation_text,
        "the first three variants lose " +
        ", ".join(pct(row["delta_vsr"]) for row in strict[:2]) +
        ", and " + pct(strict[2]["delta_vsr"]) + " points",
        "strict ablation deltas",
    )
    assert_text(
        ablation_text,
        "respectively " + "/".join(str(row["family_wins"]) for row in strict) +
        " losses (two-sided sign " +
        "$p=" + "/".join(p_fmt(row["family_sign_p"], 5) for row in strict) + "$)",
        "strict ablation family tests",
    )
    v6_ablation = data["v6_ablation"]
    if v6_ablation["events"] != 128 or v6_ablation["task_version_families"] != 16:
        raise ValueError("v6 optimized ablation coverage changed")
    optimized = {row["method"]: row for row in v6_ablation["method_metrics"]}
    optimized_pairs = {
        row["contrast"]: row for row in v6_ablation["paired_comparisons"]
    }
    full = optimized["hgt_rank_ensemble"]
    additive = optimized["hgt_rank_ensemble_additive"]
    no_edge = optimized["hgt_rank_ensemble_no_hyperedges"]
    assert_text(
        ablation_text,
        rf"Full, retrained additive, and retrained no-edge variants reach {pct(full['vsr'])}\%, {pct(additive['vsr'])}\%, and {pct(no_edge['vsr'])}\%",
        "optimized ablation rates",
    )
    additive_pair = optimized_pairs["hgt_rank_ensemble_minus_hgt_rank_ensemble_additive"]
    no_edge_pair = optimized_pairs["hgt_rank_ensemble_minus_hgt_rank_ensemble_no_hyperedges"]
    assert_text(
        ablation_text,
        "Full minus additive is $" + pct(additive_pair["delta_vsr"]) +
        rf"$ points (family-cluster CI $[{pct(additive_pair['family_cluster_ci_low'])},{pct(additive_pair['family_cluster_ci_high'])}]$)",
        "optimized additive interval",
    )
    if no_edge_pair["delta_vsr"] != 0 or no_edge_pair["family_cluster_ci_low"] != 0:
        raise ValueError("optimized no-edge tie changed")
    if additive_pair["holm_family_p"] != 1 or no_edge_pair["holm_family_p"] != 1:
        raise ValueError("optimized ablation Holm result changed")
    assert_text(
        ablation_text,
        "both Holm-adjusted family sign tests have $p=1$",
        "optimized ablation multiplicity",
    )
    assert_text(
        ablation_text,
        rf"Their selected bundles agree with full on {pct(additive_pair['bundle_agreement'])}\% and {pct(no_edge_pair['bundle_agreement'])}\% of events",
        "optimized ablation selection agreement",
    )
    pure = optimized["hgt_rank_ensemble_pure"]
    pure_additive = optimized["hgt_rank_ensemble_additive_pure"]
    pure_no_edge = optimized["hgt_rank_ensemble_no_hyperedges_pure"]
    assert_text(
        ablation_text,
        rf"full/additive/no-edge configurations reach only {pct(pure['vsr'])}/{pct(pure_additive['vsr'])}/{pct(pure_no_edge['vsr'])}\%",
        "neural-only ablation",
    )
    residual_pair = optimized_pairs["hgt_rank_ensemble_minus_hgt_rank_ensemble_pure"]
    assert_text(
        ablation_text,
        "diagnostic changes VSR by $" + f"{100 * residual_pair['delta_vsr']:+.1f}" +
        rf"$ points (CI $[{pct(residual_pair['family_cluster_ci_low'])},{pct(residual_pair['family_cluster_ci_high'])}]$)",
        "lexical-residual diagnostic",
    )
    head_off = optimized_pairs["hgt_rank_ensemble_minus_hgt_rank_ensemble_head_off"]
    if head_off["bundle_agreement"] != 1 or head_off["delta_vsr"] != 0:
        raise ValueError("complement-head-off selections changed")
    assert_text(
        ablation_text,
        "Turning off the complement head in the trained full model reproduces all 128 full selections and outcomes",
        "complement-head-off diagnostic",
    )
    rank_off = optimized_pairs["hgt_rank_ensemble_minus_hgt_rank_ensemble_ranking_off"]
    assert_text(
        ablation_text,
        rf"frozen residual configuration still reaches {pct(optimized['hgt_rank_ensemble_ranking_off']['vsr'])}\% on the fresh-query set. Full minus ranking-loss-off is only $+{pct(rank_off['delta_vsr'])}$ points (family CI $[{pct(rank_off['family_cluster_ci_low'])},{pct(rank_off['family_cluster_ci_high'])}]$; family sign $p=1$), with {pct(rank_off['bundle_agreement'])}\% bundle agreement",
        "ranking-loss-off diagnostic",
    )

    retention_text = excerpt(paper, "Group-aware retention reaches", r"\begin{table*}[t]")
    independent = online_pairs[
        "group_aware_minus_independent_relevance_online"
    ]
    lru = online_pairs["group_aware_minus_lru_online"]
    matched = online_pairs[
        "group_aware_minus_group_bm25_residual_online"
    ]
    assert_text(
        retention_text,
        f"paired advantage over independent relevance is {pct(independent['delta_vsr'])} points " +
        rf"(95\% family-cluster CI [{pct(independent['family_cluster_95ci'][0])},{pct(independent['family_cluster_95ci'][1])}])",
        "online independent-relevance inference",
    )
    assert_text(
        retention_text,
        f"{pct(lru['delta_vsr'])}-point difference from LRU/LFU has an interval crossing zero",
        "online LRU inference",
    )
    assert_text(
        retention_text,
        "group-aware--group-BM25 difference is $" + f"{100 * matched['delta_vsr']:+.1f}" +
        rf"$ points (32-family CI $[{pct(matched['family_cluster_95ci'][0])},{pct(matched['family_cluster_95ci'][1])}]$; {matched['wins']} wins, {matched['losses']} losses, and {matched['ties']} ties)",
        "group-BM25 inference",
    )
    extensions = excerpt(paper, "Three past-only rolling windows test", r"\subsection{Threats to Validity}")
    rolling = data["online"]["rolling_windows"]
    hgt_windows = sorted(
        (row for row in rolling if row["method"] == "hgt_rolling_ensemble"),
        key=lambda row: row["window"],
    )
    matched_windows = sorted(
        (row for row in rolling if row["method"] == "version_bm25_bundle"),
        key=lambda row: row["window"],
    )
    if [row["window"] for row in hgt_windows] != [1, 2, 3] or [
        row["window"] for row in matched_windows
    ] != [1, 2, 3]:
        raise ValueError("rolling-window coverage changed")
    assert_text(
        extensions,
        "Its VSR is " + "/".join(pct(row["vsr"]) for row in hgt_windows) +
        r"\%, versus " + "/".join(pct(row["vsr"]) for row in matched_windows) +
        r"\% for matched BM25 plus closure",
        "rolling-window comparison",
    )
    lpo = {
        (row["provider_id"], row["method"]): row
        for row in data["online"]["leave_provider_out"]
    }
    hgt_all = lpo["ALL", "hgt_leave_provider_out"]
    bm25_all = lpo["ALL", "version_bm25_bundle"]
    assert_text(
        extensions,
        rf"\corm\ obtains {pct(hgt_all['vsr'])}\% VSR (family CI [{pct(hgt_all['vsr_ci_low'])},{pct(hgt_all['vsr_ci_high'])}]) over 64 held-out incidents, versus {pct(bm25_all['vsr'])}\% [{pct(bm25_all['vsr_ci_low'])},{pct(bm25_all['vsr_ci_high'])}] for matched BM25",
        "leave-provider-out interval",
    )
    lpo_pairs = {
        row["provider_id"]: row for row in data["online"]["leave_provider_out_contrasts"]
    }
    overall = lpo_pairs["ALL"]
    assert_text(
        extensions,
        f"The paired difference is {100 * overall['delta_vsr']:+.1f} points " +
        r"(95\% family CI [$" + pct(overall["delta_vsr_ci_low"]) + "$," +
        pct(overall["delta_vsr_ci_high"]) + "])",
        "leave-provider-out paired interval",
    )
    provider_rates = [
        "0.0" if abs(lpo_pairs[provider]["delta_vsr"]) < 0.0005
        else f"{100 * lpo_pairs[provider]['delta_vsr']:+.1f}"
        for provider in ("mccutchen", "mock-server", "typicode", "wiremock")
    ]
    assert_text(
        extensions,
        f"Provider deltas are {provider_rates[0]}, {provider_rates[1]}, $" +
        provider_rates[2] + f"$, and {provider_rates[3]} points",
        "leave-provider-out provider deltas",
    )
    analysis_dir = ROOT / "experiments" / SOURCES["online"]
    with (analysis_dir / "public_release_drift.csv").open(newline="") as handle:
        drift_rows = list(csv.DictReader(handle))
    group_drift = [row for row in drift_rows if row["method"] == "group_aware_online"]
    phases = ("pre_change_late", "post_change_early", "post_change_late")
    phase_rates = [
        statistics.mean(float(row["vsr"]) for row in group_drift if row["phase"] == phase)
        for phase in phases
    ]
    if len(group_drift) != 12:
        raise ValueError("public-release drift coverage changed")
    assert_text(
        extensions,
        rf"group-aware VSR falls from {pct(phase_rates[0])}\% in the last eight pre-change incidents to {pct(phase_rates[1])}\% in the first eight post-change incidents, then reaches {pct(phase_rates[2])}\% over late post-change incidents",
        "release-transition phases",
    )
    with (analysis_dir / "public_release_recovery.csv").open(newline="") as handle:
        recovery_rows = list(csv.DictReader(handle))
    attempts = sorted(
        int(row["attempts_to_first_post_change_success"])
        for row in recovery_rows
        if row["method"] == "group_aware_online" and row["censored"] == "False"
    )
    if len(attempts) != 4:
        raise ValueError("public-release recovery coverage changed")
    assert_text(
        extensions,
        f"first verified post-change success occurs after {min(attempts)}--{max(attempts)} attempts for every service (median {int(statistics.median(attempts))})",
        "release-transition recovery",
    )
    print("abstract and eight primary result passages match frozen analyses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
