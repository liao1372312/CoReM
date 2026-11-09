#!/usr/bin/env python3
"""Stratify the matched HSC factorial ablation by verified chain depth."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family-outcomes",
        type=Path,
        default=Path(
            "experiments/results_expanded_hsc_corem_v2_five_seed/family_outcomes.jsonl"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    return parser.parse_args()


def quantile(values: list[float], probability: float) -> float:
    values.sort()
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def bootstrap(values: list[float], draws: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = [
        mean(rng.choice(values) for _ in values)
        for _ in range(draws)
    ]
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")
    raw = [
        json.loads(line)
        for line in args.family_outcomes.read_text().splitlines()
        if line
    ]
    rows = [
        row
        for row in raw
        if row["experiment"] == "E2" and row["lexical"] is True
    ]
    configurations = {
        "full": (True, True),
        "no_set": (True, False),
        "no_graph": (False, True),
    }
    per_family: dict[str, dict[str, Any]] = defaultdict(dict)
    for name, (graph, set_term) in configurations.items():
        members = [
            row
            for row in rows
            if row["graph"] is graph and row["set_term"] is set_term
        ]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in members:
            grouped[str(row["family_id"])].append(row)
        if len(grouped) != 36:
            raise ValueError(f"{name} does not contain 36 test families")
        for family_id, family_rows in grouped.items():
            depths = {int(row["depth"]) for row in family_rows}
            seeds = {int(row["seed"]) for row in family_rows}
            if len(depths) != 1 or len(seeds) != 5:
                raise ValueError(f"invalid family/seed coverage: {family_id}")
            per_family[family_id]["depth"] = depths.pop()
            per_family[family_id][name] = mean(
                float(row["verified_optimum_recovery"]) for row in family_rows
            )
    strata = [
        ("1", lambda depth: depth == 1),
        ("2", lambda depth: depth == 2),
        (">=3", lambda depth: depth >= 3),
        ("all", lambda _depth: True),
    ]
    output_rows = []
    for index, (label, predicate) in enumerate(strata):
        families = [
            values for values in per_family.values() if predicate(values["depth"])
        ]
        if not families:
            output_rows.append(
                {
                    "required_support_memories": label,
                    "tasks_families": 0,
                    "status": "not_identified_no_tasks",
                }
            )
            continue
        differences = [values["full"] - values["no_set"] for values in families]
        low, high = bootstrap(differences, args.bootstrap_draws, 202730 + index)
        output_rows.append(
            {
                "required_support_memories": label,
                "tasks_families": len(families),
                "full_optimal_recovery": mean(values["full"] for values in families),
                "no_set_optimal_recovery": mean(
                    values["no_set"] for values in families
                ),
                "no_graph_optimal_recovery": mean(
                    values["no_graph"] for values in families
                ),
                "full_minus_no_set_pp": 100.0 * mean(differences),
                "ci_low_pp": 100.0 * low,
                "ci_high_pp": 100.0 * high,
                "status": (
                    "underpowered_descriptive_only"
                    if len(families) < 10
                    else "measured"
                ),
            }
        )
    args.output.mkdir(parents=True)
    fields = [
        "required_support_memories",
        "tasks_families",
        "full_optimal_recovery",
        "no_set_optimal_recovery",
        "no_graph_optimal_recovery",
        "full_minus_no_set_pp",
        "ci_low_pp",
        "ci_high_pp",
        "status",
    ]
    with (args.output / "table3_dependency_complexity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    tex = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Required supports & Tasks & Full & No set term & No graph & $\Delta$ & 95\% CI \\",
        r"\midrule",
    ]
    for row in output_rows:
        if row["tasks_families"] == 0:
            tex.append(
                f"{row['required_support_memories']} & 0 & -- & -- & -- & -- & -- \\\\"
            )
            continue
        tex.append(
            f"{row['required_support_memories']} & {row['tasks_families']} & "
            f"{100*row['full_optimal_recovery']:.1f} & "
            f"{100*row['no_set_optimal_recovery']:.1f} & "
            f"{100*row['no_graph_optimal_recovery']:.1f} & "
            f"{row['full_minus_no_set_pp']:.1f} & "
            f"[{row['ci_low_pp']:.1f}, {row['ci_high_pp']:.1f}] \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (args.output / "table3_dependency_complexity.tex").write_text(
        "\n".join(tex) + "\n"
    )
    result = {
        "status": "complete_with_singleton_stratum_not_identified",
        "metric": "offline verified optimal-composition recovery (not live VSR)",
        "rows": output_rows,
        "grouping_contract": (
            "Frozen HSC instance depth, which equals the required service-chain "
            "length; never the number selected by a method."
        ),
        "averaging_contract": (
            "Average the five training seeds within each of 36 test families, "
            "then average/bootstrap families."
        ),
        "limitations": [
            "No depth-1 test family exists, so the singleton row is not identified.",
            "Only five depth-2 families exist; that row is descriptive and underpowered.",
            "The metric is offline optimal-composition recovery and must not be relabeled live VSR.",
        ],
        "source_sha256": hashlib.sha256(args.family_outcomes.read_bytes()).hexdigest(),
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote dependency-complexity analysis to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
