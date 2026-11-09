#!/usr/bin/env python3
"""Consolidate checksum-verified online runs that share the HGT/BM25 scorer."""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_json, load_jsonl, write_checksums, write_json, write_jsonl  # noqa: E402

METHODS = (
    "group_aware_online",
    "group_bm25_residual_online",
    "independent_relevance_online",
    "lru_online",
    "lfu_online",
    "no_memory",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory: Path) -> None:
    checksums = directory / "checksums.sha256"
    if not checksums.is_file():
        raise ValueError(f"missing checksums: {directory}")
    for line in checksums.read_text().splitlines():
        digest, filename = line.split("  ", 1)
        if sha256(directory / filename) != digest:
            raise ValueError(f"checksum mismatch: {directory / filename}")


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def family_bootstrap(
    values: dict[str, float],
    families: dict[str, tuple[str, str, str]],
    *,
    draws: int,
    seed: int,
) -> list[float]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for event_id, value in values.items():
        grouped[families[event_id]].append(value)
    keys = sorted(grouped)
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sample = [rng.choice(keys) for _ in keys]
        observations = [value for key in sample for value in grouped[key]]
        estimates.append(sum(observations) / len(observations))
    return [quantile(estimates, 0.025), quantile(estimates, 0.975)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.bootstrap_draws <= 0:
        parser.error("bootstrap draws must be positive")

    events = {
        str(row["event_id"]): row
        for row in load_jsonl(args.dataset / "events.jsonl")
    }
    if len(events) != 320:
        raise ValueError("expected the complete 320-event stream")
    families = {
        event_id: (
            str(row["service_id"]),
            str(row["version_id"]),
            str(row["task_key"]),
        )
        for event_id, row in events.items()
    }

    results = []
    scored: dict[str, list[dict[str, Any]]] = {}
    banks: dict[str, list[dict[str, Any]]] = {}
    source_rows = []
    retrieval_models = []
    for directory in args.runs:
        verify(directory)
        result = load_json(directory / "result.json")
        if (
            result.get("status") != "complete_online_retention"
            or result.get("events") != 320
            or result.get("seed") != 2027
            or result.get("retrieval_budget") != 160
            or result.get("storage_budget") != 640
            or result.get("workload_window") != 16
        ):
            raise ValueError(f"incompatible online run: {directory}")
        retrieval_models.append(result["retrieval_model"])
        run_scored = load_jsonl(directory / "scored_executions.jsonl")
        run_banks = load_jsonl(directory / "bank_history.jsonl")
        for summary in result["summaries"]:
            method = str(summary["method"])
            if method in scored:
                raise ValueError(f"duplicate method across runs: {method}")
            scored[method] = [row for row in run_scored if row["method"] == method]
            banks[method] = [row for row in run_banks if row["method"] == method]
            if len(scored[method]) != 320 or len(banks[method]) != 320:
                raise ValueError(f"incomplete method trajectory: {method}")
            results.append(summary)
        source_rows.append(
            {
                "directory": str(directory.resolve()),
                "result_sha256": sha256(directory / "result.json"),
                "checksums_sha256": sha256(directory / "checksums.sha256"),
            }
        )
    if set(scored) != set(METHODS):
        raise ValueError(f"method set mismatch: {sorted(scored)}")
    if any(model != retrieval_models[0] for model in retrieval_models[1:]):
        raise ValueError("runs do not share the same frozen retrieval model")

    summaries = []
    outcomes: dict[str, dict[str, float]] = {}
    for index, method in enumerate(METHODS):
        by_event = {
            str(row["event_id"]): float(row["verified_success"])
            for row in scored[method]
        }
        if set(by_event) != set(events):
            raise ValueError(f"event coverage mismatch: {method}")
        outcomes[method] = by_event
        summary = next(row for row in results if row["method"] == method)
        ci = family_bootstrap(
            by_event,
            families,
            draws=args.bootstrap_draws,
            seed=20271000 + index,
        )
        activated = sum(int(row["activated_records"]) for row in banks[method])
        evicted = sum(int(row["evicted_records"]) for row in banks[method])
        summaries.append(
            {
                **summary,
                "churn": activated + evicted,
                "vsr_family_cluster_95ci": ci,
            }
        )

    paired = {}
    target = outcomes["group_aware_online"]
    for index, comparator in enumerate(METHODS[1:]):
        differences = {
            event_id: target[event_id] - outcomes[comparator][event_id]
            for event_id in events
        }
        paired[f"group_aware_minus_{comparator}"] = {
            "delta_vsr": sum(differences.values()) / len(differences),
            "wins": sum(value > 0 for value in differences.values()),
            "losses": sum(value < 0 for value in differences.values()),
            "ties": sum(value == 0 for value in differences.values()),
            "family_cluster_95ci": family_bootstrap(
                differences,
                families,
                draws=args.bootstrap_draws,
                seed=20272000 + index,
            ),
        }

    trajectory = []
    for end in range(20, 321, 20):
        for method in METHODS:
            members = banks[method][end - 20 : end]
            trajectory.append(
                {
                    "window": end // 20,
                    "end_event": end,
                    "method": method,
                    "cumulative_vsr": sum(
                        outcomes[method][str(row["event_id"])]
                        for row in banks[method][:end]
                    )
                    / end,
                    "window_churn": sum(
                        int(row["activated_records"]) + int(row["evicted_records"])
                        for row in members
                    ),
                }
            )

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "trajectory.jsonl", trajectory)
    write_json(
        args.output / "result.json",
        {
            "status": "complete_aligned_online_retention_analysis",
            "schema_version": "corem-hgt-aligned-online-analysis-v1",
            "events": 320,
            "family_clusters": len(set(families.values())),
            "bootstrap_draws": args.bootstrap_draws,
            "retrieval_model": retrieval_models[0],
            "summaries": summaries,
            "paired_contrasts": paired,
            "source_runs": source_rows,
            "claim_boundary": (
                "Controlled resettable four-provider stream with one chronological "
                "trajectory. Five model seeds form one frozen retrieval policy and "
                "are not independent online samples."
            ),
        },
    )
    write_checksums(args.output, ("result.json", "trajectory.jsonl"))
    print("wrote aligned online-retention analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
