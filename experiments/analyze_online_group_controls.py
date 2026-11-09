#!/usr/bin/env python3
"""Audit paired online retention outcomes against whole-group BM25 controls."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_json, load_jsonl, write_checksums, write_json  # noqa: E402
from experiments.summarize_independent_v7_confirmation import (  # noqa: E402
    quantile,
    sha256,
)

METHODS = (
    "group_aware_online",
    "group_bm25_online",
    "group_bm25_residual_online",
)


def verify_checksums(directory: Path) -> None:
    lines = (directory / "checksums.sha256").read_text().splitlines()
    if not lines:
        raise ValueError(f"empty checksums in {directory}")
    for line in lines:
        digest, filename = line.split("  ", 1)
        if sha256(directory / filename) != digest:
            raise ValueError(f"artifact checksum mismatch: {directory / filename}")


def load_method(directory: Path, method: str, event_ids: set[str]) -> dict[str, Any]:
    verify_checksums(directory)
    result = load_json(directory / "result.json")
    if result["status"] != "complete_online_retention" or result["events"] != len(
        event_ids
    ):
        raise ValueError(f"incomplete online run: {method}")
    summaries = [row for row in result["summaries"] if row["method"] == method]
    if len(summaries) != 1 or summaries[0]["n"] != len(event_ids):
        raise ValueError(f"missing method summary: {method}")
    scored = [
        row
        for row in load_jsonl(directory / "scored_executions.jsonl")
        if row["method"] == method
    ]
    executions = [
        row
        for row in load_jsonl(directory / "execution_results.jsonl")
        if row["method"] == method
    ]
    banks = [
        row
        for row in load_jsonl(directory / "bank_history.jsonl")
        if row["method"] == method
    ]
    keyed = {str(row["event_id"]): row for row in scored}
    if (
        len(keyed) != len(scored)
        or set(keyed) != event_ids
        or len(executions) != len(event_ids)
        or len(banks) != len(event_ids)
        or {str(row["event_id"]) for row in executions} != event_ids
        or {str(row["event_id"]) for row in banks} != event_ids
    ):
        raise ValueError(f"event coverage mismatch: {method}")
    if any(
        row["reset_verified"] is not True
        or row["checker_independent"] is not True
        or row["execution_error"] is not None
        or row["checker_error"] is not None
        for row in executions
    ):
        raise ValueError(f"unverified execution: {method}")
    if any(
        row["storage_budget"] != result["storage_budget"]
        or row["bank_after_cost"] > result["storage_budget"]
        for row in banks
    ):
        raise ValueError(f"bank-budget violation: {method}")
    successes = sum(int(row["verified_success"]) for row in scored)
    if successes / len(scored) != summaries[0]["vsr"]:
        raise ValueError(f"aggregate VSR mismatch: {method}")
    return {
        "result": result,
        "summary": summaries[0],
        "outcomes": {
            event_id: int(row["verified_success"]) for event_id, row in keyed.items()
        },
        "selections": {
            str(row["event_id"]): tuple(row["selected_memory_ids"])
            for row in executions
        },
        "bank_costs": [int(row["bank_after_cost"]) for row in banks],
        "source_sha256": sha256(directory / "result.json"),
        "directory": str(directory.resolve()),
    }


def paired(
    left: dict[str, int],
    right: dict[str, int],
    families: dict[str, tuple[str, str, str]],
    providers: dict[str, str],
) -> dict[str, Any]:
    differences = {event_id: left[event_id] - right[event_id] for event_id in families}
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for event_id, value in differences.items():
        grouped[families[event_id]].append(value)
    keys = sorted(grouped)
    rng = random.Random(2027)
    draws = []
    for _ in range(10000):
        sampled = [rng.choice(keys) for _ in keys]
        values = [value for key in sampled for value in grouped[key]]
        draws.append(sum(values) / len(values))
    return {
        "difference": sum(differences.values()) / len(differences),
        "net_successes": sum(differences.values()),
        "discordant_wins": sum(value == 1 for value in differences.values()),
        "discordant_losses": sum(value == -1 for value in differences.values()),
        "paired_ties": sum(value == 0 for value in differences.values()),
        "family_clusters": len(keys),
        "family_cluster_95ci": [quantile(draws, 0.025), quantile(draws, 0.975)],
        "by_provider_net_successes": {
            provider: sum(
                value
                for event_id, value in differences.items()
                if providers[event_id] == provider
            )
            for provider in sorted(set(providers.values()))
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    for method in METHODS:
        parser.add_argument(f"--{method.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    events = {
        str(row["event_id"]): row for row in load_jsonl(args.dataset / "events.jsonl")
    }
    if len(events) != 320:
        raise ValueError("expected the complete 320-event v5 stream")
    methods = {
        method: load_method(getattr(args, method), method, set(events))
        for method in METHODS
    }
    contracts = {
        (
            data["result"]["storage_budget"],
            data["result"]["retrieval_budget"],
            data["result"]["workload_window"],
            data["result"]["seed"],
        )
        for data in methods.values()
    }
    if contracts != {(640, 160, 16, 2027)}:
        raise ValueError("methods do not share the fixed budget/time contract")
    families = {
        event_id: (
            str(row["service_id"]),
            str(row["version_id"]),
            str(row["task_key"]),
        )
        for event_id, row in events.items()
    }
    providers = {event_id: str(row["provider_id"]) for event_id, row in events.items()}
    result = {
        "status": "complete_posthoc_group_retention_control",
        "claim_boundary": (
            "Already inspected v5 controlled stream; 12-event prefixes were seen "
            "before both full controls. Whole-group BM25 and residual repair "
            "are post-hoc mechanism checks, not independent confirmation."
        ),
        "events": len(events),
        "families": len(set(families.values())),
        "methods": {
            method: {
                "successes": sum(data["outcomes"].values()),
                "vsr": data["summary"]["vsr"],
                "vsr_family_cluster_95ci": paired(
                    data["outcomes"],
                    {event_id: 0 for event_id in events},
                    families,
                    providers,
                )["family_cluster_95ci"],
                "mean_tokens": data["summary"]["mean_tokens"],
                "mean_bank_cost": sum(data["bank_costs"]) / len(data["bank_costs"]),
                "peak_bank_cost": data["summary"]["peak_bank_cost"],
                "mean_update_latency_ms": data["summary"]["mean_update_latency_ms"],
                "p95_update_latency_ms": data["summary"]["p95_update_latency_ms"],
                "physical_executions_this_run": data["result"][
                    "physical_executions_this_run"
                ],
                "exact_prior_executions_used": data["result"][
                    "exact_prior_executions_used"
                ],
                "source_result": data["directory"] + "/result.json",
                "source_result_sha256": data["source_sha256"],
            }
            for method, data in methods.items()
        },
        "paired_contrasts": {
            f"group_aware_minus_{method}": paired(
                methods["group_aware_online"]["outcomes"],
                methods[method]["outcomes"],
                families,
                providers,
            )
            for method in METHODS[1:]
        },
        "whole_group_controls_selection_agreement_events": sum(
            methods["group_bm25_online"]["selections"][event_id]
            == methods["group_bm25_residual_online"]["selections"][event_id]
            for event_id in events
        ),
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "result.json", result)
    write_checksums(args.output, ("result.json",))
    print(
        f"summarized {len(events)} paired online events across {len(METHODS)} policies"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
