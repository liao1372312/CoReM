#!/usr/bin/env python3
"""Compute real-track VSR, BS, ECR, SUR, and transition recovery metrics."""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import (  # noqa: E402
    RealTrackError,
    STATIC_SCHEMA_VERSION,
    audit_real_dataset,
    load_jsonl,
    parse_utc,
    score_execution_row,
    summarize_scored_rows,
    validate_split_assignments,
    write_checksums,
    write_json,
    write_jsonl,
)
from experiments.common import clustered_bootstrap_ci, write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--executions", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument(
        "--allow-execution-superset",
        action="store_true",
        help="filter a complete longitudinal execution file to the requested split",
    )
    return parser.parse_args()


def _bootstrap_vsr(rows: list[dict[str, Any]], draws: int) -> tuple[float, float]:
    return clustered_bootstrap_ci(
        rows,
        cluster_key="incident_id",
        statistic=lambda sample: (
            sum(int(row["verified_success"]) for row in sample) / len(sample)
        ),
        draws=draws,
        seed=2027,
    )


def _method_summaries(rows: list[dict[str, Any]], draws: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    summaries = []
    for method, members in sorted(grouped.items()):
        summary = summarize_scored_rows(members)
        ci_low, ci_high = _bootstrap_vsr(members, draws)
        summaries.append(
            {
                "method": method,
                "seeds": len({int(row["seed"]) for row in members}),
                "events": len({str(row["event_id"]) for row in members}),
                "incidents": len({str(row["incident_id"]) for row in members}),
                **summary,
                "vsr_ci_low": ci_low,
                "vsr_ci_high": ci_high,
            }
        )
    return summaries


def _recovery_rows(
    rows: list[dict[str, Any]],
    events_by_id: dict[str, dict[str, Any]],
    transitions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_method_seed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method_seed[(str(row["method"]), int(row["seed"]))].append(row)
    output = []
    methods = sorted({method for method, _ in by_method_seed})
    for transition in transitions:
        transition_time = parse_utc(transition["transition_time"], "transition_time")
        for method in methods:
            pre_values: list[int] = []
            post_values: list[int] = []
            recovery_attempts: list[int] = []
            censored_seeds = 0
            seeds_observed = 0
            max_attempts = 0
            for (candidate_method, _seed), members in sorted(by_method_seed.items()):
                if candidate_method != method:
                    continue
                relevant = [
                    row
                    for row in members
                    if events_by_id[str(row["event_id"])]["provider_id"]
                    == transition["provider_id"]
                    and events_by_id[str(row["event_id"])]["service_id"]
                    == transition["service_id"]
                ]
                pre = sorted(
                    (
                        row
                        for row in relevant
                        if events_by_id[str(row["event_id"])]["version_id"]
                        == transition["from_version_id"]
                        and parse_utc(
                            events_by_id[str(row["event_id"])]["event_time"],
                            "event_time",
                        )
                        < transition_time
                    ),
                    key=lambda row: parse_utc(
                        events_by_id[str(row["event_id"])]["event_time"],
                        "event_time",
                    ),
                )
                post = sorted(
                    (
                        row
                        for row in relevant
                        if events_by_id[str(row["event_id"])]["version_id"]
                        == transition["to_version_id"]
                        and parse_utc(
                            events_by_id[str(row["event_id"])]["event_time"],
                            "event_time",
                        )
                        >= transition_time
                    ),
                    key=lambda row: parse_utc(
                        events_by_id[str(row["event_id"])]["event_time"],
                        "event_time",
                    ),
                )
                if not post:
                    continue
                seeds_observed += 1
                max_attempts = max(max_attempts, len(post))
                if pre:
                    pre_values.append(int(pre[-1]["verified_success"]))
                post_values.extend(int(row["verified_success"]) for row in post)
                first_success = next(
                    (
                        index
                        for index, row in enumerate(post, start=1)
                        if row["verified_success"]
                    ),
                    None,
                )
                if first_success is None:
                    censored_seeds += 1
                else:
                    recovery_attempts.append(first_success)
            if seeds_observed:
                pre_vsr = sum(pre_values) / len(pre_values) if pre_values else None
                post_vsr = sum(post_values) / len(post_values)
                output.append(
                    {
                        "transition_id": transition["transition_id"],
                        "provider_id": transition["provider_id"],
                        "service_id": transition["service_id"],
                        "from_version_id": transition["from_version_id"],
                        "to_version_id": transition["to_version_id"],
                        "transition_time": transition["transition_time"],
                        "method": method,
                        "seeds_observed": seeds_observed,
                        "seeds_recovered": len(recovery_attempts),
                        "censored_seeds": censored_seeds,
                        "pre_transition_last_attempt_vsr": pre_vsr,
                        "post_transition_vsr": post_vsr,
                        "post_minus_pre": (
                            post_vsr - pre_vsr if pre_vsr is not None else None
                        ),
                        "median_attempts_to_first_verified_success": (
                            statistics.median(recovery_attempts)
                            if recovery_attempts
                            else None
                        ),
                        "max_observed_post_transition_attempts": max_attempts,
                    }
                )
    return output


def _blocked_result(
    args: argparse.Namespace, audit: dict[str, Any], blockers: list[str]
) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "result.json",
        {
            "status": "blocked_no_real_metrics",
            "metrics": None,
            "target_split": args.split,
            "dataset_audit": audit,
            "blockers": blockers,
            "guardrail": "No partial score is emitted when any promotion gate fails.",
        },
    )
    write_checksums(args.output, ("result.json",))
    print("real-track analysis blocked; see result.json")
    return 2


def main() -> int:
    args = parse_args()
    audit = audit_real_dataset(args.dataset)
    if not audit.eligible:
        return _blocked_result(args, audit.to_dict(), list(audit.errors))
    events = load_jsonl(args.dataset / "events.jsonl")
    static = audit.schema_version == STATIC_SCHEMA_VERSION
    transitions = (
        [] if static else load_jsonl(args.dataset / "version_transitions.jsonl")
    )
    assignments = load_jsonl(args.splits)
    executions = load_jsonl(args.executions)
    try:
        validate_split_assignments(events, assignments, allow_test_only=static)
    except RealTrackError as error:
        return _blocked_result(args, audit.to_dict(), [str(error)])
    events_by_id = {str(row["event_id"]): row for row in events}
    target_ids = (
        {str(row["event_id"]) for row in assignments}
        if args.split == "all"
        else {str(row["event_id"]) for row in assignments if row["split"] == args.split}
    )
    blockers: list[str] = []
    scored: list[dict[str, Any]] = []
    observed_keys: set[tuple[str, int, str]] = set()
    method_seeds: set[tuple[str, int]] = set()
    ignored_outside_split = 0
    for index, row in enumerate(executions, start=1):
        event_id = str(row.get("event_id", ""))
        if event_id not in target_ids:
            if args.allow_execution_superset:
                ignored_outside_split += 1
            else:
                blockers.append(
                    f"execution[{index}] is outside target split: {event_id}"
                )
            continue
        method = row.get("method")
        seed = row.get("seed")
        if not isinstance(method, str) or not method or not isinstance(seed, int):
            blockers.append(f"execution[{index}] has invalid method or seed")
            continue
        key = (method, seed, event_id)
        if key in observed_keys:
            blockers.append(f"duplicate execution row: {key}")
            continue
        observed_keys.add(key)
        method_seeds.add((method, seed))
        token_cost = row.get("token_cost")
        latency_ms = row.get("latency_ms")
        if not isinstance(token_cost, int) or token_cost < 0:
            blockers.append(f"execution[{index}] has invalid token_cost")
            continue
        if not isinstance(latency_ms, (int, float)) or float(latency_ms) < 0:
            blockers.append(f"execution[{index}] has invalid latency_ms")
            continue
        try:
            counts = score_execution_row(row, events_by_id[event_id])
        except RealTrackError as error:
            blockers.append(f"execution[{index}] {error}")
            continue
        scored.append(
            {
                "event_id": event_id,
                "incident_id": events_by_id[event_id]["incident_id"],
                "provider_id": events_by_id[event_id]["provider_id"],
                "service_id": events_by_id[event_id]["service_id"],
                "version_id": events_by_id[event_id]["version_id"],
                "event_time": events_by_id[event_id]["event_time"],
                "method": method,
                "seed": seed,
                "token_cost": token_cost,
                "latency_ms": float(latency_ms),
                **counts,
            }
        )
    expected_keys = {
        (method, seed, event_id)
        for method, seed in method_seeds
        for event_id in target_ids
    }
    missing = expected_keys.difference(observed_keys)
    if missing:
        blockers.append(f"missing {len(missing)} method/seed/event executions")
    if not method_seeds:
        blockers.append("no valid method/seed execution groups were found")
    if blockers:
        return _blocked_result(args, audit.to_dict(), blockers)

    summaries = _method_summaries(scored, args.bootstrap_draws)
    recovery = _recovery_rows(scored, events_by_id, transitions)
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "scored_executions.jsonl", scored)
    write_csv(args.output / "method_metrics.csv", summaries)
    if not static:
        write_csv(args.output / "version_recovery.csv", recovery)
    digest = hashlib.sha256(args.executions.read_bytes()).hexdigest()
    write_json(
        args.output / "result.json",
        {
            "status": "complete_static_real_track" if static else "complete_real_track",
            "target_split": args.split,
            "dataset_audit": audit.to_dict(),
            "execution_sha256": digest,
            "ignored_executions_outside_split": ignored_outside_split,
            "methods": summaries,
            "version_transitions_with_observations": len(
                {row["transition_id"] for row in recovery}
            ),
            "version_recovery_applicable": not static,
            "metric_definitions": {
                "vsr": (
                    "Goal, every required binding, and every communication criterion "
                    "pass under an independent checker; all attempted tasks remain in "
                    "the denominator."
                ),
                "bs": "Passed required binding atoms / all required binding atoms.",
                "ecr": (
                    "Tasks whose retrieved memories plus verified metadata cover every "
                    "required evidence atom / all tasks."
                ),
                "sur": (
                    "Independently invalid attempted memory reuses / all attempted "
                    "memory reuses; null when no reuse is attempted."
                ),
            },
        },
    )
    artifact_names = ["method_metrics.csv", "result.json", "scored_executions.jsonl"]
    if not static:
        artifact_names.append("version_recovery.csv")
    write_checksums(args.output, artifact_names)
    tier = "static" if static else "longitudinal"
    print(f"wrote {tier} real-track metrics for {len(scored)} executions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
