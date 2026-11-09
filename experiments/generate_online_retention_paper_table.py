#!/usr/bin/env python3
"""Render the paper online-retention table from a verified run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ORDER = (
    ("group_aware_online", "Group-aware"),
    ("group_bm25_residual_online", "Group BM25 + residual"),
    ("independent_relevance_online", "Independent relevance"),
    ("lru_online", "LRU"),
    ("lfu_online", "LFU"),
    ("no_memory", "No memory"),
)
CHECKER_CONTRACT_ID = "public-multiservice-challenge-checker-v1"


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def verify_run_files(run: Path) -> bytes:
    required = {
        "result.json", "execution_results.jsonl",
        "scored_executions.jsonl", "bank_history.jsonl",
    }
    manifest = run / "checksums.sha256"
    if not manifest.is_file():
        raise ValueError("online run has no checksum manifest")
    listed: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or not _digest(digest) or name in listed:
            raise ValueError("malformed online run checksum manifest")
        listed[name] = digest
    if not required.issubset(listed):
        raise ValueError("online run checksum manifest is incomplete")
    for name in required:
        path = run / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != listed[name]:
            raise ValueError(f"online run checksum mismatch: {name}")
    return (run / "result.json").read_bytes()


def _fraction(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def verify_run_metrics(
    run: Path,
    result: dict[str, Any],
    methods: tuple[tuple[str, str], ...] = ORDER,
) -> None:
    """Recompute displayed metrics from the checksummed event-level rows."""
    def rows(name: str) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in (run / name).read_text(encoding="utf-8").splitlines()
            if line
        ]

    scored = rows("scored_executions.jsonl")
    banks = rows("bank_history.jsonl")
    executions = rows("execution_results.jsonl")
    summaries = {row["method"]: row for row in result["summaries"]}
    event_sets: list[set[str]] = []
    for method, _ in methods:
        groups = [
            [row for row in source if row.get("method") == method]
            for source in (scored, banks, executions)
        ]
        event_ids = [
            [str(row["event_id"]) for row in group] for group in groups
        ]
        if any(len(ids) != 320 or len(set(ids)) != 320 for ids in event_ids):
            raise ValueError(f"incomplete or duplicate event rows: {method}")
        if set(event_ids[0]) != set(event_ids[1]) or set(event_ids[0]) != set(event_ids[2]):
            raise ValueError(f"event identities differ across files: {method}")
        event_sets.append(set(event_ids[0]))
        summary = summaries[method]
        scored_rows, bank_rows, execution_rows = groups
        binding_total = sum(int(row["binding_total"]) for row in scored_rows)
        if binding_total <= 0:
            raise ValueError(f"missing binding denominator: {method}")
        expected = {
            "vsr": sum(int(row["verified_success"]) for row in scored_rows) / 320,
            "bs": sum(int(row["binding_passed"]) for row in scored_rows) / binding_total,
            "ecr": sum(int(row["evidence_complete"]) for row in scored_rows) / 320,
            "total_churn_records": sum(
                int(row["activated_records"]) + int(row["evicted_records"])
                for row in bank_rows
            ),
            "mean_update_latency_ms": sum(
                float(row["update_latency_ms"]) for row in bank_rows
            ) / 320,
        }
        for key, value in expected.items():
            observed = summary[key]
            if not math.isclose(float(observed), value, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"summary differs from event rows: {method}.{key}")
        for row in execution_rows:
            if row.get("dataset_manifest_sha256") != result.get("dataset_manifest_sha256"):
                raise ValueError(f"dataset manifest differs in execution: {method}")
            if row.get("execution_provenance") == "fixed_refusal":
                continue
            execution = row.get("execution")
            checker = row.get("checker")
            contract = result["composer_contract"]
            if (
                not isinstance(execution, dict)
                or not isinstance(checker, dict)
                or checker.get("contract_id") != result["checker_contract"]
                or execution.get("composer_model") != contract["model"]
                or execution.get("composer_prompt_version") != contract["prompt_version"]
                or execution.get("composer_config_sha256") != contract["config_sha256"]
                or execution.get("composer_catalog_sha256") != contract["catalog_sha256"]
            ):
                raise ValueError(f"execution provenance differs: {method}")
    if any(events != event_sets[0] for events in event_sets[1:]):
        raise ValueError("online methods use different event sets")


def render_table(result: dict[str, Any], digest: str) -> str:
    contract = result.get("composer_contract")
    model = result.get("retrieval_model")
    if (
        result.get("status") != "complete_online_retention"
        or result.get("events") != 320
        or result.get("storage_budget") != 640
        or result.get("retrieval_budget") != 160
        or result.get("workload_window") != 16
        or type(result.get("unique_verified_executions")) is not int
        or result["unique_verified_executions"] <= 0
        or not isinstance(model, dict)
        or model.get("kind") != "fixed_online_logistic"
        or model.get("dimension") != 256
        or not _digest(model.get("sha256"))
        or not isinstance(model.get("provenance"), str)
        or not _digest(model.get("provenance_sha256"))
        or not isinstance(contract, dict)
        or contract.get("model") != "gpt-4.1-2025-04-14"
        or not isinstance(contract.get("prompt_version"), str)
        or not all(_digest(contract.get(key)) for key in ("config_sha256", "catalog_sha256"))
        or result.get("checker_contract") != CHECKER_CONTRACT_ID
        or not _digest(result.get("dataset_manifest_sha256"))
        or not _digest(digest)
    ):
        raise ValueError("online run is not a verified paper compatible run")
    rows = result.get("summaries")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("missing online summary rows")
    by_method = {row.get("method"): row for row in rows}
    if len(by_method) != len(rows) or not set(dict(ORDER)).issubset(by_method):
        raise ValueError("online retention methods are missing or duplicated")
    for method, _ in ORDER:
        row = by_method[method]
        if row.get("n") != 320:
            raise ValueError(f"wrong event count: {method}")
        if any(not _fraction(row.get(field)) for field in ("vsr", "bs", "ecr")):
            raise ValueError(f"invalid success metric: {method}")
        if (
            type(row.get("total_churn_records")) is not int
            or row["total_churn_records"] < 0
            or not isinstance(row.get("mean_update_latency_ms"), (int, float))
            or not math.isfinite(row["mean_update_latency_ms"])
            or row["mean_update_latency_ms"] < 0
        ):
            raise ValueError(f"invalid churn or update time: {method}")
    best = {
        field: max(by_method[method][field] for method, _ in ORDER)
        for field in ("vsr", "bs", "ecr")
    }
    lines = [
        "% Generated from a checksum-verified online retention run.",
        f"% Result SHA-256: {digest}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Policy & VSR & BS & ECR & Churn & Update\\",
        r"\midrule",
    ]
    for method, label in ORDER:
        row = by_method[method]
        metrics = []
        for field in ("vsr", "bs", "ecr"):
            rendered = f"{100 * row[field]:.1f}"
            if row[field] == best[field]:
                rendered = rf"\textbf{{{rendered}}}"
            metrics.append(rendered)
        latency = (
            r"$<0.1$"
            if method == "no_memory" and row["mean_update_latency_ms"] < 0.1
            else f"{row['mean_update_latency_ms']:.1f}"
        )
        lines.append(
            f"{label} & {metrics[0]} & {metrics[1]} & {metrics[2]} & "
            f"{row['total_churn_records']} & {latency} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    raw = verify_run_files(args.run)
    result = json.loads(raw)
    rendered = render_table(result, hashlib.sha256(raw).hexdigest())
    verify_run_metrics(args.run, result)
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit("generated online-retention table is absent or stale")
        print("online-retention table matches the verified run")
        return 0
    if args.output.exists():
        raise SystemExit("refusing to overwrite an existing table")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"generated online-retention table from {args.run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
