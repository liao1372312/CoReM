#!/usr/bin/env python3
"""Compare train-pilot API-set breadth without exporting task content."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "data" / "private_corem"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_appworld_local_agent_pilot import _sample_task_ids  # noqa: E402


def summarize_sample(
    tasks: list[dict[str, object]], selected_ids: set[str]
) -> dict[str, object]:
    all_ids = [str(row["task_id"]) for row in tasks]
    if len(set(all_ids)) != len(all_ids) or not selected_ids <= set(all_ids):
        raise ValueError("invalid or missing train task identifiers")
    all_counts = [len(row["required_apis"]) for row in tasks]
    pilot_counts = [
        len(row["required_apis"])
        for row in tasks
        if str(row["task_id"]) in selected_ids
    ]
    if not all_counts or not pilot_counts or any(count < 1 for count in all_counts):
        raise ValueError("empty required-API set")
    return {
        "train_tasks": len(all_counts),
        "pilot_tasks": len(pilot_counts),
        "train_mean_required_apis": round(statistics.mean(all_counts), 3),
        "pilot_mean_required_apis": round(statistics.mean(pilot_counts), 3),
        "train_required_api_range": [min(all_counts), max(all_counts)],
        "pilot_required_api_range": [min(pilot_counts), max(pilot_counts)],
        "task_ids_exported": False,
        "task_text_exported": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-view", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--pilot-tasks", type=int, default=9)
    args = parser.parse_args()
    private_view = args.private_view.resolve()
    if not private_view.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error("view must stay under data/private_corem")
    if not 1 <= args.pilot_tasks <= 30:
        parser.error("pilot task count must be between 1 and 30")
    raw = private_view.read_bytes()
    payload = json.loads(raw)
    if payload.get("status") != "protected_local_only_train_development":
        parser.error("expected a protected train-only view")
    tasks = payload["tasks"]
    selected = set(
        _sample_task_ids([str(row["task_id"]) for row in tasks], args.seed, args.pilot_tasks)
    )
    result = summarize_sample(tasks, selected)
    result.update(
        {
            "seed": args.seed,
            "private_view_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
