"""Deterministic whole-workflow serialization for body-aware development models."""

from __future__ import annotations

import json
from typing import Any


def serialize_path_body_v1(
    candidate: dict[str, Any], memories: dict[str, dict[str, Any]]
) -> str:
    """Put API calls, request bodies, and bindings before longer evidence prose."""
    records = [memories[str(memory_id)] for memory_id in candidate["selected_memory_ids"]]
    records.sort(key=lambda row: (int(row.get("step_index") or 10_000), str(row["memory_id"])))
    operations = []
    evidence = []
    for row in records:
        plan = row.get("plan") or {}
        step = row.get("step_index") or "verify"
        if plan:
            body = json.dumps(plan.get("json", {}), sort_keys=True, separators=(",", ":"))
            bindings = json.dumps(plan.get("capture", {}), sort_keys=True, separators=(",", ":"))
            operations.append(
                f"Call {step}: {plan.get('method', '')} {plan.get('path', '')} "
                f"body {body} capture {bindings}."
            )
        evidence.append(f"Evidence {step}: {row['text']}")
    return "API sequence: " + " ".join(operations) + " Evidence: " + " ".join(evidence)
