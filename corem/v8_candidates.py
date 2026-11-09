"""Dependency-free eligibility checks for prospective v8 candidate bundles."""

from __future__ import annotations

import re
from typing import Any

from corem.real_eval import parse_utc


def validate_label_free_candidate(candidate: dict[str, Any]) -> None:
    if set(candidate) != {"event_id", "candidate_edge_id", "selected_memory_ids"}:
        raise ValueError("candidate input includes labels, outcomes, or extra fields")
    if (
        not isinstance(candidate["event_id"], str)
        or not isinstance(candidate["candidate_edge_id"], str)
        or not isinstance(candidate["selected_memory_ids"], list)
        or any(not isinstance(value, str) for value in candidate["selected_memory_ids"])
    ):
        raise ValueError("candidate identifiers must be strings")
    lowered = str(candidate).lower()
    if any(
        forbidden in lowered
        for forbidden in ("oracle", "complete_negative", "goal_satisfied")
    ) or re.search(r"(?:^|:)bundle_[ab](?::|\b)", lowered):
        raise ValueError("candidate input exposes an answer label or predictable side")


def validate_candidate(
    candidate: dict[str, Any],
    event: dict[str, Any],
    memories: dict[str, dict[str, Any]],
    edges: dict[str, dict[str, Any]],
) -> None:
    validate_label_free_candidate(candidate)
    edge_id = candidate["candidate_edge_id"]
    if edge_id not in edges:
        raise ValueError(f"candidate has no verified dependency edge: {edge_id}")
    edge = edges[edge_id]
    selected = candidate["selected_memory_ids"]
    if (
        not selected
        or len(set(selected)) != len(selected)
        or set(selected) != set(map(str, edge["members"]))
        or edge["verified"] is not True
        or edge.get("service_id", event["service_id"]) != event["service_id"]
        or event["version_id"] not in edge["version_scope"]
    ):
        raise ValueError(f"incomplete or ineligible dependency edge: {edge_id}")
    timestamp = parse_utc(event["event_time"], "event_time")
    if parse_utc(edge["valid_from"], "valid_from") > timestamp or (
        edge.get("valid_until") is not None
        and parse_utc(edge["valid_until"], "valid_until") <= timestamp
    ):
        raise ValueError(f"stale dependency edge: {edge_id}")
    for memory_id in selected:
        row = memories.get(memory_id)
        if (
            row is None
            or row["service_id"] != event["service_id"]
            or row.get("provider_id", event.get("provider_id", event["service_id"]))
            != event.get("provider_id", event["service_id"])
            or event["version_id"] not in row["version_scope"]
            or row["provenance_verified"] is not True
            or parse_utc(row["valid_from"], "valid_from") > timestamp
            or (
                row.get("valid_until") is not None
                and parse_utc(row["valid_until"], "valid_until") <= timestamp
            )
        ):
            raise ValueError(f"invalid candidate memory: {memory_id}")
