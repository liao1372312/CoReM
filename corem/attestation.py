"""Structural attestation for provenance-backed workflow hyperedges."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


def _placeholders(value: object) -> set[str]:
    if isinstance(value, str):
        return set(_PLACEHOLDER.findall(value))
    if isinstance(value, Mapping):
        output: set[str] = set()
        for member in value.values():
            output.update(_placeholders(member))
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        output = set()
        for member in value:
            output.update(_placeholders(member))
        return output
    return set()


def attest_workflow_hyperedge(
    memories_by_id: Mapping[str, Mapping[str, Any]],
    edge: Mapping[str, Any],
) -> tuple[bool, str]:
    """Check that a claimed workflow edge has an internally grounded contract.

    The check does not trust edge identifiers, relation names, or provenance
    labels.  It requires one verification contract and checks every declared
    runtime handoff against a capture in an earlier step and a placeholder in
    the target step's structured plan.
    """

    member_ids = tuple(map(str, edge.get("members", ())))
    if len(member_ids) < 2 or len(set(member_ids)) != len(member_ids):
        return False, "invalid_members"
    if any(member_id not in memories_by_id for member_id in member_ids):
        return False, "unknown_member"
    members = [memories_by_id[member_id] for member_id in member_ids]
    verification_rows = [
        row
        for row in members
        if isinstance(row.get("verification_contract"), Mapping)
    ]
    if len(verification_rows) != 1:
        return False, "verification_contract_count"
    verification_row = verification_rows[0]
    expected_operations = frozenset(
        map(str, verification_row.get("operation_ids", ()))
    )
    if expected_operations and any(
        frozenset(map(str, row.get("operation_ids", ()))) != expected_operations
        for row in members
    ):
        return False, "operation_identity_mismatch"

    steps: dict[int, Mapping[str, Any]] = {}
    for row in members:
        if row.get("role") != "workflow_step":
            continue
        try:
            step_index = int(row["step_index"])
        except (KeyError, TypeError, ValueError):
            return False, "invalid_step_index"
        if step_index in steps:
            return False, "duplicate_step_index"
        if not isinstance(row.get("plan"), Mapping):
            return False, "missing_step_plan"
        steps[step_index] = row
    if not steps:
        return False, "missing_workflow_steps"

    contract = verification_row["verification_contract"]
    handoffs = contract.get("handoffs", ())
    if not isinstance(handoffs, Sequence) or isinstance(handoffs, (str, bytes)):
        return False, "invalid_handoffs"
    for handoff in handoffs:
        if not isinstance(handoff, Mapping):
            return False, "invalid_handoff"
        name = str(handoff.get("name", "")).strip()
        try:
            target_step = int(handoff["target_step"])
        except (KeyError, TypeError, ValueError):
            return False, "invalid_handoff_target"
        if not name or target_step not in steps:
            return False, "missing_handoff_target"
        if name not in _placeholders(steps[target_step]["plan"]):
            return False, "target_does_not_consume_handoff"
        captured_upstream = any(
            step_index < target_step
            and isinstance(row.get("plan", {}).get("capture"), Mapping)
            and str(row["plan"]["capture"].get("name", "")) == name
            for step_index, row in steps.items()
        )
        if not captured_upstream:
            return False, "handoff_not_captured_upstream"
    return True, "attested"


def partition_attested_hyperedges(
    memories: Sequence[Mapping[str, Any]],
    hyperedges: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Partition raw hyperedges into attested rows and auditable rejections."""

    memories_by_id = {str(row["memory_id"]): row for row in memories}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for edge in hyperedges:
        valid, reason = attest_workflow_hyperedge(memories_by_id, edge)
        if valid:
            accepted.append(dict(edge))
        else:
            rejected.append({"edge_id": str(edge.get("edge_id", "")), "reason": reason})
    return accepted, rejected
