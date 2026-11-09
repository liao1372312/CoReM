"""Deterministic software fixtures for the real-track plugin protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class FixtureRunner:
    def reset(self, event: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "reset_verified": True,
            "snapshot_id": f"fixture:{event['service_id']}:{event['version_id']}",
        }

    def execute(
        self,
        event: Mapping[str, Any],
        memories: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        return {
            "completed": True,
            "event_id": event["event_id"],
            "selected_memory_ids": [memory["memory_id"] for memory in memories],
        }

    def snapshot(self, event: Mapping[str, Any]) -> str:
        return f"fixture:{event['service_id']}:{event['version_id']}:after"


class FixtureChecker:
    def check(
        self,
        event: Mapping[str, Any],
        execution: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        selected = list(execution.get("selected_memory_ids", []))
        return {
            "contract_id": event["checker_contract_id"],
            "goal_satisfied": execution.get("completed") is True,
            "goal_evidence_ref": f"fixture://{event['event_id']}/goal",
            "binding_checks": [
                {
                    "atom": atom,
                    "passed": True,
                    "evidence_ref": f"fixture://{event['event_id']}/binding/{index}",
                }
                for index, atom in enumerate(event["required_binding_atoms"])
            ],
            "evidence_checks": [
                {
                    "atom": atom,
                    "covered": True,
                    "evidence_ref": f"fixture://{event['event_id']}/evidence/{index}",
                }
                for index, atom in enumerate(event["required_evidence_atoms"])
            ],
            "communication_checks": [
                {
                    "criterion": criterion,
                    "passed": True,
                    "evidence_ref": (
                        f"fixture://{event['event_id']}/communication/{index}"
                    ),
                }
                for index, criterion in enumerate(event["communication_criteria"])
            ],
            "reuse_attempts": [
                {
                    "memory_id": memory_id,
                    "attempted": True,
                    "invalid_for_event": False,
                    "evidence_ref": f"fixture://{event['event_id']}/reuse/{memory_id}",
                }
                for memory_id in selected
            ],
        }


def create_runner() -> FixtureRunner:
    return FixtureRunner()


def create_checker() -> FixtureChecker:
    return FixtureChecker()

