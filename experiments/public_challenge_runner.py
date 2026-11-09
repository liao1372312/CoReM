"""Version-isolated three-step runner for the public challenge track."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from experiments.public_workflows_runner import (
    PublicWorkflowsRunner,
    _capture_value,
)


class PublicChallengeRunner(PublicWorkflowsRunner):
    @staticmethod
    def _challenge_plans(
        memories: Sequence[Mapping[str, Any]], operation_count: int
    ) -> list[tuple[int, str, dict[str, Any]]]:
        plans: dict[int, tuple[str, dict[str, Any]]] = {}
        allowed = set(range(1, operation_count + 1))
        for memory in sorted(memories, key=lambda row: str(row.get("memory_id", ""))):
            plan = memory.get("plan")
            step_index = memory.get("step_index")
            if (
                memory.get("role") != "workflow_step"
                or not isinstance(plan, dict)
                or step_index not in allowed
            ):
                continue
            method = plan.get("method")
            path = plan.get("path")
            if method not in {"GET", "POST", "PATCH"} or not isinstance(path, str):
                continue
            if not path.startswith("/") or "//" in path or "\\" in path:
                continue
            if set(plan).difference({"method", "path", "json", "capture"}):
                continue
            plans.setdefault(
                int(step_index), (str(memory.get("memory_id")), dict(plan))
            )
        return [
            (index, memory_id, plan)
            for index, (memory_id, plan) in sorted(plans.items())
        ]

    def execute(
        self,
        event: Mapping[str, Any],
        memories: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        observations = self._memory_observations(memories)
        operation_count = int(event.get("operation_count", 0))
        plans = self._challenge_plans(memories, operation_count)
        expected = list(range(1, operation_count + 1))
        if operation_count < 3 or [item[0] for item in plans] != expected:
            return {
                "completed": False,
                "failure": "selected memories do not supply every workflow step",
                "selected_memory_observations": observations,
                "steps": [],
                "bindings": {},
                "communication": {
                    "explicit_statuses": False,
                    "outcome_summary": "Workflow was not executed.",
                },
            }

        bindings: dict[str, Any] = {}
        steps: list[dict[str, Any]] = []
        for step_index, memory_id, plan in plans:
            path = self._render_path(str(plan["path"]), bindings)
            response = self._request(plan, path)
            step = {
                "step_index": step_index,
                "plan_memory_id": memory_id,
                "request": {
                    "method": plan["method"],
                    "path_template": plan["path"],
                    "path": path,
                    "json": plan.get("json"),
                },
                "response": response,
            }
            capture = plan.get("capture")
            if capture is not None:
                if not isinstance(capture, dict) or not isinstance(
                    capture.get("name"), str
                ):
                    raise ValueError("capture must declare a scalar name")
                bindings[str(capture["name"])] = _capture_value(response, capture)
                step["captured_binding"] = str(capture["name"])
            steps.append(step)
        statuses = [int(step["response"]["status"]) for step in steps]
        return {
            "completed": True,
            "steps": steps,
            "bindings": bindings,
            "selected_memory_observations": observations,
            "communication": {
                "explicit_statuses": True,
                "outcome_summary": (
                    f"Executed three-step workflow with HTTP statuses "
                    f"{statuses[0]}, {statuses[1]}, and {statuses[2]}."
                ),
            },
        }


def create_runner() -> PublicChallengeRunner:
    return PublicChallengeRunner()
