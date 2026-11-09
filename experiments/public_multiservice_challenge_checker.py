"""Independent semantic checker for the frozen four-service v5 challenge."""

from __future__ import annotations

import urllib.parse
from collections.abc import Mapping
from typing import Any

from experiments.public_challenge_checker_v2 import PublicChallengeCheckerV2

CHECKER_CONTRACT_ID = "public-multiservice-challenge-checker-v1"


def _contains_scalar(value: Any, expected: Any) -> bool:
    if value == expected:
        return True
    if isinstance(value, list):
        return any(_contains_scalar(item, expected) for item in value)
    if isinstance(value, dict):
        return any(_contains_scalar(item, expected) for item in value.values())
    return False


def _handoffs(
    contract: Mapping[str, Any],
    steps: list[dict[str, Any]],
    bindings: Mapping[str, Any],
) -> bool:
    passed = True
    for handoff in contract.get("handoffs", []):
        if not isinstance(handoff, dict):
            return False
        name = handoff.get("name")
        target = handoff.get("target_step")
        if not isinstance(name, str) or not isinstance(target, int):
            return False
        if name not in bindings or not 1 <= target <= len(steps):
            return False
        request = steps[target - 1].get("request")
        if not isinstance(request, dict):
            return False
        value = bindings[name]
        if handoff.get("target_location", "path") == "json":
            observed = _contains_scalar(request.get("json"), value)
        else:
            path = request.get("path")
            observed = isinstance(path, str) and urllib.parse.quote(
                str(value), safe=""
            ) in path
        passed = passed and observed
    return passed


def _responses(steps: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    values = [step.get("response") for step in steps]
    if not all(isinstance(value, dict) for value in values):
        return None
    return values  # type: ignore[return-value]


def _wiremock_goal(
    contract: Mapping[str, Any],
    steps: list[dict[str, Any]],
    bindings: Mapping[str, Any],
) -> bool:
    responses = _responses(steps)
    if responses is None or len(responses) != 3:
        return False
    first, second, third = responses
    if [row.get("status") for row in responses] != contract.get("statuses"):
        return False
    label = contract.get("label")
    expected_status = contract.get("mapped_status")
    mapping_id = bindings.get("mapping_id")
    mapped_status = bindings.get("mapped_status")
    first_body = first.get("body")
    second_body = second.get("body")
    third_body = third.get("body")
    return bool(
        isinstance(label, str)
        and isinstance(mapping_id, str)
        and isinstance(first_body, dict)
        and first_body.get("id") == mapping_id
        and isinstance(second_body, dict)
        and second_body.get("id") == mapping_id
        and isinstance(second_body.get("response"), dict)
        and second_body["response"].get("status")
        == mapped_status
        == expected_status
        and steps[1].get("request", {}).get("path")
        == f"/__admin/mappings/{mapping_id}"
        and steps[2].get("request", {}).get("path")
        == f"/proof/{label}/{mapped_status}"
        and isinstance(third_body, dict)
        and third_body.get("proof") == label
    )


def _mockserver_goal(
    contract: Mapping[str, Any],
    steps: list[dict[str, Any]],
    bindings: Mapping[str, Any],
) -> bool:
    responses = _responses(steps)
    if responses is None or len(responses) != 3:
        return False
    first, _second, third = responses
    if [row.get("status") for row in responses] != contract.get("statuses"):
        return False
    label = contract.get("label")
    expectation_id = bindings.get("expectation_id")
    clear_status = bindings.get("clear_status")
    first_body = first.get("body")
    third_body = third.get("body")
    second_request = steps[1].get("request")
    return bool(
        isinstance(label, str)
        and isinstance(expectation_id, str)
        and isinstance(first_body, list)
        and len(first_body) == 2
        and isinstance(first_body[0], dict)
        and first_body[0].get("id") == expectation_id
        and isinstance(first_body[1], dict)
        and isinstance(first_body[1].get("httpRequest"), dict)
        and first_body[1]["httpRequest"].get("path")
        == f"/proof/{label}/200"
        and clear_status == 200
        and isinstance(second_request, dict)
        and second_request.get("json") == {"id": expectation_id}
        and steps[2].get("request", {}).get("path")
        == f"/proof/{label}/{clear_status}"
        and third.get("status") == contract.get("proof_status")
        and isinstance(third_body, dict)
        and third_body.get("proof") == label
    )


class PublicMultiserviceChallengeChecker:
    def __init__(self) -> None:
        self.base = PublicChallengeCheckerV2()

    def check(
        self, event: Mapping[str, Any], execution: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        checked = dict(self.base.check(event, execution))
        checked["contract_id"] = CHECKER_CONTRACT_ID
        contract = event.get("workflow_contract")
        if not isinstance(contract, dict) or contract.get("kind") not in {
            "wiremock_mapping_status",
            "mockserver_selective_clear",
        }:
            return checked

        raw_steps = execution.get("steps")
        steps = raw_steps if isinstance(raw_steps, list) else []
        raw_bindings = execution.get("bindings")
        bindings = raw_bindings if isinstance(raw_bindings, dict) else {}
        ordered = [step.get("step_index") for step in steps] == [1, 2, 3]
        handoffs_ok = _handoffs(contract, steps, bindings)
        responses = _responses(steps)
        status_ok = bool(
            responses is not None
            and [row.get("status") for row in responses] == contract.get("statuses")
        )
        binding_results = {
            "binding:step_order": ordered,
            "binding:runtime_handoffs": handoffs_ok,
            "binding:status_codes": status_ok,
        }
        checked["binding_checks"] = [
            {
                "atom": atom,
                "passed": binding_results.get(atom, False),
                "evidence_ref": f"checker://binding/{atom}",
            }
            for atom in event["required_binding_atoms"]
        ]
        kind = contract.get("kind")
        semantic_goal = (
            _wiremock_goal(contract, steps, bindings)
            if kind == "wiremock_mapping_status"
            else _mockserver_goal(contract, steps, bindings)
        )
        checked["goal_satisfied"] = bool(
            execution.get("completed") is True
            and ordered
            and handoffs_ok
            and semantic_goal
        )
        return checked


def create_checker() -> PublicMultiserviceChallengeChecker:
    return PublicMultiserviceChallengeChecker()
