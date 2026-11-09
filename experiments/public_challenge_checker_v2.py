"""Versioned checker that repairs the infeasible beta mutation contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from experiments.public_challenge_checker import (
    PublicChallengeChecker,
    _contains_record,
    _records,
    _type_matches,
)


def _create_fetch_query_goal(
    contract: Mapping[str, Any],
    steps: list[dict[str, Any]],
    bindings: Mapping[str, Any],
) -> bool:
    if len(steps) != 3:
        return False
    responses = [step.get("response") for step in steps]
    if not all(isinstance(response, dict) for response in responses):
        return False
    first, second, third = responses
    if [response.get("status") for response in responses] != contract.get("statuses"):
        return False
    first_body, second_body, third_body = (
        first.get("body"),
        second.get("body"),
        third.get("body"),
    )
    created_id = bindings.get("created_id")
    post_title = bindings.get("post_title")
    return bool(
        isinstance(first_body, dict)
        and isinstance(second_body, dict)
        and first_body.get("id") == created_id == second_body.get("id")
        and first_body.get("title") == second_body.get("title") == post_title == "Echo"
        and second_body.get("views") == 7
        and _contains_record(
            _records(third_body), id=created_id, title=post_title, views=7
        )
        and _type_matches(created_id, str(contract.get("id_type")))
    )


class PublicChallengeCheckerV2:
    def __init__(self) -> None:
        self.base = PublicChallengeChecker()

    def check(
        self, event: Mapping[str, Any], execution: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        checked = dict(self.base.check(event, execution))
        checked["contract_id"] = "public-versioned-challenge-checker-v2"
        contract = event.get("workflow_contract")
        if (
            isinstance(contract, dict)
            and contract.get("kind") == "json_create_fetch_query"
        ):
            raw_steps = execution.get("steps")
            steps = raw_steps if isinstance(raw_steps, list) else []
            raw_bindings = execution.get("bindings")
            bindings = raw_bindings if isinstance(raw_bindings, dict) else {}
            bindings_pass = all(
                row.get("passed") is True for row in checked["binding_checks"]
            )
            checked["goal_satisfied"] = bool(
                execution.get("completed") is True
                and bindings_pass
                and _create_fetch_query_goal(contract, steps, bindings)
            )
        return checked


def create_checker() -> PublicChallengeCheckerV2:
    return PublicChallengeCheckerV2()
