"""Independent checker for three-step public challenge workflows."""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    return False


def _records(body: Any) -> list[dict[str, Any]] | None:
    value = body.get("data") if isinstance(body, dict) and "data" in body else body
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        return None
    return value


def _contains_record(records: list[dict[str, Any]] | None, **expected: Any) -> bool:
    return bool(
        records is not None
        and any(
            all(row.get(key) == value for key, value in expected.items())
            for row in records
        )
    )


def _url_ends(body: Any, suffix: str) -> bool:
    return isinstance(body, dict) and urllib.parse.urlparse(
        str(body.get("url", ""))
    ).path.endswith(suffix)


def _goal(
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
    kind = contract.get("kind")

    if kind == "json_paginate_title_query":
        page = _records(first_body)
        post_id = bindings.get("post_id")
        post_title = bindings.get("post_title")
        return bool(
            page is not None
            and len(page) == 2
            and post_id == page[0].get("id")
            and isinstance(second_body, dict)
            and second_body.get("id") == post_id
            and second_body.get("title") == post_title == "alpha"
            and _contains_record(_records(third_body), id=post_id, title=post_title)
            and _type_matches(post_id, str(contract.get("id_type")))
        )
    if kind == "json_create_update_fetch":
        created_id = bindings.get("created_id")
        updated_id = bindings.get("updated_id")
        return bool(
            isinstance(first_body, dict)
            and isinstance(second_body, dict)
            and isinstance(third_body, dict)
            and first_body.get("id") == created_id == updated_id
            and second_body.get("id") == updated_id == third_body.get("id")
            and first_body.get("title") == "Draft"
            and second_body.get("title") == third_body.get("title") == "Echo"
            and third_body.get("views") == 7
            and _type_matches(created_id, str(contract.get("id_type")))
        )
    if kind == "json_relationship_title_query":
        post_id = bindings.get("post_id")
        post_title = bindings.get("post_title")
        return bool(
            isinstance(first_body, dict)
            and isinstance(first_body.get("post"), dict)
            and first_body["post"].get("id") == post_id
            and isinstance(second_body, dict)
            and second_body.get("id") == post_id
            and second_body.get("title") == post_title == "alpha"
            and _contains_record(_records(third_body), id=post_id, title=post_title)
            and _type_matches(post_id, str(contract.get("id_type")))
        )
    if kind == "json_update_views_query":
        post_id = bindings.get("post_id")
        views = bindings.get("views")
        return bool(
            isinstance(first_body, dict)
            and isinstance(second_body, dict)
            and first_body.get("id") == post_id == second_body.get("id")
            and second_body.get("views") == views == 10
            and second_body.get("title") == "updated"
            and _contains_record(_records(third_body), id=post_id, views=views)
            and _type_matches(post_id, str(contract.get("id_type")))
        )
    reflected_status = bindings.get("reflected_status")
    if kind == "httpbin_uuid_status_chain":
        value = bindings.get("uuid")
        return bool(
            isinstance(first_body, dict)
            and first_body.get("uuid") == value
            and isinstance(value, str)
            and UUID_PATTERN.fullmatch(value)
            and _url_ends(second_body, f"/anything/{value}")
            and reflected_status == 200
            and _url_ends(third_body, "/anything/status-200")
        )
    if kind == "httpbin_base64_status_chain":
        value = bindings.get("decoded")
        return bool(
            first.get("text") == value == "CoReM"
            and _url_ends(second_body, "/anything/CoReM")
            and reflected_status == 200
            and _url_ends(third_body, "/anything/status-200")
        )
    if kind == "httpbin_header_status_chain":
        value = bindings.get("release_token")
        return bool(
            isinstance(first_body, dict)
            and first_body.get("X-Corem-Token") == [value]
            and value == "release-proof"
            and _url_ends(second_body, "/anything/release-proof")
            and reflected_status == 200
            and _url_ends(third_body, "/anything/status-200")
        )
    if kind == "httpbin_trailer_status_chain":
        value = bindings.get("availability_status")
        return bool(
            value == contract.get("trailers_status")
            and _url_ends(second_body, f"/anything/status-{value}")
            and reflected_status == 200
            and _url_ends(third_body, "/anything/final-200")
        )
    return False


class PublicChallengeChecker:
    def check(
        self, event: Mapping[str, Any], execution: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        contract = event.get("workflow_contract")
        if not isinstance(contract, dict):
            raise TypeError("event has no workflow contract")
        raw_steps = execution.get("steps")
        steps = raw_steps if isinstance(raw_steps, list) else []
        raw_bindings = execution.get("bindings")
        bindings = raw_bindings if isinstance(raw_bindings, dict) else {}
        ordered = [step.get("step_index") for step in steps] == [1, 2, 3]
        handoffs_ok = True
        for handoff in contract.get("handoffs", []):
            if not isinstance(handoff, dict):
                handoffs_ok = False
                continue
            name = handoff.get("name")
            target = handoff.get("target_step")
            value = bindings.get(name) if isinstance(name, str) else None
            request = (
                steps[target - 1].get("request")
                if isinstance(target, int)
                and 1 <= target <= len(steps)
                and isinstance(steps[target - 1], dict)
                else None
            )
            path = request.get("path") if isinstance(request, dict) else None
            handoffs_ok = handoffs_ok and bool(
                value is not None
                and isinstance(path, str)
                and urllib.parse.quote(str(value), safe="") in path
            )
        statuses = [
            step.get("response", {}).get("status")
            if isinstance(step, dict) and isinstance(step.get("response"), dict)
            else None
            for step in steps
        ]
        status_ok = statuses == contract.get("statuses")
        goal = bool(
            execution.get("completed") is True
            and ordered
            and handoffs_ok
            and _goal(contract, steps, bindings)
        )
        binding_results = {
            "binding:step_order": ordered,
            "binding:runtime_handoffs": handoffs_ok,
            "binding:status_codes": status_ok,
        }

        observations = execution.get("selected_memory_observations")
        if not isinstance(observations, list):
            observations = []
        event_time = _utc(str(event["event_time"]))
        valid_atoms: set[str] = set()
        reuse_attempts = []
        for memory in observations:
            if not isinstance(memory, dict) or not isinstance(
                memory.get("memory_id"), str
            ):
                continue
            scopes = memory.get("version_scope")
            valid_from = memory.get("valid_from")
            valid_until = memory.get("valid_until")
            version_valid = (
                isinstance(scopes, list) and event.get("version_id") in scopes
            )
            service_valid = memory.get("service_id") == event.get("service_id")
            time_valid = isinstance(valid_from, str) and _utc(valid_from) <= event_time
            if isinstance(valid_until, str):
                time_valid = time_valid and event_time <= _utc(valid_until)
            valid = bool(
                version_valid
                and service_valid
                and time_valid
                and memory.get("provenance_verified") is True
            )
            if valid and isinstance(memory.get("evidence_atoms"), list):
                valid_atoms.update(str(atom) for atom in memory["evidence_atoms"])
            reuse_attempts.append(
                {
                    "memory_id": memory["memory_id"],
                    "attempted": True,
                    "invalid_for_event": not valid,
                    "evidence_ref": f"checker://reuse/{memory['memory_id']}",
                }
            )

        communication = execution.get("communication")
        if not isinstance(communication, dict):
            communication = {}
        communication_results = {
            "communication:all_statuses": communication.get("explicit_statuses")
            is True,
            "communication:workflow_summary": isinstance(
                communication.get("outcome_summary"), str
            )
            and "three-step" in str(communication.get("outcome_summary")),
        }
        return {
            "contract_id": "public-versioned-challenge-checker-v1",
            "goal_satisfied": goal,
            "goal_evidence_ref": f"checker://challenge/{event['event_id']}",
            "binding_checks": [
                {
                    "atom": atom,
                    "passed": binding_results.get(atom, False),
                    "evidence_ref": f"checker://binding/{atom}",
                }
                for atom in event["required_binding_atoms"]
            ],
            "evidence_checks": [
                {
                    "atom": atom,
                    "covered": atom in valid_atoms,
                    "evidence_ref": f"checker://evidence/{atom}",
                }
                for atom in event["required_evidence_atoms"]
            ],
            "communication_checks": [
                {
                    "criterion": criterion,
                    "passed": communication_results.get(criterion, False),
                    "evidence_ref": f"checker://communication/{criterion}",
                }
                for criterion in event["communication_criteria"]
            ],
            "reuse_attempts": reuse_attempts,
        }


def create_checker() -> PublicChallengeChecker:
    return PublicChallengeChecker()
