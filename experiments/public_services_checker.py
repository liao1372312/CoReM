"""Independent, atom-level checker for public version-service replays."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs, urlparse


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    return False


def _goal(contract: Mapping[str, Any], response: Mapping[str, Any]) -> bool:
    status = response.get("status")
    body = response.get("body")
    if status != contract.get("status"):
        return False
    kind = contract.get("kind")
    if kind == "json_pagination":
        records = body.get("data") if isinstance(body, dict) else body
        expected_container = contract.get("container")
        container_ok = (
            isinstance(body, dict)
            if expected_container == "paged"
            else isinstance(body, list)
        )
        return bool(
            container_ok
            and isinstance(records, list)
            and len(records) == contract.get("count")
            and all(
                isinstance(row, dict)
                and _type_matches(row.get("id"), str(contract.get("id_type")))
                for row in records
            )
        )
    if kind in {"json_record", "json_create"}:
        return bool(
            isinstance(body, dict)
            and body.get("title") == contract.get("title")
            and _type_matches(body.get("id"), str(contract.get("id_type")))
            and (kind != "json_create" or body.get("views") == contract.get("views"))
        )
    if kind == "json_comments":
        return bool(
            isinstance(body, list)
            and len(body) == contract.get("count")
            and all(
                isinstance(row, dict)
                and _type_matches(row.get("id"), str(contract.get("id_type")))
                for row in body
            )
        )
    if kind == "httpbin_anything":
        if not isinstance(body, dict):
            return False
        query = parse_qs(urlparse(str(body.get("url", ""))).query)
        return bool(
            body.get("method") == contract.get("method")
            and str(body.get("url", "")).endswith(str(contract.get("path_suffix")))
            and query.get("mode") == [contract.get("query_value")]
        )
    if kind == "status_empty":
        return response.get("text") == ""
    if kind == "plain_text":
        return response.get("text") == contract.get("text")
    if kind == "trailers":
        if contract.get("available"):
            return isinstance(body, dict) and body.get("method") == "GET"
        return isinstance(body, dict) and body.get("status_code") == 404
    return False


class PublicServicesChecker:
    def check(
        self, event: Mapping[str, Any], execution: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        contract = event.get("task_contract")
        response = execution.get("response")
        request = execution.get("request")
        if not isinstance(contract, dict):
            raise TypeError("event has no task contract")
        if not isinstance(response, dict):
            response = {}
        if not isinstance(request, dict):
            request = {}
        goal = bool(execution.get("completed") is True and _goal(contract, response))
        method_ok = request.get("method") in {"GET", "POST"}
        path_ok = isinstance(request.get("path"), str) and str(
            request["path"]
        ).startswith("/")
        status_ok = response.get("status") == contract.get("status")
        binding_results = {
            "binding:http_method": method_ok,
            "binding:http_path": path_ok,
            "binding:status_code": status_ok,
        }

        observations = execution.get("selected_memory_observations")
        if not isinstance(observations, list):
            observations = []
        valid_atoms: set[str] = set()
        reuse_attempts = []
        event_time = _utc(str(event["event_time"]))
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
            provenance_valid = memory.get("provenance_verified") is True
            valid = version_valid and service_valid and time_valid and provenance_valid
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
            "communication:explicit_status": communication.get("explicit_status")
            is True,
            "communication:outcome_summary": isinstance(
                communication.get("outcome_summary"), str
            )
            and "HTTP" in str(communication.get("outcome_summary")),
        }
        return {
            "contract_id": "public-versioned-services-checker-v1",
            "goal_satisfied": goal,
            "goal_evidence_ref": f"checker://response/{event['event_id']}",
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


def create_checker() -> PublicServicesChecker:
    return PublicServicesChecker()
