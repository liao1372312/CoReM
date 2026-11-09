"""Independent live final-state checker for two v8 development pilot workflows.

Scored operation-family contracts have not yet been designed or accepted.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

SAFE_NAME = re.compile(r"[A-Za-z0-9_-]+\Z")
MISSING = object()


def _at(value: Any, path: list[str | int]) -> Any:
    current = value
    for component in path:
        if isinstance(current, dict) and isinstance(component, str):
            current = current.get(component, MISSING)
        elif (
            isinstance(current, list)
            and isinstance(component, int)
            and 0 <= component < len(current)
        ):
            current = current[component]
        else:
            return MISSING
    return current


def _assertions_hold(value: Any, assertions: Any) -> bool:
    if not isinstance(assertions, list) or not assertions:
        raise ValueError("checker assertions must be a nonempty list")
    for assertion in assertions:
        if not isinstance(assertion, dict) or not isinstance(
            assertion.get("path"), list
        ):
            return False
        observed = _at(value, assertion["path"])
        if set(assertion) == {"path", "equals"}:
            if observed is MISSING or observed != assertion["equals"]:
                return False
        elif set(assertion) == {"path", "absent"}:
            if assertion["absent"] is not True or observed is not MISSING:
                return False
        elif set(assertion) == {"path", "nonempty"}:
            if (
                assertion["nonempty"] is not True
                or observed is MISSING
                or not isinstance(observed, (str, list, dict))
                or not observed
            ):
                return False
        else:
            return False
    return True


def _read_json(base_url: str, path: str, token: str = "") -> tuple[int, Any]:
    if not base_url.startswith("http://127.0.0.1:"):
        raise ValueError("checker endpoint must be local")
    headers = {"Accept": "application/json"}
    if token:
        headers["X-Vault-Token"] = token
    request = urllib.request.Request(base_url + path, headers=headers, method="GET")
    try:
        response = urllib.request.urlopen(request, timeout=10)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        status = int(response.status)
        raw = response.read()
    try:
        return status, json.loads(raw) if raw else None
    except json.JSONDecodeError as error:
        raise ValueError("checker received non-JSON service state") from error


def _assertion_contract(
    contract: Mapping[str, Any],
    execution: Mapping[str, Any],
    base_url: str,
    token: str,
    service_id: str,
) -> dict[str, Any]:
    expected_statuses = contract.get("trace_statuses")
    steps = execution.get("steps")
    if (
        execution.get("completed") is not True
        or not isinstance(steps, list)
        or not isinstance(expected_statuses, list)
        or [step.get("response", {}).get("status") for step in steps]
        != expected_statuses
    ):
        return {"goal_satisfied": False, "reason": "trace status mismatch"}
    trace_assertions = contract.get("trace_assertions", [])
    if trace_assertions and not _assertions_hold(execution, trace_assertions):
        return {"goal_satisfied": False, "reason": "trace assertion mismatch"}
    checks = contract.get("state_checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("assertion contract needs independent state checks")
    results = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict) or set(check) != {"path", "status", "assertions"}:
            raise ValueError("invalid state-check contract")
        path = check["path"]
        if (
            not isinstance(path, str)
            or not path.startswith("/")
            or path.startswith("//")
            or "\\" in path
        ):
            raise ValueError("unsafe checker state path")
        if service_id == "openbao" and not token:
            raise ValueError("missing OpenBao checker token")
        status, body = _read_json(base_url, path, token)
        passed = status == check["status"] and _assertions_hold(
            body, check["assertions"]
        )
        results.append(
            {
                "check_index": index,
                "passed": passed,
                "observed_status": status,
                "expected_status": check["status"],
                "evidence_ref": f"checker://v8/state/{index}",
            }
        )
    return {
        "goal_satisfied": all(item["passed"] for item in results),
        "contract_id": "independent-v8-assertions-v1",
        "state_checks": results,
        "trace_statuses_verified": True,
    }


class IndependentV8Checker:
    def check(
        self,
        event: Mapping[str, Any],
        execution: Mapping[str, Any],
        context: Mapping[str, str],
    ) -> dict[str, Any]:
        contract = event.get("workflow_contract")
        if not isinstance(contract, dict):
            raise ValueError("v8 pilot requires a private workflow contract")
        kind = contract.get("kind")
        service_id = str(event.get("service_id"))
        if context.get("service_id") != service_id:
            raise ValueError("checker service context mismatch")
        base_url = str(context.get("base_url", ""))
        token = str(context.get("token", ""))
        if kind == "assertions_v1":
            return _assertion_contract(contract, execution, base_url, token, service_id)
        steps = execution.get("steps")
        if not isinstance(steps, list) or len(steps) != 3:
            return {"goal_satisfied": False, "reason": "incomplete workflow trace"}
        statuses = [step.get("response", {}).get("status") for step in steps]
        if statuses != [200, 200, 200] or execution.get("completed") is not True:
            return {"goal_satisfied": False, "reason": "operation failure"}
        if kind == "pilot_qdrant_point" and service_id == "qdrant":
            collection = contract.get("collection")
            point_id = contract.get("point_id")
            expected_tag = contract.get("tag")
            if (
                not isinstance(collection, str)
                or SAFE_NAME.fullmatch(collection) is None
                or not isinstance(point_id, int)
                or isinstance(point_id, bool)
                or not isinstance(expected_tag, str)
            ):
                raise ValueError("invalid Qdrant pilot contract")
            status, state = _read_json(
                base_url,
                f"/collections/{collection}/points/{point_id}?with_payload=true",
            )
            point = state.get("result") if isinstance(state, dict) else None
            query_body = steps[2].get("response", {}).get("body")
            query_points = (
                query_body.get("result", {}).get("points")
                if isinstance(query_body, dict)
                and isinstance(query_body.get("result"), dict)
                else None
            )
            valid = bool(
                status == 200
                and isinstance(point, dict)
                and point.get("id") == point_id
                and isinstance(point.get("payload"), dict)
                and point["payload"].get("tag") == expected_tag
                and isinstance(query_points, list)
                and query_points
                and query_points[0].get("id") == point_id
            )
        elif kind == "pilot_openbao_kv" and service_id == "openbao":
            secret_key = contract.get("secret_key")
            expected_marker = contract.get("marker")
            if (
                not isinstance(secret_key, str)
                or SAFE_NAME.fullmatch(secret_key) is None
                or not isinstance(expected_marker, str)
                or not token
            ):
                raise ValueError("invalid OpenBao pilot contract")
            status, state = _read_json(
                base_url, f"/v1/secret/data/{secret_key}", token
            )
            meta_status, metadata = _read_json(
                base_url, f"/v1/secret/metadata/{secret_key}", token
            )
            data = state.get("data", {}).get("data") if isinstance(state, dict) else None
            metadata_data = metadata.get("data") if isinstance(metadata, dict) else None
            traced_data = steps[1].get("response", {}).get("body")
            traced_marker = (
                traced_data.get("data", {}).get("data", {}).get("marker")
                if isinstance(traced_data, dict)
                else None
            )
            valid = bool(
                status == 200
                and meta_status == 200
                and isinstance(data, dict)
                and data.get("marker") == expected_marker
                and isinstance(metadata_data, dict)
                and metadata_data.get("current_version") == 1
                and traced_marker == expected_marker
            )
        else:
            raise ValueError(f"unsupported v8 pilot contract: {kind}")
        return {
            "goal_satisfied": valid,
            "reason": "independent live state and trace agree" if valid else "goal mismatch",
            "contract_id": "independent-v8-pilot-checker-v1",
            "evidence_ref": f"checker://v8-pilot/{event['event_id']}",
        }


def create_checker() -> IndependentV8Checker:
    return IndependentV8Checker()
