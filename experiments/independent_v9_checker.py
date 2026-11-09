"""Independent live-state checker for prospective Typesense/Solr v9 workflows."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

from experiments.independent_v8_checker import _assertions_hold


def _read_state(base_url: str, path: str, api_key: str) -> tuple[int, Any]:
    if not base_url.startswith("http://127.0.0.1:"):
        raise ValueError("checker endpoint must be local")
    if not path.startswith("/") or path.startswith("//") or "\\" in path:
        raise ValueError("unsafe checker state path")
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-TYPESENSE-API-KEY"] = api_key
    request = urllib.request.Request(base_url + path, headers=headers, method="GET")
    try:
        response = urllib.request.urlopen(request, timeout=15)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        raw = response.read()
        status = int(response.status)
    try:
        body = json.loads(raw) if raw else None
    except json.JSONDecodeError as error:
        raise ValueError("checker received non-JSON live state") from error
    return status, body


class IndependentV9Checker:
    """Verify trace and fresh service state before the runner tears it down."""

    def check(
        self,
        event: Mapping[str, Any],
        execution: Mapping[str, Any],
        context: Mapping[str, str],
    ) -> dict[str, Any]:
        service = str(event.get("service_id"))
        if service not in {"typesense", "solr"} or context.get("service_id") != service:
            raise ValueError("checker service context mismatch")
        api_key = str(context.get("api_key", ""))
        if service == "typesense" and not api_key:
            raise ValueError("missing Typesense checker API key")
        if service == "solr" and api_key:
            raise ValueError("unexpected Solr checker API key")
        contract = event.get("workflow_contract")
        if not isinstance(contract, dict) or contract.get("kind") != "assertions_v1":
            raise ValueError("v9 requires an assertion workflow contract")
        steps = execution.get("steps")
        statuses = contract.get("trace_statuses")
        if (
            execution.get("completed") is not True
            or not isinstance(steps, list)
            or not isinstance(statuses, list)
            or len(steps) != int(event.get("operation_count", -1))
            or [step.get("response", {}).get("status") for step in steps] != statuses
        ):
            return {"goal_satisfied": False, "reason": "trace status mismatch"}
        trace_assertions = contract.get("trace_assertions", [])
        if trace_assertions and not _assertions_hold(execution, trace_assertions):
            return {"goal_satisfied": False, "reason": "trace assertion mismatch"}
        checks = contract.get("state_checks")
        if not isinstance(checks, list) or not checks:
            raise ValueError("v9 contract needs live state checks")
        results = []
        for index, check in enumerate(checks):
            if not isinstance(check, dict) or set(check) != {"path", "status", "assertions"}:
                raise ValueError("invalid v9 state-check contract")
            path = check["path"]
            if not isinstance(path, str):
                raise ValueError("state-check path must be text")
            status, body = _read_state(str(context.get("base_url", "")), path, api_key)
            passed = status == check["status"] and _assertions_hold(body, check["assertions"])
            results.append(
                {
                    "check_index": index,
                    "passed": passed,
                    "observed_status": status,
                    "expected_status": check["status"],
                    "evidence_ref": f"checker://v9/state/{index}",
                }
            )
        return {
            "goal_satisfied": all(row["passed"] for row in results),
            "contract_id": "independent-v9-assertions-v1",
            "state_checks": results,
            "trace_statuses_verified": True,
        }


def create_checker() -> IndependentV9Checker:
    return IndependentV9Checker()
