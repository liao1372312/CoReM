from __future__ import annotations

from unittest.mock import patch

import pytest

from experiments.independent_v9_checker import IndependentV9Checker


def _event(service: str) -> dict:
    return {
        "service_id": service,
        "operation_count": 1,
        "workflow_contract": {
            "kind": "assertions_v1",
            "trace_statuses": [200],
            "trace_assertions": [],
            "state_checks": [
                {"path": "/state", "status": 200, "assertions": [{"path": ["owner"], "equals": "target"}]}
            ],
        },
    }


def _execution() -> dict:
    return {"completed": True, "steps": [{"response": {"status": 200}}]}


def test_live_checker_accepts_matching_state_and_rejects_wrong_state() -> None:
    checker = IndependentV9Checker()
    context = {"service_id": "typesense", "base_url": "http://127.0.0.1:1234", "api_key": "secret"}
    with patch("experiments.independent_v9_checker._read_state", return_value=(200, {"owner": "target"})):
        result = checker.check(_event("typesense"), _execution(), context)
    assert result["goal_satisfied"] is True
    with patch("experiments.independent_v9_checker._read_state", return_value=(200, {"owner": "decoy"})):
        result = checker.check(_event("typesense"), _execution(), context)
    assert result["goal_satisfied"] is False


def test_live_checker_rejects_bad_trace_without_reading_state() -> None:
    checker = IndependentV9Checker()
    execution = _execution()
    execution["steps"][0]["response"]["status"] = 400
    context = {"service_id": "solr", "base_url": "http://127.0.0.1:1234", "api_key": ""}
    with patch("experiments.independent_v9_checker._read_state") as read_state:
        result = checker.check(_event("solr"), execution, context)
    assert result["goal_satisfied"] is False
    read_state.assert_not_called()


def test_live_checker_requires_scoped_service_context() -> None:
    checker = IndependentV9Checker()
    with pytest.raises(ValueError, match="service context mismatch"):
        checker.check(_event("typesense"), _execution(), {"service_id": "solr", "base_url": "http://127.0.0.1:1"})
