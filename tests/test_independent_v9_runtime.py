"""Opt-in live tests on inspected pilot patterns, never v9 confirmation data."""

from __future__ import annotations

import os
import unittest

from experiments.independent_v9_checker import IndependentV9Checker
from experiments.independent_v9_runner import IndependentV9Runner


@unittest.skipUnless(
    os.environ.get("COREM_RUN_V9_INTEGRATION") == "1",
    "set COREM_RUN_V9_INTEGRATION=1 for isolated Typesense/Solr pilot tests",
)
class IndependentV9RuntimeTests(unittest.TestCase):
    def _run(self, service: str, positive: bool) -> bool:
        if service == "typesense":
            version = "30.2"
            collection = "corem_v9_inspected_pilot"
            title = "target" if positive else "decoy"
            plans = [
                {"method": "POST", "path": "/collections", "json": {"name": collection, "fields": [{"name": "title", "type": "string"}]}},
                {"method": "POST", "path": f"/collections/{collection}/documents", "json": {"id": "1", "title": title}},
                {"method": "GET", "path": f"/collections/{collection}/documents/1"},
            ]
            statuses = [201, 201, 200]
            state_path = f"/collections/{collection}/documents/1"
            assertions = [{"path": ["title"], "equals": "target"}]
        else:
            version = "9.10.1"
            core = "corem_v9_inspected_pilot"
            title = "target" if positive else "decoy"
            plans = [
                {"method": "GET", "path": f"/solr/admin/cores?action=CREATE&name={core}&instanceDir={core}&configSet=_default&wt=json"},
                {"method": "POST", "path": f"/solr/{core}/update/json/docs?commit=true", "json": {"id": "1", "title": title}},
                {"method": "GET", "path": f"/solr/{core}/select?q=id%3A1&wt=json"},
            ]
            statuses = [200, 200, 200]
            state_path = f"/solr/{core}/select?q=id%3A1&wt=json"
            assertions = [{"path": ["response", "docs", 0, "title"], "equals": ["target"]}]
        event = {
            "event_id": f"v9-inspected-pilot-{service}",
            "service_id": service,
            "version_id": version,
            "operation_count": 3,
            "workflow_contract": {
                "kind": "assertions_v1",
                "trace_statuses": statuses,
                "trace_assertions": [],
                "state_checks": [{"path": state_path, "status": 200, "assertions": assertions}],
            },
        }
        memories = [
            {"memory_id": f"{service}:step-{index}", "role": "workflow_step", "step_index": index, "plan": plan}
            for index, plan in enumerate(plans, 1)
        ]
        runner = IndependentV9Runner()
        runner.reset(event)
        try:
            execution = runner.execute(event, memories)
            result = IndependentV9Checker().check(event, execution, runner.checker_context())
            return bool(result["goal_satisfied"])
        finally:
            runner.snapshot(event)

    def test_typesense_positive_and_negative(self) -> None:
        self.assertTrue(self._run("typesense", True))
        self.assertFalse(self._run("typesense", False))

    def test_solr_positive_and_negative(self) -> None:
        self.assertTrue(self._run("solr", True))
        self.assertFalse(self._run("solr", False))
