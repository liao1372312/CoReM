"""Fail-closed checks for prospective v8 candidate-set structure."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from corem.real_eval import load_jsonl
from experiments.audit_independent_v8_structure import audit


class IndependentV8StructureTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = Path("data/public_independent_v8_alias_role_swap_development_v1")
        self.events = load_jsonl(directory / "events.jsonl")
        self.query_rows = load_jsonl(directory / "query_only_events.jsonl")
        self.memories = load_jsonl(directory / "memories.jsonl")
        self.edges = load_jsonl(directory / "dependency_hyperedges.jsonl")
        self.candidates = load_jsonl(directory / "candidate_selections.jsonl")

    def audit(self, *, candidates: list[dict] | None = None) -> dict:
        return audit(
            self.events,
            self.query_rows,
            self.memories,
            self.edges,
            candidates if candidates is not None else self.candidates,
            {("qdrant", "alias_role_swap_development_only")},
        )

    def test_rejects_inspected_small_pilot(self) -> None:
        result = self.audit()
        self.assertEqual(result["status"], "not_ready")
        self.assertTrue(any("128 incidents" in issue for issue in result["issues"]))
        self.assertTrue(any("overlaps inspected" in issue for issue in result["issues"]))

    def test_rejects_wrong_service_candidate(self) -> None:
        edges = self.edges
        self.edges = copy.deepcopy(edges)
        self.edges[0]["service_id"] = "openbao"
        result = self.audit()
        self.assertTrue(any("invalid candidate" in issue for issue in result["issues"]))


if __name__ == "__main__":
    unittest.main()
