"""Guard label-free path scoring and temporal candidate eligibility."""

from __future__ import annotations

import unittest

from experiments.score_dependency_path_reranker import validate_candidate
from experiments.path_serialization import serialize_path_body_v1
from experiments.score_v7_path_nli_development import serialize_path


class DependencyPathRerankerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.event = {
            "event_id": "example:event-1",
            "service_id": "example",
            "version_id": "1.0",
            "event_time": "2026-10-18T00:01:00Z",
        }
        self.candidate = {
            "event_id": "example:event-1",
            "candidate_edge_id": "example:edge-1",
            "selected_memory_ids": ["example:record-1", "example:record-2"],
        }
        self.memories = {
            memory_id: {
                "memory_id": memory_id,
                "service_id": "example",
                "version_scope": ["1.0"],
                "provenance_verified": True,
                "valid_from": "2026-10-17T00:00:00Z",
                "valid_until": None,
                "step_index": index,
                "text": f"Do step {index}.",
                "plan": {"method": "POST", "path": f"/step/{index}"},
            }
            for index, memory_id in enumerate(self.candidate["selected_memory_ids"], 1)
        }
        self.edges = {
            "example:edge-1": {
                "members": list(self.candidate["selected_memory_ids"]),
                "verified": True,
                "version_scope": ["1.0"],
                "valid_from": "2026-10-17T00:00:00Z",
                "valid_until": None,
            }
        }

    def test_valid_candidate_and_ordered_path(self) -> None:
        validate_candidate(self.candidate, self.event, self.memories, self.edges)
        path = serialize_path(self.candidate, self.memories)
        self.assertLess(path.index("Step 1"), path.index("Step 2"))
        self.assertIn("POST /step/2", path)

    def test_rejects_stale_or_incomplete_candidates(self) -> None:
        self.edges["example:edge-1"]["valid_until"] = "2026-10-18T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "stale dependency edge"):
            validate_candidate(self.candidate, self.event, self.memories, self.edges)
        self.edges["example:edge-1"]["valid_until"] = None
        self.candidate["selected_memory_ids"] = ["example:record-1"]
        with self.assertRaisesRegex(ValueError, "incomplete or ineligible"):
            validate_candidate(self.candidate, self.event, self.memories, self.edges)

    def test_rejects_cross_service_or_provider_bundle(self) -> None:
        self.edges["example:edge-1"]["service_id"] = "another-service"
        with self.assertRaisesRegex(ValueError, "incomplete or ineligible"):
            validate_candidate(self.candidate, self.event, self.memories, self.edges)
        self.edges["example:edge-1"]["service_id"] = "example"
        self.memories["example:record-1"]["provider_id"] = "another-provider"
        with self.assertRaisesRegex(ValueError, "invalid candidate memory"):
            validate_candidate(self.candidate, self.event, self.memories, self.edges)

    def test_body_aware_serialization_distinguishes_same_route(self) -> None:
        left = serialize_path_body_v1(self.candidate, self.memories)
        self.memories["example:record-2"]["plan"]["json"] = {
            "actions": [{"create_alias": {"collection_name": "target"}}]
        }
        right = serialize_path_body_v1(self.candidate, self.memories)
        self.assertNotEqual(left, right)
        self.assertIn('"collection_name":"target"', right)


if __name__ == "__main__":
    unittest.main()
