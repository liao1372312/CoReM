"""Leakage guards for the independently sourced pilot selector."""

from __future__ import annotations

import unittest

from experiments.generate_public_multiservice_hgt_ensemble_selections import (
    lexical_bundle_score,
    project_handoff_relation_vocabulary,
    rank_candidates,
    validate_label_free_candidate,
    validate_query_only_event,
)


class V7SelectionTieBreakTest(unittest.TestCase):
    def test_lexical_sum_uses_all_bundle_steps(self) -> None:
        lexical = {"shared": 10.0, "correct": 2.0, "wrong": 0.0}
        self.assertEqual(
            lexical_bundle_score(lexical, frozenset({"shared", "correct"}), "max"),
            lexical_bundle_score(lexical, frozenset({"shared", "wrong"}), "max"),
        )
        self.assertGreater(
            lexical_bundle_score(lexical, frozenset({"shared", "correct"}), "sum"),
            lexical_bundle_score(lexical, frozenset({"shared", "wrong"}), "sum"),
        )

    def test_query_view_excludes_checker_and_task_key(self) -> None:
        row = {
            "event_id": "e1",
            "service_id": "gitea",
            "provider_id": "gitea",
            "version_id": "28.1.0",
            "event_time": "2026-10-18T00:00:00Z",
            "query": "Create a repository",
        }
        validate_query_only_event(row)
        with self.assertRaises(ValueError):
            validate_query_only_event({**row, "workflow_contract": {"title": "gold"}})

    def test_label_free_input_rejects_gold_fields_and_ids(self) -> None:
        good = {
            "event_id": "e1",
            "candidate_edge_id": "candidate-a3f9c1",
            "selected_memory_ids": ["record_1"],
        }
        validate_label_free_candidate(good)
        with self.assertRaises(ValueError):
            validate_label_free_candidate({**good, "method": "oracle"})
        with self.assertRaises(ValueError):
            validate_label_free_candidate({**good, "candidate_edge_id": "oracle"})
        with self.assertRaises(ValueError):
            validate_label_free_candidate({**good, "candidate_edge_id": "e1:bundle_a"})

    def test_unseen_handoff_lengths_project_to_frozen_relation(self) -> None:
        edges = [
            {"edge_id": "e1", "relation_type": "3-step-runtime-handoff"},
            {"edge_id": "e2", "relation_type": "7-step-runtime-handoff"},
        ]
        projected = project_handoff_relation_vocabulary(edges, "collapse_to_training")
        self.assertEqual(
            [row["relation_type"] for row in projected],
            ["three-step-runtime-handoff"] * 2,
        )
        self.assertEqual(edges[0]["relation_type"], "3-step-runtime-handoff")
        with self.assertRaises(ValueError):
            project_handoff_relation_vocabulary(
                [{"edge_id": "bad", "relation_type": "unknown"}],
                "collapse_to_training",
            )

    def test_seeded_tie_break_is_order_invariant(self) -> None:
        rows = [
            {"score": 0.5, "selected_memory_ids": ["bundle_a"]},
            {"score": 0.5, "selected_memory_ids": ["bundle_b"]},
        ]
        expected = rank_candidates(rows, "event-1", 2027, "seeded_hash")
        reversed_input = rank_candidates(
            list(reversed(rows)), "event-1", 2027, "seeded_hash"
        )
        self.assertEqual(expected, reversed_input)

    def test_score_dominates_hash(self) -> None:
        rows = [
            {"score": 0.2, "selected_memory_ids": ["bundle_a"]},
            {"score": 0.8, "selected_memory_ids": ["bundle_b"]},
        ]
        self.assertEqual(
            rank_candidates(rows, "event-1", 2027, "seeded_hash")[0], rows[1]
        )


if __name__ == "__main__":
    unittest.main()
