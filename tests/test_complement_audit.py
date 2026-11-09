"""Regression tests for matched complement-score diagnostics."""

from __future__ import annotations

import unittest

from experiments.audit_complement_influence import audit, verify_frozen_pair


class ComplementAuditTests(unittest.TestCase):
    @staticmethod
    def row(event: str, candidate: str, probability: float, lexical: float) -> dict:
        return {
            "event_id": event,
            "selected_memory_ids": [candidate],
            "ensemble_probability": probability,
            "bm25_normalized": lexical,
            "score": 0.9 * probability + 0.1 * lexical,
        }

    def test_score_change_without_selection_change(self) -> None:
        event = {"e1": {"service_id": "s", "task_key": "t"}}
        full = [self.row("e1", "a", 0.8, 1.0), self.row("e1", "b", 0.2, 0.0)]
        off = [self.row("e1", "a", 0.7, 1.0), self.row("e1", "b", 0.1, 0.0)]
        result = audit(full, off, event)
        self.assertEqual(result["candidate_probabilities_changed"], 2)
        self.assertEqual(result["events_with_winner_change"], 0)
        self.assertEqual(result["full_winner_matches_bm25"], 1)

    def test_head_can_change_winner(self) -> None:
        event = {"e1": {"service_id": "s", "task_key": "t"}}
        full = [self.row("e1", "a", 0.8, 0.0), self.row("e1", "b", 0.6, 1.0)]
        off = [self.row("e1", "a", 0.5, 0.0), self.row("e1", "b", 0.6, 1.0)]
        result = audit(full, off, event)
        self.assertEqual(result["events_with_winner_change"], 1)
        self.assertEqual(result["events_with_any_ranking_change"], 1)

    def test_rejects_unmatched_candidates_and_lexical_scores(self) -> None:
        event = {"e1": {"service_id": "s", "task_key": "t"}}
        full = [self.row("e1", "a", 0.8, 1.0)]
        with self.assertRaisesRegex(ValueError, "candidate pools differ"):
            audit(full, [self.row("e1", "b", 0.8, 1.0)], event)
        with self.assertRaisesRegex(ValueError, "lexical control differs"):
            audit(full, [self.row("e1", "a", 0.8, 0.0)], event)

    def test_rejects_different_model_states(self) -> None:
        common = {
            "dataset": "d",
            "target_split": "test",
            "model_seeds": [17],
            "model_states": [{"sha256": "a"}],
            "neural_weight": 0.9,
            "bm25_weight": 0.1,
            "selection_rows": 1,
        }
        full = {**common, "method": "hgt_rank_ensemble"}
        off = {
            **common,
            "model_states": [{"sha256": "b"}],
            "state_variant": "hgt_corem",
            "model_variant": "hgt_additive",
        }
        with self.assertRaisesRegex(ValueError, "model_states"):
            verify_frozen_pair(full, off)


if __name__ == "__main__":
    unittest.main()
