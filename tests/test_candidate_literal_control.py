"""Checks for the predeclared literal-value complete-candidate control."""

from __future__ import annotations

import unittest

from experiments.build_v8_alias_development_pilot import build_candidate
from experiments.score_complete_candidate_literals import literal_score


class CandidateLiteralControlTests(unittest.TestCase):
    def test_alias_role_swap_is_a_literal_tie(self) -> None:
        query = (
            "Route v8_active to v8_alias_target and v8_spare to "
            "v8_alias_decoy."
        )
        scores = []
        for side, target in enumerate(("v8_alias_decoy", "v8_alias_target")):
            rows, _edge, candidate = build_candidate(
                side,
                target=target,
                event_id="v8-dev:qdrant:alias-role-swap-test",
                balanced_actions=True,
            )
            memories = {str(row["memory_id"]): row for row in rows}
            scores.append(literal_score(query, candidate, memories))
        self.assertEqual(scores[0], scores[1])

    def test_literal_match_uses_plan_values(self) -> None:
        candidate = {"selected_memory_ids": ["first"]}
        memories = {"first": {"plan": {"json": {"target": "desired-state"}}}}
        self.assertGreater(literal_score("Set desired-state", candidate, memories), 0)
        self.assertEqual(literal_score("Set a different state", candidate, memories), 0)


if __name__ == "__main__":
    unittest.main()
