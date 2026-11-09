from __future__ import annotations

import unittest

from experiments.run_tau3_external_baseline_adapters import (
    TaskContext,
    beam_select,
    operation_priors,
    topk_select,
)


def fixture_context() -> TaskContext:
    memories = {
        "m1": {
            "token_cost": 6,
            "operation_ids": ["op1"],
            "provenance_verified": True,
            "version_scope": ["v1"],
        },
        "m2": {
            "token_cost": 5,
            "operation_ids": ["op2"],
            "provenance_verified": True,
            "version_scope": ["v1"],
        },
        "m3": {
            "token_cost": 4,
            "operation_ids": ["op3"],
            "provenance_verified": True,
            "version_scope": ["v1"],
        },
    }
    return TaskContext(
        task={
            "task_id": "q1",
            "domain": "retail",
            "active_versions": ["v1"],
            "gold_memory_ids": ["m2"],
            "gold_operation_ids": ["op2"],
        },
        bank=("m1", "m2", "m3"),
        memories=memories,
        item_values={
            "bm25": {"m1": 1.0, "m2": 0.8, "m3": 0.7},
            "dense_hash": {"m1": 1.0, "m2": 0.8, "m3": 0.7},
            "hybrid_rrf": {"m1": 1.0, "m2": 0.8, "m3": 0.7},
        },
        declared=frozenset({"m1", "m3"}),
        duplicate_groups=(frozenset({"m2", "m3"}),),
        operation_priors={"op1": 0.6, "op2": 0.4, "op3": 0.7},
        bank_operations=frozenset({"op1", "op2", "op3"}),
    )


class ExternalAdapterTests(unittest.TestCase):
    def test_topk_and_beam_enforce_budget(self) -> None:
        context = fixture_context()
        selected, _, _ = topk_select(context, "bm25", 10)
        self.assertEqual(selected, ("m1", "m3"))
        self.assertLessEqual(context.cost(selected), 10)

        selected, _, _ = beam_select(
            context,
            budget=10,
            beam_width=8,
            score_many=lambda candidates: [float(len(row)) for row in candidates],
        )
        self.assertEqual(len(selected), 2)
        self.assertLessEqual(context.cost(selected), 10)

    def test_declared_features_ignore_answer_side_gold_fields(self) -> None:
        context = fixture_context()
        gold_only = context.feature_dict(("m2",))
        declared = context.feature_dict(("m1", "m3"))
        self.assertEqual(gold_only["declared_coverage"], 0.0)
        self.assertEqual(gold_only["declared_complete"], 0.0)
        self.assertEqual(declared["declared_coverage"], 1.0)
        self.assertEqual(declared["declared_complete"], 1.0)

    def test_operation_priors_use_training_observations(self) -> None:
        rows = [
            {"selected_operation_ids": ["op1"], "outcome": 1},
            {"selected_operation_ids": ["op1", "op2"], "outcome": 0},
            {"selected_operation_ids": ["op2"], "outcome": 0},
        ]
        priors = operation_priors(rows)
        self.assertGreater(priors["op1"], priors["op2"])


if __name__ == "__main__":
    unittest.main()
