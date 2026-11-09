"""Metric-extension parsing and aggregation stay execution grounded."""

from __future__ import annotations

import unittest

from experiments.analyze_tau3_external_baselines import enrich_rows, method_summary
from experiments.tau3_selection_ids import selected_action_spec


class Tau3MetricExtensionTests(unittest.TestCase):
    def test_action_spec_separates_current_extra_and_stale(self) -> None:
        self.assertEqual(
            selected_action_spec(
                [
                    "mem:tau3:retail:abc:a2",
                    "mem:tau3:retail:abc:controlled-extra-3",
                    "mem:tau3:retail:abc:controlled-stale-1",
                ]
            ),
            ((2,), (3,), (1,)),
        )

    def test_metrics_require_current_successful_calls(self) -> None:
        task_id = "tau3:retail:abc"
        current = f"mem:{task_id}:a1"
        stale = f"mem:{task_id}:controlled-stale-1"
        tasks = {
            task_id: {
                "required_atoms": [f"goal:{task_id}"],
                "verified_atoms": [],
                "active_versions": ["current"],
            }
        }
        memories = {
            current: {
                "evidence_atoms": [f"goal:{task_id}"],
                "version_scope": ["current"],
            },
            stale: {
                "evidence_atoms": [f"goal:{task_id}"],
                "version_scope": ["old"],
                "controlled_stale": True,
                "current_version_invalid": True,
            },
        }
        annotations = {
            task_id: {
                "required_evidence_memory_ids": [current],
                "binding_atoms_by_memory": {
                    current: [{"atom": "operation"}, {"atom": "argument"}]
                },
                "binding_atom_count": 2,
            }
        }
        base = {
            "task_id": task_id,
            "method": "bm25_topk",
            "seed": 17,
            "outcome": 0,
            "token_cost": 10,
            "latency_ms": 1.0,
            "execution_error_count": 0,
            "assertion_error_count": 0,
            "adapter_level": "retrieval_control",
        }
        rows = enrich_rows(
            [
                {
                    **base,
                    "selected_memory_ids": [current, stale],
                    "call_checks": [
                        {
                            "kind": "current",
                            "source_action_index": 1,
                            "passed": True,
                        }
                    ],
                }
            ],
            tasks,
            memories,
            {task_id: frozenset({current})},
            annotations,
        )
        self.assertEqual(rows[0]["binding_passed"], 2)
        self.assertEqual(rows[0]["evidence_complete"], 1)
        self.assertEqual(rows[0]["stale_attempts"], 1)
        summary = method_summary("bm25_topk", rows, bootstrap_draws=10)
        self.assertEqual(summary["bs"], 1.0)
        self.assertEqual(summary["ecr"], 1.0)
        self.assertEqual(summary["sur"], 0.5)


if __name__ == "__main__":
    unittest.main()
