"""Checks for matched representation-by-utility summary logic."""

from __future__ import annotations

import unittest

from experiments.summarize_public_multiservice_factorial import summarize


class FactorialSummaryTests(unittest.TestCase):
    def test_paired_main_effects_and_interaction(self) -> None:
        events = {
            "e1": {"service_id": "s", "version_id": "v", "task_key": "a"},
            "e2": {"service_id": "s", "version_id": "v", "task_key": "b"},
        }
        labels = {
            "hgt_corem": [1, 1],
            "hgt_additive": [1, 0],
            "text_complement": [0, 1],
            "text_additive": [0, 0],
        }
        rows = {
            cell: {
                event_id: {
                    "event_id": event_id,
                    "label": outcome,
                    "selected_memory_ids": [f"{cell}-{event_id}"],
                }
                for event_id, outcome in zip(events, outcomes, strict=True)
            }
            for cell, outcomes in labels.items()
        }
        result = summarize(rows, events)
        self.assertEqual(result["events"], 2)
        self.assertEqual(result["cells"]["hgt_corem"]["vsr"], 1.0)
        self.assertEqual(
            result["paired_contrasts"]["complement_with_graph"]["difference"],
            0.5,
        )
        self.assertEqual(result["paired_contrasts"]["interaction"]["difference"], 0.0)

    def test_rejects_unmatched_events(self) -> None:
        rows = {
            "hgt_corem": {"e1": {"label": 1}},
            "hgt_additive": {"e2": {"label": 1}},
            "text_complement": {"e1": {"label": 1}},
            "text_additive": {"e1": {"label": 1}},
        }
        with self.assertRaisesRegex(ValueError, "identical events"):
            summarize(rows, {})


if __name__ == "__main__":
    unittest.main()
