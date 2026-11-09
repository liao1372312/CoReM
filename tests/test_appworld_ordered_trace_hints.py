"""Synthetic, non-protected checks for ordered AppWorld trace hints."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.build_appworld_train_ordered_trace_hints import (
    compact_api_sequence,
    ordered_api_names,
    route_pattern,
)


class OrderedTraceHintTests(unittest.TestCase):
    def test_dynamic_route_is_anchored(self) -> None:
        pattern = route_pattern("/calendar/events/{event_id}")
        self.assertIsNotNone(pattern.fullmatch("/calendar/events/42"))
        self.assertIsNone(pattern.fullmatch("/calendar/events/42/extra"))

    def test_ordered_resolution_and_compression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.jsonl"
            rows = [
                {"method": "GET", "url": "/calendar/events/42", "data": {}},
                {"method": "GET", "url": "/calendar/events/43", "data": {}},
                {"method": "POST", "url": "/calendar/events", "data": {}},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            docs = [
                ("GET", route_pattern("/calendar/events/{id}"), "calendar.get_event"),
                ("POST", route_pattern("/calendar/events"), "calendar.create_event"),
            ]
            names = ordered_api_names(path, docs)
            self.assertEqual(
                names,
                ["calendar.get_event", "calendar.get_event", "calendar.create_event"],
            )
            self.assertEqual(
                compact_api_sequence(names),
                "calendar.get_event x2 -> calendar.create_event",
            )

    def test_unmatched_call_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.jsonl"
            path.write_text(json.dumps({"method": "DELETE", "url": "/unknown"}) + "\n")
            with self.assertRaises(ValueError):
                ordered_api_names(path, [])


if __name__ == "__main__":
    unittest.main()
