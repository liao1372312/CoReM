"""Structural checks for post-inspection matched-literal development tasks."""

from __future__ import annotations

import json
import re
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from corem.real_eval import load_jsonl


class RoleSwapDevelopmentTests(unittest.TestCase):
    def test_candidates_have_identical_plan_literal_multisets(self) -> None:
        dataset = Path("data/public_v8_role_swap_development_v1")
        memories = {
            row["memory_id"]: row for row in load_jsonl(dataset / "memories.jsonl")
        }
        grouped = defaultdict(list)
        for row in load_jsonl(dataset / "candidate_selections.jsonl"):
            grouped[row["event_id"]].append(row)
        self.assertEqual(len(grouped), 24)
        for event_id, candidates in grouped.items():
            with self.subTest(event_id=event_id):
                self.assertEqual(len(candidates), 2)
                bags = []
                plans = []
                for candidate in candidates:
                    rows = [memories[memory_id] for memory_id in candidate["selected_memory_ids"]]
                    self.assertEqual(sum(row["token_cost"] for row in rows), 144)
                    plan_text = " ".join(
                        json.dumps(row.get("plan", {}), sort_keys=True) for row in rows
                    )
                    plans.append(plan_text)
                    bags.append(Counter(re.findall(r"[a-z0-9_]+", plan_text.casefold())))
                self.assertEqual(bags[0], bags[1])
                self.assertNotEqual(plans[0], plans[1])


if __name__ == "__main__":
    unittest.main()
