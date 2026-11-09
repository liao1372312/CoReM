import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/public_independent_v10_e2"
COMMITMENT = ROOT / "experiments/results_independent_v10_e2_selection_commitment.json"
RESULTS = ROOT / "experiments/results_independent_v10_e2_factorial_v1"
DIAGNOSTIC = ROOT / "experiments/results_independent_v10_e2_posthoc_semantic_diagnostic"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class IndependentV10E2ProtocolTests(unittest.TestCase):
    def test_public_confirmation_set_is_complete_and_label_free(self) -> None:
        manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
        events = load_jsonl(DATASET / "events.jsonl")
        queries = load_jsonl(DATASET / "query_only_events.jsonl")
        candidates = load_jsonl(DATASET / "candidate_selections.jsonl")
        self.assertEqual(manifest["status"], "prospective_outcome_hidden_confirmation")
        self.assertFalse(manifest["outcomes_exposed_to_selectors"])
        self.assertEqual(len(events), 128)
        self.assertEqual(len(queries), 128)
        self.assertEqual(len(candidates), 256)
        self.assertEqual({row["provider_id"] for row in events}, {"solr", "typesense"})
        self.assertEqual(len({row["task_key"] for row in events}), 16)
        self.assertTrue(all(row["observation_tier"] == "prospective_outcome_hidden_confirmation" for row in events))
        forbidden = {"oracle", "complete_negative", "goal_satisfied", "label"}
        for path in (DATASET / "query_only_events.jsonl", DATASET / "candidate_selections.jsonl"):
            lowered = path.read_text(encoding="utf-8").lower()
            self.assertFalse(any(token in lowered for token in forbidden))

    def test_all_factorial_cells_are_hash_committed(self) -> None:
        commitment = json.loads(COMMITMENT.read_text(encoding="utf-8"))
        self.assertEqual(
            commitment["status"],
            "all_factorial_cells_committed_before_outcome_execution",
        )
        self.assertFalse(commitment["outcomes_loaded"])
        self.assertEqual(commitment["events"], 128)
        self.assertEqual(len(commitment["cells"]), 8)
        for method, entry in commitment["cells"].items():
            directory = ROOT / f"experiments/results_independent_v10_e2_{method}_selections"
            for field, name in (
                ("manifest_sha256", "manifest.json"),
                ("selections_sha256", "selections.jsonl"),
                ("candidate_scores_sha256", "candidate_scores.jsonl"),
            ):
                observed = hashlib.sha256((directory / name).read_bytes()).hexdigest()
                self.assertEqual(observed, entry[field])

    def test_outcome_hidden_factorial_matches_paper(self) -> None:
        result = json.loads((RESULTS / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(
            result["status"], "complete_outcome_hidden_event_confirmation"
        )
        self.assertEqual(result["events"], 128)
        self.assertEqual(result["candidate_sets"], 256)
        self.assertEqual(result["family_clusters"], 16)
        self.assertEqual(len(result["rows"]), 8)
        self.assertEqual(
            [row["verified_successes"] for row in result["rows"]],
            [55, 56, 53, 54, 58, 61, 56, 54],
        )
        self.assertAlmostEqual(
            result["main_effects_vsr_pp"]["graph_on_minus_off"], 2.5390625
        )
        comparison = next(
            row
            for row in result["paired_comparisons"]
            if row["contrast"] == "graph_complement_minus_text_additive"
        )
        self.assertEqual(
            (
                comparison["event_wins"],
                comparison["event_losses"],
                comparison["event_ties"],
            ),
            (18, 17, 93),
        )
        self.assertEqual(comparison["event_exact_sign_p"], 1.0)
        paper = (ROOT / "CoReM_WWW2027_Main_Expanded_Experiments.tex").read_text(
            encoding="utf-8"
        )
        for expected in (
            "On  & On  & On  & 43.0 & 43.0 & .253",
            "Off & Off & On  & 42.2 & 42.2 & .253",
            "On  & Off & Off & 47.7 & 47.7 & .253",
            "earlier lexical effect and double-off result do not replicate",
        ):
            self.assertIn(expected, paper)

    def test_result_checksums(self) -> None:
        for line in (RESULTS / "checksums.sha256").read_text(
            encoding="utf-8"
        ).splitlines():
            digest, name = line.split("  ", 1)
            observed = hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
            self.assertEqual(observed, digest)

    def test_semantic_result_is_explicitly_post_outcome(self) -> None:
        result = json.loads((DIAGNOSTIC / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "complete_post_outcome_diagnostic")
        self.assertIn("outcomes were inspected", result["claim_boundary"])
        self.assertEqual(result["candidate_physical_executions"], 256)
        self.assertEqual(
            {
                name: row["verified_successes"]
                for name, row in result["methods"].items()
            },
            {
                "bm25_sum_text_and_plan": 53,
                "trained_difference_path_nli": 128,
                "base_difference_path_nli": 128,
            },
        )


if __name__ == "__main__":
    unittest.main()
