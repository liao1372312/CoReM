"""The paper table accepts only a complete, contract-matched run."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.generate_online_retention_paper_table import (
    ORDER,
    render_table,
    verify_run_files,
    verify_run_metrics,
)


class OnlineRetentionPaperTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result = {
            "status": "complete_online_retention",
            "events": 320,
            "storage_budget": 640,
            "retrieval_budget": 160,
            "workload_window": 16,
            "unique_verified_executions": 1,
            "dataset_manifest_sha256": "d" * 64,
            "retrieval_model": {
                "kind": "fixed_online_logistic", "dimension": 256,
                "sha256": "e" * 64, "provenance": "frozen.provenance.json",
                "provenance_sha256": "f" * 64,
            },
            "composer_contract": {
                "model": "gpt-4.1-2025-04-14",
                "prompt_version": "corem-composer-v2",
                "config_sha256": "a" * 64,
                "catalog_sha256": "b" * 64,
            },
            "checker_contract": "public-multiservice-challenge-checker-v1",
            "summaries": [
                {
                    "method": method, "n": 320,
                    "vsr": 0.25, "bs": 0.5, "ecr": 0.2,
                    "total_churn_records": 10,
                    "mean_update_latency_ms": 1.5,
                }
                for method, _ in ORDER
            ],
        }

    def test_renders_only_paper_compatible_metrics(self) -> None:
        rendered = render_table(self.result, "c" * 64)
        self.assertIn("Group-aware &", rendered)
        self.assertIn(r"\end{tabular}", rendered)
        self.result["retrieval_model"]["kind"] = "five_seed_hgt_bm25_ensemble"
        with self.assertRaises(ValueError):
            render_table(self.result, "c" * 64)

    def test_rejects_missing_composer_contract_and_incomplete_run(self) -> None:
        self.result["composer_contract"] = None
        with self.assertRaises(ValueError):
            render_table(self.result, "c" * 64)
        self.result["composer_contract"] = {
            "model": "gpt-4.1-2025-04-14",
            "prompt_version": "corem-composer-v2",
            "config_sha256": "a" * 64,
            "catalog_sha256": "b" * 64,
        }
        self.result["events"] = 319
        with self.assertRaises(ValueError):
            render_table(self.result, "c" * 64)

    def test_verifies_underlying_artifact_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            names = (
                "result.json", "execution_results.jsonl",
                "scored_executions.jsonl", "bank_history.jsonl",
            )
            (run / "result.json").write_text(json.dumps(self.result), encoding="utf-8")
            for name in names[1:]:
                (run / name).write_text("{}\n", encoding="utf-8")
            (run / "checksums.sha256").write_text(
                "".join(
                    f"{hashlib.sha256((run / name).read_bytes()).hexdigest()}  {name}\n"
                    for name in names
                ),
                encoding="utf-8",
            )
            self.assertEqual(verify_run_files(run), (run / "result.json").read_bytes())
            (run / names[1]).write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_run_files(run)

    def test_summary_is_reconciled_against_event_rows(self) -> None:
        method = ORDER[0][0]
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            scored = [{
                "method": method, "event_id": f"event-{index}",
                "verified_success": 0, "binding_passed": 0,
                "binding_total": 1, "evidence_complete": 0,
            } for index in range(320)]
            banks = [{
                "method": method, "event_id": f"event-{index}",
                "activated_records": 0, "evicted_records": 0,
                "update_latency_ms": 1.0,
            } for index in range(320)]
            executions = [{
                "method": method, "event_id": f"event-{index}",
                "dataset_manifest_sha256": "d" * 64,
                "execution_provenance": "fixed_refusal",
            } for index in range(320)]
            for name, records in (
                ("scored_executions.jsonl", scored),
                ("bank_history.jsonl", banks),
                ("execution_results.jsonl", executions),
            ):
                (run / name).write_text(
                    "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
                )
            summary = self.result["summaries"][0]
            summary.update(vsr=0.0, bs=0.0, ecr=0.0,
                           total_churn_records=0, mean_update_latency_ms=1.0)
            verify_run_metrics(run, self.result, (ORDER[0],))
            summary["vsr"] = 0.25
            with self.assertRaisesRegex(ValueError, "summary differs"):
                verify_run_metrics(run, self.result, (ORDER[0],))


if __name__ == "__main__":
    unittest.main()
