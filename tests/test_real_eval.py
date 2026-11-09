from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from corem.real_eval import (
    RealTrackError,
    audit_real_dataset,
    grouped_chronological_split,
    score_execution_row,
    stratified_grouped_chronological_split,
    summarize_scored_rows,
    validate_native_export,
    validate_split_assignments,
    write_json,
    write_jsonl,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _event(index: int) -> dict[str, object]:
    provider = "provider-a" if index % 2 else "provider-b"
    service = "service-a" if index % 2 else "service-b"
    version = "v1" if index <= 4 else "v2"
    return {
        "event_id": f"event-{index}",
        "incident_id": f"incident-{index}",
        "duplicate_group": None,
        "provider_id": provider,
        "service_id": service,
        "version_id": version,
        "event_time": f"2024-01-{index:02d}T00:00:00Z",
        "query": f"perform task {index}",
        "required_binding_atoms": ["binding:user_id"],
        "required_evidence_atoms": ["evidence:postcondition"],
        "communication_criteria": ["communication:explicit_confirmation"],
        "checker_contract_id": "checker-v1",
    }


def _make_dataset(root: Path) -> list[dict[str, object]]:
    events = [_event(index) for index in range(1, 9)]
    memories = [
        {
            "memory_id": f"memory-{index}",
            "service_id": "service-a" if index % 2 else "service-b",
            "version_scope": ["v1" if index <= 2 else "v2"],
            "valid_from": "2023-12-01T00:00:00Z",
            "valid_until": None,
            "evidence_atoms": ["evidence:postcondition"],
            "provenance_uri": f"https://example.test/memory/{index}",
            "content_digest": f"sha256:{index:064d}",
            "provenance_verified": True,
        }
        for index in range(1, 5)
    ]
    transitions = [
        {
            "transition_id": "transition-a",
            "provider_id": "provider-a",
            "service_id": "service-a",
            "from_version_id": "v1",
            "to_version_id": "v2",
            "transition_time": "2024-01-04T12:00:00Z",
            "evidence_uri": "https://example.test/change/a",
            "independently_observed": True,
        },
        {
            "transition_id": "transition-b",
            "provider_id": "provider-b",
            "service_id": "service-b",
            "from_version_id": "v1",
            "to_version_id": "v2",
            "transition_time": "2024-01-05T12:00:00Z",
            "evidence_uri": "https://example.test/change/b",
            "independently_observed": True,
        },
    ]
    root.mkdir()
    write_jsonl(root / "events.jsonl", events)
    write_jsonl(root / "memories.jsonl", memories)
    write_jsonl(root / "version_transitions.jsonl", transitions)
    files = {
        "events.jsonl": {
            "records": len(events),
            "sha256": _digest(root / "events.jsonl"),
        },
        "memories.jsonl": {
            "records": len(memories),
            "sha256": _digest(root / "memories.jsonl"),
        },
        "version_transitions.jsonl": {
            "records": len(transitions),
            "sha256": _digest(root / "version_transitions.jsonl"),
        },
    }
    write_json(
        root / "manifest.json",
        {
            "schema_version": "corem-real-eval-v1",
            "real_world_observations": True,
            "provenance": {
                "source_uri": "https://example.test/archive",
                "license": "test-only",
                "immutable_source_id": "test-fixture-do-not-report",
                "collection_start_utc": "2024-01-01T00:00:00Z",
                "collection_end_utc": "2024-01-08T00:00:00Z",
            },
            "execution": {"resettable": True, "runner_contract": "runner-v1"},
            "checker": {
                "independent_from_runner": True,
                "contract_id": "checker-v1",
                "implementation_digest": "sha256:test-checker",
            },
            "files": files,
        },
    )
    return events


def _checker(
    passed: bool, *, invalid: bool, evidence_covered: bool = True
) -> dict[str, object]:
    return {
        "contract_id": "checker-v1",
        "goal_satisfied": passed,
        "goal_evidence_ref": "artifact://goal",
        "binding_checks": [
            {
                "atom": "binding:user_id",
                "passed": passed,
                "evidence_ref": "artifact://binding",
            }
        ],
        "evidence_checks": [
            {
                "atom": "evidence:postcondition",
                "covered": evidence_covered,
                "evidence_ref": "artifact://evidence",
            }
        ],
        "communication_checks": [
            {
                "criterion": "communication:explicit_confirmation",
                "passed": True,
                "evidence_ref": "artifact://communication",
            }
        ],
        "reuse_attempts": [
            {
                "memory_id": "memory-1",
                "attempted": True,
                "invalid_for_event": invalid,
                "evidence_ref": "artifact://reuse",
            }
        ],
    }


class RealDatasetAuditTests(unittest.TestCase):
    def test_complete_contract_is_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "real"
            _make_dataset(root)
            report = audit_real_dataset(root)
        self.assertTrue(report.eligible, report.errors)
        self.assertEqual(report.counts["events"], 8)
        self.assertEqual(report.counts["providers"], 2)

    def test_missing_contract_files_are_blocked_without_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = audit_real_dataset(Path(directory))
        self.assertFalse(report.eligible)
        self.assertIn("missing manifest.json", report.errors)
        self.assertIn("missing events.jsonl", report.errors)

    def test_non_utc_event_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "real"
            events = _make_dataset(root)
            events[0]["event_time"] = "2024-01-01T00:00:00"
            write_jsonl(root / "events.jsonl", events)
            manifest = __import__("json").loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            manifest["files"]["events.jsonl"]["sha256"] = _digest(root / "events.jsonl")
            write_json(root / "manifest.json", manifest)
            report = audit_real_dataset(root)
        self.assertFalse(report.eligible)
        self.assertTrue(any("explicit UTC" in error for error in report.errors))

    def test_static_public_release_requires_no_fictitious_transition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "static"
            events = _make_dataset(root)
            for event in events:
                event["version_id"] = "v1"
            write_jsonl(root / "events.jsonl", events)
            (root / "version_transitions.jsonl").unlink()
            manifest = __import__("json").loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            manifest["schema_version"] = "corem-real-static-v1"
            manifest["benchmark_mode"] = "static_public_release"
            manifest["natural_drift_claim"] = False
            manifest["event_time_semantics"] = "Controlled order, not traffic time"
            manifest["claim_boundary"] = "Pinned services; no observed drift"
            manifest["files"].pop("version_transitions.jsonl")
            manifest["files"]["events.jsonl"]["sha256"] = _digest(root / "events.jsonl")
            write_json(root / "manifest.json", manifest)
            report = audit_real_dataset(root)
        self.assertTrue(report.eligible, report.errors)
        self.assertEqual(report.counts["version_transitions"], 0)

    def test_static_dataset_rejects_unfounded_drift_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "static"
            _make_dataset(root)
            manifest = __import__("json").loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            manifest["schema_version"] = "corem-real-static-v1"
            manifest["benchmark_mode"] = "static_public_release"
            manifest["natural_drift_claim"] = True
            manifest["event_time_semantics"] = "Controlled order"
            manifest["claim_boundary"] = "No natural traffic"
            write_json(root / "manifest.json", manifest)
            report = audit_real_dataset(root)
        self.assertFalse(report.eligible)
        self.assertTrue(any("natural drift" in error for error in report.errors))


class ChronologicalSplitTests(unittest.TestCase):
    def test_test_only_split_is_allowed_only_when_explicit(self) -> None:
        events = [_event(index) for index in range(1, 3)]
        assignments = [
            {"event_id": event["event_id"], "split": "test"} for event in events
        ]
        with self.assertRaisesRegex(RealTrackError, "empty splits"):
            validate_split_assignments(events, assignments)
        validate_split_assignments(events, assignments, allow_test_only=True)

    def test_incidents_and_duplicates_never_cross_splits(self) -> None:
        events = [_event(index) for index in range(1, 9)]
        events[1]["duplicate_group"] = "near-duplicate-a"
        events[2]["duplicate_group"] = "near-duplicate-a"
        assignments = grouped_chronological_split(events)
        validate_split_assignments(events, assignments)
        by_id = {row["event_id"]: row["split"] for row in assignments}
        self.assertEqual(by_id["event-2"], by_id["event-3"])
        self.assertEqual(set(by_id.values()), {"train", "tune", "validation", "test"})

    def test_split_rejects_too_few_independent_groups(self) -> None:
        events = [_event(index) for index in range(1, 5)]
        for event in events:
            event["incident_id"] = "one-incident"
        with self.assertRaisesRegex(RealTrackError, "fewer than four"):
            grouped_chronological_split(events)

    def test_stratified_split_keeps_each_service_in_every_split(self) -> None:
        events = []
        for service_index, service in enumerate(("service-a", "service-b")):
            for index in range(1, 9):
                event = _event(index)
                event["event_id"] = f"{service}-{index}"
                event["incident_id"] = f"{service}-incident-{index}"
                event["service_id"] = service
                event["provider_id"] = f"provider-{service_index}"
                event["event_time"] = f"2024-0{service_index + 1}-{index:02d}T00:00:00Z"
                events.append(event)
        assignments = stratified_grouped_chronological_split(events)
        validate_split_assignments(events, assignments)
        observed = {(row["service_id"], row["split"]) for row in assignments}
        self.assertEqual(len(observed), 8)
        self.assertTrue(all("split_stratum" in row for row in assignments))


class RealMetricTests(unittest.TestCase):
    def test_metrics_use_independent_atom_level_checks(self) -> None:
        event = _event(1)
        first = score_execution_row(
            {
                "checker": _checker(True, invalid=True),
                "checker_independent": True,
                "reset_verified": True,
                "execution_error": None,
                "selected_memory_ids": ["memory-1"],
            },
            event,
        )
        second = score_execution_row(
            {
                "checker": _checker(False, invalid=False, evidence_covered=False),
                "checker_independent": True,
                "reset_verified": True,
                "execution_error": None,
                "selected_memory_ids": ["memory-1"],
            },
            event,
        )
        rows = [
            {**first, "latency_ms": 10.0, "token_cost": 100},
            {**second, "latency_ms": 30.0, "token_cost": 300},
        ]
        summary = summarize_scored_rows(rows)
        self.assertEqual(summary["vsr"], 0.5)
        self.assertEqual(summary["bs"], 0.5)
        self.assertEqual(summary["ecr"], 0.5)
        self.assertEqual(summary["evidence_atom_coverage"], 0.5)
        self.assertEqual(summary["sur"], 0.5)
        self.assertEqual(summary["mean_tokens"], 200.0)

    def test_missing_checker_atom_blocks_scoring(self) -> None:
        checker = _checker(True, invalid=False)
        checker["binding_checks"] = []
        with self.assertRaisesRegex(RealTrackError, "exactly match"):
            score_execution_row(
                {
                    "checker": checker,
                    "checker_independent": True,
                    "reset_verified": True,
                    "selected_memory_ids": ["memory-1"],
                },
                _event(1),
            )

    def test_shared_checker_and_runner_cannot_be_promoted(self) -> None:
        with self.assertRaisesRegex(RealTrackError, "checker_independent"):
            score_execution_row(
                {
                    "checker": _checker(True, invalid=False),
                    "checker_independent": False,
                    "reset_verified": True,
                    "selected_memory_ids": ["memory-1"],
                },
                _event(1),
            )


class NativeExportTests(unittest.TestCase):
    def test_complete_label_blind_native_export_is_valid(self) -> None:
        events = [_event(1), _event(2)]
        manifest = {
            "system_name": "NativeSystem",
            "source_repository": "https://example.test/native.git",
            "source_commit": "a" * 40,
            "method": "native_system",
            "native_unmodified_core": True,
            "token_budget": 1024,
            "storage_budget": 4096,
            "llm_call_budget": 1,
            "offline_training_examples": 100,
            "seeds": [7],
        }
        selections = [
            {
                "event_id": event["event_id"],
                "seed": 7,
                "method": "native_system",
                "selected_memory_ids": ["memory-1"],
                "token_cost": 64,
                "active_bank_cost": 1024,
                "llm_calls": 1,
                "latency_ms": 2.5,
            }
            for event in events
        ]
        result = validate_native_export(
            events,
            manifest,
            selections,
            split_event_ids={"event-1", "event-2"},
        )
        self.assertTrue(result["valid"], result["errors"])

    def test_label_leakage_and_missing_rows_are_rejected(self) -> None:
        events = [_event(1), _event(2)]
        manifest = {
            "system_name": "NativeSystem",
            "source_repository": "https://example.test/native.git",
            "source_commit": "a" * 40,
            "method": "native_system",
            "native_unmodified_core": True,
            "token_budget": 32,
            "storage_budget": 4096,
            "llm_call_budget": 1,
            "offline_training_examples": 100,
            "seeds": [7],
        }
        result = validate_native_export(
            events,
            manifest,
            [
                {
                    "event_id": "event-1",
                    "seed": 7,
                    "method": "native_system",
                    "selected_memory_ids": ["memory-1"],
                    "token_cost": 64,
                    "active_bank_cost": 1024,
                    "llm_calls": 1,
                    "latency_ms": 2.5,
                    "outcome": 1,
                }
            ],
            split_event_ids={"event-1", "event-2"},
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("label fields" in error for error in result["errors"]))
        self.assertTrue(any("missing" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
