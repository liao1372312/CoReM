"""Fail-closed checks for the prospective v9 input and freeze boundaries."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments.audit_independent_v9_structure import audit
from experiments.export_v9_selector_view import SELECTOR_FILES
from experiments.freeze_independent_v9_candidate import _gate
from experiments.score_v9_lexical_tie_hybrid import _inputs


def test_v9_structural_audit_rejects_query_projection_leak(tmp_path: Path) -> None:
    source = Path("data/public_independent_v9_staging_v7")
    target = tmp_path / "leaked"
    shutil.copytree(source, target)
    query_path = target / "query_only_events.jsonl"
    rows = [json.loads(line) for line in query_path.read_text().splitlines()]
    rows[0]["workflow_contract"] = {"leak": True}
    query_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = audit(target, 256)
    assert result["status"] == "not_ready"
    assert any("query-only projection" in issue for issue in result["issues"])


def test_v9_selector_input_matches_inspected_development_pool() -> None:
    events, _memories, candidates, scores = _inputs(
        Path("data/public_independent_v8_candidate_v1"),
        Path("experiments/results_v8_candidate_v1_bm25_plan_selections/candidate_scores.jsonl"),
        160,
    )
    assert len(events) == len(candidates) == 128
    assert len(scores) == 256


def test_v9_trace_repaired_candidate_keeps_balanced_public_structure() -> None:
    result = audit(Path("data/public_independent_v9_staging_v7"), 256)
    assert result["status"] == "structurally_ready_for_physical_gate"
    assert not result["issues"]
    assert len(result["metrics"]["family_counts"]) == 16


def test_v9_freeze_rejects_superseded_404_negative_trace() -> None:
    private = Path("data/private_corem/v9_staging_v6")
    with pytest.raises(ValueError, match="nonconforming physical gate row"):
        _gate(
            private / "negative_gate_v1",
            Path("data/public_independent_v9_staging_v6"),
            private / "complete_negative_selections.jsonl",
            0,
        )


def test_v9_selector_view_excludes_contracts_and_private_labels() -> None:
    directory = Path("data/public_independent_v9_staging_v7_selector_view")
    assert {path.name for path in directory.iterdir()} == {
        *SELECTOR_FILES, "manifest.json", "checksums.sha256",
    }
    assert all(
        set(json.loads(line)) == {
            "event_id", "service_id", "provider_id", "version_id", "event_time", "query",
        }
        for line in (directory / "query_only_events.jsonl").read_text().splitlines()
    )
