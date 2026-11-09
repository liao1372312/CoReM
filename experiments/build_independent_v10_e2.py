#!/usr/bin/env python3
"""Build an outcome-hidden 128-incident E2 confirmation set."""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import sha256_file, write_checksums, write_json, write_jsonl  # noqa: E402
from experiments.build_independent_v9_staging import _candidate  # noqa: E402
from experiments.build_independent_v9_staging_v5 import RETRIEVAL_BUDGET, SOURCES, VERSIONS  # noqa: E402
from experiments.build_independent_v9_staging_v7 import _case  # noqa: E402

START = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-label-output", type=Path, required=True)
    parser.add_argument("--salt-file", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.private_label_output.exists():
        parser.error("refusing to overwrite v10 data")
    salt = args.salt_file.read_bytes()
    if len(salt) != 32:
        raise ValueError("private assignment salt must be 32 bytes")

    events: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    memories: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    splits: list[dict[str, str]] = []
    oracle: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    for family_index, family in enumerate(SOURCES):
        sides = [0] * 4 + [1] * 4
        seed = hashlib.sha256(salt + f"v10-e2-assignment|{family}".encode()).digest()
        random.Random(int.from_bytes(seed[:8], "big")).shuffle(sides)
        for offset in range(8):
            number = offset + 9
            event_id = f"v10-e2:{family}:event-{number:03d}"
            service, query, plans, contract = _case(family, number, True)
            event = {
                "event_id": event_id,
                "incident_id": event_id,
                "service_id": service,
                "provider_id": service,
                "version_id": VERSIONS[service],
                "event_time": (
                    START + timedelta(minutes=family_index * 8 + offset)
                ).isoformat().replace("+00:00", "Z"),
                "query": query,
                "task_key": family,
                "operation_count": len(plans),
                "workflow_contract": contract,
                "checker_contract_id": "independent-v9-assertions-v1",
                "required_evidence_atoms": [
                    f"role:step{index}" for index in range(1, len(plans) + 1)
                ]
                + ["role:verify"],
                "required_binding_atoms": [],
                "communication_criteria": [],
                "observation_tier": "prospective_outcome_hidden_confirmation",
            }
            events.append(event)
            queries.append(
                {
                    key: event[key]
                    for key in (
                        "event_id",
                        "service_id",
                        "provider_id",
                        "version_id",
                        "event_time",
                        "query",
                    )
                }
            )
            splits.append({"event_id": event_id, "split": "test"})
            for side in (0, 1):
                correct = side == sides[offset]
                _, _, candidate_plans, _ = _case(family, number, correct)
                rows, edge, candidate = _candidate(
                    salt,
                    event_id,
                    family,
                    service,
                    side,
                    candidate_plans,
                    SOURCES[family],
                )
                memories.extend(rows)
                edges.append(edge)
                candidates.append(candidate)
                selection = {
                    "run_id": f"v10-e2:{event_id}:{side}",
                    "event_id": event_id,
                    "method": "oracle" if correct else "complete_negative",
                    "candidate_edge_id": edge["edge_id"],
                    "selected_memory_ids": candidate["selected_memory_ids"],
                    "token_cost": 24 * len(candidate["selected_memory_ids"]),
                }
                (oracle if correct else negative).append(selection)

    if Counter(row["service_id"] for row in events) != {
        "typesense": 64,
        "solr": 64,
    }:
        raise ValueError("provider balance failed")
    args.output.mkdir(parents=True)
    files = {
        "events.jsonl": events,
        "query_only_events.jsonl": queries,
        "memories.jsonl": memories,
        "dependency_hyperedges.jsonl": edges,
        "candidate_selections.jsonl": candidates,
        "splits.jsonl": splits,
    }
    for name, rows in files.items():
        write_jsonl(args.output / name, rows)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": "corem-independent-v10-e2-v1",
            "status": "prospective_outcome_hidden_confirmation",
            "claim_boundary": (
                "New incidents 9--16 from sixteen previously designed Solr/Typesense "
                "families; this tests untouched-event replication, not unseen-family "
                "or natural-production generalization."
            ),
            "families": list(SOURCES),
            "events": len(events),
            "candidate_rows": len(candidates),
            "retrieval_budget": RETRIEVAL_BUDGET,
            "source_uris": SOURCES,
            "assignment_salt_commitment_sha256": hashlib.sha256(salt).hexdigest(),
            "generator_sha256": sha256_file(Path(__file__)),
            "outcomes_exposed_to_selectors": False,
        },
    )
    write_checksums(args.output, (*files, "manifest.json"))

    args.private_label_output.mkdir(parents=True, mode=0o700)
    write_jsonl(args.private_label_output / "oracle_selections.jsonl", oracle)
    write_jsonl(
        args.private_label_output / "complete_negative_selections.jsonl", negative
    )
    write_json(
        args.private_label_output / "manifest.json",
        {
            "status": "private_v10_e2_labels_unopened_by_selectors",
            "rows_per_label": len(oracle),
            "public_manifest_sha256": sha256_file(args.output / "manifest.json"),
        },
    )
    write_checksums(
        args.private_label_output,
        (
            "oracle_selections.jsonl",
            "complete_negative_selections.jsonl",
            "manifest.json",
        ),
    )
    print(f"wrote {len(events)} outcome-hidden v10 E2 incidents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
