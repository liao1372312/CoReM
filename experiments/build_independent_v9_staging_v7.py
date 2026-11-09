#!/usr/bin/env python3
"""Repair three v9 hard-negative traces without changing their live-state goals.

Superseded v6 remains immutable: its 24 failing negatives had a final GET
404, so they were not clean state-only hard negatives. This new candidate
uses a stable read endpoint for both sides and needs full fresh gates.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import sha256_file, write_checksums, write_json, write_jsonl  # noqa: E402
from experiments.build_independent_v9_staging import _candidate  # noqa: E402
from experiments.build_independent_v9_staging_v5 import RETRIEVAL_BUDGET, SOURCES, START, VERSIONS, _case as _v6_case  # noqa: E402


def _case(family: str, number: int, positive: bool) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
    service, query, plans, contract = _v6_case(family, number, positive)
    suffix = f"{number:03d}"
    if family == "solr_schema_field_assignment":
        plans[-1] = {"method": "GET", "path": f"/solr/v9_schema_target_{suffix}/schema?wt=json"}
        query = query.replace("inspect the target field.", "inspect both schemas.")
    elif family == "solr_dynamic_field_assignment":
        plans[-1] = {"method": "GET", "path": f"/solr/v9_dyn_target_{suffix}/schema?wt=json"}
        query = query.replace("inspect target.", "inspect both schemas.")
    elif family == "typesense_document_delete_assignment":
        plans[-1] = {"method": "GET", "path": f"/collections/v9_delete_target_{suffix}"}
    return service, query, plans, contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-label-output", type=Path, required=True)
    parser.add_argument("--salt-file", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.private_label_output.exists():
        parser.error("refusing to overwrite v9 staging files")
    salt = args.salt_file.read_bytes()
    if len(salt) != 32:
        raise ValueError("private assignment salt must be 32 bytes")
    events, queries, memories, edges, candidates, oracle, negative = [], [], [], [], [], [], []
    for family_index, family in enumerate(SOURCES):
        sides = [0] * 4 + [1] * 4
        seed = hashlib.sha256(salt + f"v9-v7-assignment|{family}".encode()).digest()
        random.Random(int.from_bytes(seed[:8], "big")).shuffle(sides)
        for offset in range(8):
            number = offset + 1
            event_id = f"v9-staging-v7:{family}:event-{number:03d}"
            service, query, plans, contract = _case(family, number, True)
            event = {
                "event_id": event_id, "incident_id": event_id,
                "service_id": service, "provider_id": service,
                "version_id": VERSIONS[service],
                "event_time": (START + timedelta(minutes=family_index * 8 + offset)).isoformat().replace("+00:00", "Z"),
                "query": query, "task_key": family, "operation_count": len(plans),
                "workflow_contract": contract, "checker_contract_id": "independent-v9-assertions-v1",
                "required_evidence_atoms": [f"role:step{index}" for index in range(1, len(plans) + 1)] + ["role:verify"],
                "required_binding_atoms": [], "observation_tier": "prospective_unscored_staging",
            }
            events.append(event)
            queries.append({key: event[key] for key in (
                "event_id", "service_id", "provider_id", "version_id", "event_time", "query",
            )})
            for side in (0, 1):
                correct = side == sides[offset]
                _service, _query, candidate_plans, _unused_contract = _case(family, number, correct)
                rows, edge, candidate = _candidate(
                    salt, event_id, family, service, side, candidate_plans, SOURCES[family],
                )
                memories.extend(rows)
                edges.append(edge)
                candidates.append(candidate)
                selection = {
                    "run_id": f"v9-staging-v7:{event_id}:{side}",
                    "event_id": event_id, "method": "oracle" if correct else "complete_negative",
                    "candidate_edge_id": edge["edge_id"],
                    "selected_memory_ids": candidate["selected_memory_ids"],
                    "token_cost": 24 * len(candidate["selected_memory_ids"]),
                }
                (oracle if correct else negative).append(selection)
    if Counter(row["service_id"] for row in events) != {"typesense": 64, "solr": 64}:
        raise ValueError("provider balance failed")
    args.output.mkdir(parents=True)
    files = {
        "events.jsonl": events, "query_only_events.jsonl": queries,
        "memories.jsonl": memories, "dependency_hyperedges.jsonl": edges,
        "candidate_selections.jsonl": candidates,
    }
    for name, rows in files.items():
        write_jsonl(args.output / name, rows)
    write_json(args.output / "manifest.json", {
        "status": "prospective_unscored_staging_only",
        "claim_boundary": "Sixteen prospective families after trace repair; no method score or independent method claim yet.",
        "families": list(SOURCES), "events": len(events), "candidate_rows": len(candidates),
        "retrieval_budget": RETRIEVAL_BUDGET,
        "source_uris": SOURCES, "assignment_salt_commitment_sha256": hashlib.sha256(salt).hexdigest(),
        "generator_sha256": sha256_file(Path(__file__)),
        "supersedes": "data/public_independent_v9_staging_v6",
        "repair_scope": ["solr_schema_field_assignment", "solr_dynamic_field_assignment", "typesense_document_delete_assignment"],
    })
    write_checksums(args.output, (*files, "manifest.json"))
    args.private_label_output.mkdir(parents=True, mode=0o700)
    write_jsonl(args.private_label_output / "oracle_selections.jsonl", oracle)
    write_jsonl(args.private_label_output / "complete_negative_selections.jsonl", negative)
    write_json(args.private_label_output / "manifest.json", {
        "status": "private_v9_staging_labels", "rows_per_label": len(oracle),
        "public_manifest_sha256": sha256_file(args.output / "manifest.json"),
    })
    write_checksums(args.private_label_output, (
        "oracle_selections.jsonl", "complete_negative_selections.jsonl", "manifest.json",
    ))
    print(f"wrote {len(events)} prospective unscored v9 staging events after trace repair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
