#!/usr/bin/env python3
"""Generate a prospective sixteen-family Typesense/Solr task-validity set.

This is not frozen or scored. Every family needs a clean live oracle/negative
gate and a later preselection freeze before any method result can be claimed.
"""

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
from experiments.build_independent_v9_staging_v2 import VERSIONS, _contract, _core_create  # noqa: E402
from experiments.build_independent_v9_staging_v4 import SOURCES as BASE_SOURCES, _case as _base_case  # noqa: E402

SOURCES = {
    **BASE_SOURCES,
    "typesense_schema_field_assignment": "https://typesense.org/docs/30.2/api/collections.html",
    "typesense_document_patch_assignment": "https://typesense.org/docs/30.2/api/documents.html",
    "typesense_document_delete_assignment": "https://typesense.org/docs/30.2/api/documents.html",
    "solr_atomic_update_assignment": "https://solr.apache.org/guide/solr/9_10/indexing-guide/partial-document-updates.html",
    "solr_document_delete_assignment": "https://solr.apache.org/guide/solr/9_10/indexing-guide/indexing-with-update-handlers.html",
    "solr_request_handler_assignment": "https://solr.apache.org/guide/solr/9_10/configuration-guide/config-api.html",
}
START = datetime(2026, 10, 18, 16, 0, tzinfo=UTC)
RETRIEVAL_BUDGET = 256


def _ts_create(name: str, fields: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"method": "POST", "path": "/collections", "json": {
        "name": name, "fields": fields or [{"name": "title", "type": "string"}],
    }}


def _ts_document(
    collection: str, doc_id: str, title: str = "item", stage: str | None = None
) -> dict[str, Any]:
    body = {"id": doc_id, "title": title}
    if stage is not None:
        body["stage"] = stage
    return {"method": "POST", "path": f"/collections/{collection}/documents", "json": body}


def _solr_select(core: str, doc_id: str) -> str:
    return f"/solr/{core}/select?q=id%3A{doc_id}&wt=json"


def _case(family: str, number: int, positive: bool) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
    if family in BASE_SOURCES:
        return _base_case(family, number, positive)
    suffix = f"{number:03d}"
    if family == "typesense_schema_field_assignment":
        target, decoy = f"v9_fields_target_{suffix}", f"v9_fields_decoy_{suffix}"
        amber, violet = f"amber_{suffix}", f"violet_{suffix}"
        left, right = (amber, violet) if positive else (violet, amber)
        query = f"Create collections {target} and {decoy}; add string field {amber} to {target} and {violet} to {decoy}; inspect both schemas."
        plans = [_ts_create(name) for name in (target, decoy)] + [
            {"method": "PATCH", "path": f"/collections/{name}", "json": {"fields": [{"name": field, "type": "string"}]}}
            for name, field in ((target, left), (decoy, right))
        ] + [{"method": "GET", "path": f"/collections/{target}"}]
        checks = [
            {"path": f"/collections/{name}", "status": 200, "assertions": [
                {"path": ["name"], "equals": name},
                {"path": ["fields", 1, "name"], "equals": field},
            ]}
            for name, field in ((target, amber), (decoy, violet))
        ]
        return "typesense", query, plans, _contract([201, 201, 200, 200, 200], checks)
    if family == "typesense_document_patch_assignment":
        collection = f"v9_patch_{suffix}"
        amber, violet = f"amber{suffix}", f"violet{suffix}"
        left, right = (amber, violet) if positive else (violet, amber)
        query = f"Create collection {collection} with documents 1 and 2; patch stage of document 1 to {amber} and document 2 to {violet}; inspect both."
        plans = [_ts_create(collection, [{"name": "title", "type": "string"}, {"name": "stage", "type": "string"}])]
        plans += [_ts_document(collection, doc_id, stage="draft") for doc_id in ("1", "2")]
        plans += [
            {"method": "PATCH", "path": f"/collections/{collection}/documents/{doc_id}", "json": {"stage": value}}
            for doc_id, value in (("1", left), ("2", right))
        ]
        plans.append({"method": "GET", "path": f"/collections/{collection}/documents/1"})
        checks = [
            {"path": f"/collections/{collection}/documents/{doc_id}", "status": 200, "assertions": [
                {"path": ["id"], "equals": doc_id},
                {"path": ["stage"], "equals": value},
            ]}
            for doc_id, value in (("1", amber), ("2", violet))
        ]
        return "typesense", query, plans, _contract([201, 201, 201, 200, 200, 200], checks)
    if family == "typesense_document_delete_assignment":
        target, decoy = f"v9_delete_target_{suffix}", f"v9_delete_decoy_{suffix}"
        left_id, right_id = ("1", "2") if positive else ("2", "1")
        query = f"Create collections {target} and {decoy}, each with documents 1 and 2; delete document 1 only from {target} and document 2 only from {decoy}; verify survivors."
        plans = [_ts_create(name) for name in (target, decoy)]
        plans += [_ts_document(name, doc_id) for name in (target, decoy) for doc_id in ("1", "2")]
        plans += [
            {"method": "DELETE", "path": f"/collections/{name}/documents/{doc_id}"}
            for name, doc_id in ((target, left_id), (decoy, right_id))
        ]
        plans.append({"method": "GET", "path": f"/collections/{target}/documents/2"})
        checks = [
            {"path": f"/collections/{name}/documents/{doc_id}", "status": status,
             "assertions": ([{"path": ["id"], "absent": True}] if status == 404 else [{"path": ["id"], "equals": doc_id}])}
            for name, doc_id, status in ((target, "1", 404), (target, "2", 200), (decoy, "1", 200), (decoy, "2", 404))
        ]
        return "typesense", query, plans, _contract([201] * 6 + [200, 200, 200], checks)
    if family == "solr_atomic_update_assignment":
        core = f"v9_atomic_{suffix}"
        amber, violet = f"amber{suffix}", f"violet{suffix}"
        left, right = (amber, violet) if positive else (violet, amber)
        query = f"Create Solr core {core} with documents 1 and 2; atomically set stage_s of document 1 to {amber} and document 2 to {violet}; inspect both."
        plans = [_core_create(core), {"method": "POST", "path": f"/solr/{core}/update?commit=true", "json": [
            {"id": "1", "stage_s": "draft"}, {"id": "2", "stage_s": "draft"},
        ]}]
        plans += [
            {"method": "POST", "path": f"/solr/{core}/update?commit=true", "json": [
                {"id": doc_id, "stage_s": {"set": value}},
            ]}
            for doc_id, value in (("1", left), ("2", right))
        ]
        plans.append({"method": "GET", "path": _solr_select(core, "1")})
        checks = [
            {"path": _solr_select(core, doc_id), "status": 200, "assertions": [
                {"path": ["response", "numFound"], "equals": 1},
                {"path": ["response", "docs", 0, "id"], "equals": doc_id},
                {"path": ["response", "docs", 0, "stage_s"], "equals": value},
            ]}
            for doc_id, value in (("1", amber), ("2", violet))
        ]
        return "solr", query, plans, _contract([200] * 5, checks)
    if family == "solr_document_delete_assignment":
        target, decoy = f"v9_sdel_target_{suffix}", f"v9_sdel_decoy_{suffix}"
        left_id, right_id = ("1", "2") if positive else ("2", "1")
        query = f"Create Solr cores {target} and {decoy}, each with documents 1 and 2; delete 1 from {target} and 2 from {decoy}, retaining the other documents."
        plans = [_core_create(name) for name in (target, decoy)]
        plans += [
            {"method": "POST", "path": f"/solr/{name}/update?commit=true", "json": [
                {"id": "1", "title": "first"}, {"id": "2", "title": "second"},
            ]}
            for name in (target, decoy)
        ]
        plans += [
            {"method": "POST", "path": f"/solr/{name}/update?commit=true", "json": {"delete": {"id": doc_id}}}
            for name, doc_id in ((target, left_id), (decoy, right_id))
        ]
        plans.append({"method": "GET", "path": _solr_select(target, "2")})
        checks = [
            {"path": _solr_select(name, doc_id), "status": 200, "assertions": [
                {"path": ["response", "numFound"], "equals": expected},
            ]}
            for name, doc_id, expected in ((target, "1", 0), (target, "2", 1), (decoy, "1", 1), (decoy, "2", 0))
        ]
        return "solr", query, plans, _contract([200] * 7, checks)
    if family == "solr_request_handler_assignment":
        target, decoy = f"v9_handler_target_{suffix}", f"v9_handler_decoy_{suffix}"
        small, large = 4 + number, 20 + number
        left, right = (small, large) if positive else (large, small)
        query = f"Create Solr cores {target} and {decoy}; add each core's own /corem_<core> search handler with default rows {small} on {target} and {large} on {decoy}; inspect both overlays."
        plans = [_core_create(name) for name in (target, decoy)] + [
            {"method": "POST", "path": f"/solr/{name}/config", "json": {"add-requesthandler": {
                "name": f"/corem_{name}", "class": "solr.SearchHandler", "defaults": {"rows": value},
            }}}
            for name, value in ((target, left), (decoy, right))
        ] + [{"method": "GET", "path": f"/solr/{target}/config/overlay?wt=json"}]
        checks = [
            {"path": f"/solr/{name}/config/overlay?wt=json", "status": 200, "assertions": [
                {"path": ["overlay", "requestHandler", f"/corem_{name}", "defaults", "rows"], "equals": value},
            ]}
            for name, value in ((target, small), (decoy, large))
        ]
        return "solr", query, plans, _contract([200] * 5, checks)
    raise ValueError(f"unsupported v9 staging family: {family}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-label-output", type=Path, required=True)
    parser.add_argument("--salt-file", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.private_label_output.exists():
        parser.error("refusing to overwrite v9 staging")
    salt = args.salt_file.read_bytes()
    if len(salt) != 32:
        raise ValueError("private assignment salt must be 32 bytes")
    events, queries, memories, edges, candidates, oracle, negative = [], [], [], [], [], [], []
    for family_index, family in enumerate(SOURCES):
        sides = [0] * 4 + [1] * 4
        seed = hashlib.sha256(salt + f"v9-v5-assignment|{family}".encode()).digest()
        random.Random(int.from_bytes(seed[:8], "big")).shuffle(sides)
        for offset in range(8):
            number = offset + 1
            event_id = f"v9-staging-v5:{family}:event-{number:03d}"
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
            queries.append({key: event[key] for key in ("event_id", "service_id", "provider_id", "version_id", "event_time", "query")})
            for side in (0, 1):
                correct = side == sides[offset]
                _service, _query, candidate_plans, _unused_contract = _case(family, number, correct)
                rows, edge, candidate = _candidate(
                    salt, event_id, family, service, side, candidate_plans, SOURCES[family]
                )
                memories.extend(rows)
                edges.append(edge)
                candidates.append(candidate)
                selection = {
                    "run_id": f"v9-staging-v5:{event_id}:{side}",
                    "event_id": event_id, "method": "oracle" if correct else "complete_negative",
                    "candidate_edge_id": edge["edge_id"],
                    "selected_memory_ids": candidate["selected_memory_ids"],
                    "token_cost": 24 * len(candidate["selected_memory_ids"]),
                }
                (oracle if correct else negative).append(selection)
    if Counter(row["service_id"] for row in events) != {"typesense": 64, "solr": 64}:
        raise ValueError("provider balance failed")
    args.output.mkdir(parents=True)
    files = {"events.jsonl": events, "query_only_events.jsonl": queries,
             "memories.jsonl": memories, "dependency_hyperedges.jsonl": edges,
             "candidate_selections.jsonl": candidates}
    for name, rows in files.items():
        write_jsonl(args.output / name, rows)
    write_json(args.output / "manifest.json", {
        "status": "prospective_unscored_staging_only",
        "claim_boundary": "Sixteen prospective families; no complete physical gates, method scores, independent method claim, or submission claim yet.",
        "families": list(SOURCES), "events": len(events), "candidate_rows": len(candidates),
        "retrieval_budget": RETRIEVAL_BUDGET,
        "source_uris": SOURCES, "assignment_salt_commitment_sha256": hashlib.sha256(salt).hexdigest(),
        "generator_sha256": sha256_file(Path(__file__)),
    })
    write_checksums(args.output, (*files, "manifest.json"))
    args.private_label_output.mkdir(parents=True, mode=0o700)
    write_jsonl(args.private_label_output / "oracle_selections.jsonl", oracle)
    write_jsonl(args.private_label_output / "complete_negative_selections.jsonl", negative)
    write_json(args.private_label_output / "manifest.json", {
        "status": "private_v9_staging_labels", "rows_per_label": len(oracle),
        "public_manifest_sha256": sha256_file(args.output / "manifest.json"),
    })
    write_checksums(args.private_label_output, ("oracle_selections.jsonl", "complete_negative_selections.jsonl", "manifest.json"))
    print(f"wrote {len(events)} prospective unscored v9 staging events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
