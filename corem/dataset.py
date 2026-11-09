"""Streaming loader for the normalized private CoReM dataset."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from .types import (
    DependencyHyperedge,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    MemoryRecord,
    OutcomeExample,
    Request,
)

_OUTCOME_SUFFIX = re.compile(
    r"\s*Observed endpoint test (?:succeeded|failed)\.\s*$", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    name: str
    schema_version: str
    private: bool
    files: dict[str, dict[str, Any]]


class PrivateCoremDataset:
    """Read generated JSONL lazily and convert rows to CoReM objects."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file() or not (self.root / "PRIVATE").is_file():
            raise FileNotFoundError(f"not a private CoReM dataset: {self.root}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("private") is not True:
            raise ValueError("dataset manifest must explicitly set private=true")

    @property
    def summary(self) -> DatasetSummary:
        return DatasetSummary(
            name=self.manifest["name"],
            schema_version=self.manifest["schema_version"],
            private=True,
            files=self.manifest["files"],
        )

    def validate_integrity(self) -> dict[str, bool]:
        checksum_path = self.root / "checksums.sha256"
        if not checksum_path.is_file():
            raise FileNotFoundError(checksum_path)
        checksums_ok = True
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            expected, filename = line.split("  ", 1)
            digest = hashlib.sha256()
            with (self.root / filename).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            checksums_ok &= digest.hexdigest() == expected

        record_counts_ok = True
        for filename, metadata in self.manifest["files"].items():
            expected = metadata.get("records")
            if expected is None:
                continue
            with (self.root / filename).open(encoding="utf-8") as handle:
                observed = sum(bool(line.strip()) for line in handle)
            record_counts_ok &= observed == expected
        return {
            "checksums_ok": checksums_ok,
            "record_counts_ok": record_counts_ok,
        }

    def iter_rows(self, filename: str) -> Iterator[dict[str, Any]]:
        path = self.root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from error
                if not isinstance(row, dict):
                    raise TypeError(f"expected an object at {path}:{line_number}")
                yield row

    def iter_memory_records(
        self,
        *,
        splits: set[str] | None = None,
        sources: set[str] | None = None,
        outcome_training_only: bool = False,
        sanitize_outcome_text: bool = False,
    ) -> Iterator[MemoryRecord]:
        for row in self.iter_rows("memory_records.jsonl"):
            if splits is not None and row["split"] not in splits:
                continue
            if sources is not None and row["source"] not in sources:
                continue
            if outcome_training_only and not row["eligible_for_outcome_training"]:
                continue
            yield self._memory_from_row(
                row, sanitize_outcome_text=sanitize_outcome_text
            )

    def iter_requests(
        self,
        *,
        splits: set[str] | None = None,
        sources: set[str] | None = None,
    ) -> Iterator[Request]:
        for row in self.iter_rows("retrieval_tasks.jsonl"):
            if splits is not None and row["split"] not in splits:
                continue
            if sources is not None and row["source"] not in sources:
                continue
            yield self._request_from_row(row)

    def iter_outcome_examples(
        self,
        *,
        splits: set[str] | None = None,
        limit: int | None = None,
    ) -> Iterator[OutcomeExample]:
        emitted = 0
        for row in self.iter_rows("set_outcomes.jsonl"):
            if splits is not None and row["split"] not in splits:
                continue
            if row.get("scope") != "singleton_endpoint_only":
                raise ValueError("unexpected non-singleton row in observed outcomes")
            request = Request(
                request_id=row["task_id"],
                text=row["query"],
                required_atoms=frozenset(row["required_atoms"]),
                timestamp=float(row["event_time"] or 0.0),
                active_versions=frozenset(row.get("active_versions", [])),
            )
            yield OutcomeExample(
                request=request,
                selected_ids=frozenset(row["selected_memory_ids"]),
                outcome=int(row["outcome"]),
                actually_executed=bool(row.get("actually_executed", True)),
                independently_verified=bool(row.get("independently_verified", True)),
                eligible_for_training=bool(row.get("eligible_for_training", True)),
                eligible_for_interaction_training=bool(
                    row.get("eligible_for_interaction_training", False)
                ),
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                return

    def get_task(self, task_id: str) -> dict[str, Any]:
        for row in self.iter_rows("retrieval_tasks.jsonl"):
            if row["task_id"] == task_id:
                return row
        raise KeyError(task_id)

    def load_task_graph(
        self,
        task_id: str,
        *,
        min_hyperedge_confidence: float = 0.75,
    ) -> tuple[Request, EvidenceGraph, frozenset[str]]:
        task = self.get_task(task_id)
        memory_ids = frozenset(task["gold_memory_ids"])
        graph = self.build_graph(
            memory_ids=memory_ids,
            task_id=task_id,
            min_hyperedge_confidence=min_hyperedge_confidence,
        )
        return self._request_from_row(task), graph, memory_ids

    def build_graph(
        self,
        *,
        memory_ids: frozenset[str] | None = None,
        splits: set[str] | None = None,
        sources: set[str] | None = None,
        task_id: str | None = None,
        min_hyperedge_confidence: float = 0.75,
        include_duplicates: bool = True,
        include_hyperedges: bool = True,
        max_duplicate_pairs: int = 10_000,
        max_records: int | None = None,
        include_operation_entities: bool = False,
        include_schema_entities: bool = False,
        sanitize_outcome_text: bool = False,
    ) -> EvidenceGraph:
        graph = EvidenceGraph()
        loaded_ids: set[str] = set()
        records_by_operation: dict[str, list[str]] = {}
        for row in self.iter_rows("memory_records.jsonl"):
            if memory_ids is not None and row["memory_id"] not in memory_ids:
                continue
            if splits is not None and row["split"] not in splits:
                continue
            if sources is not None and row["source"] not in sources:
                continue
            record = self._memory_from_row(
                row, sanitize_outcome_text=sanitize_outcome_text
            )
            graph.add_record(record)
            loaded_ids.add(record.record_id)
            for operation_id in record.operation_ids:
                records_by_operation.setdefault(operation_id, []).append(
                    record.record_id
                )
            if max_records is not None and len(loaded_ids) >= max_records:
                break

        if include_operation_entities:
            self._add_operation_evidence(
                graph,
                records_by_operation,
                include_schema_entities=include_schema_entities,
            )

        if include_hyperedges:
            for row in self.iter_rows("dependency_hyperedges.jsonl"):
                if task_id is not None and row["task_id"] != task_id:
                    continue
                if not row.get("admissible_default", False):
                    continue
                if float(row.get("confidence", 0.0)) < min_hyperedge_confidence:
                    continue
                members = frozenset(row["members"])
                if not members.issubset(loaded_ids):
                    continue
                graph.add_hyperedge(
                    DependencyHyperedge(
                        edge_id=row["edge_id"],
                        members=members,
                        evidence_atoms=frozenset(row["evidence_atoms"]),
                        relation_type=row["relation_type"],
                        provenance=row["provenance"],
                        verified=bool(row.get("verified", True)),
                        signature=str(row.get("signature", "")),
                        version_scope=frozenset(row.get("version_scope", [])),
                        valid_from=float(row.get("valid_from") or 0.0),
                        valid_until=(
                            float(row["valid_until"])
                            if row.get("valid_until") is not None
                            else None
                        ),
                    )
                )

        if include_duplicates and len(loaded_ids) > 1:
            pair_count = 0
            for row in self.iter_rows("duplicate_groups.jsonl"):
                members = sorted(loaded_ids.intersection(row["memory_ids"]))
                for left, right in itertools.combinations(members, 2):
                    graph.add_duplicate(left, right)
                    pair_count += 1
                    if pair_count >= max_duplicate_pairs:
                        return graph
        return graph

    def _add_operation_evidence(
        self,
        graph: EvidenceGraph,
        records_by_operation: dict[str, list[str]],
        *,
        include_schema_entities: bool,
    ) -> None:
        needed = set(records_by_operation)
        if not needed:
            return
        found: set[str] = set()
        for row in self.iter_rows("operations.jsonl"):
            operation_id = str(row["operation_id"])
            if operation_id not in needed:
                continue
            found.add(operation_id)
            provenance = json.dumps(
                row.get("provenance", {"source": row.get("source", "unknown")}),
                ensure_ascii=False,
                sort_keys=True,
            )
            operation_node = f"operation-node:{operation_id}"
            operation_text = " ".join(
                str(value)
                for value in (
                    row.get("method", ""),
                    row.get("endpoint", ""),
                    row.get("name", ""),
                    row.get("description", ""),
                    row.get("category", ""),
                )
                if value
            )
            graph.add_entity(
                EvidenceNode(
                    operation_node,
                    "operation",
                    operation_text or operation_id,
                    provenance,
                )
            )
            for record_id in records_by_operation[operation_id]:
                graph.add_edge(
                    EvidenceEdge(
                        record_id,
                        operation_node,
                        "documents_operation",
                        graph.records[record_id].provenance,
                    )
                )

            service_id = str(row.get("service_id") or operation_id)
            service_node = f"service-node:{service_id}"
            if service_node not in graph.entities:
                graph.add_entity(
                    EvidenceNode(
                        service_node,
                        "service",
                        f"service {service_id}",
                        provenance,
                    )
                )
            graph.add_edge(
                EvidenceEdge(
                    operation_node,
                    service_node,
                    "belongs_to_service",
                    provenance,
                )
            )

            if include_schema_entities:
                self._add_schema_entities(graph, row, operation_node, provenance)
        missing = needed.difference(found)
        if missing:
            raise ValueError(
                f"memory records reference {len(missing)} missing operations; "
                f"examples: {sorted(missing)[:3]}"
            )

    @staticmethod
    def _add_schema_entities(
        graph: EvidenceGraph,
        row: dict[str, Any],
        operation_node: str,
        provenance: str,
    ) -> None:
        operation_id = str(row["operation_id"])
        schemas = (
            (
                "input_constraint",
                "requires_input",
                row.get("required_input_atoms", []),
            ),
            ("output_schema", "produces_output", row.get("output_atoms", [])),
        )
        for node_type, relation_type, atoms in schemas:
            for atom in sorted(set(map(str, atoms))):
                digest = hashlib.sha256(atom.encode("utf-8")).hexdigest()[:16]
                node_id = f"schema-node:{operation_id}:{node_type}:{digest}"
                graph.add_entity(EvidenceNode(node_id, node_type, atom, provenance))
                graph.add_edge(
                    EvidenceEdge(
                        operation_node,
                        node_id,
                        relation_type,
                        provenance,
                    )
                )

    @staticmethod
    def _memory_from_row(
        row: dict[str, Any], *, sanitize_outcome_text: bool = False
    ) -> MemoryRecord:
        text = str(row["text"])
        if sanitize_outcome_text:
            text = _OUTCOME_SUFFIX.sub("", text)
        return MemoryRecord(
            record_id=row["memory_id"],
            text=text,
            evidence_atoms=frozenset(row["evidence_atoms"]),
            operation_ids=frozenset(row["operation_ids"]),
            token_cost=int(row["token_cost"]),
            storage_cost=int(row["storage_cost"]),
            timestamp=float(row["event_time"] or 0.0),
            provenance=str(row["provenance"]),
            service_scope=frozenset(row["operation_ids"]),
            version_scope=frozenset(row.get("version_scope", [])),
            valid_from=float(row.get("valid_from") or 0.0),
            valid_until=(
                float(row["valid_until"])
                if row.get("valid_until") is not None
                else None
            ),
            execution_outcome=(
                int(row["observed_outcome"])
                if row.get("observed_outcome") is not None
                else None
            ),
            provenance_verified=bool(
                row.get("provenance_verified", row.get("admissible", True))
            ),
            admissible=bool(row["admissible"]),
        )

    @staticmethod
    def _request_from_row(row: dict[str, Any]) -> Request:
        return Request(
            request_id=row["task_id"],
            text=row["query"],
            required_atoms=frozenset(row["required_atoms"]),
            verified_atoms=frozenset(row.get("verified_atoms", [])),
            timestamp=float(row["event_time"] or 0.0),
            active_versions=frozenset(row.get("active_versions", [])),
        )


class PublicCoremDataset(PrivateCoremDataset):
    """Load a redistributable CoReM dataset with verified set-level labels."""

    SUPPORTED_LABEL_SCOPES: ClassVar[set[str]] = {
        "multi_operation_db_state",
        "multi_operation_state_criteria",
    }

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file() or not (self.root / "PUBLIC").is_file():
            raise FileNotFoundError(f"not a public CoReM dataset: {self.root}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("private") is not False:
            raise ValueError("dataset manifest must explicitly set private=false")
        if self.manifest.get("label_scope") not in self.SUPPORTED_LABEL_SCOPES:
            raise ValueError("unsupported public dataset label scope")

    @property
    def summary(self) -> DatasetSummary:
        return DatasetSummary(
            name=self.manifest["name"],
            schema_version=self.manifest["schema_version"],
            private=False,
            files=self.manifest["files"],
        )

    def iter_outcome_examples(
        self,
        *,
        splits: set[str] | None = None,
        limit: int | None = None,
    ) -> Iterator[OutcomeExample]:
        emitted = 0
        for row in self.iter_rows("set_outcomes.jsonl"):
            if splits is not None and row["split"] not in splits:
                continue
            if row.get("scope") != self.manifest.get("label_scope"):
                raise ValueError("unexpected outcome scope in public dataset")
            if len(row["selected_memory_ids"]) < 1:
                raise ValueError("public set outcomes must select at least one memory")
            request = Request(
                request_id=row["task_id"],
                text=row["query"],
                required_atoms=frozenset(row["required_atoms"]),
                verified_atoms=frozenset(row.get("verified_atoms", [])),
                timestamp=float(row["event_time"] or 0.0),
                active_versions=frozenset(row.get("active_versions", [])),
            )
            yield OutcomeExample(
                request=request,
                selected_ids=frozenset(row["selected_memory_ids"]),
                outcome=int(row["outcome"]),
                actually_executed=bool(row.get("actually_executed", False)),
                independently_verified=bool(
                    row.get("independently_verified", False)
                ),
                eligible_for_training=bool(row.get("eligible_for_training", False)),
                eligible_for_interaction_training=bool(
                    row.get("eligible_for_interaction_training", False)
                ),
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a private or public CoReM dataset"
    )
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    dataset: PrivateCoremDataset | PublicCoremDataset
    if (args.root / "PUBLIC").is_file():
        dataset = PublicCoremDataset(args.root)
    else:
        dataset = PrivateCoremDataset(args.root)
    summary = dataset.summary
    integrity = dataset.validate_integrity()
    first = next(dataset.iter_outcome_examples(limit=1), None)
    print(
        json.dumps(
            {
                "name": summary.name,
                "schema_version": summary.schema_version,
                "private": summary.private,
                "files": summary.files,
                "first_observed_example": first.request.request_id if first else None,
                "integrity": integrity,
                "status": "valid" if all(integrity.values()) else "invalid",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
