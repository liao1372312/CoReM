#!/usr/bin/env python3
"""Build the private, normalized CoReM research dataset.

Inputs are read-only.  The builder deliberately separates observed singleton
outcomes from non-executed composition labels so weak supervision cannot be
mistaken for verified execution evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO
from urllib.parse import urlsplit, urlunsplit


SCHEMA_VERSION = "1.0.0"
TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+|[\u3400-\u9fff]")
ATOM_RE = re.compile(r"[^a-z0-9_\u3400-\u9fff]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/lxc/next_work/A/WWW/dataset"),
        help="source dataset root (read-only)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/private_corem"),
        help="new output directory; must not already exist",
    )
    return parser.parse_args()


def compact_text(*parts: Any) -> str:
    return " ".join(str(part).strip() for part in parts if part is not None and str(part).strip())


def estimate_tokens(text: str) -> int:
    return max(1, len(TOKEN_RE.findall(text)))


def normalize_atom(value: Any) -> str:
    text = ATOM_RE.sub("_", str(value).casefold()).strip("_")
    return text[:160]


def stable_hash(value: str, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def stable_split(key: str) -> str:
    bucket = int(stable_hash(key, 8), 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def normalize_url(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return value.strip().casefold().rstrip("/")
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    return urlunsplit((parsed.scheme.casefold(), parsed.netloc.casefold(), path, "", ""))


def json_dump_line(handle: TextIO, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(path: Path, source_root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(source_root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def hsc_split(index: int) -> str:
    return "train" if index < 8000 else "val" if index < 9000 else "test"


def output_fields_from_schema(summary: dict[str, Any]) -> list[str]:
    text = summary.get("text")
    if not text:
        return []
    try:
        schema = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return []
    properties = schema.get("properties") if isinstance(schema, dict) else None
    return sorted(normalize_atom(key) for key in properties or {} if normalize_atom(key))


def parameter_names(parameters: Iterable[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            normalized
            for parameter in parameters or ()
            if (normalized := normalize_atom(parameter.get("name", "")))
        }
    )


def operation_atom(operation_id: str) -> str:
    return f"operation:{operation_id}"


def slot_atom(task_id: str, step_id: Any) -> str:
    return f"slot:{task_id}:{step_id}"


def binding_atom(name: str) -> str:
    return f"binding:{name}"


def task_dependency_edges(record: dict[str, Any]) -> list[tuple[int, int]]:
    """Normalize heterogeneous five-domain dependency annotations."""
    valid_ids = {int(task["id"]) for task in record.get("TaskList", []) if str(task.get("id", "")).isdigit()}
    edges: set[tuple[int, int]] = set()

    def numbers(value: Any) -> list[int]:
        if isinstance(value, bool) or value is None:
            return []
        if isinstance(value, int):
            return [value]
        if isinstance(value, str):
            return [int(item) for item in re.findall(r"\d+", value)]
        if isinstance(value, list):
            result: list[int] = []
            for item in value:
                result.extend(numbers(item))
            return result
        return []

    for task in record.get("TaskList", []):
        child = int(task["id"])
        for key in ("depends_on", "dependencies"):
            for parent in numbers(task.get(key)):
                edges.add((parent, child))
        value = task.get("dependency")
        if isinstance(value, dict):
            source = numbers(value.get("from"))
            target = numbers(value.get("to"))
            if source and target:
                edges.add((source[0], target[0]))
            else:
                for parent in numbers(value.get("depends_on")):
                    edges.add((parent, child))
        else:
            for parent in numbers(value):
                edges.add((parent, child))

    top = record.get("dependencies")
    if isinstance(top, dict):
        for child_value, parents in top.items():
            child_numbers = numbers(child_value)
            if child_numbers:
                for parent in numbers(parents):
                    edges.add((parent, child_numbers[0]))
    elif isinstance(top, list):
        for sentence in top:
            values = numbers(sentence)
            if len(values) >= 2:
                child = values[0]
                for parent in values[1:]:
                    edges.add((parent, child))
    top_single = record.get("dependency")
    if isinstance(top_single, dict):
        child = numbers(top_single.get("id"))
        if child:
            for parent in numbers(top_single.get("depends_on")):
                edges.add((parent, child[0]))
    return sorted(
        (parent, child)
        for parent, child in edges
        if parent in valid_ids and child in valid_ids and parent != child
    )


class DatasetBuilder:
    def __init__(self, source: Path, output: Path) -> None:
        self.source = source.resolve()
        self.output = output.resolve()
        self.temp = Path(tempfile.mkdtemp(prefix=".private_corem_", dir=self.output.parent))
        self.operations: dict[str, dict[str, Any]] = {}
        self.operation_interface_index: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.memory_ids: set[str] = set()
        self.memory_by_operation: dict[str, list[str]] = defaultdict(list)
        self.counts: Counter[str] = Counter()
        self.task_splits: Counter[str] = Counter()
        self.task_sources: Counter[str] = Counter()
        self.memory_sources: Counter[str] = Counter()
        self.hyperedge_grounding: Counter[str] = Counter()
        self.handles: dict[str, TextIO] = {}

    def __enter__(self) -> "DatasetBuilder":
        if self.output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {self.output}")
        for filename in (
            "memory_records.jsonl",
            "retrieval_tasks.jsonl",
            "dependency_hyperedges.jsonl",
            "set_outcomes.jsonl",
            "weak_set_labels.jsonl",
            "duplicate_groups.jsonl",
        ):
            self.handles[filename] = (self.temp / filename).open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles.values():
            handle.close()
        if exc_type is not None:
            shutil.rmtree(self.temp)

    def write(self, filename: str, row: dict[str, Any]) -> None:
        json_dump_line(self.handles[filename], row)
        self.counts[filename] += 1

    def add_operation(self, row: dict[str, Any]) -> None:
        operation_id = row["operation_id"]
        if operation_id not in self.operations:
            self.operations[operation_id] = row
            key = (row.get("method", "").upper(), normalize_url(row.get("endpoint", "")))
            if key[1]:
                self.operation_interface_index[key].append(operation_id)

    def add_memory(self, row: dict[str, Any]) -> None:
        memory_id = row["memory_id"]
        if memory_id in self.memory_ids:
            raise ValueError(f"duplicate memory id: {memory_id}")
        self.memory_ids.add(memory_id)
        for operation_id in row["operation_ids"]:
            if operation_id not in self.operations:
                raise KeyError(f"memory references unknown operation: {operation_id}")
            self.memory_by_operation[operation_id].append(memory_id)
        self.memory_sources[row["source"]] += 1
        self.write("memory_records.jsonl", row)

    def add_task(self, row: dict[str, Any]) -> None:
        missing = set(row.get("gold_memory_ids", ())).difference(self.memory_ids)
        if missing:
            raise KeyError(f"task references unknown memories: {sorted(missing)[:3]}")
        missing_ops = set(row.get("gold_operation_ids", ())).difference(self.operations)
        if missing_ops:
            raise KeyError(f"task references unknown operations: {sorted(missing_ops)[:3]}")
        self.task_splits[row["split"]] += 1
        self.task_sources[row["source"]] += 1
        self.write("retrieval_tasks.jsonl", row)

    def process_rapidapi(self) -> None:
        processed = self.source / "processed"
        feedback_by_endpoint: dict[str, dict[str, Any]] = {}
        feedback_order: dict[str, int] = {}
        with (processed / "feedback_events.jsonl").open(encoding="utf-8") as handle:
            for sequence, line in enumerate(handle):
                event = json.loads(line)
                feedback_by_endpoint[event["endpoint_id"]] = event
                feedback_order[event["endpoint_id"]] = sequence

        endpoints: dict[str, dict[str, Any]] = {}
        with (processed / "endpoints.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                endpoint = json.loads(line)
                endpoint_id = endpoint["endpoint_id"]
                operation_id = f"rapidapi:{endpoint_id}"
                feedback = feedback_by_endpoint.get(endpoint_id)
                outputs = set(output_fields_from_schema(endpoint.get("schema_summary", {})))
                if feedback:
                    outputs.update(
                        normalize_atom(key)
                        for key in feedback.get("response_summary", {}).get("top_keys", [])
                        if normalize_atom(key)
                    )
                input_atoms = parameter_names(
                    endpoint.get("required_parameters", []) + endpoint.get("optional_parameters", [])
                )
                text = endpoint.get("interface_text") or compact_text(
                    endpoint.get("endpoint_name"), endpoint.get("endpoint_description")
                )
                self.add_operation(
                    {
                        "operation_id": operation_id,
                        "service_id": endpoint["service_id"],
                        "source": "rapidapi_processed",
                        "name": endpoint["endpoint_name"],
                        "description": endpoint["endpoint_description"],
                        "category": endpoint["category"],
                        "method": endpoint["method"].upper(),
                        "endpoint": endpoint["url"],
                        "input_atoms": input_atoms,
                        "required_input_atoms": parameter_names(endpoint.get("required_parameters", [])),
                        "output_atoms": sorted(outputs),
                        "split": endpoint["split"],
                        "qos": {},
                        "provenance": {
                            "source_path": "processed/endpoints.jsonl",
                            "schema_available": bool(endpoint.get("schema_summary", {}).get("available")),
                        },
                    }
                )
                outcome = feedback.get("success") if feedback else None
                evidence_atoms = [operation_atom(operation_id), *(f"output:{name}" for name in sorted(outputs))]
                memory_id = f"mem:rapidapi:{endpoint_id}"
                memory_text = compact_text(
                    text,
                    "Observed endpoint test succeeded." if outcome is True else
                    "Observed endpoint test failed." if outcome is False else
                    "No observed endpoint test is available.",
                )
                self.add_memory(
                    {
                        "memory_id": memory_id,
                        "source": "rapidapi_processed",
                        "text": memory_text,
                        "operation_ids": [operation_id],
                        "evidence_atoms": evidence_atoms,
                        "token_cost": estimate_tokens(memory_text),
                        "storage_cost": estimate_tokens(memory_text) + 2 * len(evidence_atoms),
                        "event_time": None,
                        "sequence_index": feedback_order.get(endpoint_id),
                        "time_quality": "missing",
                        "provenance": feedback["event_id"] if feedback else "endpoint_spec_only",
                        "observed_outcome": outcome,
                        "split": endpoint["split"],
                        "admissible": True,
                        "live_validated": False,
                        "eligible_for_outcome_training": outcome is not None,
                    }
                )
                endpoints[endpoint_id] = endpoint

        with (processed / "selection_tasks.jsonl").open(encoding="utf-8") as handle:
            for sequence, line in enumerate(handle):
                task = json.loads(line)
                positive = task["positive_endpoint_id"]
                operation_id = f"rapidapi:{positive}"
                memory_id = f"mem:rapidapi:{positive}"
                required_atoms = [operation_atom(operation_id)]
                row = {
                    "task_id": f"rapidapi:{task['task_id']}",
                    "source": "rapidapi_selection",
                    "query": task["query"],
                    "required_atoms": required_atoms,
                    "verified_atoms": [],
                    "candidate_operation_ids": [f"rapidapi:{key}" for key in task["candidate_endpoint_ids"]],
                    "gold_operation_ids": [operation_id],
                    "gold_memory_ids": [memory_id],
                    "dependency_edge_ids": [],
                    "split": task["split"],
                    "event_time": None,
                    "sequence_index": sequence,
                    "time_quality": "deterministic_file_order_only",
                    "supervision": "weak_endpoint_selection",
                    "is_executed_composition": False,
                }
                self.add_task(row)
                feedback = feedback_by_endpoint.get(positive)
                if feedback:
                    self.write(
                        "set_outcomes.jsonl",
                        {
                            "example_id": f"outcome:{feedback['event_id']}",
                            "task_id": row["task_id"],
                            "source": "rapidapi_observed_endpoint_test",
                            "query": row["query"],
                            "required_atoms": required_atoms,
                            "selected_memory_ids": [memory_id],
                            "outcome": int(bool(feedback["success"])),
                            "split": task["split"],
                            "event_time": None,
                            "sequence_index": feedback_order[positive],
                            "time_quality": "missing",
                            "scope": "singleton_endpoint_only",
                            "eligible_for_interaction_training": False,
                        },
                    )

    def process_emp(self) -> None:
        path = self.source / "emp_data2_synthetic_compositions.jsonl"
        records = [json.loads(line) for line in path.open(encoding="utf-8")]
        for record in records:
            for api in record["apis"]:
                operation_id = f"emp:{api['record_id']}"
                text = compact_text(api.get("api_name"), api.get("description"))
                self.add_operation(
                    {
                        "operation_id": operation_id,
                        "service_id": f"emp:{api.get('service_en') or api.get('domain')}",
                        "source": "emp_catalog",
                        "name": api.get("api_name", ""),
                        "description": api.get("description", ""),
                        "category": api.get("category") or api.get("domain", ""),
                        "method": str(api.get("method", "")).upper(),
                        "endpoint": api.get("endpoint", ""),
                        "input_atoms": parameter_names(api.get("parameters", [])),
                        "required_input_atoms": parameter_names(api.get("required_parameters", [])),
                        "output_atoms": parameter_names(api.get("outputs", [])),
                        "split": "private_catalog",
                        "qos": {},
                        "provenance": {
                            "source_path": "EMP/data2.xlsx",
                            "executable_claim": bool(api.get("executable")),
                        },
                    }
                )

        for sequence, record in enumerate(records):
            task_id = f"emp:{record['id']}"
            split = stable_split(task_id)
            api_by_step = {int(api["step_id"]): api for api in record["apis"]}
            task_by_step = {int(task["id"]): task for task in record["TaskList"]}
            dependency_edges = sorted(
                (int(parent), int(task["id"]))
                for task in record["TaskList"]
                for parent in task.get("depends_on", [])
                if int(parent) in task_by_step
            )
            binding_by_source: dict[int, set[str]] = defaultdict(set)
            edge_rows: list[dict[str, Any]] = []
            required_atoms = [slot_atom(task_id, step_id) for step_id in sorted(task_by_step)]
            for parent, child in dependency_edges:
                outputs = set(parameter_names(api_by_step[parent].get("outputs", [])))
                inputs = set(parameter_names(api_by_step[child].get("parameters", [])))
                exact = sorted(outputs.intersection(inputs))
                for name in exact:
                    binding_by_source[parent].add(name)
                    required_atoms.append(binding_atom(name))
                grounding = "output_input_exact" if exact else "declared_synthetic_dependency"
                edge_id = f"dep:{task_id}:{parent}:{child}"
                edge_rows.append(
                    {
                        "edge_id": edge_id,
                        "task_id": task_id,
                        "source": "emp_synthetic",
                        "members": [f"mem:{task_id}:step:{parent}", f"mem:{task_id}:step:{child}"],
                        "evidence_atoms": [*(binding_atom(name) for name in exact), slot_atom(task_id, child)],
                        "relation_type": "binding_dependency" if exact else "declared_dependency",
                        "grounding": grounding,
                        "confidence": 1.0 if exact else 0.5,
                        "admissible_default": bool(exact),
                        "provenance": "TaskList.depends_on + API field intersection" if exact else "TaskList.depends_on",
                        "split": split,
                    }
                )

            gold_memories: list[str] = []
            gold_operations: list[str] = []
            for step_id in sorted(task_by_step):
                api = api_by_step[step_id]
                task = task_by_step[step_id]
                operation_id = f"emp:{api['record_id']}"
                memory_id = f"mem:{task_id}:step:{step_id}"
                evidence = [
                    operation_atom(operation_id),
                    slot_atom(task_id, step_id),
                    *(f"output:{name}" for name in parameter_names(api.get("outputs", []))),
                    *(binding_atom(name) for name in sorted(binding_by_source[step_id])),
                ]
                text = compact_text(task.get("description"), api.get("api_name"), api.get("description"))
                self.add_memory(
                    {
                        "memory_id": memory_id,
                        "source": "emp_synthetic",
                        "text": text,
                        "operation_ids": [operation_id],
                        "evidence_atoms": sorted(set(evidence)),
                        "token_cost": estimate_tokens(text),
                        "storage_cost": estimate_tokens(text) + 2 * len(set(evidence)),
                        "event_time": None,
                        "sequence_index": sequence,
                        "time_quality": "synthetic_task_order",
                        "provenance": task_id,
                        "observed_outcome": None,
                        "split": split,
                        "admissible": True,
                        "live_validated": False,
                        "eligible_for_outcome_training": False,
                    }
                )
                gold_memories.append(memory_id)
                gold_operations.append(operation_id)
            for edge in edge_rows:
                self.hyperedge_grounding[edge["grounding"]] += 1
                self.write("dependency_hyperedges.jsonl", edge)
            row = {
                "task_id": task_id,
                "source": "emp_synthetic",
                "query": record["task"],
                "required_atoms": sorted(set(required_atoms)),
                "verified_atoms": [],
                "candidate_operation_ids": gold_operations,
                "gold_operation_ids": gold_operations,
                "gold_memory_ids": gold_memories,
                "dependency_edge_ids": [edge["edge_id"] for edge in edge_rows],
                "split": split,
                "event_time": None,
                "sequence_index": sequence,
                "time_quality": "synthetic_task_order",
                "supervision": "synthetic_composition_gold_proxy",
                "is_executed_composition": False,
                "metadata": {
                    "composition_type": record["metadata"].get("composition_type"),
                    "composition_relation": record["metadata"].get("composition_relation"),
                    "depth": record["metadata"].get("depth"),
                },
            }
            self.add_task(row)
            self.write(
                "weak_set_labels.jsonl",
                {
                    "label_id": f"weak:{task_id}",
                    "task_id": task_id,
                    "source": "emp_synthetic",
                    "selected_memory_ids": gold_memories,
                    "selected_operation_ids": gold_operations,
                    "proxy_label": 1,
                    "proxy_type": "synthetic_selected_composition",
                    "is_executed": False,
                    "split": split,
                    "eligible_for_bce_outcome_training": False,
                },
            )

    def _resolve_candidate_operation(self, candidate: dict[str, Any]) -> str:
        method = str(candidate.get("method", "")).upper()
        endpoint = str(candidate.get("endpoint", ""))
        matches = self.operation_interface_index.get((method, normalize_url(endpoint)), [])
        rapid_matches = [key for key in matches if key.startswith("rapidapi:")]
        if len(rapid_matches) == 1:
            return rapid_matches[0]
        identity = f"{method}|{normalize_url(endpoint)}|{candidate.get('api_name', '')}"
        operation_id = f"five:{stable_hash(identity, 16)}"
        if operation_id not in self.operations:
            description = candidate.get("description", "")
            self.add_operation(
                {
                    "operation_id": operation_id,
                    "service_id": f"five:{stable_hash(str(candidate.get('api_name', '')), 16)}",
                    "source": "five_domain_unmatched_candidate",
                    "name": candidate.get("api_name", ""),
                    "description": description,
                    "category": candidate.get("domain", ""),
                    "method": method,
                    "endpoint": endpoint,
                    "input_atoms": parameter_names(candidate.get("required_parameters", [])),
                    "required_input_atoms": parameter_names(candidate.get("required_parameters", [])),
                    "output_atoms": [],
                    "split": "private_catalog",
                    "qos": {},
                    "provenance": {"source_path": "five_domain_1400_gold.jsonl"},
                }
            )
        return operation_id

    def process_five_domain(self) -> None:
        path = self.source / "five_domain_1400_gold.jsonl"
        for sequence, line in enumerate(path.open(encoding="utf-8")):
            record = json.loads(line)
            task_id = f"five:{record['id']}"
            split = stable_split(task_id)
            task_by_step = {int(task["id"]): task for task in record["TaskList"]}
            candidates_by_step: dict[int, list[str]] = {}
            selected_by_step: dict[int, tuple[str, dict[str, Any], dict[str, Any]]] = {}
            for task, gold in zip(record["TaskList"], record["gold_api_candidates"]):
                step_id = int(task["id"])
                candidates_by_step[step_id] = [
                    self._resolve_candidate_operation(candidate)
                    for candidate in task.get("top_candidates", [])
                ]
                selection = gold.get("selection", {})
                candidate = selection.get("candidate", {})
                operation_id = self._resolve_candidate_operation(candidate)
                selected_by_step[step_id] = (operation_id, candidate, selection)

            gold_memories: list[str] = []
            gold_operations: list[str] = []
            for step_id in sorted(task_by_step):
                operation_id, candidate, selection = selected_by_step[step_id]
                memory_id = f"mem:{task_id}:step:{step_id}"
                text = compact_text(
                    task_by_step[step_id].get("description"),
                    candidate.get("api_name"),
                    candidate.get("description"),
                )
                evidence = [operation_atom(operation_id), slot_atom(task_id, step_id)]
                self.add_memory(
                    {
                        "memory_id": memory_id,
                        "source": "five_domain_weak_gold",
                        "text": text,
                        "operation_ids": [operation_id],
                        "evidence_atoms": evidence,
                        "token_cost": estimate_tokens(text),
                        "storage_cost": estimate_tokens(text) + 2 * len(evidence),
                        "event_time": None,
                        "sequence_index": sequence,
                        "time_quality": "source_file_order_only",
                        "provenance": record["metadata"].get("source_file", task_id),
                        "observed_outcome": None,
                        "split": split,
                        "admissible": True,
                        "live_validated": False,
                        "eligible_for_outcome_training": False,
                        "selection_confidence": selection.get("confidence"),
                        "fallback_selection": str(selection.get("rationale", "")).startswith("Fallback"),
                    }
                )
                gold_memories.append(memory_id)
                gold_operations.append(operation_id)

            edge_rows = []
            for parent, child in task_dependency_edges(record):
                if parent not in selected_by_step or child not in selected_by_step:
                    continue
                edge = {
                    "edge_id": f"dep:{task_id}:{parent}:{child}",
                    "task_id": task_id,
                    "source": "five_domain_weak_gold",
                    "members": [f"mem:{task_id}:step:{parent}", f"mem:{task_id}:step:{child}"],
                    "evidence_atoms": [slot_atom(task_id, child)],
                    "relation_type": "declared_dependency",
                    "grounding": "heterogeneous_annotation_unverified",
                    "confidence": 0.4,
                    "admissible_default": False,
                    "provenance": "normalized dependency/dependencies/depends_on field",
                    "split": split,
                }
                edge_rows.append(edge)
                self.hyperedge_grounding[edge["grounding"]] += 1
                self.write("dependency_hyperedges.jsonl", edge)

            candidate_ids = sorted({key for values in candidates_by_step.values() for key in values})
            row = {
                "task_id": task_id,
                "source": "five_domain_weak_gold",
                "query": record.get("user_query") or record.get("task", ""),
                "required_atoms": [slot_atom(task_id, step_id) for step_id in sorted(task_by_step)],
                "verified_atoms": [],
                "candidate_operation_ids": candidate_ids,
                "gold_operation_ids": gold_operations,
                "gold_memory_ids": gold_memories,
                "dependency_edge_ids": [edge["edge_id"] for edge in edge_rows],
                "split": split,
                "event_time": None,
                "sequence_index": sequence,
                "time_quality": "source_file_order_only",
                "supervision": "llm_or_fallback_selection_proxy",
                "is_executed_composition": False,
                "metadata": {"paper_domain": record["metadata"].get("paper_domain")},
            }
            self.add_task(row)
            self.write(
                "weak_set_labels.jsonl",
                {
                    "label_id": f"weak:{task_id}",
                    "task_id": task_id,
                    "source": "five_domain_weak_gold",
                    "selected_memory_ids": gold_memories,
                    "selected_operation_ids": gold_operations,
                    "proxy_label": 1,
                    "proxy_type": "llm_or_fallback_selected_composition",
                    "is_executed": False,
                    "split": split,
                    "eligible_for_bce_outcome_training": False,
                },
            )

    def process_hsc(self) -> None:
        root = self.source / "HSC-main"
        with (root / "raw_models.json").open(encoding="utf-8") as handle:
            raw = json.load(handle)
        with (root / "normalized_model.json").open(encoding="utf-8") as handle:
            normalized = json.load(handle)
        with (root / "requirements.json").open(encoding="utf-8") as handle:
            requirements = json.load(handle)
        with (root / "workflow.json").open(encoding="utf-8") as handle:
            workflows = json.load(handle)
        with (root / "best_solution.json").open(encoding="utf-8") as handle:
            solutions = json.load(handle)

        suffix_index: dict[str, list[tuple[str, list[Any]]]] = defaultdict(list)
        for full_name, values in raw.items():
            suffix_index[full_name.rsplit("/", 1)[-1]].append((full_name, values))

        for index, service in enumerate(normalized):
            operation_id = f"hsc:{index:05d}"
            matches = [
                item
                for item in suffix_index.get(service["service_name"], [])
                if len(item[1]) > 7 and item[1][7] == service["function"]
            ]
            full_name = matches[0][0] if len(matches) == 1 else None
            raw_values = matches[0][1] if len(matches) == 1 else None
            provider = full_name.split("/", 1)[0] if full_name and "/" in full_name else None
            description = raw_values[0] if raw_values else service["function"]
            self.add_operation(
                {
                    "operation_id": operation_id,
                    "service_id": operation_id,
                    "source": "hsc",
                    "name": service["service_name"],
                    "description": description,
                    "category": service["function"],
                    "method": "MODEL_CALL",
                    "endpoint": full_name or service["service_name"],
                    "input_atoms": [],
                    "required_input_atoms": [],
                    "output_atoms": [],
                    "split": "auxiliary_catalog",
                    "qos": {
                        key: service[key]
                        for key in (
                            "normalized_downloads",
                            "normalized_likes",
                            "normalized_response_times",
                            "normalized_waiting_times",
                            "reliabilities",
                            "successabilities",
                        )
                    },
                    "provenance": {
                        "source_path": "HSC-main/normalized_model.json",
                        "array_index": index,
                        "raw_full_name": full_name,
                        "provider": provider,
                    },
                }
            )

        solution_by_task = {item["req_number"]: item for item in solutions}
        for index, (query, workflow) in enumerate(zip(requirements, workflows)):
            task_id = f"hsc:{index:05d}"
            split = hsc_split(index)
            functions = [function for stage in workflow["flow"] for function in stage]
            solution = solution_by_task[index]
            selected_indices = solution["best_solution"]
            if len(functions) != len(selected_indices):
                raise ValueError(f"HSC flattened workflow mismatch at task {index}")
            gold_memories: list[str] = []
            gold_operations: list[str] = []
            required_atoms: list[str] = []
            for slot, (function, selected_index) in enumerate(zip(functions, selected_indices), start=1):
                operation_id = f"hsc:{selected_index:05d}"
                operation = self.operations[operation_id]
                if operation["category"] != function:
                    raise ValueError(f"HSC gold function mismatch at task {index}, slot {slot}")
                memory_id = f"mem:{task_id}:slot:{slot}"
                atom = slot_atom(task_id, slot)
                text = compact_text(query, function, operation["name"], operation["description"])
                evidence = [operation_atom(operation_id), atom, f"function:{normalize_atom(function)}"]
                self.add_memory(
                    {
                        "memory_id": memory_id,
                        "source": "hsc_optimizer_gold",
                        "text": text,
                        "operation_ids": [operation_id],
                        "evidence_atoms": evidence,
                        "token_cost": estimate_tokens(text),
                        "storage_cost": estimate_tokens(text) + 2 * len(evidence),
                        "event_time": None,
                        "sequence_index": index,
                        "time_quality": "dataset_index_only",
                        "provenance": f"HSC best_solution[{index}]",
                        "observed_outcome": None,
                        "split": split,
                        "admissible": True,
                        "live_validated": False,
                        "eligible_for_outcome_training": False,
                    }
                )
                gold_memories.append(memory_id)
                gold_operations.append(operation_id)
                required_atoms.append(atom)
            row = {
                "task_id": task_id,
                "source": "hsc_optimizer_gold",
                "query": query,
                "required_atoms": required_atoms,
                "verified_atoms": [],
                "candidate_operation_ids": [],
                "candidate_function_paths": functions,
                "gold_operation_ids": gold_operations,
                "gold_memory_ids": gold_memories,
                "dependency_edge_ids": [],
                "split": split,
                "event_time": None,
                "sequence_index": index,
                "time_quality": "dataset_index_only",
                "supervision": "qos_optimizer_gold_proxy",
                "is_executed_composition": False,
                "metadata": {
                    "objective": workflow["obj_func"],
                    "constraints": workflow["cons"],
                    "best_value": solution["best_value"],
                    "solver_time_seconds": solution["time"],
                },
            }
            self.add_task(row)
            self.write(
                "weak_set_labels.jsonl",
                {
                    "label_id": f"weak:{task_id}",
                    "task_id": task_id,
                    "source": "hsc_optimizer_gold",
                    "selected_memory_ids": gold_memories,
                    "selected_operation_ids": gold_operations,
                    "proxy_utility": solution["best_value"],
                    "proxy_type": "qos_optimizer_solution",
                    "is_executed": False,
                    "split": split,
                    "eligible_for_bce_outcome_training": False,
                },
            )

    def finalize(self) -> None:
        with (self.temp / "operations.jsonl").open("w", encoding="utf-8") as handle:
            for operation_id in sorted(self.operations):
                json_dump_line(handle, self.operations[operation_id])
                self.counts["operations.jsonl"] += 1

        for operation_id, members in sorted(self.memory_by_operation.items()):
            if len(members) > 1:
                self.write(
                    "duplicate_groups.jsonl",
                    {
                        "group_id": f"dup:{operation_id}",
                        "kind": "same_operation_experience",
                        "operation_id": operation_id,
                        "memory_ids": members,
                        "member_count": len(members),
                        "confidence": 1.0,
                    },
                )

        for handle in self.handles.values():
            handle.flush()

        source_files = [
            self.source / "processed" / "services.jsonl",
            self.source / "processed" / "endpoints.jsonl",
            self.source / "processed" / "feedback_events.jsonl",
            self.source / "processed" / "selection_tasks.jsonl",
            self.source / "emp_data2_synthetic_compositions.jsonl",
            self.source / "five_domain_1400_gold.jsonl",
            self.source / "EMP" / "data2.xlsx",
            self.source / "HSC-main" / "raw_models.json",
            self.source / "HSC-main" / "normalized_model.json",
            self.source / "HSC-main" / "requirements.json",
            self.source / "HSC-main" / "workflow.json",
            self.source / "HSC-main" / "best_solution.json",
        ]
        quality = {
            "schema_version": SCHEMA_VERSION,
            "counts": dict(sorted(self.counts.items())),
            "task_counts_by_source": dict(sorted(self.task_sources.items())),
            "task_counts_by_split": dict(sorted(self.task_splits.items())),
            "memory_counts_by_source": dict(sorted(self.memory_sources.items())),
            "hyperedge_counts_by_grounding": dict(sorted(self.hyperedge_grounding.items())),
            "verified_invariants": {
                "unique_memory_ids": True,
                "all_memory_operations_exist": True,
                "all_task_gold_memories_exist": True,
                "hsc_gold_indices_and_functions_match": True,
                "weak_and_observed_labels_are_separate": True,
            },
            "known_limitations": [
                "Observed outcomes are singleton RapidAPI endpoint tests, not composition executions.",
                "No source provides trustworthy event timestamps for chronological evaluation.",
                "Only exact EMP output/input matches are admissible dependency hyperedges by default.",
                "HSC and five-domain labels are optimizer/LLM proxies and cannot train outcome BCE.",
                "The dataset is private; source permissions must be checked before redistribution.",
            ],
        }
        (self.temp / "quality_report.json").write_text(
            json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "name": "private_corem",
            "schema_version": SCHEMA_VERSION,
            "private": True,
            "redistribution_allowed": False,
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(self.source),
            "sources": [source_fingerprint(path, self.source) for path in source_files],
            "files": {
                path.name: {"bytes": path.stat().st_size, "records": self.counts.get(path.name)}
                for path in sorted(self.temp.glob("*.jsonl"))
            },
            "quality_report": "quality_report.json",
            "builder": "scripts/build_private_corem_dataset.py",
        }
        (self.temp / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.temp / "README.md").write_text(DATASET_README, encoding="utf-8")
        (self.temp / "CODEBOOK.md").write_text(CODEBOOK, encoding="utf-8")
        (self.temp / "PRIVATE").write_text(
            "PRIVATE RESEARCH DATASET\nDo not redistribute without verifying every source license.\n",
            encoding="utf-8",
        )
        checksum_lines = []
        for path in sorted(self.temp.iterdir()):
            if path.is_file() and path.name != "checksums.sha256":
                checksum_lines.append(f"{sha256_file(path)}  {path.name}")
        (self.temp / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        self.temp.rename(self.output)


DATASET_README = """# Private CoReM Dataset

This directory is a private, normalized derivative of the local RapidAPI,
EMP, five-domain, and HSC sources. It is generated by
`scripts/build_private_corem_dataset.py`; do not edit generated JSONL files.

Crucially, `set_outcomes.jsonl` contains only observed singleton endpoint-test
outcomes. Non-executed composition selections are isolated in
`weak_set_labels.jsonl` and are never valid labels for CoReM's execution BCE.
Only dependency hyperedges with `admissible_default=true` have exact
output-to-input evidence. Event times remain null where the source has no
trustworthy timestamp.

Use the dataset for representation/retrieval pretraining and static proxy
evaluation. A faithful interaction/retention experiment still requires
replayed or live, independently verified composition executions over time.

Run `python3 -m corem.dataset data/private_corem` for a validation summary.
See `CODEBOOK.md` and `quality_report.json` for schemas and limitations.
"""


CODEBOOK = """# Dataset Codebook

## `operations.jsonl`

One normalized callable operation. `input_atoms` and `output_atoms` are schema
field names, not proof of compatibility. `qos` is populated for HSC services.

## `memory_records.jsonl`

One candidate memory record with operation IDs, evidence atoms, estimated token
and storage costs, provenance, split, and outcome-training eligibility. A null
`observed_outcome` must never be converted to a negative label.

## `retrieval_tasks.jsonl`

Requests with candidate/gold operation IDs, gold memory IDs, required evidence
atoms, dependency edges, split, and supervision type. `event_time` is null when
the original source has no trustworthy time.

## `dependency_hyperedges.jsonl`

Two-record dependency groups. `output_input_exact` edges have exact normalized
field overlap and are admissible by default. Declared or heterogeneous weak
annotations are retained for auditing but disabled by default.

## `set_outcomes.jsonl`

Observed binary labels from individual endpoint tests. These train unary terms
only (`eligible_for_interaction_training=false`) and do not validate a composed
workflow.

## `weak_set_labels.jsonl`

Synthetic, LLM-selected, or optimizer-selected composition sets. They support
ranking/pretraining diagnostics but are explicitly excluded from execution
outcome BCE.

## `duplicate_groups.jsonl`

Memory records referring to the same normalized operation. Expand groups into
pairs only within the active candidate bank to avoid quadratic storage.
"""


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    required = (
        source / "processed" / "endpoints.jsonl",
        source / "emp_data2_synthetic_compositions.jsonl",
        source / "five_domain_1400_gold.jsonl",
        source / "HSC-main" / "workflow.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source files: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with DatasetBuilder(source, output) as builder:
        builder.process_rapidapi()
        builder.process_emp()
        builder.process_five_domain()
        builder.process_hsc()
        builder.finalize()
    print(json.dumps({"output": str(output), "status": "created"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

