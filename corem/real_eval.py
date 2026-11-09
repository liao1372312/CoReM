"""Strict contracts for real, longitudinal CoReM evaluation.

This module deliberately separates observations, retrieval selections, service
execution, and independent checking.  It never upgrades proxy labels into real
workflow metrics.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

UTC = timezone.utc

SCHEMA_VERSION = "corem-real-eval-v1"
STATIC_SCHEMA_VERSION = "corem-real-static-v1"
REQUIRED_FILES = ("events.jsonl", "memories.jsonl", "version_transitions.jsonl")
STATIC_REQUIRED_FILES = ("events.jsonl", "memories.jsonl")
SPLIT_ORDER = ("train", "tune", "validation", "test")
SPLIT_RATIOS = (0.60, 0.10, 0.10, 0.20)


class RealTrackError(ValueError):
    """Raised when a real-track artifact violates its declared contract."""


class ServiceRunner(Protocol):
    """Interface implemented by an explicitly selected service plugin."""

    def reset(self, event: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def execute(
        self,
        event: Mapping[str, Any],
        memories: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]: ...

    def snapshot(self, event: Mapping[str, Any]) -> str: ...


class IndependentChecker(Protocol):
    """Interface implemented separately from the service runner."""

    def check(
        self,
        event: Mapping[str, Any],
        execution: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class AuditReport:
    dataset: str
    eligible: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    counts: dict[str, int]
    schema_version: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "eligible": self.eligible,
            "status": "eligible_real_track" if self.eligible else "blocked",
            "schema_version": self.schema_version,
            "counts": self.counts,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise RealTrackError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise RealTrackError(f"expected a JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    handle = path.open(encoding="utf-8")
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RealTrackError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise RealTrackError(f"expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(directory: Path, filenames: Sequence[str]) -> None:
    lines = [f"{sha256_file(directory / name)}  {name}" for name in sorted(filenames)]
    (directory / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise RealTrackError(f"{field} must be a non-empty ISO-8601 UTC string")
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(
            text.replace("Z", "+00:00")  # noqa: FURB162 -- Python 3.10 support
        )
    except ValueError as error:
        raise RealTrackError(f"{field} is not valid ISO-8601: {value!r}") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise RealTrackError(f"{field} must include an explicit UTC offset")
    return parsed.astimezone(UTC)


def _nonempty_string(row: Mapping[str, Any], field: str, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RealTrackError(f"{label}.{field} must be a non-empty string")
    return value


def _string_list(
    row: Mapping[str, Any], field: str, label: str, *, allow_empty: bool = False
) -> list[str]:
    value = row.get(field)
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a" if allow_empty else "a non-empty"
        raise RealTrackError(f"{label}.{field} must be {qualifier} list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise RealTrackError(f"{label}.{field} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise RealTrackError(f"{label}.{field} contains duplicates")
    return value


def _record_errors(
    rows: Sequence[dict[str, Any]],
    identifier: str,
    kind: str,
    validator: Callable[[dict[str, Any], str], None],
) -> tuple[list[str], set[str]]:
    errors: list[str] = []
    identifiers: set[str] = set()
    for index, row in enumerate(rows, start=1):
        label = f"{kind}[{index}]"
        try:
            row_id = _nonempty_string(row, identifier, label)
            if row_id in identifiers:
                raise RealTrackError(f"duplicate {identifier}: {row_id}")
            identifiers.add(row_id)
            validator(row, label)
        except RealTrackError as error:
            errors.append(str(error))
    return errors, identifiers


def _validate_event(row: dict[str, Any], label: str) -> None:
    for field in (
        "incident_id",
        "provider_id",
        "service_id",
        "version_id",
        "query",
        "checker_contract_id",
    ):
        _nonempty_string(row, field, label)
    parse_utc(row.get("event_time"), f"{label}.event_time")
    _string_list(row, "required_binding_atoms", label)
    _string_list(row, "required_evidence_atoms", label)
    _string_list(row, "communication_criteria", label)
    duplicate_group = row.get("duplicate_group")
    if duplicate_group is not None and (
        not isinstance(duplicate_group, str) or not duplicate_group.strip()
    ):
        raise RealTrackError(f"{label}.duplicate_group must be null or a string")


def _validate_memory(row: dict[str, Any], label: str) -> None:
    for field in ("service_id", "provenance_uri", "content_digest"):
        _nonempty_string(row, field, label)
    _string_list(row, "version_scope", label)
    _string_list(row, "evidence_atoms", label)
    valid_from = parse_utc(row.get("valid_from"), f"{label}.valid_from")
    valid_until_value = row.get("valid_until")
    if valid_until_value is not None:
        valid_until = parse_utc(valid_until_value, f"{label}.valid_until")
        if valid_until <= valid_from:
            raise RealTrackError(f"{label}.valid_until must follow valid_from")
    if row.get("provenance_verified") is not True:
        raise RealTrackError(f"{label}.provenance_verified must be true")


def _validate_transition(row: dict[str, Any], label: str) -> None:
    for field in (
        "provider_id",
        "service_id",
        "from_version_id",
        "to_version_id",
        "evidence_uri",
    ):
        _nonempty_string(row, field, label)
    if row["from_version_id"] == row["to_version_id"]:
        raise RealTrackError(f"{label} must change version")
    parse_utc(row.get("transition_time"), f"{label}.transition_time")
    if row.get("independently_observed") is not True:
        raise RealTrackError(f"{label}.independently_observed must be true")


def _legacy_candidate_diagnostics(
    dataset: Path, counts: dict[str, int], warnings: list[str]
) -> None:
    """Describe useful legacy rows without treating them as contract records."""

    candidates = {
        "feedback_events.jsonl": (
            "candidate_feedback_events",
            ("event_time", "provider_id", "version_id", "incident_id"),
        ),
        "endpoints.jsonl": (
            "candidate_endpoints",
            ("version_scope", "valid_from", "provenance_uri", "evidence_atoms"),
        ),
    }
    for filename, (count_name, required_fields) in candidates.items():
        path = dataset / filename
        if not path.is_file():
            continue
        try:
            rows = load_jsonl(path)
        except RealTrackError as error:
            warnings.append(f"could not profile legacy {filename}: {error}")
            continue
        counts[count_name] = len(rows)
        coverage = {
            field: sum(row.get(field) is not None for row in rows)
            for field in required_fields
        }
        warnings.append(
            f"legacy {filename} found ({len(rows)} rows); required-field coverage: "
            + ", ".join(f"{field}={coverage[field]}" for field in required_fields)
            + ". These rows are candidates only, not promoted real-track records."
        )


def audit_real_dataset(root: str | Path) -> AuditReport:
    """Audit whether a directory may support promoted real-track results."""

    dataset = Path(root).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    counts = {"events": 0, "memories": 0, "version_transitions": 0}
    manifest: dict[str, Any] = {}
    manifest_path = dataset / "manifest.json"
    if not manifest_path.is_file():
        errors.append("missing manifest.json")
    else:
        try:
            manifest = load_json(manifest_path)
        except RealTrackError as error:
            errors.append(str(error))

    static = manifest.get("schema_version") == STATIC_SCHEMA_VERSION
    required_files = STATIC_REQUIRED_FILES if static else REQUIRED_FILES
    for filename in required_files:
        if not (dataset / filename).is_file():
            errors.append(f"missing {filename}")
    if errors:
        _legacy_candidate_diagnostics(dataset, counts, warnings)
        return AuditReport(
            dataset=str(dataset),
            eligible=False,
            errors=tuple(errors),
            warnings=tuple(warnings),
            counts=counts,
            schema_version=manifest.get("schema_version"),
        )

    if manifest.get("schema_version") not in {SCHEMA_VERSION, STATIC_SCHEMA_VERSION}:
        errors.append(
            f"schema_version must equal {SCHEMA_VERSION!r} or {STATIC_SCHEMA_VERSION!r}"
        )
    if static and (dataset / "version_transitions.jsonl").exists():
        errors.append("static benchmark must not declare version transitions")
    if static:
        if manifest.get("benchmark_mode") != "static_public_release":
            errors.append("static benchmark_mode must be 'static_public_release'")
        if manifest.get("natural_drift_claim") is not False:
            errors.append("static benchmark must explicitly deny natural drift")
        for field in ("event_time_semantics", "claim_boundary"):
            if not isinstance(manifest.get(field), str) or not manifest[field].strip():
                errors.append(f"static manifest.{field} is required")
    if manifest.get("real_world_observations") is not True:
        errors.append("manifest.real_world_observations must be true")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("manifest.provenance must be an object")
    else:
        for field in ("source_uri", "license", "immutable_source_id"):
            if (
                not isinstance(provenance.get(field), str)
                or not provenance[field].strip()
            ):
                errors.append(f"manifest.provenance.{field} is required")
        for field in ("collection_start_utc", "collection_end_utc"):
            try:
                parse_utc(provenance.get(field), f"manifest.provenance.{field}")
            except RealTrackError as error:
                errors.append(str(error))
    execution = manifest.get("execution")
    if not isinstance(execution, dict):
        errors.append("manifest.execution must be an object")
    else:
        if execution.get("resettable") is not True:
            errors.append("manifest.execution.resettable must be true")
        if not isinstance(execution.get("runner_contract"), str):
            errors.append("manifest.execution.runner_contract is required")
    checker = manifest.get("checker")
    if not isinstance(checker, dict):
        errors.append("manifest.checker must be an object")
    else:
        if checker.get("independent_from_runner") is not True:
            errors.append("manifest.checker.independent_from_runner must be true")
        for field in ("contract_id", "implementation_digest"):
            if not isinstance(checker.get(field), str) or not checker[field].strip():
                errors.append(f"manifest.checker.{field} is required")

    try:
        events = load_jsonl(dataset / "events.jsonl")
        memories = load_jsonl(dataset / "memories.jsonl")
        transitions = (
            [] if static else load_jsonl(dataset / "version_transitions.jsonl")
        )
    except RealTrackError as error:
        errors.append(str(error))
        events, memories, transitions = [], [], []
    counts.update(
        events=len(events), memories=len(memories), version_transitions=len(transitions)
    )
    if not events:
        errors.append("events.jsonl must not be empty")
    if not memories:
        errors.append("memories.jsonl must not be empty")
    if not static and not transitions:
        errors.append("version_transitions.jsonl must not be empty")

    event_errors, _ = _record_errors(events, "event_id", "event", _validate_event)
    memory_errors, _ = _record_errors(memories, "memory_id", "memory", _validate_memory)
    transition_errors, _ = _record_errors(
        transitions, "transition_id", "transition", _validate_transition
    )
    errors.extend(event_errors + memory_errors + transition_errors)

    services: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in events:
        if all(field in row for field in ("provider_id", "service_id", "version_id")):
            services[(str(row["provider_id"]), str(row["service_id"]))].add(
                str(row["version_id"])
            )
    transition_services = {
        (str(row.get("provider_id")), str(row.get("service_id"))) for row in transitions
    }
    if not static:
        for service, versions in sorted(services.items()):
            if len(versions) < 2:
                errors.append(
                    f"service {service[0]}/{service[1]} has fewer than two observed versions"
                )
            if service not in transition_services:
                errors.append(
                    f"service {service[0]}/{service[1]} has no observed transition"
                )
    provider_count = len({provider for provider, _ in services})
    counts["providers"] = provider_count
    counts["services"] = len(services)
    if provider_count < 2:
        errors.append("at least two providers are required for provider holdout")

    declared_files = manifest.get("files")
    if not isinstance(declared_files, dict):
        errors.append("manifest.files must contain checksums and record counts")
    else:
        rows_by_file = {
            "events.jsonl": events,
            "memories.jsonl": memories,
        }
        if not static:
            rows_by_file["version_transitions.jsonl"] = transitions
        for filename, rows in rows_by_file.items():
            metadata = declared_files.get(filename)
            if not isinstance(metadata, dict):
                errors.append(f"manifest.files.{filename} is required")
                continue
            if metadata.get("records") != len(rows):
                errors.append(f"manifest count mismatch for {filename}")
            expected = metadata.get("sha256")
            if not isinstance(expected, str) or expected != sha256_file(
                dataset / filename
            ):
                errors.append(f"manifest checksum mismatch for {filename}")

    return AuditReport(
        dataset=str(dataset),
        eligible=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        counts=counts,
        schema_version=manifest.get("schema_version"),
    )


class _DisjointSet:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def grouped_chronological_split(
    events: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create leakage-safe 60/10/10/20 chronological assignments.

    Incident and duplicate relations are transitively closed.  A split boundary
    is accepted only when every event on its left is no later than every event
    on its right.
    """

    if len(events) < 4:
        raise RealTrackError("at least four events are required")
    event_ids = [_nonempty_string(row, "event_id", "event") for row in events]
    if len(event_ids) != len(set(event_ids)):
        raise RealTrackError("event_id values must be unique")
    dsu = _DisjointSet(event_ids)
    relation_members: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in events:
        event_id = str(row["event_id"])
        incident = _nonempty_string(row, "incident_id", f"event {event_id}")
        relation_members[("incident", incident)].append(event_id)
        duplicate = row.get("duplicate_group")
        if duplicate:
            relation_members[("duplicate", str(duplicate))].append(event_id)
        parse_utc(row.get("event_time"), f"event {event_id}.event_time")
    for members in relation_members.values():
        for member in members[1:]:
            dsu.union(members[0], member)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        grouped[dsu.find(str(row["event_id"]))].append(row)
    groups = []
    for root, members in grouped.items():
        times = [parse_utc(row["event_time"], "event_time") for row in members]
        groups.append(
            {
                "root": root,
                "members": members,
                "start": min(times),
                "end": max(times),
            }
        )
    groups.sort(key=lambda group: (group["start"], group["end"], group["root"]))
    if len(groups) < 4:
        raise RealTrackError(
            "incident/duplicate closure leaves fewer than four independent groups"
        )

    prefix_max: list[datetime] = []
    latest = groups[0]["end"]
    for group in groups:
        latest = max(latest, group["end"])
        prefix_max.append(latest)
    suffix_min = [groups[-1]["start"] for _ in groups]
    earliest = groups[-1]["start"]
    for index in range(len(groups) - 1, -1, -1):
        earliest = min(earliest, groups[index]["start"])
        suffix_min[index] = earliest
    candidates = [
        index
        for index in range(1, len(groups))
        if prefix_max[index - 1] <= suffix_min[index]
    ]
    if len(candidates) < 3:
        raise RealTrackError(
            "fewer than three leakage-safe chronological boundaries exist"
        )

    cumulative_events = []
    count = 0
    for group in groups:
        count += len(group["members"])
        cumulative_events.append(count)
    targets = [sum(SPLIT_RATIOS[:index]) * len(events) for index in (1, 2, 3)]
    cuts: list[int] = []
    for target_index, target in enumerate(targets):
        remaining = 2 - target_index
        options = [
            candidate
            for position, candidate in enumerate(candidates)
            if (not cuts or candidate > cuts[-1])
            and len(candidates) - position - 1 >= remaining
        ]
        if not options:
            raise RealTrackError("cannot allocate four non-empty chronological splits")
        cuts.append(
            min(
                options,
                key=lambda candidate: (
                    abs(cumulative_events[candidate - 1] - target),
                    candidate,
                ),
            )
        )

    assignments: list[dict[str, Any]] = []
    boundaries = (0, *cuts, len(groups))
    for split_index, split in enumerate(SPLIT_ORDER):
        for group in groups[boundaries[split_index] : boundaries[split_index + 1]]:
            for row in group["members"]:
                assignments.append(
                    {
                        "event_id": row["event_id"],
                        "incident_id": row["incident_id"],
                        "duplicate_group": row.get("duplicate_group"),
                        "provider_id": row["provider_id"],
                        "service_id": row["service_id"],
                        "version_id": row["version_id"],
                        "event_time": row["event_time"],
                        "split": split,
                    }
                )
    return sorted(assignments, key=lambda row: str(row["event_id"]))


def stratified_grouped_chronological_split(
    events: Sequence[dict[str, Any]],
    *,
    fields: Sequence[str] = ("service_id",),
) -> list[dict[str, Any]]:
    """Split independently inside declared strata while preserving chronology.

    Public services commonly have non-overlapping release calendars.  A single
    global time boundary would then confound provider identity with split.  The
    stratum label is persisted so validation can enforce chronology and group
    isolation within exactly the same populations used during assignment.
    """

    if not fields:
        raise RealTrackError("at least one stratification field is required")
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for index, event in enumerate(events, start=1):
        key = tuple(
            _nonempty_string(event, field, f"event[{index}]") for field in fields
        )
        grouped[key].append(event)
    if len(grouped) < 2:
        raise RealTrackError("stratified splitting requires at least two strata")

    assignments: list[dict[str, Any]] = []
    for key, members in sorted(grouped.items()):
        label = "|".join(
            f"{field}={value}" for field, value in zip(fields, key, strict=True)
        )
        for row in grouped_chronological_split(members):
            assignments.append(
                {
                    **row,
                    "split_stratum": label,
                    "stratification_fields": list(fields),
                }
            )
    return sorted(assignments, key=lambda row: str(row["event_id"]))


def validate_split_assignments(
    events: Sequence[dict[str, Any]],
    assignments: Sequence[dict[str, Any]],
    *,
    allow_test_only: bool = False,
) -> None:
    event_ids = {str(row["event_id"]) for row in events}
    assignment_ids = [str(row.get("event_id")) for row in assignments]
    if set(assignment_ids) != event_ids or len(assignment_ids) != len(event_ids):
        raise RealTrackError("split assignments must cover every event exactly once")
    if (
        allow_test_only
        and assignments
        and all(row.get("split") == "test" for row in assignments)
    ):
        return
    by_id = {str(row["event_id"]): row for row in events}
    split_groups: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    split_times: dict[tuple[str, str], list[datetime]] = defaultdict(list)
    strata: set[str] = set()
    for row in assignments:
        split = str(row.get("split"))
        if split not in SPLIT_ORDER:
            raise RealTrackError(f"unknown split {split!r}")
        stratum_value = row.get("split_stratum", "__global__")
        if not isinstance(stratum_value, str) or not stratum_value:
            raise RealTrackError("split_stratum must be a non-empty string")
        stratum = stratum_value
        strata.add(stratum)
        event = by_id[str(row["event_id"])]
        split_groups[(stratum, split)].add(("incident", str(event["incident_id"])))
        if event.get("duplicate_group"):
            split_groups[(stratum, split)].add(
                ("duplicate", str(event["duplicate_group"]))
            )
        split_times[(stratum, split)].append(
            parse_utc(event["event_time"], "event_time")
        )
    for stratum in sorted(strata):
        missing_splits = [
            split for split in SPLIT_ORDER if not split_times[(stratum, split)]
        ]
        if missing_splits:
            raise RealTrackError(
                f"stratum {stratum!r} has empty splits: {missing_splits}"
            )
        for left_index, left in enumerate(SPLIT_ORDER):
            for right in SPLIT_ORDER[left_index + 1 :]:
                overlap = split_groups[(stratum, left)].intersection(
                    split_groups[(stratum, right)]
                )
                if overlap:
                    raise RealTrackError(
                        f"group leakage between {left} and {right} in {stratum}"
                    )
            if left_index < len(SPLIT_ORDER) - 1:
                right = SPLIT_ORDER[left_index + 1]
                if max(split_times[(stratum, left)]) > min(
                    split_times[(stratum, right)]
                ):
                    raise RealTrackError(
                        f"chronology violation between {left} and {right} in {stratum}"
                    )


def load_plugin(specification: str) -> Any:
    """Load an explicitly supplied ``module:factory`` plugin."""

    if ":" not in specification:
        raise RealTrackError("plugin must use module:factory syntax")
    module_name, attribute_name = specification.split(":", 1)
    if not module_name or not attribute_name:
        raise RealTrackError("plugin must use module:factory syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise RealTrackError(f"plugin factory is not callable: {specification}")
    return factory()


def _exact_boolean_checks(
    checker: Mapping[str, Any],
    field: str,
    required: Sequence[str],
    atom_field: str,
    pass_field: str,
) -> tuple[int, int]:
    checks = checker.get(field)
    if not isinstance(checks, list):
        raise RealTrackError(f"checker.{field} must be a list")
    observed: dict[str, bool] = {}
    for index, check in enumerate(checks, start=1):
        if not isinstance(check, dict):
            raise RealTrackError(f"checker.{field}[{index}] must be an object")
        atom = _nonempty_string(check, atom_field, f"checker.{field}[{index}]")
        passed = check.get(pass_field)
        if not isinstance(passed, bool):
            raise RealTrackError(
                f"checker.{field}[{index}].{pass_field} must be boolean"
            )
        _nonempty_string(check, "evidence_ref", f"checker.{field}[{index}]")
        if atom in observed:
            raise RealTrackError(f"duplicate {atom_field} in checker.{field}: {atom}")
        observed[atom] = passed
    if set(observed) != set(required):
        raise RealTrackError(
            f"checker.{field} must exactly match declared {atom_field} values"
        )
    return sum(observed.values()), len(required)


def score_execution_row(
    row: Mapping[str, Any], event: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one independent checker record and return additive counts."""

    checker = row.get("checker")
    if not isinstance(checker, dict):
        raise RealTrackError("execution row is missing checker output")
    if row.get("checker_independent") is not True:
        raise RealTrackError("checker_independent must be true")
    if row.get("reset_verified") is not True:
        raise RealTrackError("reset_verified must be true")
    if checker.get("contract_id") != event.get("checker_contract_id"):
        raise RealTrackError("checker contract does not match the event")
    goal = checker.get("goal_satisfied")
    if not isinstance(goal, bool):
        raise RealTrackError("checker.goal_satisfied must be boolean")
    _nonempty_string(checker, "goal_evidence_ref", "checker")
    binding_passed, binding_total = _exact_boolean_checks(
        checker,
        "binding_checks",
        list(event["required_binding_atoms"]),
        "atom",
        "passed",
    )
    evidence_covered, evidence_total = _exact_boolean_checks(
        checker,
        "evidence_checks",
        list(event["required_evidence_atoms"]),
        "atom",
        "covered",
    )
    communication_passed, communication_total = _exact_boolean_checks(
        checker,
        "communication_checks",
        list(event["communication_criteria"]),
        "criterion",
        "passed",
    )
    reuse_attempts = checker.get("reuse_attempts")
    if not isinstance(reuse_attempts, list):
        raise RealTrackError("checker.reuse_attempts must be a list")
    stale_attempts = 0
    attempted_reuses = 0
    selected_memory_ids = row.get("selected_memory_ids")
    if not isinstance(selected_memory_ids, list) or any(
        not isinstance(memory_id, str) for memory_id in selected_memory_ids
    ):
        raise RealTrackError("execution row selected_memory_ids must be a string list")
    selected_set = set(selected_memory_ids)
    for index, attempt in enumerate(reuse_attempts, start=1):
        if not isinstance(attempt, dict):
            raise RealTrackError(f"checker.reuse_attempts[{index}] must be an object")
        memory_id = _nonempty_string(attempt, "memory_id", f"reuse_attempt[{index}]")
        if memory_id not in selected_set:
            raise RealTrackError(
                f"reuse_attempt[{index}] references an unselected memory"
            )
        _nonempty_string(attempt, "evidence_ref", f"reuse_attempt[{index}]")
        attempted = attempt.get("attempted")
        invalid = attempt.get("invalid_for_event")
        if not isinstance(attempted, bool) or not isinstance(invalid, bool):
            raise RealTrackError("reuse attempted/invalid_for_event must be boolean")
        if invalid and not attempted:
            raise RealTrackError("an unattempted reuse cannot count as stale use")
        attempted_reuses += int(attempted)
        stale_attempts += int(attempted and invalid)
    verified_success = bool(
        goal
        and binding_passed == binding_total
        and communication_passed == communication_total
        and not row.get("execution_error")
    )
    return {
        "verified_success": int(verified_success),
        "goal_satisfied": int(goal),
        "binding_passed": binding_passed,
        "binding_total": binding_total,
        "evidence_covered": evidence_covered,
        "evidence_total": evidence_total,
        "evidence_complete": int(evidence_covered == evidence_total),
        "communication_passed": communication_passed,
        "communication_total": communication_total,
        "communication_success": int(communication_passed == communication_total),
        "attempted_reuses": attempted_reuses,
        "stale_attempts": stale_attempts,
        "stale_workflow": int(stale_attempts > 0),
    }


def summarize_scored_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RealTrackError("cannot summarize an empty result set")

    def ratio(numerator: str, denominator: str) -> float | None:
        total = sum(int(row[denominator]) for row in rows)
        return sum(int(row[numerator]) for row in rows) / total if total else None

    latencies = sorted(float(row["latency_ms"]) for row in rows)
    p95_index = max(0, math.ceil(0.95 * len(latencies)) - 1)
    return {
        "n": len(rows),
        "vsr": sum(int(row["verified_success"]) for row in rows) / len(rows),
        "goal_success_rate": sum(int(row["goal_satisfied"]) for row in rows)
        / len(rows),
        "bs": ratio("binding_passed", "binding_total"),
        "ecr": sum(int(row["evidence_complete"]) for row in rows) / len(rows),
        "evidence_atom_coverage": ratio("evidence_covered", "evidence_total"),
        "communication_success_rate": sum(
            int(row["communication_success"]) for row in rows
        )
        / len(rows),
        "sur": ratio("stale_attempts", "attempted_reuses"),
        "stale_workflow_rate": sum(int(row["stale_workflow"]) for row in rows)
        / len(rows),
        "mean_tokens": sum(float(row["token_cost"]) for row in rows) / len(rows),
        "p95_latency_ms": latencies[p95_index],
    }


def validate_native_export(
    events: Sequence[dict[str, Any]],
    manifest: Mapping[str, Any],
    selections: Sequence[dict[str, Any]],
    *,
    split_event_ids: set[str],
    known_memory_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate that a claimed native run is complete and label blind."""

    errors: list[str] = []
    for field in ("system_name", "source_repository", "source_commit", "method"):
        if not isinstance(manifest.get(field), str) or not str(manifest[field]).strip():
            errors.append(f"manifest.{field} is required")
    if manifest.get("native_unmodified_core") is not True:
        errors.append("manifest.native_unmodified_core must be true")
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or not (
        7 <= len(source_commit) <= 64
        and all(character in "0123456789abcdefABCDEF" for character in source_commit)
    ):
        errors.append("manifest.source_commit must be a 7-64 character hex revision")
    budget = manifest.get("token_budget")
    if not isinstance(budget, int) or budget <= 0:
        errors.append("manifest.token_budget must be a positive integer")
    seeds = manifest.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(not isinstance(seed, int) for seed in seeds)
    ):
        errors.append("manifest.seeds must be a non-empty integer list")
        seeds = []
    for field in (
        "storage_budget",
        "llm_call_budget",
        "offline_training_examples",
    ):
        value = manifest.get(field)
        if not isinstance(value, int) or value < 0:
            errors.append(f"manifest.{field} must be a non-negative integer")
    prohibited = {
        "outcome",
        "goal_satisfied",
        "gold_memory_ids",
        "required_binding_atoms",
        "required_evidence_atoms",
        "verified_success",
    }
    observed: set[tuple[str, int]] = set()
    for index, row in enumerate(selections, start=1):
        leaked = prohibited.intersection(row)
        if leaked:
            errors.append(f"selection[{index}] contains label fields: {sorted(leaked)}")
        event_id = str(row.get("event_id", ""))
        seed = row.get("seed")
        if event_id not in split_event_ids:
            errors.append(f"selection[{index}] has unknown/non-target event_id")
        if seed not in seeds:
            errors.append(f"selection[{index}] has undeclared seed")
        if row.get("method") != manifest.get("method"):
            errors.append(f"selection[{index}] method does not match manifest")
        key = (event_id, seed) if isinstance(seed, int) else (event_id, -1)
        if key in observed:
            errors.append(f"duplicate selection for event/seed: {key}")
        observed.add(key)
        selected = row.get("selected_memory_ids")
        if not isinstance(selected, list) or any(
            not isinstance(item, str) for item in selected
        ):
            errors.append(f"selection[{index}].selected_memory_ids must be a list")
        elif known_memory_ids is not None:
            unknown = set(selected).difference(known_memory_ids)
            if unknown:
                errors.append(
                    f"selection[{index}] references unknown memories: {sorted(unknown)}"
                )
        token_cost = row.get("token_cost")
        if not isinstance(token_cost, int) or token_cost < 0:
            errors.append(f"selection[{index}].token_cost must be non-negative")
        elif isinstance(budget, int) and token_cost > budget:
            errors.append(f"selection[{index}] exceeds the token budget")
        latency = row.get("latency_ms")
        if not isinstance(latency, (int, float)) or float(latency) < 0:
            errors.append(f"selection[{index}].latency_ms must be non-negative")
        llm_calls = row.get("llm_calls")
        if not isinstance(llm_calls, int) or llm_calls < 0:
            errors.append(f"selection[{index}].llm_calls must be non-negative")
        elif isinstance(manifest.get("llm_call_budget"), int) and llm_calls > int(
            manifest["llm_call_budget"]
        ):
            errors.append(f"selection[{index}] exceeds the LLM-call budget")
        bank_cost = row.get("active_bank_cost")
        if not isinstance(bank_cost, int) or bank_cost < 0:
            errors.append(f"selection[{index}].active_bank_cost must be non-negative")
        elif isinstance(manifest.get("storage_budget"), int) and bank_cost > int(
            manifest["storage_budget"]
        ):
            errors.append(f"selection[{index}] exceeds the storage budget")
    expected = {(event_id, seed) for event_id in split_event_ids for seed in seeds}
    missing = expected.difference(observed)
    extra = observed.difference(expected)
    if missing:
        errors.append(f"missing {len(missing)} event/seed selections")
    if extra:
        errors.append(f"found {len(extra)} unexpected event/seed selections")
    known_events = {str(row["event_id"]) for row in events}
    if not split_event_ids.issubset(known_events):
        errors.append("split assignments reference unknown dataset events")
    return {
        "status": "valid_native_export" if not errors else "blocked",
        "valid": not errors,
        "errors": errors,
        "expected_rows": len(expected),
        "observed_rows": len(selections),
    }
