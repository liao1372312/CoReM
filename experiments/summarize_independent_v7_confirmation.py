#!/usr/bin/env python3
"""Audit and summarize the precommitted independent-v7 method comparison."""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_json, load_jsonl, write_checksums, write_json  # noqa: E402

FACTORIAL = ("graph_complement", "graph_additive", "text_complement", "text_additive")
COMPARATORS = ("candidate_bm25_max", "candidate_bm25_sum", "version_bm25_closure")
METHOD_LABELS = {"version_bm25_closure": "version_bm25_bundle"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relocated_path(
    stored_path: str, frozen_root: Path, current_root: Path = REPOSITORY_ROOT
) -> Path:
    """Map a manifest's original absolute repo path into this checkout."""
    if stored_path.startswith("/"):
        path = PurePosixPath(stored_path)
        original_root = PurePosixPath(str(frozen_root).replace("\\", "/"))
        if not original_root.is_absolute():
            raise ValueError(f"frozen repository root is not absolute: {frozen_root}")
        try:
            relative = path.relative_to(original_root)
        except ValueError as error:
            raise ValueError(f"frozen path escapes repository: {path}") from error
        return current_root.joinpath(*relative.parts)
    path = Path(stored_path)
    if not path.is_absolute():
        raise ValueError(f"frozen path is not absolute: {path}")
    try:
        relative = path.relative_to(frozen_root)
    except ValueError as error:
        raise ValueError(f"frozen path escapes repository: {path}") from error
    return current_root / relative


def verify_frozen_sources(
    freeze: Any, frozen_root: Path, current_root: Path = REPOSITORY_ROOT
) -> int:
    """Fail closed if any source, model, input, or runtime pinned at freeze changed."""
    checked: set[Path] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("path"), str) and isinstance(
                node.get("sha256"), str
            ):
                path = relocated_path(node["path"], frozen_root, current_root)
                if path not in checked:
                    if sha256(path) != node["sha256"]:
                        raise ValueError(f"frozen source changed: {path}")
                    checked.add(path)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(freeze)
    return len(checked)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def paired_contrast(
    outcomes: dict[str, dict[str, int]],
    families: dict[str, str],
    coefficients: dict[str, int],
) -> dict[str, Any]:
    event_ids = sorted(families)
    differences = {
        event_id: sum(
            coeff * outcomes[method][event_id] for method, coeff in coefficients.items()
        )
        for event_id in event_ids
    }
    grouped: dict[str, list[int]] = defaultdict(list)
    for event_id, difference in differences.items():
        grouped[families[event_id]].append(difference)
    family_keys = sorted(grouped)
    rng = random.Random(2027)
    draws = []
    for _ in range(10000):
        sampled = [rng.choice(family_keys) for _ in family_keys]
        values = [value for key in sampled for value in grouped[key]]
        draws.append(sum(values) / len(values))
    mean = sum(differences.values()) / len(differences)
    result = {
        "difference": mean,
        "net_successes": sum(differences.values()),
        "family_cluster_95ci": [quantile(draws, 0.025), quantile(draws, 0.975)],
        "family_clusters": len(family_keys),
        "by_family_net_successes": {
            family: sum(grouped[family]) for family in family_keys
        },
    }
    if len(coefficients) == 2 and set(coefficients.values()) == {1, -1}:
        result["discordant_wins"] = sum(value == 1 for value in differences.values())
        result["discordant_losses"] = sum(value == -1 for value in differences.values())
        result["paired_ties"] = sum(value == 0 for value in differences.values())
    return result


def load_verified_method(
    method: str,
    entry: dict[str, Any],
    analysis_dir: Path,
    frozen_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    selection_file = relocated_path(entry["selections"]["path"], frozen_root)
    manifest_file = relocated_path(entry["manifest"]["path"], frozen_root)
    if sha256(selection_file) != entry["selections"]["sha256"]:
        raise ValueError(f"committed selections changed: {method}")
    if sha256(manifest_file) != entry["manifest"]["sha256"]:
        raise ValueError(f"committed manifest changed: {method}")
    selections = load_jsonl(selection_file)
    if len(selections) != entry["rows"]:
        raise ValueError(f"committed selection count changed: {method}")
    result = load_json(analysis_dir / "result.json")
    if result["status"] != "complete_static_real_track":
        raise ValueError(f"incomplete formal analysis: {method}: {result['status']}")
    execution_dir = analysis_dir.parent / analysis_dir.name.replace(
        "_analysis_v2", "_cached_execution_v2"
    )
    if not execution_dir.exists():
        execution_dir = analysis_dir.parent / analysis_dir.name.replace(
            "_analysis_v2", "_execution_v2"
        )
    execution_file = execution_dir / "execution_results.jsonl"
    if result["execution_sha256"] != sha256(execution_file):
        raise ValueError(f"execution hash mismatch: {method}")
    manifest_path = execution_dir / "manifest.json"
    if manifest_path.exists():
        execution_manifest = load_json(manifest_path)
        if execution_manifest["selection_sha256"] != sha256(selection_file):
            raise ValueError(f"execution not linked to committed selection: {method}")
    else:
        run_manifest = load_json(execution_dir / "run_manifest.json")
        if (
            relocated_path(run_manifest["selections"], frozen_root).resolve()
            != selection_file.resolve()
        ):
            raise ValueError(f"physical execution not linked to selection: {method}")
    scored = load_jsonl(analysis_dir / "scored_executions.jsonl")
    scored_by_event = {str(row["event_id"]): row for row in scored}
    selected_by_event = {str(row["event_id"]): row for row in selections}
    if (
        len(scored_by_event) != len(scored)
        or len(selected_by_event) != len(selections)
        or set(scored_by_event) != set(selected_by_event)
    ):
        raise ValueError(f"missing or duplicate scored events: {method}")
    if any(row["method"] != METHOD_LABELS.get(method, method) for row in scored):
        raise ValueError(f"method label mismatch: {method}")
    return selected_by_event, scored_by_event


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument(
        "--analysis", action="append", required=True, metavar="METHOD=DIR"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    commitment = load_json(args.commitment)
    if (
        commitment["status"] != "all_methods_committed_before_outcome_mapping"
        or commitment["outcomes_loaded"]
    ):
        raise ValueError("all-method pre-outcome commitment is absent")
    freeze = commitment["preselection_freeze"]
    frozen_root = Path(freeze["path"]).parents[1]
    freeze_path = relocated_path(freeze["path"], frozen_root)
    if sha256(freeze_path) != freeze["sha256"]:
        raise ValueError("preselection freeze changed")
    frozen_manifest = load_json(freeze_path)
    frozen_source_count = verify_frozen_sources(frozen_manifest, frozen_root)
    frozen_dataset = relocated_path(
        frozen_manifest["dataset"]["events.jsonl"]["path"], frozen_root
    ).parent
    if args.dataset.resolve() != frozen_dataset.resolve():
        raise ValueError("dataset is not the one pinned at preselection freeze")
    directories: dict[str, Path] = {}
    for item in args.analysis:
        method, separator, path = item.partition("=")
        if not separator or method in directories:
            parser.error(f"expected unique METHOD=DIR, got {item}")
        directories[method] = Path(path)
    if set(directories) != set(commitment["methods"]):
        raise ValueError("analysis set does not match all committed methods")
    events = {
        str(row["event_id"]): row for row in load_jsonl(args.dataset / "events.jsonl")
    }
    if len(events) != commitment["test_incidents"]:
        raise ValueError("dataset/test incident count mismatch")
    selected = {}
    scored = {}
    for method, directory in directories.items():
        selected[method], scored[method] = load_verified_method(
            method, commitment["methods"][method], directory, frozen_root
        )
        if set(scored[method]) != set(events):
            raise ValueError(f"method does not cover every test incident: {method}")
    families = {event_id: str(row["task_key"]) for event_id, row in events.items()}
    outcomes = {
        method: {
            event_id: int(row["verified_success"]) for event_id, row in rows.items()
        }
        for method, rows in scored.items()
    }
    methods = {}
    for method, rows in scored.items():
        successes = sum(outcomes[method].values())
        selection_latencies = [
            float(row["latency_ms"]) for row in selected[method].values()
        ]
        if any(value < 0 or not math.isfinite(value) for value in selection_latencies):
            raise ValueError(f"invalid selection latency: {method}")
        methods[method] = {
            "successes": successes,
            "events": len(rows),
            "vsr": successes / len(rows),
            "mean_tokens": sum(float(row["token_cost"]) for row in rows.values())
            / len(rows),
            "mean_selection_latency_ms": sum(selection_latencies)
            / len(selection_latencies),
            "p95_selection_latency_ms": quantile(selection_latencies, 0.95),
            "by_provider": {
                provider: {
                    "successes": sum(
                        outcomes[method][event_id]
                        for event_id, event in events.items()
                        if event["provider_id"] == provider
                    ),
                    "events": sum(
                        event["provider_id"] == provider for event in events.values()
                    ),
                }
                for provider in sorted({row["provider_id"] for row in events.values()})
            },
            "by_lexical_stratum": {
                stratum: {
                    "successes": sum(
                        outcomes[method][event_id]
                        for event_id, event in events.items()
                        if event["lexical_stratum"] == stratum
                    ),
                    "events": sum(
                        event["lexical_stratum"] == stratum for event in events.values()
                    ),
                }
                for stratum in sorted(
                    {row["lexical_stratum"] for row in events.values()}
                )
            },
            "by_family": {
                family: {
                    "successes": sum(
                        outcomes[method][event_id]
                        for event_id in rows
                        if families[event_id] == family
                    ),
                    "events": sum(value == family for value in families.values()),
                }
                for family in sorted(set(families.values()))
            },
        }
    contrasts = {
        f"graph_complement_minus_{other}": paired_contrast(
            outcomes, families, {"graph_complement": 1, other: -1}
        )
        for other in COMPARATORS
    }
    contrasts.update(
        {
            "complement_with_graph": paired_contrast(
                outcomes, families, {"graph_complement": 1, "graph_additive": -1}
            ),
            "complement_without_graph": paired_contrast(
                outcomes, families, {"text_complement": 1, "text_additive": -1}
            ),
            "graph_with_complement": paired_contrast(
                outcomes, families, {"graph_complement": 1, "text_complement": -1}
            ),
            "graph_without_complement": paired_contrast(
                outcomes, families, {"graph_additive": 1, "text_additive": -1}
            ),
            "factorial_interaction": paired_contrast(
                outcomes,
                families,
                {
                    "graph_complement": 1,
                    "graph_additive": -1,
                    "text_complement": -1,
                    "text_additive": 1,
                },
            ),
        }
    )
    result = {
        "status": "complete_independent_v7_confirmatory_analysis",
        "claim_boundary": "Controlled synthetic workflows on independently implemented Gitea/Meilisearch service snapshots; not naturally observed production drift. All methods selected before outcome mapping. No post-test tuning is valid on this set.",
        "selection_latency_boundary": "Measured selection-only wall time excludes model/checkpoint loading and service execution; it is not end-to-end workflow latency.",
        "events": len(events),
        "families": len(set(families.values())),
        "commitment_sha256": sha256(args.commitment),
        "frozen_sources_sha256_verified": frozen_source_count,
        "methods": methods,
        "paired_contrasts": contrasts,
        "sources": {
            method: {
                "analysis_result": str((path / "result.json").resolve()),
                "sha256": sha256(path / "result.json"),
            }
            for method, path in directories.items()
        },
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "result.json", result)
    write_checksums(args.output, ("result.json",))
    print(
        f"summarized {len(events)} committed events across {len(directories)} methods"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
