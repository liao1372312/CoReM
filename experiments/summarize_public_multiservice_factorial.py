#!/usr/bin/env python3
"""Summarize fixed-weight, matched 2x2 validation decisions."""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_json, load_jsonl, write_checksums, write_json  # noqa: E402

CELLS = ("hgt_corem", "hgt_additive", "text_complement", "text_additive")


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def cluster_ci(
    differences: dict[str, float], families: dict[str, tuple[str, ...]]
) -> tuple[float, float]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for event_id, value in differences.items():
        grouped[families[event_id]].append(value)
    keys = sorted(grouped)
    rng = random.Random(2027)
    draws = []
    for _ in range(10000):
        sampled = [rng.choice(keys) for _ in keys]
        values = [value for key in sampled for value in grouped[key]]
        draws.append(sum(values) / len(values))
    return quantile(draws, 0.025), quantile(draws, 0.975)


def summarize(
    rows: dict[str, dict[str, dict[str, Any]]],
    events: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if set(rows) != set(CELLS):
        raise ValueError("all four factorial cells are required")
    event_ids = set(rows[CELLS[0]])
    if not event_ids or any(set(rows[cell]) != event_ids for cell in CELLS):
        raise ValueError("factorial cells do not cover identical events")
    families = {
        event_id: (
            str(events[event_id]["service_id"]),
            str(events[event_id]["version_id"]),
            str(events[event_id]["task_key"]),
        )
        for event_id in event_ids
    }
    outcomes = {
        cell: {event_id: int(rows[cell][event_id]["label"]) for event_id in event_ids}
        for cell in CELLS
    }
    full_bundles = {
        event_id: tuple(rows["hgt_corem"][event_id]["selected_memory_ids"])
        for event_id in event_ids
    }
    metrics = {
        cell: {
            "successes": sum(outcomes[cell].values()),
            "vsr": sum(outcomes[cell].values()) / len(event_ids),
            "bundle_agreement_with_full": sum(
                tuple(rows[cell][event_id]["selected_memory_ids"])
                == full_bundles[event_id]
                for event_id in event_ids
            )
            / len(event_ids),
        }
        for cell in CELLS
    }
    formulas = {
        "complement_with_graph": (1, -1, 0, 0),
        "complement_without_graph": (0, 0, 1, -1),
        "graph_with_complement": (1, 0, -1, 0),
        "graph_without_complement": (0, 1, 0, -1),
        "interaction": (1, -1, -1, 1),
    }
    contrasts = {}
    for name, coefficients in formulas.items():
        difference = {
            event_id: sum(
                coefficient * outcomes[cell][event_id]
                for cell, coefficient in zip(CELLS, coefficients, strict=True)
            )
            for event_id in event_ids
        }
        low, high = cluster_ci(difference, families)
        contrasts[name] = {
            "difference": sum(difference.values()) / len(event_ids),
            "family_cluster_95ci": [low, high],
        }
    return {
        "status": "complete_exploratory_factorial_validation",
        "claim_boundary": (
            "Old inspected v5 validation data, not an independent provider/task "
            "confirmation; fixed candidates and weights, no test-label tuning."
        ),
        "events": len(event_ids),
        "task_version_families": len(set(families.values())),
        "cells": metrics,
        "paired_contrasts": contrasts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--track", choices=("validation", "executed_test"), default="validation"
    )
    for cell in CELLS:
        parser.add_argument(f"--{cell.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")

    directories = {cell: getattr(args, cell) for cell in CELLS}
    if args.track == "validation":
        results = {
            cell: load_json(path / "result.json") for cell, path in directories.items()
        }
        weights = {float(row["selected_neural_weight"]) for row in results.values()}
        seeds = {tuple(row["seeds"]) for row in results.values()}
    else:
        results = {
            cell: load_json(path / "manifest.json")
            for cell, path in directories.items()
        }
        weights = {float(row["neural_weight"]) for row in results.values()}
        seeds = {tuple(row["model_seeds"]) for row in results.values()}
    if len(weights) != 1 or len(seeds) != 1:
        raise ValueError("factorial cells must share fixed weight and model seeds")
    if any(results[cell].get("state_variant") != cell for cell in CELLS):
        raise ValueError("factorial cell is labeled with a different model variant")
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    for cell, path in directories.items():
        if args.track == "validation":
            selected = load_jsonl(path / "selected_validation.jsonl")
        else:
            selection_file = path / "selections.jsonl"
            execution_dir = Path(f"{path}_test")
            analysis_dir = Path(f"{path}_test_analysis")
            execution_file = execution_dir / "execution_results.jsonl"
            execution_manifest = load_json(execution_dir / "manifest.json")
            analysis_result = load_json(analysis_dir / "result.json")
            if (
                execution_manifest["selection_sha256"]
                != hashlib.sha256(selection_file.read_bytes()).hexdigest()
            ):
                raise ValueError(f"execution provenance does not match {cell}")
            if (
                analysis_result["status"] != "complete_real_track"
                or analysis_result["execution_sha256"]
                != hashlib.sha256(execution_file.read_bytes()).hexdigest()
            ):
                raise ValueError(f"analysis provenance does not match {cell}")
            selections = load_jsonl(selection_file)
            scored = load_jsonl(analysis_dir / "scored_executions.jsonl")
            scored_by_event = {str(row["event_id"]): row for row in scored}
            if len(scored_by_event) != len(scored) or {
                str(row["event_id"]) for row in selections
            } != set(scored_by_event):
                raise ValueError(f"scored test events are incomplete in {cell}")
            selected = [
                {
                    **row,
                    "label": scored_by_event[str(row["event_id"])]["verified_success"],
                }
                for row in selections
            ]
        keyed = {str(row["event_id"]): row for row in selected}
        if len(keyed) != len(selected):
            raise ValueError(f"duplicate validation event in {cell}")
        rows[cell] = keyed
    events = {
        str(row["event_id"]): row for row in load_jsonl(args.dataset / "events.jsonl")
    }
    result = summarize(rows, events)
    result["track"] = args.track
    if args.track == "executed_test":
        result["status"] = "complete_posthoc_factorial_executed_test"
        result["claim_boundary"] = (
            "Post-hoc factorial analysis on the already inspected v6 test set, "
            "which reuses v5 providers, task families, memory bank, and checker. "
            "Exact event/bundle executions are independently checked, but this "
            "is not an independent new-provider or new-task confirmation."
        )
    result["neural_weight"] = weights.pop()
    result["model_seeds"] = list(seeds.pop())
    result["sources"] = {}
    for cell, path in directories.items():
        manifest_path = path / (
            "result.json" if args.track == "validation" else "manifest.json"
        )
        source = {
            "path": str(manifest_path.resolve()),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }
        if args.track == "executed_test":
            analysis_path = Path(f"{path}_test_analysis") / "result.json"
            source["analysis_path"] = str(analysis_path.resolve())
            source["analysis_sha256"] = hashlib.sha256(
                analysis_path.read_bytes()
            ).hexdigest()
        result["sources"][cell] = source
    args.output.mkdir(parents=True)
    write_json(args.output / "result.json", result)
    write_checksums(args.output, ("result.json",))
    print(
        f"summarized {result['events']} {args.track} events at fixed neural "
        f"weight {result['neural_weight']:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
