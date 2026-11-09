#!/usr/bin/env python3
"""Build a private, same-source mirrored-index edge control for v23 TRAIN hints.

This is an exploratory information control, not a verified false graph or a
new task score. It reads only the frozen train-only views and exports no text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "data" / "private_corem"
EDGE_MARKER = (
    "\nPotential source-code def-use edges between occurrence numbers "
    "(not verified runtime bindings):\n"
)


def mirror_edges(edges: tuple[tuple[int, int], ...], count: int) -> tuple[tuple[int, int], ...]:
    """Reflect occurrence labels while retaining forward-directed edge shape."""
    if count < 1:
        raise ValueError("source must contain API calls")
    if any(not 1 <= producer < consumer <= count for producer, consumer in edges):
        raise ValueError("invalid input edge")
    return tuple(sorted((count + 1 - consumer, count + 1 - producer) for producer, consumer in edges))


def mirrored_hint(order_hint: str, dependency_hint: str) -> tuple[str, int, int]:
    prefix, separator, edge_text = dependency_hint.partition(EDGE_MARKER)
    if not separator or prefix != order_hint:
        raise ValueError("paired hints do not share an exact source/order prefix")
    numbered = re.findall(r"(?m)^(\d+)\. [A-Za-z_]\w*\.[A-Za-z_]\w*$", order_hint)
    count = len(numbered)
    if [int(number) for number in numbered] != list(range(1, count + 1)):
        raise ValueError("source API occurrence labels are malformed")
    if edge_text == "None detected in this source.":
        edges: tuple[tuple[int, int], ...] = ()
    else:
        if not re.fullmatch(r"\d+->\d+(?:, \d+->\d+)*", edge_text):
            raise ValueError("source edge format is malformed")
        edges = tuple(
            (int(producer), int(consumer))
            for producer, consumer in (
                edge.split("->") for edge in edge_text.split(", ")
            )
        )
    mirrored = mirror_edges(edges, count)
    mirrored_text = (
        ", ".join(f"{producer}->{consumer}" for producer, consumer in mirrored)
        if mirrored
        else "None detected in this source."
    )
    return (
        order_hint + EDGE_MARKER + mirrored_text,
        len(edges),
        len(set(edges) & set(mirrored)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order-view", required=True, type=Path)
    parser.add_argument("--dependency-view", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = [args.order_view.resolve(), args.dependency_view.resolve(), args.output.resolve()]
    if any(not path.is_relative_to(PRIVATE_ROOT.resolve()) for path in paths):
        parser.error("all views must remain under data/private_corem")
    if len(set(paths)) != 3 or paths[2].exists():
        parser.error("refusing to alias or overwrite a view")
    order = json.loads(paths[0].read_text())
    dependency = json.loads(paths[1].read_text())
    for key in ("split", "seed", "num_tasks", "selection_audit", "train_split_sha256"):
        if order.get(key) != dependency.get(key):
            raise ValueError(f"paired metadata mismatch: {key}")
    if order.get("split") != "train" or order.get("seed") != 2027:
        raise ValueError("control is restricted to the frozen seed-2027 train pilot")
    if order.get("method") != "bm25_strict_source_static_api_order_only" or dependency.get(
        "method"
    ) != "bm25_strict_source_static_api_order_plus_candidate_defuse":
        raise ValueError("unexpected paired methods")
    order_hints, dependency_hints = order["hints"], dependency["hints"]
    if set(order_hints) != set(dependency_hints) or len(order_hints) != order["num_tasks"]:
        raise ValueError("paired target keys differ")

    hints: dict[str, str] = {}
    total_edges = overlap_edges = 0
    for key in sorted(order_hints):
        hints[key], count, overlap = mirrored_hint(order_hints[key], dependency_hints[key])
        total_edges += count
        overlap_edges += overlap
    payload = {
        **order,
        "method": "bm25_strict_source_static_api_order_plus_mirrored_index_edges",
        "hints": hints,
        "control_origin_order_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        "control_origin_dependency_sha256": hashlib.sha256(paths[1].read_bytes()).hexdigest(),
        "control_builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "control_total_edges": total_edges,
        "control_edge_overlap_with_original": overlap_edges,
        "control_is_verified_false_graph": False,
        "target_ground_truth_loaded": False,
        "dev_or_test_task_loaded": False,
    }
    paths[2].parent.mkdir(parents=True, exist_ok=True)
    paths[2].write_text(json.dumps(payload, sort_keys=True) + "\n")
    print(
        f"Mirrored-index train control frozen for {len(hints)} targets; "
        f"{total_edges} candidate edges, {overlap_edges} also present in originals."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
