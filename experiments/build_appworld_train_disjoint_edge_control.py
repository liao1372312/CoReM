#!/usr/bin/env python3
"""Create a label-blind, disjoint static-edge information control on TRAIN.

Edges preserve forward occurrence order and the count per selected source,
but have no overlap with the source-extracted candidates. They are not
verified false runtime bindings or evidence of target-task utility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "data" / "private_corem"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.build_appworld_train_mirrored_edge_control import EDGE_MARKER  # noqa: E402


def disjoint_hint(order_hint: str, dependency_hint: str, seed: int) -> tuple[str, int]:
    prefix, separator, edge_text = dependency_hint.partition(EDGE_MARKER)
    if not separator or prefix != order_hint:
        raise ValueError("paired hints do not share an exact source/order prefix")
    numbered = re.findall(r"(?m)^(\d+)\. [A-Za-z_]\w*\.[A-Za-z_]\w*$", order_hint)
    count = len(numbered)
    if [int(value) for value in numbered] != list(range(1, count + 1)):
        raise ValueError("malformed numbered API occurrences")
    if edge_text == "None detected in this source.":
        original: set[tuple[int, int]] = set()
    else:
        if not re.fullmatch(r"\d+->\d+(?:, \d+->\d+)*", edge_text):
            raise ValueError("malformed original edge list")
        original = {
            tuple(map(int, token.split("->"))) for token in edge_text.split(", ")
        }
        if len(original) != len(edge_text.split(", ")):
            raise ValueError("duplicate original edge")
    legal = {
        (producer, consumer)
        for producer in range(1, count + 1)
        for consumer in range(producer + 1, count + 1)
    }
    if not original <= legal:
        raise ValueError("original edge is not a legal forward occurrence")
    available = sorted(legal - original)
    if len(available) < len(original):
        raise ValueError("no same-count disjoint control is possible")
    source_seed = seed ^ int(hashlib.sha256(order_hint.encode()).hexdigest()[:16], 16)
    chosen = sorted(random.Random(source_seed).sample(available, len(original)))
    if len(chosen) != len(original) or set(chosen) & original:
        raise AssertionError("disjoint edge construction failed")
    rendered = (
        ", ".join(f"{producer}->{consumer}" for producer, consumer in chosen)
        if chosen else "None detected in this source."
    )
    return order_hint + EDGE_MARKER + rendered, len(chosen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order-view", required=True, type=Path)
    parser.add_argument("--dependency-view", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2027)
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
    if order.get("split") != "train" or order.get("seed") != args.seed or args.seed != 2027:
        raise ValueError("control is restricted to the frozen train pilot")
    if order.get("method") != "bm25_strict_source_static_api_order_only" or dependency.get(
        "method"
    ) != "bm25_strict_source_static_api_order_plus_candidate_defuse":
        raise ValueError("unexpected paired methods")
    order_hints, dependency_hints = order["hints"], dependency["hints"]
    if set(order_hints) != set(dependency_hints) or len(order_hints) != 9:
        raise ValueError("paired target keys differ")
    hints: dict[str, str] = {}
    edge_count = 0
    for key in sorted(order_hints):
        hints[key], count = disjoint_hint(order_hints[key], dependency_hints[key], args.seed)
        edge_count += count
    payload = {
        **order,
        "method": "bm25_strict_source_static_api_order_plus_disjoint_index_edges",
        "hints": hints,
        "control_origin_order_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        "control_origin_dependency_sha256": hashlib.sha256(paths[1].read_bytes()).hexdigest(),
        "control_builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "control_total_edges": edge_count,
        "control_edge_overlap_with_original": 0,
        "control_is_verified_false_graph": False,
        "target_ground_truth_loaded": False,
        "dev_or_test_task_loaded": False,
    }
    paths[2].parent.mkdir(parents=True, exist_ok=True)
    paths[2].write_text(json.dumps(payload, sort_keys=True) + "\n")
    print(
        f"Disjoint-edge train control frozen for {len(hints)} targets; "
        f"{edge_count} forward edges, zero original-edge overlap."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
