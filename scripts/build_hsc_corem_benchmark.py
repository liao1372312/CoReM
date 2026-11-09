#!/usr/bin/env python3
"""Freeze an HSC benchmark for the expanded CoReM experiments.

The selector never receives ``labels.jsonl``.  Candidate identifiers are opaque,
and the positive candidate is shuffled with deterministic hard negatives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import write_checksums, write_json, write_jsonl  # noqa: E402

SEED = 20271022
SPLIT_COUNTS = {"train": 40, "validation": 12, "test": 12}
MAX_CANDIDATES = 20
QOS_FIELDS = (
    ("average number of downloads", "normalized_downloads", False),
    ("average number of likes", "normalized_likes", False),
    ("average response time", "normalized_response_times", True),
    ("average waiting time", "normalized_waiting_times", True),
    ("reliability", "reliabilities", False),
    ("successability", "successabilities", False),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def opaque_id(prefix: str, *parts: object) -> str:
    payload = "|".join(map(str, parts)).encode()
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:16]}"


def depth_band(depth: int) -> str:
    if depth in (2, 3):
        return "2-3"
    if depth in (4, 5):
        return "4-5"
    if depth in (6, 7):
        return "6-7"
    raise ValueError(f"unsupported workflow depth: {depth}")


def quality(row: dict[str, Any]) -> float:
    return (
        float(row["normalized_downloads"])
        + float(row["normalized_likes"])
        + (1.0 - float(row["normalized_response_times"]))
        + (1.0 - float(row["normalized_waiting_times"]))
        + float(row["reliabilities"])
        + float(row["successabilities"])
    ) / 6.0


def candidate_sequences(
    gold: tuple[int, ...],
    categories: tuple[str, ...],
    alternatives: dict[str, tuple[int, ...]],
    workflow: dict[str, Any],
    services: list[dict[str, Any]],
    rng: random.Random,
) -> list[tuple[int, ...]]:
    """Return gold plus 19 objective-near but strictly inferior candidates."""

    def objective(sequence: tuple[int, ...]) -> float:
        rows = [services[index] for index in sequence]
        value = 0.0
        for name, field, inverse in QOS_FIELDS:
            average = sum(float(row[field]) for row in rows) / len(rows)
            utility = 1.0 - average if inverse else average
            value += float(workflow["obj_func"][name]) * utility
        return value

    sequences = {gold}
    depth = len(gold)
    # Single-stage substitutions are the hardest controlled negatives.
    for stage in range(depth):
        for replacement in alternatives[categories[stage]][:8]:
            if replacement == gold[stage]:
                continue
            item = list(gold)
            item[stage] = replacement
            sequences.add(tuple(item))
    # Add multi-stage substitutions when shallow workflows have too few singles.
    attempts = 0
    while len(sequences) < 500 and attempts < 12000:
        attempts += 1
        item = list(gold)
        replace_count = rng.randint(1, min(depth, 3))
        for stage in rng.sample(range(depth), replace_count):
            pool = [
                value
                for value in alternatives[categories[stage]][:24]
                if value != gold[stage]
            ]
            if pool:
                item[stage] = rng.choice(pool)
        sequences.add(tuple(item))
    gold_value = objective(gold)
    negatives = [
        (objective(sequence), sequence)
        for sequence in sequences
        if sequence != gold and objective(sequence) < gold_value - 1e-6
    ]
    # Closest inferior compositions are controlled hard negatives.  The rule is
    # frozen before selecting the new held-out source indices.
    negatives.sort(key=lambda item: (-item[0], item[1]))
    if len(negatives) < MAX_CANDIDATES - 1:
        raise ValueError("could not generate 19 strictly inferior candidates")
    return [gold, *(sequence for _, sequence in negatives[: MAX_CANDIDATES - 1])]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--exclude-dataset",
        type=Path,
        help="Previously inspected frozen benchmark whose HSC source indices are excluded.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    source_files = {
        name: args.source / name
        for name in (
            "requirements.json",
            "workflow.json",
            "best_solution.json",
            "normalized_model.json",
        )
    }
    missing = [str(path) for path in source_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing HSC inputs: {missing}")
    requirements = json.loads(source_files["requirements.json"].read_text())
    workflows = json.loads(source_files["workflow.json"].read_text())
    solutions = json.loads(source_files["best_solution.json"].read_text())
    services = json.loads(source_files["normalized_model.json"].read_text())
    if not (len(requirements) == len(workflows) == len(solutions) == 10000):
        raise ValueError("unexpected HSC aligned-array lengths")

    by_function: dict[str, list[int]] = defaultdict(list)
    for index, service in enumerate(services):
        by_function[str(service["function"])].append(index)
    alternatives = {
        function: tuple(
            sorted(indices, key=lambda index: (-quality(services[index]), index))
        )
        for function, indices in by_function.items()
    }

    excluded_indices: set[int] = set()
    if args.exclude_dataset is not None:
        excluded_indices = {
            int(json.loads(line)["source_index"])
            for line in (args.exclude_dataset / "instances.jsonl").read_text().splitlines()
            if line.strip()
        }

    eligible: dict[str, list[int]] = defaultdict(list)
    for index, (workflow, solution) in enumerate(zip(workflows, solutions)):
        if index in excluded_indices:
            continue
        categories = tuple(
            str(category) for stage in workflow["flow"] for category in stage
        )
        gold = tuple(map(int, solution["best_solution"]))
        depth = len(categories)
        if depth < 2 or depth > 7 or len(gold) != depth:
            continue
        if any(services[item]["function"] != category for item, category in zip(gold, categories)):
            raise ValueError(f"HSC category/solution mismatch at {index}")
        # Some narrow HSC functions expose only one or two services.  Retain a
        # workflow when the Cartesian product still supports 20 distinct
        # complete compositions; no individual stage is required to have four.
        if math.prod(len(alternatives[category]) for category in categories) < MAX_CANDIDATES:
            continue
        eligible[depth_band(depth)].append(index)

    selected: dict[int, str] = {}
    frozen_sequences: dict[int, list[tuple[int, ...]]] = {}
    rng = random.Random(args.seed)
    for band in ("2-3", "4-5", "6-7"):
        pool = list(eligible[band])
        rng.shuffle(pool)
        cursor = 0
        # Choose test first, then validation, so later training-size changes cannot
        # alter either held-out manifest.
        for split in ("test", "validation", "train"):
            count = SPLIT_COUNTS[split]
            members = []
            while len(members) < count and cursor < len(pool):
                index = pool[cursor]
                cursor += 1
                workflow = workflows[index]
                categories = tuple(
                    str(category)
                    for stage in workflow["flow"]
                    for category in stage
                )
                gold = tuple(map(int, solutions[index]["best_solution"]))
                try:
                    sequences = candidate_sequences(
                        gold,
                        categories,
                        alternatives,
                        workflow,
                        services,
                        random.Random(f"{args.seed}:{index}"),
                    )
                except ValueError:
                    continue
                frozen_sequences[index] = sequences
                members.append(index)
            if len(members) != count:
                raise ValueError(f"insufficient eligible HSC workflows in {band}")
            selected.update({index: split for index in members})

    instance_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for source_index, split in sorted(selected.items()):
        workflow = workflows[source_index]
        solution = solutions[source_index]
        categories = tuple(
            str(category) for stage in workflow["flow"] for category in stage
        )
        gold = tuple(map(int, solution["best_solution"]))
        family_id = opaque_id("family", args.seed, source_index)
        family_rng = random.Random(f"{args.seed}:{source_index}")
        sequences = frozen_sequences[source_index]
        candidates = []
        for sequence in sequences:
            candidate_id = opaque_id("bundle", args.seed, source_index, *sequence)
            candidates.append(candidate_id)
            candidate_rows.append(
                {
                    "family_id": family_id,
                    "candidate_id": candidate_id,
                    "services": [
                        {
                            "stage": stage,
                            "service_index": service_index,
                            **services[service_index],
                        }
                        for stage, service_index in enumerate(sequence, start=1)
                    ],
                }
            )
        family_rng.shuffle(candidates)
        gold_candidate = opaque_id("bundle", args.seed, source_index, *gold)
        instance_rows.append(
            {
                "family_id": family_id,
                "source_index": source_index,
                "split": split,
                "depth": len(categories),
                "depth_band": depth_band(len(categories)),
                "requirement": str(requirements[source_index]),
                "categories": list(categories),
                "objective": workflow["obj_func"],
                "constraints": workflow["cons"],
                "candidate_ids": candidates,
            }
        )
        label_rows.append(
            {
                "family_id": family_id,
                "gold_candidate_id": gold_candidate,
                "best_value": float(solution["best_value"]),
            }
        )

    split_counter = Counter(row["split"] for row in instance_rows)
    band_counter = Counter(
        (row["split"], row["depth_band"]) for row in instance_rows
    )
    if split_counter != Counter({"train": 120, "validation": 36, "test": 36}):
        raise ValueError(f"unexpected split counts: {split_counter}")
    if any(value != SPLIT_COUNTS[split] for (split, _), value in band_counter.items()):
        raise ValueError(f"unbalanced depth bands: {band_counter}")
    if any(len(row["candidate_ids"]) != MAX_CANDIDATES for row in instance_rows):
        raise ValueError("candidate count invariant failed")
    if len({row["family_id"] for row in instance_rows}) != len(instance_rows):
        raise ValueError("family identifiers are not unique")

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "instances.jsonl", instance_rows)
    write_jsonl(args.output / "candidates.jsonl", candidate_rows)
    write_jsonl(args.output / "labels.jsonl", label_rows)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": "corem-hsc-expanded-v2",
            "dataset_name": "HSC CoReM frozen multi-candidate benchmark",
            "seed": args.seed,
            "source": str(args.source.resolve()),
            "source_sha256": {
                name: sha256(path) for name, path in source_files.items()
            },
            "excluded_source_indices": len(excluded_indices),
            "candidate_construction": (
                "Gold plus the 19 closest generated compositions with strictly "
                "lower documented weighted QoS utility; identifiers are then shuffled."
            ),
            "instances": len(instance_rows),
            "candidates_per_instance": MAX_CANDIDATES,
            "split_counts": dict(sorted(split_counter.items())),
            "depth_band_counts": {
                f"{split}:{band}": value
                for (split, band), value in sorted(band_counter.items())
            },
            "label_blinding": (
                "Gold candidate identifiers are stored only in labels.jsonl; "
                "selectors consume instances.jsonl and candidates.jsonl."
            ),
            "outcome_contract": (
                "A success is exact recovery of HSC's independently supplied "
                "best_solution, not live API execution."
            ),
            "license": "CC BY 4.0 (HSC source dataset)",
        },
    )
    (args.output / "README.md").write_text(
        "# Frozen HSC CoReM Benchmark\n\n"
        "This generated benchmark contains 120 training, 36 validation, and 36 "
        "test workflows, balanced over 2--3, 4--5, and 6--7 operations. Each "
        "workflow has 20 opaque, shuffled candidate compositions. The selector "
        "must not read `labels.jsonl`; evaluation uses it only after selection. "
        "Success means exact recovery of HSC's supplied optimal composition and "
        "does not imply live API execution.\n"
    )
    write_checksums(
        args.output,
        ("README.md", "candidates.jsonl", "instances.jsonl", "labels.jsonl", "manifest.json"),
    )
    print(
        f"wrote {len(instance_rows)} frozen HSC workflows and "
        f"{len(candidate_rows)} candidates to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
