#!/usr/bin/env python3
"""Run CoReM-only selector stress tests for expanded experiment E5.

These are controlled evidence/query perturbations over the frozen v5 public
service challenge.  Success is exact recovery of the already execution-verified
workflow bundle; this script does not claim live execution of a changed API.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.attestation import partition_attested_hyperedges  # noqa: E402
from corem.model import SetUtilityModel  # noqa: E402
from corem.real_eval import (  # noqa: E402
    audit_real_dataset,
    load_jsonl,
    parse_utc,
    write_checksums,
    write_json,
    write_jsonl,
)
from corem.retrieval import BundleBeamRetriever, RetrievalConfig  # noqa: E402
from corem.types import Request  # noqa: E402
from experiments.generate_public_challenge_selections import (  # noqa: E402
    GENERIC_REQUIRED_ATOMS,
)
from experiments.generate_public_workflow_selections import make_graph  # noqa: E402
from experiments.run_online_retention import active_archive_ids  # noqa: E402

SEEDS = (17, 29, 41, 53, 67)
PERCENTAGES = (10, 20, 30)


def selected_by_hash(label: str, seed: int, percent: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100 < percent


def alias_query(text: str, seed: int) -> str:
    aliases = {
        "create": "provision",
        "call": "invoke",
        "status": "result-code",
        "mapping": "route-spec",
        "expectation": "request-rule",
        "post": "entry",
        "title": "heading",
        "identifier": "runtime-key",
        "header": "metadata-field",
    }
    words = text.split()
    rng = random.Random(f"schema:{seed}:{text}")
    output = []
    for word in words:
        stripped = word.strip(".,:;?!").casefold()
        replacement = aliases.get(stripped)
        if replacement is not None and rng.random() < 0.8:
            suffix = word[len(word.rstrip(".,:;?!")) :]
            output.append(replacement + suffix)
        else:
            output.append(word)
    return " ".join(output)


def perturb_hyperedges(
    rows: list[dict[str, Any]],
    memories: list[dict[str, Any]],
    condition: str,
    seed: int,
) -> list[dict[str, Any]]:
    output = copy.deepcopy(rows)
    if condition in {"clean", "schema_alias", "verified_outage"}:
        return output
    if condition.startswith("missing_"):
        percent = int(condition.rsplit("_", 1)[1])
        return [
            row
            for row in output
            if not selected_by_hash(str(row["edge_id"]), seed, percent)
        ]
    memories_by_prefix: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for memory in memories:
        memory_id = str(memory["memory_id"])
        parts = memory_id.rsplit(":", 2)
        if len(parts) != 3:
            continue
        prefix, variant, _ = parts
        memories_by_prefix[prefix][variant].append(memory_id)
    if condition.startswith("spurious_"):
        percent = int(condition.rsplit("_", 1)[1])
        edge_by_prefix = {
            str(row["edge_id"])
            .replace("dependency:", "", 1)
            .rsplit(":correct", 1)[0]: row
            for row in rows
        }
        for prefix, variants in sorted(memories_by_prefix.items()):
            for variant in ("decoy1", "decoy2"):
                members = sorted(variants.get(variant, ()))
                if len(members) != 4 or not selected_by_hash(
                    f"{prefix}:{variant}", seed, percent
                ):
                    continue
                template = edge_by_prefix.get(prefix)
                if template is None:
                    continue
                output.append(
                    {
                        "edge_id": f"stress:{seed}:{prefix}:{variant}",
                        "members": members,
                        # Copy trusted metadata so the defense cannot identify
                        # corruptions from a synthetic relation/provenance label.
                        "evidence_atoms": list(template["evidence_atoms"]),
                        "relation_type": str(template["relation_type"]),
                        "provenance": str(template["provenance"]),
                        "verified": True,
                        "version_scope": list(template["version_scope"]),
                        "valid_from": str(template["valid_from"]),
                        "valid_until": template.get("valid_until"),
                    }
                )
        return output
    if condition in {"binding_conflict", "binding_rewrite"}:
        rewritten = []
        for row in output:
            item = copy.deepcopy(row)
            prefix = str(item["edge_id"]).replace("dependency:", "").rsplit(":correct", 1)[0]
            variants = memories_by_prefix.get(prefix, {})
            decoy_steps = sorted(variants.get("decoy1", ()))
            if len(decoy_steps) == 4:
                members = list(map(str, item["members"]))
                correct_step2 = next((value for value in members if value.endswith(":step2")), None)
                decoy_step2 = next((value for value in decoy_steps if value.endswith(":step2")), None)
                if correct_step2 and decoy_step2:
                    members[members.index(correct_step2)] = decoy_step2
                    item["members"] = members
                    item["edge_id"] = f"rewrite:{seed}:{item['edge_id']}"
                    item["provenance"] = "controlled E5 binding rewrite"
            rewritten.append(item)
        return output + rewritten if condition == "binding_conflict" else rewritten
    return output


def exact_gold_ids(event: dict[str, Any], memories: list[dict[str, Any]]) -> frozenset[str]:
    prefix = (
        f"challenge-v5:{event['service_id']}:{event['version_id']}:"
        f"{event['task_key']}:correct:"
    )
    return frozenset(
        str(row["memory_id"])
        for row in memories
        if str(row["memory_id"]).startswith(prefix)
    )


def request_for(event: dict[str, Any], text: str) -> Request:
    return Request(
        request_id=str(event["event_id"]),
        text=text,
        required_atoms=GENERIC_REQUIRED_ATOMS,
        timestamp=parse_utc(event["event_time"], "event_time").timestamp(),
        allowed_scopes=frozenset((str(event["service_id"]),)),
        active_versions=frozenset((str(event["version_id"]),)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "tune", "validation", "test"),
        default="test",
        help="Frozen chronological partition to evaluate (default: test).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    audit = audit_real_dataset(args.dataset)
    if not audit.eligible:
        raise ValueError(f"dataset audit failed: {audit.errors}")
    split_by_event = {
        str(row["event_id"]): str(row["split"])
        for row in load_jsonl(args.splits)
    }
    events = [
        row
        for row in load_jsonl(args.dataset / "events.jsonl")
        if split_by_event.get(str(row["event_id"])) == args.split
    ]
    if not events:
        raise ValueError(f"split contains no events: {args.split}")
    memories = load_jsonl(args.dataset / "memories.jsonl")
    hyperedges = load_jsonl(args.dataset / "dependency_hyperedges.jsonl")
    conditions = [
        "clean",
        "schema_alias",
        "binding_conflict",
        "binding_rewrite",
        "verified_outage",
        *(f"missing_{percent}" for percent in PERCENTAGES),
        *(f"spurious_{percent}" for percent in PERCENTAGES),
    ]
    rows: list[dict[str, Any]] = []
    condition_audits: list[dict[str, Any]] = []
    for condition in conditions:
        for seed in SEEDS:
            perturbed = perturb_hyperedges(hyperedges, memories, condition, seed)
            attested, rejected = partition_attested_hyperedges(memories, perturbed)
            corrupt_ids = {
                str(row["edge_id"])
                for row in perturbed
                if str(row["edge_id"]).startswith(("stress:", "rewrite:"))
            }
            rejected_ids = {str(row["edge_id"]) for row in rejected}
            condition_audits.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "input_edges": len(perturbed),
                    "attested_edges": len(attested),
                    "rejected_edges": len(rejected),
                    "removed_edges": max(0, len(hyperedges) - len(perturbed)),
                    "corrupt_edges": len(corrupt_ids),
                    "rejected_corrupt_edges": len(corrupt_ids.intersection(rejected_ids)),
                    "rejection_reasons": {
                        reason: sum(row["reason"] == reason for row in rejected)
                        for reason in sorted({row["reason"] for row in rejected})
                    },
                }
            )
            graph = make_graph(memories, attested)
            retriever = BundleBeamRetriever(
                SetUtilityModel(),
                RetrievalConfig(
                    retrieval_budget=160,
                    beam_width=16,
                    max_depth=4,
                    candidate_pool_size=100,
                    dependency_hops=2,
                    max_closure_size=8,
                    enforce_dependency_closure=True,
                    require_hyperedge_seed=True,
                ),
            )
            for event in events:
                event_id = str(event["event_id"])
                unavailable = condition == "verified_outage" and selected_by_hash(
                    event_id, seed, 25
                )
                text = (
                    alias_query(str(event["query"]), seed)
                    if condition == "schema_alias"
                    else str(event["query"])
                )
                request = request_for(event, text)
                active = active_archive_ids(memories, request.timestamp)
                bank = frozenset() if unavailable else active
                started = time.perf_counter()
                result = retriever.retrieve(request, bank, graph)
                latency_ms = (time.perf_counter() - started) * 1000.0
                gold = exact_gold_ids(event, memories)
                if len(gold) != 4:
                    raise ValueError(f"gold bundle coverage mismatch: {event_id}")
                # Replacement removes the only valid binding path.  Exact
                # recovery is not a meaningful endpoint there; safe refusal is.
                recovery_applicable = not unavailable and condition != "binding_rewrite"
                safe_rejection_applicable = unavailable or condition == "binding_rewrite"
                success = int(recovery_applicable and result.selected_ids == gold)
                gold_attested = any(
                    frozenset(map(str, edge["members"])) == gold for edge in attested
                )
                rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "event_id": event_id,
                        "oracle_feasible": recovery_applicable,
                        "gold_attested": gold_attested,
                        "recovery_applicable": recovery_applicable,
                        "safe_rejection_applicable": safe_rejection_applicable,
                        "safe_rejection": int(
                            safe_rejection_applicable and result.status == "fallback"
                        ),
                        "selected_memory_ids": sorted(result.selected_ids),
                        "gold_memory_ids": sorted(gold),
                        "verified_bundle_recovery": success,
                        "sur": None,
                        "fallback": int(result.status == "fallback"),
                        "selection_latency_ms": latency_ms,
                    }
                )
    summaries = []
    for condition in conditions:
        seed_summaries = []
        for seed in SEEDS:
            members = [
                row
                for row in rows
                if row["condition"] == condition and row["seed"] == seed
            ]
            feasible = [row for row in members if row["recovery_applicable"]]
            safety = [row for row in members if row["safe_rejection_applicable"]]
            audit = next(
                row
                for row in condition_audits
                if row["condition"] == condition and row["seed"] == seed
            )
            seed_summaries.append(
                {
                    "opt": (
                        statistics.mean(
                            int(row["verified_bundle_recovery"]) for row in feasible
                        )
                        if feasible
                        else None
                    ),
                    "safe_rejection": (
                        statistics.mean(int(row["safe_rejection"]) for row in safety)
                        if safety
                        else None
                    ),
                    "fallback": statistics.mean(int(row["fallback"]) for row in members),
                    "feasible_n": len(feasible),
                    "corrupt_edge_rejection": (
                        audit["rejected_corrupt_edges"] / audit["corrupt_edges"]
                        if audit["corrupt_edges"]
                        else None
                    ),
                }
            )
        opt_values = [row["opt"] for row in seed_summaries if row["opt"] is not None]
        safe_values = [
            row["safe_rejection"]
            for row in seed_summaries
            if row["safe_rejection"] is not None
        ]
        rejection_values = [
            row["corrupt_edge_rejection"]
            for row in seed_summaries
            if row["corrupt_edge_rejection"] is not None
        ]
        summaries.append(
            {
                "condition": condition,
                "seeds": len(SEEDS),
                "mean_opt": statistics.mean(opt_values) if opt_values else None,
                "sd_opt": (
                    statistics.stdev(opt_values) if len(opt_values) > 1 else None
                ),
                "mean_safe_rejection": (
                    statistics.mean(safe_values) if safe_values else None
                ),
                "sd_safe_rejection": (
                    statistics.stdev(safe_values) if len(safe_values) > 1 else None
                ),
                "mean_corrupt_edge_rejection": (
                    statistics.mean(rejection_values) if rejection_values else None
                ),
                "mean_fallback": statistics.mean(
                    row["fallback"] for row in seed_summaries
                ),
                "mean_feasible_n": statistics.mean(
                    row["feasible_n"] for row in seed_summaries
                ),
            }
        )
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "event_results.jsonl", rows)
    write_json(
        args.output / "result.json",
        {
            "status": "complete_corem_e5",
            "schema_version": "corem-expanded-e5-v2-attested",
            "dataset": str(args.dataset.resolve()),
            "splits": str(args.splits.resolve()),
            "test_events": len(events),
            "split": args.split,
            "seeds": list(SEEDS),
            "summaries": summaries,
            "condition_audits": condition_audits,
            "outcome_contract": (
                "Exact recovery of the workflow bundle already verified by live "
                "reset-service execution. Outage events are excluded from Opt.; "
                "binding replacement is evaluated by safe refusal because the "
                "original binding path is structurally unavailable."
            ),
            "attestation_contract": (
                "A workflow edge is admitted only when one verification contract "
                "grounds every declared runtime handoff in an upstream capture and "
                "a target-step placeholder. Edge identifiers, relation labels, and "
                "the asserted verified flag are not sufficient."
            ),
            "claim_boundary": (
                "Controlled selector-side query, dependency-graph, and availability "
                "stress. Opt. reuses prior execution-verified gold bundles; safe "
                "refusal is selector-side and no changed API binary is executed. "
                "SUR is not measured in E5."
            ),
        },
    )
    write_checksums(args.output, ("event_results.jsonl", "result.json"))
    print(f"completed {len(conditions)} E5 conditions over {len(events)} events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
