#!/usr/bin/env python3
"""Run CoReM-only E1, E2, and E7 on the frozen HSC benchmark."""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.encoding import StableHashEncoder  # noqa: E402
from corem.model import SetUtilityModel, TrainingConfig  # noqa: E402
from corem.real_eval import (  # noqa: E402
    load_json,
    load_jsonl,
    write_checksums,
    write_json,
    write_jsonl,
)
from corem.retrieval import BundleBeamRetriever, RetrievalConfig  # noqa: E402
from corem.types import (  # noqa: E402
    DependencyHyperedge,
    EvidenceEdge,
    EvidenceGraph,
    MemoryRecord,
    OutcomeExample,
    Request,
)

SEEDS = (17, 29, 41, 53, 67)
CANDIDATE_COUNTS = (2, 5, 10, 20)
TRAINING_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
TOKEN_RE = re.compile(r"[\w./:-]+", re.UNICODE)
QOS_FIELDS = (
    ("average number of downloads", "normalized_downloads", False),
    ("average number of likes", "normalized_likes", False),
    ("average response time", "normalized_response_times", True),
    ("average waiting time", "normalized_waiting_times", True),
    ("reliability", "reliabilities", False),
    ("successability", "successabilities", False),
)


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def verify_checksums(directory: Path) -> None:
    for line in (directory / "checksums.sha256").read_text().splitlines():
        digest, filename = line.split("  ", 1)
        observed = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        if observed != digest:
            raise ValueError(f"checksum mismatch: {directory / filename}")


def qualitative(value: float, *, inverse: bool = False) -> str:
    adjusted = 1.0 - value if inverse else value
    if adjusted >= 0.75:
        return "high"
    if adjusted >= 0.45:
        return "medium"
    return "low"


def service_text(service: dict[str, Any]) -> str:
    return (
        f"service {service['service_name']} provides {service['function']}; "
        f"downloads {qualitative(float(service['normalized_downloads']))}; "
        f"likes {qualitative(float(service['normalized_likes']))}; "
        f"response-speed {qualitative(float(service['normalized_response_times']), inverse=True)}; "
        f"waiting-speed {qualitative(float(service['normalized_waiting_times']), inverse=True)}; "
        f"reliability {qualitative(float(service['reliabilities']))}; "
        f"successability {qualitative(float(service['successabilities']))}."
    )


@dataclass(frozen=True)
class FamilyView:
    family_id: str
    request: Request
    candidate_ids: tuple[str, ...]
    members: dict[str, frozenset[str]]
    sequences: dict[str, tuple[int, ...]]
    gold_candidate_id: str
    split: str
    depth: int
    depth_band: str


def build_graph(
    instances: Iterable[dict[str, Any]],
    candidates_by_family: dict[str, list[dict[str, Any]]],
    labels: dict[str, str],
    *,
    graph_enabled: bool,
) -> tuple[EvidenceGraph, dict[str, FamilyView], dict[tuple[str, str], float]]:
    graph = EvidenceGraph()
    views: dict[str, FamilyView] = {}
    qos_scores: dict[tuple[str, str], float] = {}
    pending_edges: list[EvidenceEdge] = []
    pending_hyperedges: list[DependencyHyperedge] = []
    for instance in instances:
        family_id = str(instance["family_id"])
        categories = tuple(map(str, instance["categories"]))
        required_atoms = frozenset(
            f"stage:{stage}:{category}"
            for stage, category in enumerate(categories, start=1)
        )
        objective_text = " ".join(
            f"objective {key} weight {value}"
            for key, value in sorted(instance["objective"].items())
            if float(value) != 0.0
        )
        constraint_text = " ".join(
            f"constraint {key} threshold {value}"
            for key, value in sorted(instance["constraints"].items())
            if float(value) != 0.0
        )
        request = Request(
            request_id=family_id,
            text=f"{instance['requirement']} {objective_text} {constraint_text}",
            required_atoms=required_atoms,
            operation_ids=frozenset(categories),
            active_versions=frozenset((family_id,)),
        )
        members: dict[str, frozenset[str]] = {}
        sequences: dict[str, tuple[int, ...]] = {}
        rows_by_id = {
            str(row["candidate_id"]): row
            for row in candidates_by_family[family_id]
        }
        for candidate_id in map(str, instance["candidate_ids"]):
            candidate = rows_by_id[candidate_id]
            record_ids: list[str] = []
            sequence: list[int] = []
            for service in candidate["services"]:
                stage = int(service["stage"])
                category = str(service["function"])
                record_id = f"{candidate_id}:stage-{stage}"
                record_ids.append(record_id)
                sequence.append(int(service["service_index"]))
                graph.add_record(
                    MemoryRecord(
                        record_id=record_id,
                        text=service_text(service),
                        evidence_atoms=frozenset((f"stage:{stage}:{category}",)),
                        operation_ids=frozenset((category,)),
                        token_cost=24,
                        storage_cost=24,
                        provenance="HSC CC-BY-4.0 frozen candidate",
                        service_scope=frozenset((family_id,)),
                        version_scope=frozenset((family_id,)),
                    )
                )
                qos_scores[(family_id, record_id)] = sum(
                    float(instance["objective"][name])
                    * (
                        1.0 - float(service[field])
                        if inverse
                        else float(service[field])
                    )
                    for name, field, inverse in QOS_FIELDS
                )
            candidate_members = frozenset(record_ids)
            members[candidate_id] = candidate_members
            sequences[candidate_id] = tuple(sequence)
            pending_hyperedges.append(
                DependencyHyperedge(
                    edge_id=f"dependency:{candidate_id}",
                    members=candidate_members,
                    evidence_atoms=required_atoms,
                    relation_type="hsc-workflow-composition",
                    provenance="HSC workflow structure",
                    version_scope=frozenset((family_id,)),
                )
            )
            if graph_enabled:
                for left, right in zip(record_ids, record_ids[1:]):
                    pending_edges.append(
                        EvidenceEdge(
                            source=left,
                            target=right,
                            relation_type="workflow-next",
                            provenance="HSC workflow order",
                        )
                    )
        views[family_id] = FamilyView(
            family_id=family_id,
            request=request,
            candidate_ids=tuple(map(str, instance["candidate_ids"])),
            members=members,
            sequences=sequences,
            gold_candidate_id=labels[family_id],
            split=str(instance["split"]),
            depth=int(instance["depth"]),
            depth_band=str(instance["depth_band"]),
        )
    for edge in pending_edges:
        graph.add_edge(edge)
    for edge in pending_hyperedges:
        graph.add_hyperedge(edge)
    return graph, views, qos_scores


def training_examples(
    views: dict[str, FamilyView],
    *,
    seed: int,
    fraction: float,
    set_enabled: bool,
) -> list[OutcomeExample]:
    rng = random.Random(seed)
    by_band: dict[str, list[FamilyView]] = defaultdict(list)
    for view in views.values():
        if view.split == "train":
            by_band[view.depth_band].append(view)
    selected: list[FamilyView] = []
    for band in sorted(by_band):
        members = sorted(by_band[band], key=lambda item: item.family_id)
        rng.shuffle(members)
        keep = max(1, round(len(members) * fraction))
        selected.extend(members[:keep])
    examples: list[OutcomeExample] = []
    for view in selected:
        negative_ids = [
            candidate_id
            for candidate_id in view.candidate_ids
            if candidate_id != view.gold_candidate_id
        ]
        negative = negative_ids[rng.randrange(len(negative_ids))]
        for candidate_id, outcome in ((view.gold_candidate_id, 1), (negative, 0)):
            examples.append(
                OutcomeExample(
                    request=view.request,
                    selected_ids=view.members[candidate_id],
                    outcome=outcome,
                    eligible_for_interaction_training=set_enabled,
                )
            )
    rng.shuffle(examples)
    return examples


class LexicalResidualModel:
    """Fixed .9 model/.1 lexical mixture used by the selected CoReM variant."""

    def __init__(self, base: SetUtilityModel, weight: float) -> None:
        self.base = base
        self.weight = weight

    def semantic_similarity(self, request: Request, record_id: str, graph: EvidenceGraph) -> float:
        return self.base.semantic_similarity(request, record_id, graph)

    @staticmethod
    def _lexical(request: Request, selected: Iterable[str], graph: EvidenceGraph) -> float:
        query = set(TOKEN_RE.findall(request.text.casefold()))
        document: set[str] = set()
        for record_id in selected:
            document.update(TOKEN_RE.findall(graph.records[record_id].text.casefold()))
        if not query or not document:
            return 0.0
        overlap = len(query.intersection(document)) / math.sqrt(len(query) * len(document))
        return 1.0 / (1.0 + math.exp(-8.0 * (overlap - 0.08)))

    def probability(self, request: Request, selected: Iterable[str], graph: EvidenceGraph) -> float:
        selected_set = frozenset(selected)
        base = self.base.probability(request, selected_set, graph)
        lexical = self._lexical(request, selected_set, graph)
        return (1.0 - self.weight) * base + self.weight * lexical

    def probabilities(
        self,
        request: Request,
        selected_sets: Sequence[frozenset[str]],
        graph: EvidenceGraph,
    ) -> tuple[float, ...]:
        return tuple(self.probability(request, selected, graph) for selected in selected_sets)

    def relaxed_probability_and_gradient(
        self, request: Request, inclusion: dict[str, float], graph: EvidenceGraph
    ) -> tuple[float, dict[str, float]]:
        return self.base.relaxed_probability_and_gradient(request, inclusion, graph)


class QoSAwareSetUtilityModel(SetUtilityModel):
    """Set utility with outcome-independent HSC QoS evidence in its encoder score."""

    def __init__(
        self,
        *,
        encoder: StableHashEncoder,
        qos_scores: dict[tuple[str, str], float],
    ) -> None:
        super().__init__(encoder=encoder)
        self.qos_scores = qos_scores

    def semantic_similarity(
        self, request: Request, record_id: str, graph: EvidenceGraph
    ) -> float:
        lexical_graph = super().semantic_similarity(request, record_id, graph)
        qos = self.qos_scores[(request.request_id, record_id)]
        # HSC objective weights sum to one. Map their documented utility to the
        # cosine range and retain a small graph/text channel.
        return 0.85 * (2.0 * qos - 1.0) + 0.15 * lexical_graph


def fit_model(
    graph: EvidenceGraph,
    views: dict[str, FamilyView],
    *,
    seed: int,
    graph_enabled: bool,
    set_enabled: bool,
    lexical_enabled: bool,
    training_fraction: float,
    qos_scores: dict[tuple[str, str], float],
) -> tuple[LexicalResidualModel, dict[str, Any]]:
    rng = random.Random(seed)
    base = QoSAwareSetUtilityModel(
        encoder=StableHashEncoder(message_layers=2 if graph_enabled else 0),
        qos_scores=qos_scores,
    )
    for weights in (
        base.base_weights,
        base.unary_weights,
        base.complement_weights,
        base.redundancy_weights,
    ):
        for index in range(len(weights)):
            weights[index] += rng.uniform(-0.02, 0.02)
    examples = training_examples(
        views,
        seed=seed,
        fraction=training_fraction,
        set_enabled=set_enabled,
    )
    started = time.perf_counter()
    fit = base.fit(
        examples,
        graph,
        TrainingConfig(
            learning_rate=0.03,
            epochs=32,
            calibration_weight=0.1,
            relation_weight=0.05 if graph_enabled else 0.0,
            interaction_weight=0.001 if set_enabled else 0.0,
            l2_weight=0.0001,
            patience=8,
        ),
    )
    if not set_enabled:
        # Closure and candidate exposure remain fixed; only learned set utility is removed.
        base.complement_weights[:] = [-40.0, 0.0, 0.0, 0.0, 0.0]
    model = LexicalResidualModel(base, 0.1 if lexical_enabled else 0.0)
    return model, {
        "training_examples": len(examples),
        "fit_epochs": fit.epochs,
        "fit_converged": fit.converged,
        "final_loss": fit.losses[-1],
        "fit_seconds": time.perf_counter() - started,
    }


def candidate_pool(view: FamilyView, count: int) -> tuple[str, ...]:
    negatives = [item for item in view.candidate_ids if item != view.gold_candidate_id]
    chosen = [view.gold_candidate_id, *negatives[: count - 1]]
    # Opaque deterministic ordering prevents the positive from occupying a fixed position.
    return tuple(sorted(chosen, key=lambda item: hashlib.sha256(item.encode()).hexdigest()))


def evaluate(
    model: LexicalResidualModel,
    graph: EvidenceGraph,
    views: dict[str, FamilyView],
    *,
    candidate_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for view in sorted(views.values(), key=lambda item: item.family_id):
        if view.split != "test":
            continue
        pool = candidate_pool(view, candidate_count)
        bank = frozenset(record_id for item in pool for record_id in view.members[item])
        retriever = BundleBeamRetriever(
            model,  # type: ignore[arg-type]
            RetrievalConfig(
                retrieval_budget=view.depth * 24,
                beam_width=8,
                max_depth=2,
                candidate_pool_size=200,
                dependency_hops=2,
                max_closure_size=7,
                enforce_dependency_closure=True,
                context_cost_weight=0.05,
            ),
        )
        started = time.perf_counter()
        result = retriever.retrieve(view.request, bank, graph)
        latency_ms = (time.perf_counter() - started) * 1000.0
        selected_sequence = None
        selected_candidate = None
        for candidate_id in pool:
            if result.selected_ids == view.members[candidate_id]:
                selected_candidate = candidate_id
                selected_sequence = view.sequences[candidate_id]
                break
        gold_sequence = view.sequences[view.gold_candidate_id]
        success = int(selected_sequence == gold_sequence)
        candidate_probabilities = {
            candidate_id: model.probability(
                view.request, view.members[candidate_id], graph
            )
            for candidate_id in pool
        }
        brier = statistics.mean(
            (
                probability
                - float(candidate_id == view.gold_candidate_id)
            )
            ** 2
            for candidate_id, probability in candidate_probabilities.items()
        )
        top1 = max(
            candidate_probabilities,
            key=lambda item: (candidate_probabilities[item], item),
        )
        rows.append(
            {
                "family_id": view.family_id,
                "depth": view.depth,
                "depth_band": view.depth_band,
                "candidate_count": candidate_count,
                "selected_candidate_id": selected_candidate,
                "gold_candidate_id": view.gold_candidate_id,
                "verified_optimum_recovery": success,
                "evidence_complete_recall": success,
                "top1_correct": int(top1 == view.gold_candidate_id),
                "brier": brier,
                "token_cost": result.token_cost,
                "selection_latency_ms": latency_ms,
                "retrieval_status": result.status,
                "evaluations": result.evaluations,
            }
        )
    if len(rows) != 36:
        raise ValueError(f"expected 36 test families, observed {len(rows)}")
    latencies = [float(row["selection_latency_ms"]) for row in rows]
    summary = {
        "families": len(rows),
        "candidate_count": candidate_count,
        "vsr": statistics.mean(int(row["verified_optimum_recovery"]) for row in rows),
        "ecr": statistics.mean(int(row["evidence_complete_recall"]) for row in rows),
        "top1": statistics.mean(int(row["top1_correct"]) for row in rows),
        "brier": statistics.mean(float(row["brier"]) for row in rows),
        "mean_tokens": statistics.mean(int(row["token_cost"]) for row in rows),
        "selection_p50_ms": quantile(latencies, 0.50),
        "selection_p95_ms": quantile(latencies, 0.95),
        "fallback_rate": statistics.mean(row["retrieval_status"] != "ok" for row in rows),
    }
    return summary, rows


def aggregate(records: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for group, members in sorted(grouped.items()):
        item = {key: value for key, value in zip(keys, group)}
        for metric in ("vsr", "ecr", "top1", "brier", "mean_tokens", "selection_p50_ms", "selection_p95_ms", "fallback_rate"):
            values = [float(row[metric]) for row in members]
            item[f"mean_{metric}"] = statistics.mean(values)
            item[f"sd_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
        item["seeds"] = len(members)
        output.append(item)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}")
        return 2
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    verify_checksums(args.dataset)
    manifest = load_json(args.dataset / "manifest.json")
    if manifest.get("schema_version") not in {
        "corem-hsc-expanded-v1",
        "corem-hsc-expanded-v2",
    }:
        raise ValueError("unexpected benchmark schema")
    instances = load_jsonl(args.dataset / "instances.jsonl")
    candidates_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(args.dataset / "candidates.jsonl"):
        candidates_by_family[str(row["family_id"])].append(row)
    labels = {
        str(row["family_id"]): str(row["gold_candidate_id"])
        for row in load_jsonl(args.dataset / "labels.jsonl")
    }
    if set(labels) != {str(row["family_id"]) for row in instances}:
        raise ValueError("label coverage mismatch")

    train_instances = []
    test_instances = []
    for instance in instances:
        if instance["split"] == "train":
            family_id = str(instance["family_id"])
            gold = labels[family_id]
            negatives = [
                item for item in instance["candidate_ids"] if item != gold
            ]
            # Fitting uses one executed positive/negative pair per family.  The
            # other candidate records are not exposed to the fitted graph.
            train_instances.append(
                {**instance, "candidate_ids": [gold, negatives[0]]}
            )
        elif instance["split"] == "test":
            test_instances.append(instance)
    fit_graphs = {}
    fit_views_by_graph = {}
    qos_by_graph = {}
    eval_graphs = {}
    eval_views_by_graph = {}
    for enabled in (False, True):
        fit_graph, fit_views, fit_qos = build_graph(
            train_instances,
            candidates_by_family,
            labels,
            graph_enabled=enabled,
        )
        eval_graph, eval_views, eval_qos = build_graph(
            test_instances,
            candidates_by_family,
            labels,
            graph_enabled=enabled,
        )
        fit_graphs[enabled] = fit_graph
        fit_views_by_graph[enabled] = fit_views
        eval_graphs[enabled] = eval_graph
        eval_views_by_graph[enabled] = eval_views
        qos_by_graph[enabled] = {**fit_qos, **eval_qos}

    all_records: list[dict[str, Any]] = []
    family_records: list[dict[str, Any]] = []
    model_cache: dict[tuple[int, bool, bool, bool, float], LexicalResidualModel] = {}
    fit_cache: dict[tuple[int, bool, bool, bool, float], dict[str, Any]] = {}

    def run_configuration(
        *,
        seed: int,
        graph_enabled: bool,
        set_enabled: bool,
        lexical_enabled: bool,
        fraction: float,
        counts: Sequence[int],
        experiment: str,
    ) -> None:
        key = (seed, graph_enabled, set_enabled, lexical_enabled, fraction)
        if key not in model_cache:
            model, fit_info = fit_model(
                fit_graphs[graph_enabled],
                fit_views_by_graph[graph_enabled],
                seed=seed,
                graph_enabled=graph_enabled,
                set_enabled=set_enabled,
                lexical_enabled=lexical_enabled,
                training_fraction=fraction,
                qos_scores=qos_by_graph[graph_enabled],
            )
            model_cache[key] = model
            fit_cache[key] = fit_info
        for count in counts:
            summary, rows = evaluate(
                model_cache[key],
                eval_graphs[graph_enabled],
                eval_views_by_graph[graph_enabled],
                candidate_count=count,
            )
            record = {
                "experiment": experiment,
                "seed": seed,
                "graph": graph_enabled,
                "set_term": set_enabled,
                "lexical": lexical_enabled,
                "training_fraction": fraction,
                **fit_cache[key],
                **summary,
            }
            all_records.append(record)
            for row in rows:
                family_records.append({**record, **row})

    # E1: full CoReM over nested candidate inventories.
    for seed in args.seeds:
        run_configuration(
            seed=seed,
            graph_enabled=True,
            set_enabled=True,
            lexical_enabled=True,
            fraction=1.0,
            counts=CANDIDATE_COUNTS,
            experiment="E1",
        )
    # E2: all 2x2x2 CoReM component configurations on the 20-candidate test.
    for seed in args.seeds:
        for graph_enabled in (True, False):
            for set_enabled in (True, False):
                for lexical_enabled in (True, False):
                    run_configuration(
                        seed=seed,
                        graph_enabled=graph_enabled,
                        set_enabled=set_enabled,
                        lexical_enabled=lexical_enabled,
                        fraction=1.0,
                        counts=(20,),
                        experiment="E2",
                    )
    # E7: full CoReM at four fractions; the 100% model is reused exactly.
    for seed in args.seeds:
        for fraction in TRAINING_FRACTIONS:
            run_configuration(
                seed=seed,
                graph_enabled=True,
                set_enabled=True,
                lexical_enabled=True,
                fraction=fraction,
                counts=(20,),
                experiment="E7",
            )

    e1 = aggregate([row for row in all_records if row["experiment"] == "E1"], ("candidate_count",))
    e2 = aggregate(
        [row for row in all_records if row["experiment"] == "E2"],
        ("graph", "set_term", "lexical"),
    )
    e7 = aggregate(
        [row for row in all_records if row["experiment"] == "E7"],
        ("training_fraction",),
    )
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "run_summaries.jsonl", all_records)
    write_jsonl(args.output / "family_outcomes.jsonl", family_records)
    write_json(
        args.output / "result.json",
        {
            "status": "complete_corem_only_e1_e2_e7",
            "schema_version": "corem-expanded-hsc-results-v1",
            "dataset": str(args.dataset.resolve()),
            "dataset_checksums_sha256": hashlib.sha256(
                (args.dataset / "checksums.sha256").read_bytes()
            ).hexdigest(),
            "seeds": args.seeds,
            "test_families": 36,
            "e1": e1,
            "e2": e2,
            "e7": e7,
            "outcome_contract": manifest["outcome_contract"],
            "claim_boundary": (
                "Offline optimal-composition recovery on HSC. Results are not "
                "live API executions and must not be labeled VSR without this qualifier."
            ),
        },
    )
    write_checksums(
        args.output, ("family_outcomes.jsonl", "result.json", "run_summaries.jsonl")
    )
    print(f"completed E1/E2/E7 CoReM runs in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
