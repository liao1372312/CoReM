"""Small reproducible demonstration runnable with ``python -m corem``."""

from __future__ import annotations

import json

from .model import SetUtilityModel
from .retention import GroupAwareRetention, RetentionConfig
from .retrieval import BundleBeamRetriever, RetrievalConfig
from .types import DependencyHyperedge, EvidenceGraph, MemoryRecord, Request


def main() -> None:
    graph = EvidenceGraph()
    graph.add_records(
        (
            MemoryRecord(
                "order-binding",
                "Map the created order identifier into the payment call",
                frozenset({"order_id_binding"}),
                frozenset({"create_order", "pay_order"}),
                token_cost=40,
                storage_cost=50,
                provenance="trace-17",
            ),
            MemoryRecord(
                "payment-state",
                "Shipment creation requires a captured payment state",
                frozenset({"payment_precondition"}),
                frozenset({"pay_order", "create_shipment"}),
                token_cost=45,
                storage_cost=55,
                provenance="trace-18",
            ),
            MemoryRecord(
                "unrelated",
                "A catalog lookup can filter products by color",
                frozenset({"catalog_filter"}),
                frozenset({"search_catalog"}),
                token_cost=30,
                storage_cost=35,
                provenance="trace-9",
            ),
        )
    )
    graph.add_hyperedge(
        DependencyHyperedge(
            "order-payment-chain",
            frozenset({"order-binding", "payment-state"}),
            frozenset({"order_id_binding", "payment_precondition"}),
            "binding-chain",
            "traces-17-18",
        )
    )
    request = Request(
        "ship-order",
        "Pay for an order and create its shipment",
        frozenset({"order_id_binding", "payment_precondition"}),
        operation_ids=frozenset({"pay_order", "create_shipment"}),
    )
    model = SetUtilityModel()
    retriever = BundleBeamRetriever(
        model,
        RetrievalConfig(retrieval_budget=100, beam_width=8, max_depth=3),
    )
    result = retriever.retrieve(request, frozenset(graph.records), graph)
    retainer = GroupAwareRetention(retriever, RetentionConfig(storage_budget=110))
    retained = retainer.optimize(
        frozenset({"unrelated"}),
        frozenset({"order-binding", "payment-state"}),
        (request,),
        graph,
    )
    print(
        json.dumps(
            {
                "retrieval": {
                    "status": result.status,
                    "selected": sorted(result.selected_ids),
                    "probability": round(result.probability, 4),
                    "objective": round(result.objective, 4),
                },
                "retention": {
                    "retained": sorted(retained.retained_ids),
                    "bank_value": round(retained.bank_value, 4),
                    "accepted_swaps": retained.accepted_swaps,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
