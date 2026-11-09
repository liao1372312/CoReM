#!/usr/bin/env python3
"""Build unscored v8 candidate families for real-service validity staging.

No model or baseline scores are read. Private oracle/negative assignments are
written separately from the label-free selector view.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import os
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import write_checksums, write_json, write_jsonl  # noqa: E402

BASE_TIME = datetime(2026, 10, 18, 14, 0, tzinfo=UTC)
VALID_FROM = "2026-10-17T00:00:00Z"
FAMILY_SOURCE = {
    "qdrant_snapshot": "https://api.qdrant.tech/v-1-15-x/api-reference/snapshots/create-snapshot",
    "qdrant_collection_update": "https://api.qdrant.tech/v-1-15-x/api-reference/collections/update-collection",
    "qdrant_payload_set": "https://api.qdrant.tech/v-1-15-x/api-reference/points/set-payload",
    "qdrant_payload_delete": "https://api.qdrant.tech/v-1-15-x/api-reference/points/delete-payload",
    "qdrant_vector_update": "https://api.qdrant.tech/v-1-15-x/api-reference/points/update-vectors",
    "qdrant_filtered_count": "https://api.qdrant.tech/v-1-15-x/api-reference/points/count-points",
    "qdrant_point_delete": "https://api.qdrant.tech/v-1-15-x/api-reference/points/delete-points",
    "qdrant_payload_facet": "https://api.qdrant.tech/v-1-15-x/api-reference/points/facet",
    "openbao_soft_delete": "https://openbao.org/docs/secrets/kv/kv-v2/",
    "openbao_undelete": "https://openbao.org/docs/secrets/kv/kv-v2/",
    "openbao_destroy": "https://openbao.org/docs/secrets/kv/kv-v2/",
    "openbao_cas_swap": "https://openbao.org/docs/secrets/kv/kv-v2/",
    "openbao_kv_patch": "https://openbao.org/docs/secrets/kv/kv-v2/",
    "openbao_custom_metadata": "https://openbao.org/docs/secrets/kv/kv-v2/",
    "openbao_transit_encrypt": "https://openbao.org/docs/secrets/transit/",
    "openbao_transit_rewrap": "https://openbao.org/docs/secrets/transit/",
}
VERSIONS = {"qdrant": "1.15.5", "openbao": "2.7.0"}


def _opaque(salt: bytes, event_id: str, side: int) -> str:
    digest = hmac.new(salt, f"candidate|{event_id}|{side}".encode(), hashlib.sha256).hexdigest()
    return f"{event_id}:candidate-{digest[:24]}"


def _positive_sides(salt: bytes, family: str) -> list[int]:
    digest = hmac.new(salt, f"assignment|{family}".encode(), hashlib.sha256).digest()
    sides = [0] * 4 + [1] * 4
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(sides)
    return sides


def _case(
    family: str, number: int, *, positive: bool
) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
    suffix = f"{number:02d}"
    if family == "qdrant_snapshot":
        target = f"v8snap_target_{suffix}"
        decoy = f"v8snap_decoy_{suffix}"
        chosen = target if positive else decoy
        query = f"Create collections {target} and {decoy}; snapshot {target} and inspect its snapshot list."
        plans = [
            {"method": "PUT", "path": f"/collections/{target}", "json": {"vectors": {"size": 2, "distance": "Cosine"}}},
            {"method": "PUT", "path": f"/collections/{decoy}", "json": {"vectors": {"size": 2, "distance": "Cosine"}}},
            {"method": "POST", "path": f"/collections/{chosen}/snapshots"},
            {"method": "GET", "path": f"/collections/{target}/snapshots"},
        ]
        contract = {
            "kind": "assertions_v1",
            "trace_statuses": [200, 200, 200, 200],
            "trace_assertions": [],
            "state_checks": [
                {
                    "path": f"/collections/{target}/snapshots",
                    "status": 200,
                    "assertions": [{"path": ["result"], "nonempty": True}],
                },
                {
                    "path": f"/collections/{decoy}/snapshots",
                    "status": 200,
                    "assertions": [{"path": ["result"], "equals": []}],
                },
            ],
        }
        return "qdrant", query, plans, contract
    if family == "qdrant_collection_update":
        collection = f"v8cfg_{suffix}"
        desired = 24000 + number * 100
        offered = desired if positive else desired + 5000
        query = f"Create Qdrant collection {collection}, set its optimizer indexing threshold to {desired}, then inspect the collection configuration."
        plans = [
            {"method": "PUT", "path": f"/collections/{collection}", "json": {"vectors": {"size": 2, "distance": "Cosine"}}},
            {"method": "PATCH", "path": f"/collections/{collection}", "json": {"optimizers_config": {"indexing_threshold": offered}}},
            {"method": "GET", "path": f"/collections/{collection}"},
        ]
        contract = {
            "kind": "assertions_v1",
            "trace_statuses": [200, 200, 200],
            "trace_assertions": [],
            "state_checks": [
                {
                    "path": f"/collections/{collection}",
                    "status": 200,
                    "assertions": [
                        {"path": ["result", "config", "optimizer_config", "indexing_threshold"], "equals": desired}
                    ],
                }
            ],
        }
        return "qdrant", query, plans, contract
    if family == "qdrant_payload_set":
        collection = f"v8payloadset_{suffix}"
        chosen = 1 if positive else 2
        query = f"Create collection {collection}, insert two points, set point 1 stage to published, and read it back."
        plans = [
            {"method": "PUT", "path": f"/collections/{collection}", "json": {"vectors": {"size": 2, "distance": "Cosine"}}},
            {"method": "PUT", "path": f"/collections/{collection}/points?wait=true", "json": {"points": [
                {"id": 1, "vector": [1.0, 0.0], "payload": {"owner": "alpha"}},
                {"id": 2, "vector": [0.0, 1.0], "payload": {"owner": "beta"}},
            ]}},
            {"method": "POST", "path": f"/collections/{collection}/points/payload?wait=true", "json": {"payload": {"stage": "published"}, "points": [chosen]}},
            {"method": "GET", "path": f"/collections/{collection}/points/1?with_payload=true"},
        ]
        contract = {
            "kind": "assertions_v1",
            "trace_statuses": [200, 200, 200, 200],
            "trace_assertions": [],
            "state_checks": [
                {"path": f"/collections/{collection}/points/1?with_payload=true", "status": 200, "assertions": [
                    {"path": ["result", "payload", "owner"], "equals": "alpha"},
                    {"path": ["result", "payload", "stage"], "equals": "published"},
                ]},
                {"path": f"/collections/{collection}/points/2?with_payload=true", "status": 200, "assertions": [
                    {"path": ["result", "payload", "stage"], "absent": True},
                ]},
            ],
        }
        return "qdrant", query, plans, contract
    if family == "qdrant_payload_delete":
        collection = f"v8payloaddel_{suffix}"
        chosen = 1 if positive else 2
        query = f"Create collection {collection} with two points; remove only point 1's obsolete payload field while retaining owner."
        plans = [
            {"method": "PUT", "path": f"/collections/{collection}", "json": {"vectors": {"size": 2, "distance": "Cosine"}}},
            {"method": "PUT", "path": f"/collections/{collection}/points?wait=true", "json": {"points": [
                {"id": 1, "vector": [1.0, 0.0], "payload": {"owner": "alpha", "obsolete": "retire"}},
                {"id": 2, "vector": [0.0, 1.0], "payload": {"owner": "beta", "obsolete": "retire"}},
            ]}},
            {"method": "POST", "path": f"/collections/{collection}/points/payload/delete?wait=true", "json": {"keys": ["obsolete"], "points": [chosen]}},
            {"method": "GET", "path": f"/collections/{collection}/points/1?with_payload=true"},
        ]
        contract = {
            "kind": "assertions_v1",
            "trace_statuses": [200, 200, 200, 200],
            "trace_assertions": [],
            "state_checks": [
                {"path": f"/collections/{collection}/points/1?with_payload=true", "status": 200, "assertions": [
                    {"path": ["result", "payload", "owner"], "equals": "alpha"},
                    {"path": ["result", "payload", "obsolete"], "absent": True},
                ]},
                {"path": f"/collections/{collection}/points/2?with_payload=true", "status": 200, "assertions": [
                    {"path": ["result", "payload", "obsolete"], "equals": "retire"},
                ]},
            ],
        }
        return "qdrant", query, plans, contract
    if family == "qdrant_vector_update":
        collection = f"v8vector_{suffix}"
        chosen = 1 if positive else 2
        query = f"Create collection {collection} with two points; update only point 1's vector to [0, 1], preserving point 2."
        plans = [
            {"method": "PUT", "path": f"/collections/{collection}", "json": {"vectors": {"size": 2, "distance": "Cosine"}}},
            {"method": "PUT", "path": f"/collections/{collection}/points?wait=true", "json": {"points": [
                {"id": 1, "vector": [1.0, 0.0], "payload": {"owner": "alpha"}},
                {"id": 2, "vector": [1.0, 0.0], "payload": {"owner": "beta"}},
            ]}},
            {"method": "PUT", "path": f"/collections/{collection}/points/vectors?wait=true", "json": {"points": [
                {"id": chosen, "vector": [0.0, 1.0]}
            ]}},
            {"method": "GET", "path": f"/collections/{collection}/points/1?with_vector=true"},
        ]
        contract = {
            "kind": "assertions_v1", "trace_statuses": [200] * 4, "trace_assertions": [],
            "state_checks": [
                {"path": f"/collections/{collection}/points/1?with_vector=true", "status": 200, "assertions": [
                    {"path": ["result", "vector"], "equals": [0.0, 1.0]},
                    {"path": ["result", "payload", "owner"], "equals": "alpha"},
                ]},
                {"path": f"/collections/{collection}/points/2?with_vector=true", "status": 200, "assertions": [
                    {"path": ["result", "vector"], "equals": [1.0, 0.0]},
                ]},
            ],
        }
        return "qdrant", query, plans, contract
    if family == "qdrant_filtered_count":
        collection = f"v8count_{suffix}"
        color = "red" if positive else "blue"
        query = f"Create collection {collection} with two red and one blue points; count only red points, expecting 2."
        plans = [
            {"method": "PUT", "path": f"/collections/{collection}", "json": {"vectors": {"size": 2, "distance": "Cosine"}}},
            {"method": "PUT", "path": f"/collections/{collection}/points?wait=true", "json": {"points": [
                {"id": 1, "vector": [1.0, 0.0], "payload": {"color": "red"}},
                {"id": 2, "vector": [0.0, 1.0], "payload": {"color": "red"}},
                {"id": 3, "vector": [1.0, 1.0], "payload": {"color": "blue"}},
            ]}},
            {"method": "POST", "path": f"/collections/{collection}/points/count", "json": {"filter": {"must": [
                {"key": "color", "match": {"value": color}}
            ]}, "exact": True}},
            {"method": "GET", "path": f"/collections/{collection}/points/1?with_payload=true"},
        ]
        contract = {
            "kind": "assertions_v1", "trace_statuses": [200] * 4,
            "trace_assertions": [{"path": ["steps", 2, "response", "body", "result", "count"], "equals": 2}],
            "state_checks": [
                {"path": f"/collections/{collection}/points/1?with_payload=true", "status": 200, "assertions": [
                    {"path": ["result", "payload", "color"], "equals": "red"},
                ]},
                {"path": f"/collections/{collection}/points/3?with_payload=true", "status": 200, "assertions": [
                    {"path": ["result", "payload", "color"], "equals": "blue"},
                ]},
            ],
        }
        return "qdrant", query, plans, contract
    if family == "qdrant_point_delete":
        collection = f"v8pointdel_{suffix}"
        chosen = 1 if positive else 2
        query = f"Create collection {collection} with two points; delete point 1, but retain point 2."
        plans = [
            {"method": "PUT", "path": f"/collections/{collection}", "json": {"vectors": {"size": 2, "distance": "Cosine"}}},
            {"method": "PUT", "path": f"/collections/{collection}/points?wait=true", "json": {"points": [
                {"id": 1, "vector": [1.0, 0.0], "payload": {"owner": "alpha"}},
                {"id": 2, "vector": [0.0, 1.0], "payload": {"owner": "beta"}},
            ]}},
            {"method": "POST", "path": f"/collections/{collection}/points/delete?wait=true", "json": {"points": [chosen]}},
            {"method": "GET", "path": f"/collections/{collection}/points/2?with_payload=true"},
        ]
        contract = {
            "kind": "assertions_v1", "trace_statuses": [200] * 4, "trace_assertions": [],
            "state_checks": [
                {"path": f"/collections/{collection}/points/1?with_payload=true", "status": 404, "assertions": [
                    {"path": ["status", "error"], "nonempty": True},
                ]},
                {"path": f"/collections/{collection}/points/2?with_payload=true", "status": 200, "assertions": [
                    {"path": ["result", "payload", "owner"], "equals": "beta"},
                ]},
            ],
        }
        return "qdrant", query, plans, contract
    if family == "qdrant_payload_facet":
        collection = f"v8facet_{suffix}"
        color = "red" if positive else "blue"
        query = f"Create collection {collection}, index color, and facet only red points, expecting a red bucket of size 2."
        plans = [
            {"method": "PUT", "path": f"/collections/{collection}", "json": {"vectors": {"size": 2, "distance": "Cosine"}}},
            {"method": "PUT", "path": f"/collections/{collection}/points?wait=true", "json": {"points": [
                {"id": 1, "vector": [1.0, 0.0], "payload": {"color": "red"}},
                {"id": 2, "vector": [0.0, 1.0], "payload": {"color": "red"}},
                {"id": 3, "vector": [1.0, 1.0], "payload": {"color": "blue"}},
            ]}},
            {"method": "PUT", "path": f"/collections/{collection}/index?wait=true", "json": {"field_name": "color", "field_schema": "keyword"}},
            {"method": "POST", "path": f"/collections/{collection}/facet", "json": {"key": "color", "exact": True, "filter": {"must": [
                {"key": "color", "match": {"value": color}}
            ]}}},
            {"method": "GET", "path": f"/collections/{collection}"},
        ]
        contract = {
            "kind": "assertions_v1", "trace_statuses": [200] * 5,
            "trace_assertions": [
                {"path": ["steps", 3, "response", "body", "result", "hits", 0, "value"], "equals": "red"},
                {"path": ["steps", 3, "response", "body", "result", "hits", 0, "count"], "equals": 2},
            ],
            "state_checks": [
                {"path": f"/collections/{collection}", "status": 200, "assertions": [
                    {"path": ["result", "payload_schema", "color", "data_type"], "equals": "keyword"},
                ]},
                {"path": f"/collections/{collection}/points/1?with_payload=true", "status": 200, "assertions": [
                    {"path": ["result", "payload", "color"], "equals": "red"},
                ]},
            ],
        }
        return "qdrant", query, plans, contract
    if family == "openbao_soft_delete":
        key = f"v8_softdelete_{suffix}"
        marker = f"checkpoint-{suffix}"
        version = 1 if positive else 2
        query = f"Write {marker} to OpenBao key {key}; soft-delete version 1 and inspect its metadata."
        plans = [
            {"method": "POST", "path": f"/v1/secret/data/{key}", "json": {"data": {"marker": marker}}},
            {"method": "POST", "path": f"/v1/secret/delete/{key}", "json": {"versions": [version]}},
            {"method": "GET", "path": f"/v1/secret/metadata/{key}"},
        ]
        contract = {
            "kind": "assertions_v1",
            "trace_statuses": [200, 204, 200],
            "trace_assertions": [],
            "state_checks": [
                {
                    "path": f"/v1/secret/data/{key}?version=1",
                    "status": 404,
                    "assertions": [
                        {"path": ["data", "data"], "equals": None},
                        {"path": ["data", "metadata", "deletion_time"], "nonempty": True},
                    ],
                },
                {
                    "path": f"/v1/secret/metadata/{key}",
                    "status": 200,
                    "assertions": [
                        {"path": ["data", "versions", "1", "deletion_time"], "nonempty": True},
                        {"path": ["data", "versions", "1", "destroyed"], "equals": False},
                    ],
                },
            ],
        }
        return "openbao", query, plans, contract
    if family == "openbao_undelete":
        key = f"v8_undelete_{suffix}"
        marker = f"restore-{suffix}"
        version = 1 if positive else 2
        query = f"Write {marker} to OpenBao key {key}, soft-delete version 1, then undelete version 1 and inspect metadata."
        plans = [
            {"method": "POST", "path": f"/v1/secret/data/{key}", "json": {"data": {"marker": marker}}},
            {"method": "POST", "path": f"/v1/secret/delete/{key}", "json": {"versions": [1]}},
            {"method": "POST", "path": f"/v1/secret/undelete/{key}", "json": {"versions": [version]}},
            {"method": "GET", "path": f"/v1/secret/metadata/{key}"},
        ]
        contract = {
            "kind": "assertions_v1",
            "trace_statuses": [200, 204, 204, 200],
            "trace_assertions": [],
            "state_checks": [
                {
                    "path": f"/v1/secret/data/{key}?version=1",
                    "status": 200,
                    "assertions": [
                        {"path": ["data", "data", "marker"], "equals": marker},
                        {"path": ["data", "metadata", "deletion_time"], "equals": ""},
                    ],
                },
                {
                    "path": f"/v1/secret/metadata/{key}",
                    "status": 200,
                    "assertions": [
                        {"path": ["data", "versions", "1", "deletion_time"], "equals": ""},
                        {"path": ["data", "versions", "1", "destroyed"], "equals": False},
                    ],
                },
            ],
        }
        return "openbao", query, plans, contract
    if family == "openbao_destroy":
        key = f"v8_destroy_{suffix}"
        marker = f"dispose-{suffix}"
        version = 1 if positive else 2
        query = f"Write {marker} to OpenBao key {key}, irreversibly destroy version 1, and inspect version metadata."
        plans = [
            {"method": "POST", "path": f"/v1/secret/data/{key}", "json": {"data": {"marker": marker}}},
            {"method": "POST", "path": f"/v1/secret/destroy/{key}", "json": {"versions": [version]}},
            {"method": "GET", "path": f"/v1/secret/metadata/{key}"},
        ]
        contract = {
            "kind": "assertions_v1",
            "trace_statuses": [200, 204, 200],
            "trace_assertions": [],
            "state_checks": [
                {"path": f"/v1/secret/data/{key}?version=1", "status": 404, "assertions": [
                    {"path": ["data", "data"], "equals": None},
                    {"path": ["data", "metadata", "destroyed"], "equals": True},
                ]},
                {"path": f"/v1/secret/metadata/{key}", "status": 200, "assertions": [
                    {"path": ["data", "versions", "1", "destroyed"], "equals": True},
                ]},
            ],
        }
        return "openbao", query, plans, contract
    if family == "openbao_cas_swap":
        primary = f"v8_cas_primary_{suffix}"
        decoy = f"v8_cas_decoy_{suffix}"
        target_value = f"approved-{suffix}"
        decoy_value = f"rejected-{suffix}"
        first = target_value if positive else decoy_value
        second = decoy_value if positive else target_value
        query = f"Create keys {primary} and {decoy}; update {primary} to {target_value} and {decoy} to {decoy_value} with their returned CAS versions."
        plans = [
            {"method": "POST", "path": f"/v1/secret/data/{primary}", "json": {"data": {"marker": "base-primary"}}, "capture": {"name": "primary_version", "source": "body_path", "path": ["data", "version"]}},
            {"method": "POST", "path": f"/v1/secret/data/{decoy}", "json": {"data": {"marker": "base-decoy"}}, "capture": {"name": "decoy_version", "source": "body_path", "path": ["data", "version"]}},
            {"method": "POST", "path": f"/v1/secret/data/{primary}", "json": {"options": {"cas": "{primary_version}"}, "data": {"marker": first}}},
            {"method": "POST", "path": f"/v1/secret/data/{decoy}", "json": {"options": {"cas": "{decoy_version}"}, "data": {"marker": second}}},
            {"method": "GET", "path": f"/v1/secret/data/{primary}"},
        ]
        contract = {
            "kind": "assertions_v1",
            "trace_statuses": [200, 200, 200, 200, 200],
            "trace_assertions": [
                {"path": ["bindings", "primary_version"], "equals": 1},
                {"path": ["bindings", "decoy_version"], "equals": 1},
                {"path": ["steps", 2, "request", "json", "options", "cas"], "equals": 1},
                {"path": ["steps", 3, "request", "json", "options", "cas"], "equals": 1},
            ],
            "state_checks": [
                {"path": f"/v1/secret/data/{primary}", "status": 200, "assertions": [
                    {"path": ["data", "data", "marker"], "equals": target_value},
                    {"path": ["data", "metadata", "version"], "equals": 2},
                ]},
                {"path": f"/v1/secret/data/{decoy}", "status": 200, "assertions": [
                    {"path": ["data", "data", "marker"], "equals": decoy_value},
                    {"path": ["data", "metadata", "version"], "equals": 2},
                ]},
            ],
        }
        return "openbao", query, plans, contract
    if family == "openbao_kv_patch":
        key = f"v8_patch_{suffix}"
        stage = "published" if positive else "archived"
        query = f"Write OpenBao key {key} with owner alpha and stage draft; patch only stage to published, preserving owner."
        plans = [
            {"method": "POST", "path": f"/v1/secret/data/{key}", "json": {"data": {"owner": "alpha", "stage": "draft"}}},
            {"method": "PATCH", "path": f"/v1/secret/data/{key}", "content_type": "application/merge-patch+json", "json": {"data": {"stage": stage}}},
            {"method": "GET", "path": f"/v1/secret/data/{key}"},
        ]
        contract = {
            "kind": "assertions_v1", "trace_statuses": [200, 200, 200], "trace_assertions": [],
            "state_checks": [{"path": f"/v1/secret/data/{key}", "status": 200, "assertions": [
                {"path": ["data", "data", "owner"], "equals": "alpha"},
                {"path": ["data", "data", "stage"], "equals": "published"},
                {"path": ["data", "metadata", "version"], "equals": 2},
            ]}],
        }
        return "openbao", query, plans, contract
    if family == "openbao_custom_metadata":
        key = f"v8_custommeta_{suffix}"
        owner = "alpha" if positive else "beta"
        query = f"Write OpenBao key {key}; attach custom metadata owner=alpha while preserving its secret value."
        plans = [
            {"method": "POST", "path": f"/v1/secret/data/{key}", "json": {"data": {"marker": "original"}}},
            {"method": "POST", "path": f"/v1/secret/metadata/{key}", "json": {"custom_metadata": {"owner": owner}}},
            {"method": "GET", "path": f"/v1/secret/metadata/{key}"},
        ]
        contract = {
            "kind": "assertions_v1", "trace_statuses": [200, 204, 200], "trace_assertions": [],
            "state_checks": [
                {"path": f"/v1/secret/metadata/{key}", "status": 200, "assertions": [
                    {"path": ["data", "custom_metadata", "owner"], "equals": "alpha"},
                ]},
                {"path": f"/v1/secret/data/{key}", "status": 200, "assertions": [
                    {"path": ["data", "data", "marker"], "equals": "original"},
                    {"path": ["data", "metadata", "version"], "equals": 1},
                ]},
            ],
        }
        return "openbao", query, plans, contract
    if family == "openbao_transit_encrypt":
        target = f"v8_transit_target_{suffix}"
        chosen = target if positive else f"v8_transit_decoy_{suffix}"
        plaintext = base64.b64encode(f"sealed-{suffix}".encode()).decode()
        query = f"Mount OpenBao Transit, encrypt sealed-{suffix} with key {target}, and decrypt the returned ciphertext."
        plans = [
            {"method": "POST", "path": "/v1/sys/mounts/transit", "json": {"type": "transit"}},
            {"method": "POST", "path": f"/v1/transit/encrypt/{chosen}", "json": {"plaintext": plaintext}, "capture": {"name": "ciphertext", "source": "body_path", "path": ["data", "ciphertext"]}},
            {"method": "POST", "path": f"/v1/transit/decrypt/{chosen}", "json": {"ciphertext": "{ciphertext}"}},
            {"method": "GET", "path": f"/v1/transit/keys/{chosen}"},
        ]
        contract = {
            "kind": "assertions_v1", "trace_statuses": [204, 200, 200, 200],
            "trace_assertions": [{"path": ["steps", 2, "response", "body", "data", "plaintext"], "equals": plaintext}],
            "state_checks": [{"path": f"/v1/transit/keys/{target}", "status": 200, "assertions": [
                {"path": ["data", "latest_version"], "equals": 1},
            ]}],
        }
        return "openbao", query, plans, contract
    if family == "openbao_transit_rewrap":
        target = f"v8_rewrap_target_{suffix}"
        chosen = target if positive else f"v8_rewrap_decoy_{suffix}"
        plaintext = base64.b64encode(f"rotate-{suffix}".encode()).decode()
        query = f"Mount OpenBao Transit; encrypt rotate-{suffix} with {target}, rotate that key, rewrap the ciphertext, and decrypt it."
        plans = [
            {"method": "POST", "path": "/v1/sys/mounts/transit", "json": {"type": "transit"}},
            {"method": "POST", "path": f"/v1/transit/encrypt/{chosen}", "json": {"plaintext": plaintext}, "capture": {"name": "ciphertext", "source": "body_path", "path": ["data", "ciphertext"]}},
            {"method": "POST", "path": f"/v1/transit/keys/{chosen}/rotate"},
            {"method": "POST", "path": f"/v1/transit/rewrap/{chosen}", "json": {"ciphertext": "{ciphertext}"}, "capture": {"name": "rewrapped_ciphertext", "source": "body_path", "path": ["data", "ciphertext"]}},
            {"method": "POST", "path": f"/v1/transit/decrypt/{chosen}", "json": {"ciphertext": "{rewrapped_ciphertext}"}},
        ]
        contract = {
            "kind": "assertions_v1", "trace_statuses": [204, 200, 200, 200, 200],
            "trace_assertions": [
                {"path": ["steps", 3, "response", "body", "data", "key_version"], "equals": 2},
                {"path": ["steps", 4, "response", "body", "data", "plaintext"], "equals": plaintext},
            ],
            "state_checks": [{"path": f"/v1/transit/keys/{target}", "status": 200, "assertions": [
                {"path": ["data", "latest_version"], "equals": 2},
            ]}],
        }
        return "openbao", query, plans, contract
    raise ValueError(f"unsupported staging family: {family}")


def _candidate(
    salt: bytes,
    event_id: str,
    service: str,
    family: str,
    side: int,
    plans: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    edge_id = _opaque(salt, event_id, side)
    selected = []
    memories = []
    for step_index, plan in enumerate(plans, 1):
        memory_id = f"{edge_id}:record-{step_index}"
        selected.append(memory_id)
        memories.append(
            {
                "memory_id": memory_id,
                "service_id": service,
                "provider_id": service,
                "version_scope": [VERSIONS[service]],
                "valid_from": VALID_FROM,
                "valid_until": None,
                "role": "workflow_step",
                "step_index": step_index,
                # Neutral text across positive and negative candidate sides.
                "text": f"Execute {service} workflow operation {step_index} using {plan['method']} {plan['path']}.",
                "plan": plan,
                "token_cost": 24,
                "storage_cost": 48,
                "provenance_uri": FAMILY_SOURCE[family],
                "provenance_verified": True,
                "evidence_atoms": [f"role:step{step_index}"] + (
                    [f"binding:{plan['capture']['name']}"] if "capture" in plan else []
                ),
            }
        )
    verification_id = f"{edge_id}:record-{len(plans) + 1}"
    selected.append(verification_id)
    memories.append(
        {
            "memory_id": verification_id,
            "service_id": service,
            "provider_id": service,
            "version_scope": [VERSIONS[service]],
            "valid_from": VALID_FROM,
            "valid_until": None,
            "role": "verification",
            "step_index": None,
            "text": f"Independently inspect final {service} service state.",
            "token_cost": 24,
            "storage_cost": 48,
            "provenance_uri": FAMILY_SOURCE[family],
            "provenance_verified": True,
            "evidence_atoms": ["role:verify"],
        }
    )
    edge = {
        "edge_id": edge_id,
        "members": selected,
        "service_id": service,
        "version_scope": [VERSIONS[service]],
        "valid_from": VALID_FROM,
        "valid_until": None,
        "verified": True,
        "relation_type": "candidate-complete-workflow",
        "evidence_atoms": [f"role:step{index}" for index in range(1, len(plans) + 1)] + ["role:verify"],
    }
    candidate = {
        "event_id": event_id,
        "candidate_edge_id": edge_id,
        "selected_memory_ids": selected,
    }
    return memories, edge, candidate


def _salt(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(os.urandom(32))
        except BaseException:
            raise
    salt = path.read_bytes()
    if len(salt) != 32:
        raise ValueError("private candidate-assignment salt must be 32 bytes")
    return salt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-label-output", type=Path, required=True)
    parser.add_argument("--salt-file", type=Path, required=True)
    parser.add_argument(
        "--final-candidate",
        action="store_true",
        help="Mark the full sixteen-family set for structural and formal validity gates; never a score claim.",
    )
    args = parser.parse_args()
    if args.output.exists() or args.private_label_output.exists():
        parser.error("refusing to overwrite a prior candidate or private-label output")
    salt = _salt(args.salt_file)
    events = []
    queries = []
    memories = []
    edges = []
    candidates = []
    positive_selections = []
    negative_selections = []
    families = (
        "qdrant_snapshot",
        "qdrant_collection_update",
        "qdrant_payload_set",
        "qdrant_payload_delete",
        "qdrant_vector_update",
        "qdrant_filtered_count",
        "qdrant_point_delete",
        "qdrant_payload_facet",
        "openbao_soft_delete",
        "openbao_undelete",
        "openbao_destroy",
        "openbao_cas_swap",
        "openbao_kv_patch",
        "openbao_custom_metadata",
        "openbao_transit_encrypt",
        "openbao_transit_rewrap",
    )
    for family_index, family in enumerate(families):
        sides = _positive_sides(salt, family)
        for offset in range(8):
            number = offset + 1
            event_id = f"v8-candidate:{family}:event-{number:03d}"
            positive_side = sides[offset]
            service, query, _plans, contract = _case(family, number, positive=True)
            event = {
                "event_id": event_id,
                "incident_id": event_id,
                "service_id": service,
                "provider_id": service,
                "version_id": VERSIONS[service],
                "event_time": (BASE_TIME + timedelta(minutes=family_index * 8 + offset)).isoformat().replace("+00:00", "Z"),
                "query": query,
                "task_key": family,
                "operation_count": len(_plans),
                "workflow_contract": contract,
                "checker_contract_id": "independent-v8-assertions-v1",
                "required_evidence_atoms": [f"role:step{index}" for index in range(1, len(_plans) + 1)] + ["role:verify"] + [
                    f"binding:{plan['capture']['name']}" for plan in _plans if "capture" in plan
                ],
                "required_binding_atoms": [
                    f"binding:{plan['capture']['name']}" for plan in _plans if "capture" in plan
                ],
                "observation_tier": (
                    "independent_candidate" if args.final_candidate else "independent_candidate_staging"
                ),
            }
            events.append(event)
            queries.append(
                {
                    key: event[key]
                    for key in ("event_id", "service_id", "provider_id", "version_id", "event_time", "query")
                }
            )
            for side in (0, 1):
                positive = side == positive_side
                _service, _query, plans, _contract = _case(family, number, positive=positive)
                rows, edge, candidate = _candidate(salt, event_id, service, family, side, plans)
                memories.extend(rows)
                edges.append(edge)
                candidates.append(candidate)
                selection = {
                    "run_id": f"v8-candidate-validity:{event_id}:{side}",
                    "event_id": event_id,
                    "method": "oracle" if positive else "complete_negative",
                    "selected_memory_ids": candidate["selected_memory_ids"],
                    "token_cost": 24 * len(candidate["selected_memory_ids"]),
                }
                (positive_selections if positive else negative_selections).append(selection)
    if args.final_candidate and (len(families) != 16 or len(events) != 128):
        parser.error("final candidate requires sixteen families and 128 events")
    args.output.mkdir(parents=True)
    files = {
        "events.jsonl": events,
        "query_only_events.jsonl": queries,
        "memories.jsonl": memories,
        "dependency_hyperedges.jsonl": edges,
        "candidate_selections.jsonl": candidates,
    }
    for name, rows in files.items():
        write_jsonl(args.output / name, rows)
    write_json(
        args.output / "manifest.json",
        {
            "status": "candidate_for_structural_and_validity_gates" if args.final_candidate else "candidate_staging_not_frozen",
            "claim_boundary": "No selector may score this set before structural and oracle/negative gates are accepted and the task set is frozen.",
            "families": list(families),
            "events": len(events),
            "candidate_rows": len(candidates),
            "assignment_commitment_sha256": hashlib.sha256(salt).hexdigest(),
            "positive_side_balance_per_family": "4:4",
            "real_service_sources": FAMILY_SOURCE,
        },
    )
    write_checksums(args.output, (*files, "manifest.json"))
    args.private_label_output.mkdir(parents=True, mode=0o700)
    write_jsonl(args.private_label_output / "oracle_selections.jsonl", positive_selections)
    write_jsonl(args.private_label_output / "complete_negative_selections.jsonl", negative_selections)
    write_json(
        args.private_label_output / "manifest.json",
        {
            "status": "private_candidate_gate_labels" if args.final_candidate else "private_candidate_staging_labels",
            "public_dataset_sha256": hashlib.sha256((args.output / "manifest.json").read_bytes()).hexdigest(),
            "salt_sha256": hashlib.sha256(salt).hexdigest(),
            "oracle_rows": len(positive_selections),
            "negative_rows": len(negative_selections),
        },
    )
    write_checksums(
        args.private_label_output,
        ("oracle_selections.jsonl", "complete_negative_selections.jsonl", "manifest.json"),
    )
    print(f"wrote {len(events)} unscored candidate events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
