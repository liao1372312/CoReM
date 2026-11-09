#!/usr/bin/env python3
"""Export a LOCAL-ONLY AppWorld train view for cross-environment development.

The output contains protected task and API text. Never publish or commit it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = REPOSITORY_ROOT / "data" / "private_corem"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error("protected output must stay under data/private_corem")
    if output.exists():
        parser.error("refusing to overwrite protected output")

    from appworld import load_task_ids, update_root
    from appworld.task import Task

    update_root(str(root))
    ids = load_task_ids("train")
    tasks = [
        Task.load(
            task_id,
            load_ground_truth=True,
            ground_truth_mode="full",
            include_api_response_schemas=False,
        )
        for task_id in ids
    ]
    if not tasks or any(task.ground_truth is None for task in tasks):
        raise ValueError("train tasks lack ground truth")
    docs = [
        {
            "api": f"{app}.{name}",
            "text": " ".join(
                [
                    app,
                    name.replace("_", " "),
                    str(doc["description"]),
                    " ".join(str(parameter["name"]) for parameter in doc["parameters"]),
                ]
            ),
        }
        for app, app_docs in tasks[0].api_docs.items()
        for name, doc in app_docs.items()
    ]
    documented = {row["api"] for row in docs}
    if len(documented) != len(docs):
        raise ValueError("duplicate API docs")
    memories = [
        {
            "task_id": task.id,
            "generator_id": task.generator_id,
            "instruction": task.instruction,
            "required_apis": sorted(task.ground_truth.required_apis),
        }
        for task in tasks
    ]
    if any(
        not row["required_apis"] or not set(row["required_apis"]) <= documented
        for row in memories
    ):
        raise ValueError("train task has missing API documentation")
    payload = {
        "status": "protected_local_only_train_development",
        "benchmark": "AppWorld 0.1.3.post1",
        "train_split_sha256": hashlib.sha256(
            (root / "data/datasets/train.txt").read_bytes()
        ).hexdigest(),
        "tasks": memories,
        "api_docs": docs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        f"Private train view written: {len(memories)} tasks, "
        f"{len(docs)} API docs; no dev/test task content loaded."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
