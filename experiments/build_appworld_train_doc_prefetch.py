#!/usr/bin/env python3
"""Freeze private BM25 API-documentation prefetch for AppWorld TRAIN tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = REPOSITORY_ROOT / "data" / "private_corem"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.generate_public_workflow_selections import bm25_scores  # noqa: E402
from experiments.run_appworld_local_agent_pilot import _sample_task_ids  # noqa: E402


def select_docs(
    instruction: str,
    docs: list[dict[str, Any]],
    max_docs: int,
    max_chars: int,
) -> tuple[list[str], list[str]]:
    """Rank without labels and include whole specifications in BM25 order."""
    scores = bm25_scores(
        instruction,
        [{"memory_id": doc["api"], "text": doc["index_text"]} for doc in docs],
    )
    ranked = sorted(docs, key=lambda doc: (-scores[doc["api"]], doc["api"]))
    chosen: list[str] = []
    names: list[str] = []
    used = 0
    for doc in ranked:
        if len(chosen) == max_docs:
            break
        specification = doc["specification"]
        extra = len(specification) + (2 if chosen else 0)
        if used + extra > max_chars:
            break
        chosen.append(specification)
        names.append(doc["api"])
        used += extra
    if not chosen:
        raise ValueError("top-ranked API specification exceeds prefetch budget")
    return chosen, names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--num-tasks", type=int, default=9)
    parser.add_argument("--max-docs", type=int, default=12)
    parser.add_argument("--max-chars", type=int, default=9000)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if not root.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error("AppWorld root must be private")
    if not output.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error("protected output must stay under data/private_corem")
    if output.exists():
        parser.error("refusing to overwrite protected output")
    if not 1 <= args.num_tasks <= 30:
        parser.error("invalid task count")
    if not 1 <= args.max_docs <= 50 or not 1000 <= args.max_chars <= 20000:
        parser.error("invalid doc-count or character budget")

    from appworld import load_task_ids, update_root
    from appworld.task import Task

    update_root(str(root))
    train_ids = load_task_ids("train")
    selected = _sample_task_ids(train_ids, args.seed, args.num_tasks)
    catalog_task = Task.load(train_ids[0], load_ground_truth=False)
    docs = []
    for app, app_docs in catalog_task.api_docs.items():
        for api_name, specification in app_docs.items():
            api = f"{app}.{api_name}"
            indexed = " ".join(
                [
                    app,
                    api_name.replace("_", " "),
                    str(specification["description"]),
                    " ".join(str(row["name"]) for row in specification["parameters"]),
                ]
            )
            complete = json.dumps(
                specification,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            docs.append(
                {
                    "api": api,
                    "index_text": indexed,
                    "specification": f"{api}: {complete}",
                }
            )
    if len({doc["api"] for doc in docs}) != len(docs):
        raise ValueError("duplicate API documentation")

    hints: dict[str, str] = {}
    audits: list[dict[str, object]] = []
    for task_id in selected:
        task = Task.load(task_id, load_ground_truth=False)
        chosen, names = select_docs(
            task.instruction, docs, args.max_docs, args.max_chars
        )
        task_hash = hashlib.sha256(task_id.encode()).hexdigest()
        hint = (
            "Prefetched API documentation selected from the task wording only. "
            "These are candidate APIs, not a task solution; inspect actual "
            "results and other API docs as needed.\n" + "\n\n".join(chosen)
        )
        hints[task_hash] = hint
        audits.append(
            {
                "task_sha256": task_hash,
                "selected_apis": names,
                "selected_docs": len(names),
                "specification_chars": len("\n\n".join(chosen)),
            }
        )
    payload = {
        "status": "protected_local_only_train_exploratory",
        "split": "train",
        "method": "task_instruction_bm25_complete_api_docs",
        "seed": args.seed,
        "num_tasks": len(hints),
        "max_docs": args.max_docs,
        "max_chars": args.max_chars,
        "catalog_docs": len(docs),
        "train_split_sha256": hashlib.sha256(
            (root / "data/datasets/train.txt").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection_audit": audits,
        "hints": hints,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        f"Private train API-doc prefetch frozen for {len(hints)} tasks "
        f"from {len(docs)} documented APIs; no task ground truth was loaded."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
