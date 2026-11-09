#!/usr/bin/env python3
"""Compare AppWorld API-failure feedback modes on a synthetic train call."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "data" / "private_corem"
SYNTHETIC_CODE = (
    'print(apis.spotify.show_playlist_library(access_token="synthetic_invalid_token"))'
)


def classify_feedback(output: str) -> dict[str, object]:
    status_match = re.search(r"Response status code is (\d{3})", output)
    response_object = False
    try:
        response_object = isinstance(json.loads(output), dict)
    except json.JSONDecodeError:
        pass
    return {
        "execution_failed": output.startswith("Execution failed."),
        "http_status_in_traceback": int(status_match[1]) if status_match else None,
        "json_object_returned": response_object,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-prefix", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if not root.is_relative_to(PRIVATE_ROOT.resolve()) or not output.is_relative_to(
        PRIVATE_ROOT.resolve()
    ):
        parser.error("benchmark and output paths must stay private")
    if output.exists():
        parser.error("refusing to overwrite diagnostic output")
    if not args.experiment_prefix.isidentifier():
        parser.error("experiment prefix must be a Python identifier")
    names = {
        mode: f"{args.experiment_prefix}_{'raise' if mode else 'return'}"
        for mode in (False, True)
    }
    for name in names.values():
        if (root / "experiments" / "outputs" / name).exists():
            parser.error("refusing to reuse an AppWorld experiment directory")

    from appworld import AppWorld, load_task_ids, update_root

    update_root(str(root))
    task_id = sorted(load_task_ids("train"))[0]
    modes: dict[str, dict[str, object]] = {}
    for raise_on_failure, name in names.items():
        with AppWorld(
            task_id=task_id,
            experiment_name=name,
            max_interactions=1,
            raise_on_unsafe_syntax=True,
            null_patch_unsafe_execution=True,
            load_ground_truth=False,
            raise_on_failure=raise_on_failure,
            timeout_seconds=30,
            random_seed=2027,
        ) as world:
            feedback = world.execute(SYNTHETIC_CODE)
            world.save_logs()
        modes["raise" if raise_on_failure else "return"] = classify_feedback(feedback)
    result = {
        "status": "train_only_synthetic_api_feedback_not_task_success",
        "benchmark": f"AppWorld {importlib.metadata.version('appworld')}",
        "split": "train",
        "task_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
        "synthetic_code_sha256": hashlib.sha256(SYNTHETIC_CODE.encode()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "train_split_sha256": hashlib.sha256(
            (root / "data" / "datasets" / "train.txt").read_bytes()
        ).hexdigest(),
        "modes": modes,
        "target_ground_truth_loaded": False,
        "task_content_exported": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
