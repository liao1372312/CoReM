#!/usr/bin/env python3
"""Copy only label-blind v9 inputs into a separate selector-visible directory."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corem.real_eval import sha256_file, write_checksums, write_json  # noqa: E402
from experiments.audit_independent_v9_structure import DATA_FILES  # noqa: E402

SELECTOR_FILES = (
    "query_only_events.jsonl", "memories.jsonl", "dependency_hyperedges.jsonl",
    "candidate_selections.jsonl",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--structural-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a selector view")
    audit = json.loads(args.structural_audit.read_text())
    if (
        audit.get("status") != "structurally_ready_for_physical_gate"
        or audit.get("issues")
        or audit.get("input_sha256") != {name: sha256_file(args.dataset / name) for name in DATA_FILES}
    ):
        raise ValueError("source dataset lacks a clean current structural audit")
    args.output.mkdir(parents=True)
    for name in SELECTOR_FILES:
        shutil.copy2(args.dataset / name, args.output / name)
    write_json(args.output / "manifest.json", {
        "status": "label_blind_selector_view",
        "claim_boundary": "No full workflow contracts, checker targets, private candidate labels, or execution outcomes are present.",
        "source_dataset_manifest_sha256": sha256_file(args.dataset / "manifest.json"),
        "structural_audit_sha256": sha256_file(args.structural_audit),
        "files_sha256": {name: sha256_file(args.output / name) for name in SELECTOR_FILES},
        "exporter_sha256": sha256_file(Path(__file__)),
    })
    write_checksums(args.output, (*SELECTOR_FILES, "manifest.json"))
    if {path.name for path in args.output.iterdir()} != {*SELECTOR_FILES, "manifest.json", "checksums.sha256"}:
        raise RuntimeError("selector view contains an unexpected file")
    print(f"exported {len(SELECTOR_FILES)} label-blind v9 selector inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
