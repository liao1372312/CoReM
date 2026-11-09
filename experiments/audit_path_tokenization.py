#!/usr/bin/env python3
"""Audit information loss in complete-path tokenization without loading labels."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corem.real_eval import load_jsonl, write_checksums, write_json  # noqa: E402
from experiments.path_serialization import serialize_path_body_v1  # noqa: E402
from experiments.score_v7_path_nli_development import serialize_path  # noqa: E402
from experiments.summarize_independent_v7_confirmation import sha256  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.max_length < 32:
        parser.error("--max-length must be at least 32")
    events = {str(row["event_id"]): row for row in load_jsonl(args.events)}
    memories = {str(row["memory_id"]): row for row in load_jsonl(args.memories)}
    candidates = load_jsonl(args.candidates)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        event_id = str(candidate["event_id"])
        if event_id not in events:
            raise ValueError(f"candidate references unknown event: {event_id}")
        grouped[event_id].append(index)
    if set(grouped) != set(events):
        raise ValueError("incomplete candidate pool")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    serializations = {
        "legacy": serialize_path,
        "body_v1": serialize_path_body_v1,
    }
    results = {}
    for name, serializer in serializations.items():
        premises = [serializer(row, memories) for row in candidates]
        hypotheses = [
            "This workflow fulfills the request: " + str(events[str(row["event_id"])]["query"])
            for row in candidates
        ]
        full = tokenizer(premises, hypotheses, truncation=False)["input_ids"]
        limited = tokenizer(
            premises, hypotheses, truncation=True, max_length=args.max_length
        )["input_ids"]
        lengths = sorted(map(len, full))
        results[name] = {
            "candidate_rows": len(candidates),
            "median_full_tokens": lengths[len(lengths) // 2],
            "max_full_tokens": lengths[-1],
            "truncated_rows": sum(len(ids) > args.max_length for ids in full),
            "indistinguishable_event_pairs": sum(
                len(indices) == 2 and limited[indices[0]] == limited[indices[1]]
                for indices in grouped.values()
            ),
            "events_with_two_candidates": sum(len(indices) == 2 for indices in grouped.values()),
        }
    args.output.mkdir(parents=True)
    write_json(
        args.output / "result.json",
        {
            "status": "label_blind_tokenization_audit",
            "max_length": args.max_length,
            "results": results,
            "input_sha256": {
                "events": sha256(args.events),
                "candidates": sha256(args.candidates),
                "memories": sha256(args.memories),
                "tokenizer": sha256(args.tokenizer_dir / "tokenizer.json"),
            },
        },
    )
    write_checksums(args.output, ("result.json",))
    print(f"audited {len(candidates)} candidate tokenizations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
