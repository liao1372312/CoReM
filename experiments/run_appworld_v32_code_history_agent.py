#!/usr/bin/env python3
"""Train-only agent preserving prior code while bounding old observations.

The v30 full-document observation and nested API guard are unchanged.
Only context selection changes: every previous code action remains visible,
the last four observations remain available, and older observations become
short placeholders. The latest documentation reply remains complete.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_appworld_v30_full_docs_agent as v30  # noqa: E402

_original_run_task = v30.run_task
CONTEXT_POLICY = "all_code_recent_four_observations_latest_doc_full"


def context_messages(
    messages: list[dict[str, str]], strategy: str
) -> list[dict[str, str]]:
    """Expose all code actions but limit older observation context.

    The CLI retains the v30 `preserve_task` choice so the only change is
    this explicitly versioned policy. `preserve_code` in the frozen base
    runner retains system/task and all actions, replacing older outputs.
    """
    if strategy != "preserve_task":
        raise ValueError("v32 requires the frozen preserve_task CLI setting")
    selected = v30._original_context_messages(messages, "preserve_code")
    result = [item.copy() for item in selected]
    for index, item in enumerate(result[:-1]):
        if (
            index >= 2
            and item["role"] == "user"
            and len(item["content"]) > v30.OLDER_OBSERVATION_CHARS
        ):
            item["content"] = (
                item["content"][:v30.OLDER_OBSERVATION_CHARS]
                + "\n[Older observation shortened; request details again if needed.]"
            )
    return result


def run_task(*args: object, **kwargs: object) -> dict[str, object]:
    row = _original_run_task(*args, **kwargs)
    row["context_policy"] = CONTEXT_POLICY
    row["context_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return row


def main() -> int:
    v30.context_messages = context_messages
    v30.run_task = run_task
    return v30.main()


if __name__ == "__main__":
    raise SystemExit(main())
