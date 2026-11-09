"""Parse tau3 memory identifiers without importing the replay stack."""

from __future__ import annotations

import re


ACTION_SUFFIX = re.compile(r":a(\d+)$")
CONTROLLED_EXTRA_SUFFIX = re.compile(r":controlled-extra-(\d+)$")
CONTROLLED_STALE_SUFFIX = re.compile(r":controlled-stale-(\d+)$")


def selected_action_spec(
    memory_ids: list[str],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return current, extra, and controlled-stale action indices."""
    indices: list[int] = []
    extras: list[int] = []
    stale: list[int] = []
    for memory_id in memory_ids:
        match = ACTION_SUFFIX.search(memory_id)
        if match is not None:
            indices.append(int(match.group(1)))
            continue
        extra_match = CONTROLLED_EXTRA_SUFFIX.search(memory_id)
        if extra_match is not None:
            extras.append(int(extra_match.group(1)))
            continue
        stale_match = CONTROLLED_STALE_SUFFIX.search(memory_id)
        if stale_match is not None:
            stale.append(int(stale_match.group(1)))
            continue
        raise ValueError(f"cannot recover action index from {memory_id}")
    return tuple(sorted(indices)), tuple(sorted(extras)), tuple(sorted(stale))
