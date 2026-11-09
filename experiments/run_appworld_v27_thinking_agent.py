#!/usr/bin/env python3
"""Train-only Qwen thinking-mode agent; execute only the final code block."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_appworld_local_agent_pilot as base  # noqa: E402

_original_strip_code_fences = base._strip_code_fences
_original_run_task = base._run_task


def strip_thinking(value: str) -> str:
    if "<think>" in value:
        if "</think>" not in value:
            raise ValueError("thinking response has no final answer")
        value = value.split("</think>", 1)[1]
    return _original_strip_code_fences(value)


def thinking_run_task(*args: object, **kwargs: object) -> dict[str, object]:
    result = _original_run_task(*args, **kwargs)
    result["reasoning_policy"] = "execute_final_code_only_require_closed_think_tag"
    result["reasoning_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main() -> int:
    base._strip_code_fences = strip_thinking
    base._run_task = thinking_run_task
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
