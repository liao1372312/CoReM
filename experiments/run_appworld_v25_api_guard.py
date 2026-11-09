#!/usr/bin/env python3
"""Train-only AppWorld composer with a read-only invalid-API documentation guard.

The guard reads only the APIs exposed by the current AppWorld instance. It
never reads target ground truth or substitutes a state-changing action.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_appworld_local_agent_pilot as base  # noqa: E402

_original_run_task = base._run_task
_original_execute_and_flush = base._execute_and_flush
_guard_interventions = 0


def safe_api_names(world: object) -> dict[str, set[str]]:
    """Read the task-visible API surface, not data or evaluation state."""
    return {
        str(app_name): {str(name) for name in app_apis.keys()}
        for app_name, app_apis in world.apis.items()
    }


def guard_code(
    code: str, names: Mapping[str, set[str]]
) -> tuple[str, str | None]:
    """Replace the first invalid literal apis.app.method call with a doc query."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, None
    calls = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    for call in calls:
        method = call.func
        if not isinstance(method, ast.Attribute) or not isinstance(method.value, ast.Attribute):
            continue
        app = method.value
        if not isinstance(app.value, ast.Name) or app.value.id != "apis":
            continue
        if app.attr not in names:
            return (
                "print(apis.api_docs.show_app_descriptions())",
                "invalid_app",
            )
        if method.attr not in names[app.attr]:
            if "api_docs" not in names or "show_api_descriptions" not in names["api_docs"]:
                raise ValueError("AppWorld documentation API is unavailable")
            return (
                f"print(apis.api_docs.show_api_descriptions(app_name={app.attr!r}))",
                "invalid_method",
            )
    return code, None


def guarded_execute_and_flush(world: object, code: str) -> str:
    global _guard_interventions
    executed, reason = guard_code(code, safe_api_names(world))
    observation = _original_execute_and_flush(world, executed)
    if reason is None:
        return observation
    _guard_interventions += 1
    return (
        "API guard: the proposed code was not executed because it references "
        f"an {reason.replace('_', ' ')}. The read-only documentation query "
        "below was executed instead. Choose a documented API name and inspect "
        "its detailed schema before retrying.\n" + str(observation)
    )


def guarded_run_task(*args: object, **kwargs: object) -> dict[str, object]:
    global _guard_interventions
    _guard_interventions = 0
    result = _original_run_task(*args, **kwargs)
    result["api_guard_interventions"] = _guard_interventions
    result["api_guard_policy"] = "read_only_documentation_for_invalid_literal_api_calls"
    result["api_guard_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main() -> int:
    base._execute_and_flush = guarded_execute_and_flush
    base._run_task = guarded_run_task
    status = base.main()
    # The base runner writes an exclusive private result. Its task rows now
    # carry the guard hash and intervention count; print aggregates only.
    output_index = sys.argv.index("--output") + 1
    payload = json.loads(Path(sys.argv[output_index]).read_text())
    print(
        "train-only API guard interventions:",
        sum(row["api_guard_interventions"] for row in payload["private_task_outcomes"]),
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
