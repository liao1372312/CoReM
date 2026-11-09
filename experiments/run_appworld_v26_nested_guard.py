#!/usr/bin/env python3
"""Train-only API guard that also catches nested invalid apis.app.name paths."""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_appworld_v25_api_guard as v25  # noqa: E402

_original_guarded_run_task = v25.guarded_run_task


def api_attribute_chain(node: ast.Attribute) -> list[str] | None:
    """Return names after `apis` for a direct attribute chain, if any."""
    names: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        names.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name) or current.id != "apis":
        return None
    return list(reversed(names))


def guard_code(
    code: str, names: Mapping[str, set[str]]
) -> tuple[str, str | None]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, None
    attributes = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Attribute)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    for node in attributes:
        chain = api_attribute_chain(node)
        if chain is None or len(chain) < 2:
            continue
        app, method = chain[:2]
        if app not in names:
            return "print(apis.api_docs.show_app_descriptions())", "invalid_app"
        if method not in names[app]:
            if "api_docs" not in names or "show_api_descriptions" not in names["api_docs"]:
                raise ValueError("AppWorld documentation API is unavailable")
            return (
                f"print(apis.api_docs.show_api_descriptions(app_name={app!r}))",
                "invalid_method",
            )
    return code, None


def guarded_run_task(*args: object, **kwargs: object) -> dict[str, object]:
    result = _original_guarded_run_task(*args, **kwargs)
    result["api_guard_policy"] = "read_only_documentation_for_invalid_nested_api_paths"
    result["api_guard_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main() -> int:
    v25.guard_code = guard_code
    v25.guarded_run_task = guarded_run_task
    return v25.main()


if __name__ == "__main__":
    raise SystemExit(main())
