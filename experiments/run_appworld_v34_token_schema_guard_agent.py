#!/usr/bin/env python3
"""Train-only agent guarding documented token omissions with read-only docs."""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_appworld_v30_full_docs_agent as v30  # noqa: E402
from experiments import run_appworld_v32_code_history_agent as v32  # noqa: E402
from experiments import run_appworld_v33_auth_contract_agent as v33  # noqa: E402
from experiments.run_appworld_v25_api_guard import safe_api_names  # noqa: E402
from experiments.run_appworld_v26_nested_guard import api_attribute_chain  # noqa: E402

TOKEN_GUARD_POLICY = "read_only_docs_for_documented_missing_access_token"
_auth_guard_interventions = 0
_original_run_task = v33.run_task


def token_schema_guard(
    code: str,
    world: object,
    names: Mapping[str, set[str]],
) -> tuple[str, tuple[str, str] | None]:
    """Replace the first documented token omission with its detailed docs."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, None
    calls = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    for call in calls:
        if not isinstance(call.func, ast.Attribute):
            continue
        chain = api_attribute_chain(call.func)
        if chain is None or len(chain) < 2:
            continue
        app_name, api_name = chain[:2]
        if (
            app_name == "api_docs"
            or app_name not in names
            or api_name not in names[app_name]
            or any(keyword.arg is None for keyword in call.keywords)
        ):
            continue
        doc = world.apis.api_docs.show_api_doc(
            app_name=app_name, api_name=api_name
        )
        if not isinstance(doc, dict) or not isinstance(doc.get("parameters"), list):
            raise ValueError("AppWorld detailed API documentation is malformed")
        parameter_names = {
            parameter.get("name")
            for parameter in doc["parameters"]
            if isinstance(parameter, dict)
        }
        keyword_names = {keyword.arg for keyword in call.keywords}
        if "access_token" in parameter_names and "access_token" not in keyword_names:
            return (
                "print(apis.api_docs.show_api_doc("
                f"app_name={app_name!r}, api_name={api_name!r}))",
                (app_name, api_name),
            )
    return code, None


def guarded_execute_and_flush(world: object, code: str) -> str:
    """Compose the frozen invalid-path guard with the token-schema guard."""
    global _auth_guard_interventions
    names = safe_api_names(world)
    executed, invalid_reason = v30.guard_code(code, names)
    token_target = None
    if invalid_reason is None:
        executed, token_target = token_schema_guard(executed, world, names)
    observation = str(v30._original_execute_and_flush(world, executed))
    if invalid_reason is not None:
        v30._guard_interventions += 1
        return (
            "API guard: the proposed code was not executed because it references "
            f"an {invalid_reason.replace('_', ' ')}. The read-only documentation "
            "query below was executed instead. Choose a documented API name and "
            "inspect its detailed schema before retrying.\n" + observation
        )
    if token_target is not None:
        _auth_guard_interventions += 1
        app_name, api_name = token_target
        return (
            "Authentication guard: the proposed code was not executed because "
            f"the documented {app_name}.{api_name} API requires an explicit "
            "access_token argument. Its read-only detailed documentation was "
            "returned below. Log in to that app if needed, save the returned "
            "token, then retry with access_token=<saved token>.\n" + observation
        )
    return observation


def run_task(*args: object, **kwargs: object) -> dict[str, object]:
    global _auth_guard_interventions
    _auth_guard_interventions = 0
    row = _original_run_task(*args, **kwargs)
    row["token_schema_guard_interventions"] = _auth_guard_interventions
    row["token_schema_guard_policy"] = TOKEN_GUARD_POLICY
    row["token_schema_guard_script_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    return row


def main() -> int:
    v30.base._initial_messages = v33.initial_messages
    v30.context_messages = v32.context_messages
    v30.guarded_execute_and_flush = guarded_execute_and_flush
    v30.run_task = run_task
    return v30.main()


if __name__ == "__main__":
    raise SystemExit(main())
