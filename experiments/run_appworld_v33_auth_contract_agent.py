#!/usr/bin/env python3
"""Train-only code-history agent with an explicit authentication contract."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_appworld_v30_full_docs_agent as v30  # noqa: E402
from experiments import run_appworld_v32_code_history_agent as v32  # noqa: E402

AUTH_CONTRACT = """

Authentication contract: most non-login app APIs require an `access_token`
argument on every call; a successful login does not implicitly authenticate
later calls. Before using a protected API, inspect the target app's login
documentation, obtain that app's exact credentials through documented
supervisor APIs, call login with the documented fields, print and inspect the
response, save its returned access token in a variable, and pass
`access_token=<saved token>` explicitly to every later protected call for that
app. Never reuse another app's token. HTTP 401 means the token is missing or
wrong: verify the login result and token argument before retrying. HTTP 422
means the call schema is wrong: reread that API's detailed documentation and
fix its argument names and types before retrying.
""".strip()
PROMPT_POLICY = "explicit_per_app_access_token_and_401_422_recovery"

_original_initial_messages = v30.base._initial_messages
_original_run_task = v32.run_task


def initial_messages(
    instruction: str,
    supervisor: object,
    prompt_variant: str,
    memory_hint: str | None,
    doc_hint: str | None = None,
) -> list[dict[str, str]]:
    messages = _original_initial_messages(
        instruction, supervisor, prompt_variant, memory_hint, doc_hint
    )
    result = [item.copy() for item in messages]
    result[0]["content"] = result[0]["content"] + "\n\n" + AUTH_CONTRACT
    return result


def run_task(*args: object, **kwargs: object) -> dict[str, object]:
    row = _original_run_task(*args, **kwargs)
    row["auth_prompt_policy"] = PROMPT_POLICY
    row["auth_prompt_script_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    return row


def main() -> int:
    v30.base._initial_messages = initial_messages
    v30.context_messages = v32.context_messages
    v30.run_task = run_task
    return v30.main()


if __name__ == "__main__":
    raise SystemExit(main())
