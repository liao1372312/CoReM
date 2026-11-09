#!/usr/bin/env python3
"""Train-only agent preserving the latest complete API-documentation reply."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import time
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_appworld_local_agent_pilot as base  # noqa: E402
from experiments.run_appworld_v25_api_guard import safe_api_names  # noqa: E402
from experiments.run_appworld_v26_nested_guard import guard_code  # noqa: E402

_original_execute_and_flush = base._execute_and_flush
_original_context_messages = base._context_messages
_guard_interventions = 0
MAX_COMPLETE_DOC_CHARS = 12_000
OLDER_OBSERVATION_CHARS = 2_000


def context_messages(
    messages: list[dict[str, str]], strategy: str
) -> list[dict[str, str]]:
    """Preserve task and latest full reply; compact older long observations."""
    selected = _original_context_messages(messages, strategy)
    if strategy != "preserve_task":
        raise ValueError("v30 requires preserve_task history")
    result = [item.copy() for item in selected]
    for index, item in enumerate(result[:-1]):
        if index >= 2 and item["role"] == "user" and len(item["content"]) > OLDER_OBSERVATION_CHARS:
            item["content"] = (
                item["content"][:OLDER_OBSERVATION_CHARS]
                + "\n[Older observation shortened; request details again if needed.]"
            )
    return result


def guarded_execute_and_flush(world: object, code: str) -> str:
    """Use only task-visible names; do not synthesize state-changing calls."""
    global _guard_interventions
    executed, reason = guard_code(code, safe_api_names(world))
    observation = str(_original_execute_and_flush(world, executed))
    if reason is None:
        return observation
    _guard_interventions += 1
    return (
        "API guard: the proposed code was not executed because it references "
        f"an {reason.replace('_', ' ')}. The read-only documentation query "
        "below was executed instead. Choose a documented API name and inspect "
        "its detailed schema before retrying.\n" + observation
    )


def history_observation(code: str, observation: str) -> tuple[str, bool]:
    """Retain an entire recent doc reply; fail rather than silently clip it."""
    is_doc = "apis.api_docs." in code or observation.startswith("API guard:")
    if is_doc:
        if len(observation) > MAX_COMPLETE_DOC_CHARS:
            raise ValueError("documentation reply exceeds declared full-doc limit")
        return observation, True
    return observation[-OLDER_OBSERVATION_CHARS:], False


def run_task(
    task_id: str,
    experiment_name: str,
    endpoint: str,
    model: str,
    max_steps: int,
    seed: int,
    memory_hint: str | None = None,
    context_tokens: int = 4096,
    predict_tokens: int = 256,
    prompt_variant: str = "generic",
    history_strategy: str = "legacy_tail11",
    doc_hint: str | None = None,
    raise_on_failure: bool = False,
) -> dict[str, object]:
    from appworld import AppWorld

    global _guard_interventions
    _guard_interventions = 0
    started = time.perf_counter()
    generated_steps = 0
    model_errors = 0
    execution_errors = 0
    execution_failure_messages = 0
    syntax_failure_messages = 0
    safety_failure_messages = 0
    runtime_failure_messages = 0
    generated_syntax_errors = 0
    markdown_fence_steps = 0
    empty_code_messages = 0
    api_doc_steps = 0
    api_reference_steps = 0
    full_doc_observations = 0
    max_doc_observation_chars = 0
    evaluation_error = False
    success = False
    task_completed = False
    with AppWorld(
        task_id=task_id,
        experiment_name=experiment_name,
        max_interactions=max_steps,
        max_api_calls_per_interaction=50,
        raise_on_unsafe_syntax=True,
        raise_on_failure=raise_on_failure,
        random_seed=base.APPWORLD_RANDOM_SEED,
        timeout_seconds=30,
        ground_truth_mode="minimal",
    ) as world:
        messages = base._initial_messages(
            world.task.instruction, world.task.supervisor,
            prompt_variant, memory_hint, doc_hint,
        )
        for step in range(max_steps):
            try:
                code = base._next_code(
                    endpoint, model, base._context_messages(messages, history_strategy),
                    seed + step, context_tokens, predict_tokens,
                )
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
                model_errors += 1
                break
            generated_steps += 1
            markdown_fence_steps += int("```" in code)
            try:
                ast.parse(code)
            except SyntaxError:
                generated_syntax_errors += 1
            api_doc_steps += int("apis.api_docs." in code)
            api_reference_steps += int("apis." in code)
            try:
                observation = base._execute_and_flush(world, code)
            except Exception:
                execution_errors += 1
                break
            observation_text = str(observation)
            execution_failure_messages += int(
                observation_text.startswith("Execution failed.")
            )
            syntax_failure_messages += int("Syntax error in line:" in observation_text)
            safety_failure_messages += int(
                "Usage of the following" in observation_text
                and "not allowed" in observation_text
            )
            runtime_failure_messages += int(
                observation_text.startswith("Execution failed.")
                and "Syntax error in line:" not in observation_text
                and "Usage of the following" not in observation_text
            )
            empty_code_messages += int(observation_text.startswith("No code available"))
            selected_observation, is_doc = history_observation(code, observation_text)
            full_doc_observations += int(is_doc)
            if is_doc:
                max_doc_observation_chars = max(
                    max_doc_observation_chars, len(selected_observation)
                )
            messages.append({"role": "assistant", "content": code})
            messages.append({"role": "user", "content": selected_observation})
            task_completed = bool(world.task_completed())
            if task_completed:
                break
        try:
            success = bool(world.evaluate().success)
        except Exception:
            evaluation_error = True
    return {
        "task_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
        "success": success, "task_completed": task_completed,
        "steps": generated_steps, "model_errors": model_errors,
        "execution_errors": execution_errors,
        "execution_failure_messages": execution_failure_messages,
        "syntax_failure_messages": syntax_failure_messages,
        "safety_failure_messages": safety_failure_messages,
        "runtime_failure_messages": runtime_failure_messages,
        "generated_syntax_errors": generated_syntax_errors,
        "markdown_fence_steps": markdown_fence_steps,
        "empty_code_messages": empty_code_messages,
        "api_doc_steps": api_doc_steps,
        "api_reference_steps": api_reference_steps,
        "evaluation_error": evaluation_error,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "api_guard_interventions": _guard_interventions,
        "api_guard_policy": "read_only_documentation_for_invalid_nested_api_paths",
        "api_guard_script_sha256": hashlib.sha256(
            (ROOT / "experiments" / "run_appworld_v26_nested_guard.py").read_bytes()
        ).hexdigest(),
        "observation_policy": "latest_complete_api_doc_older_replies_2000_prefix",
        "observation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "full_doc_observations": full_doc_observations,
        "max_doc_observation_chars": max_doc_observation_chars,
    }


def main() -> int:
    base._execute_and_flush = guarded_execute_and_flush
    base._context_messages = context_messages
    base._run_task = run_task
    status = base.main()
    output_index = sys.argv.index("--output") + 1
    payload = json.loads(Path(sys.argv[output_index]).read_text())
    print(json.dumps({
        "guard_interventions": sum(
            row["api_guard_interventions"] for row in payload["private_task_outcomes"]
        ),
        "full_doc_observations": sum(
            row["full_doc_observations"] for row in payload["private_task_outcomes"]
        ),
    }, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
