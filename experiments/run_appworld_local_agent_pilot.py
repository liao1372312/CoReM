#!/usr/bin/env python3
"""Run a fixed local Ollama agent on AppWorld TRAIN tasks only.

This is an end-to-end state-based development pilot, not a CoReM result.
Task text, generated code, and observations remain in the private AppWorld
root and the local model service; this script prints only aggregate metrics.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = REPOSITORY_ROOT / "data" / "private_corem"
OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/chat"
APPWORLD_RANDOM_SEED = 100  # Pin the installed AppWorld 0.1.3.post1 default.
APPWORLD_VERSION = "0.1.3.post1"


def _require_appworld_version(actual: str) -> None:
    if actual != APPWORLD_VERSION:
        raise ValueError(
            f"AppWorld version mismatch: expected {APPWORLD_VERSION}, got {actual}"
        )


class ModelServerResourceError(RuntimeError):
    """The local inference server refuses work due to GPU contention."""


class ModelContextOverflowError(RuntimeError):
    """The local inference server refuses an oversize prompt."""


SYSTEM_PROMPT = """You are an AppWorld assistant working in a Python REPL.
Solve the user's task by calling the available `apis` object. Output ONLY one
small executable Python code chunk per turn, with no Markdown fences or prose.
Inspect the available apps and API documentation when needed by calling
`apis.api_docs.show_app_descriptions()`,
`apis.api_docs.show_api_descriptions(app_name=...)`, and
`apis.api_docs.show_api_doc(app_name=..., api_name=...)`.
Read each API's parameters and response schema before changing state. Use only
AppWorld APIs, never OS files, processes, network calls, or Python packages for
the user's apps. Variables persist across turns. Check outputs before making
irreversible changes. To finish, call `apis.supervisor.complete_task()` or,
for information questions, `apis.supervisor.complete_task(answer=value)`.
Never use `import`, `from`, `open`, `requests`, `os`, `sys`, or `subprocess`.
Your FIRST reply must be exactly this one Python line:
print(apis.api_docs.show_app_descriptions())
The supervisor app can provide the user's account credentials and personal
details. Never guess an API name, argument, or response field: inspect its
documentation first. If a call fails, use its error and documentation to
repair the call. Do not call `complete_task` until a task-relevant app call
has succeeded and you have checked that the requested change or answer is
actually established. Do not end the task merely because a call failed.
Do not invent credentials, entity identifiers, or facts."""

CHECKLIST_PROMPT = """You are an AppWorld assistant in a persistent Python REPL.
Return only executable Python code, one small complete chunk per turn. No
Markdown or prose. Use `apis` to inspect and act on the user's apps. Begin
with exactly: print(apis.api_docs.show_app_descriptions())
Inspect app/API descriptions and each relevant API's argument and response
schema before use. Print returned values and inspect them before writing.
The supervisor app provides the current user's credentials and personal
details; use the supplied identity context and actual API responses instead
of guessing. Safe Python standard-library modules such as datetime, math,
json, re, and collections are allowed when needed. Never access OS files,
processes, network clients, or app-specific Python packages. Use only the
AppWorld APIs for app state changes.
Treat the user's request as a checklist: satisfy every requested condition,
including time ranges, all relevant pages, and all named entities. After a
state-changing call, read the relevant app state again. Do not call
`apis.supervisor.complete_task()` until each requested change is confirmed
by the returned state. For information tasks, pass only the requested value
as `answer`. A successful API call or the end of an example trace does not
by itself prove that the user's entire task is done. If a call fails, use
its error and the API docs to repair it. Continue until the checklist is
satisfied or the interaction budget ends."""

WORKED_EXAMPLE_PROMPT = CHECKLIST_PROMPT + """

Demonstration only; the values below are dummy values, not facts about the
current task. To answer a request for a Spotify account password, a useful
sequence is:
1. Inspect available apps and then print
   apis.api_docs.show_api_descriptions(app_name='supervisor').
2. Print apis.api_docs.show_api_doc(app_name='supervisor',
   api_name='show_account_passwords') before calling that API.
3. Run passwords = apis.supervisor.show_account_passwords(); print(passwords).
4. Select the actual row whose account_name is 'spotify', then call
   apis.supervisor.complete_task(answer=selected_password).
For the real task, use its own apps, documentation, returned identifiers, and
all requested conditions. Do not copy any demonstration account or value.
Do not finish after reading documentation or after an unsuccessful call."""


def _initial_messages(
    instruction: str,
    supervisor: object,
    prompt_variant: str,
    memory_hint: str | None,
    doc_hint: str | None = None,
) -> list[dict[str, str]]:
    if prompt_variant == "generic":
        system = SYSTEM_PROMPT
        user = "Task:\n" + instruction
    elif prompt_variant in {"checklist", "worked_example"}:
        system = (
            CHECKLIST_PROMPT
            if prompt_variant == "checklist"
            else WORKED_EXAMPLE_PROMPT
        )
        user = (
            "Supervisor identity:\n"
            f"Name: {supervisor.first_name} {supervisor.last_name}\n"
            f"Email: {supervisor.email}\n"
            f"Phone: {supervisor.phone_number}\n"
            "Task:\n" + instruction
        )
    else:
        raise ValueError("unknown prompt variant")
    if doc_hint:
        user += "\n\n" + doc_hint
    if memory_hint:
        user += "\n\n" + memory_hint
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _sample_task_ids(task_ids: list[str], seed: int, count: int) -> list[str]:
    from appworld.task import task_id_to_generator_id

    grouped: dict[str, list[str]] = {}
    for task_id in task_ids:
        grouped.setdefault(task_id_to_generator_id(task_id), []).append(task_id)
    groups = sorted(grouped)
    random.Random(seed).shuffle(groups)
    chosen = [sorted(grouped[group])[0] for group in groups[:count]]
    if len(chosen) != count:
        raise ValueError("not enough distinct train generators")
    return chosen


def _strip_code_fences(value: str) -> str:
    text = value.strip()
    fenced = re.search(
        r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL
    )
    if fenced:
        return fenced.group(1).strip()
    if text.startswith("```python"):
        text = text[len("```python") :]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _context_messages(
    messages: list[dict[str, str]], strategy: str
) -> list[dict[str, str]]:
    """Select conversation turns without mutating the stored transcript."""
    if len(messages) < 2:
        raise ValueError("conversation must contain system and task messages")
    if strategy == "legacy_tail11":
        return [messages[0], *messages[1:][-11:]]
    if strategy == "preserve_task":
        return [messages[0], messages[1], *messages[2:][-8:]]
    if strategy == "preserve_code":
        recent_observations = [
            index
            for index in range(2, len(messages))
            if messages[index]["role"] == "user"
        ][-4:]
        recent_indices = set(recent_observations)
        return [
            {
                "role": message["role"],
                "content": (
                    "[Earlier environment output omitted; rerun a read-only API if needed.]"
                    if index >= 2
                    and message["role"] == "user"
                    and index not in recent_indices
                    else message["content"]
                ),
            }
            for index, message in enumerate(messages)
        ]
    raise ValueError("unknown history strategy")


def _next_code(
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    seed: int,
    context_tokens: int = 4096,
    predict_tokens: int = 256,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_ctx": context_tokens,
            "num_predict": predict_tokens,
        },
        "keep_alive": "5m",
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 503:
            raise ModelServerResourceError("local model lacks GPU reserve") from error
        if error.code == 413:
            raise ModelContextOverflowError("local model prompt exceeds context") from error
        raise
    code = result.get("message", {}).get("content", "")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("local model returned empty code")
    stripped = _strip_code_fences(code)
    if not stripped:
        raise ValueError("local model returned no executable code")
    return stripped


def _model_digest(endpoint: str, model: str) -> str:
    tags_endpoint = endpoint.rsplit("/", 1)[0] + "/tags"
    with urllib.request.urlopen(tags_endpoint, timeout=10) as response:
        data = json.load(response)
    matches = [row["digest"] for row in data.get("models", []) if row["name"] == model]
    if len(matches) != 1:
        raise ValueError("local model name is not uniquely installed")
    return str(matches[0])


def _execute_and_flush(world: object, code: str) -> str:
    """Persist even AppWorld execution paths that return before saving logs."""
    observation = world.execute(code)
    world.save_logs()
    return observation


def _run_task(
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
        random_seed=APPWORLD_RANDOM_SEED,
        timeout_seconds=30,
        ground_truth_mode="minimal",
    ) as world:
        messages = _initial_messages(
            world.task.instruction,
            world.task.supervisor,
            prompt_variant,
            memory_hint,
            doc_hint,
        )
        for step in range(max_steps):
            try:
                code = _next_code(
                    endpoint,
                    model,
                    _context_messages(messages, history_strategy),
                    seed + step,
                    context_tokens,
                    predict_tokens,
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
                observation = _execute_and_flush(world, code)
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
            messages.append({"role": "assistant", "content": code})
            messages.append({"role": "user", "content": observation_text[-2000:]})
            task_completed = bool(world.task_completed())
            if task_completed:
                break
        try:
            success = bool(world.evaluate().success)
        except Exception:
            evaluation_error = True
    return {
        "task_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
        "success": success,
        "task_completed": task_completed,
        "steps": generated_steps,
        "model_errors": model_errors,
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
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--model", default="LLama3:latest")
    parser.add_argument("--endpoint", default=OLLAMA_ENDPOINT)
    parser.add_argument("--num-tasks", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--predict-tokens", type=int, default=256)
    parser.add_argument(
        "--prompt-variant",
        choices=("generic", "checklist", "worked_example"),
        default="generic",
    )
    parser.add_argument(
        "--history-strategy",
        choices=("legacy_tail11", "preserve_task", "preserve_code"),
        default="legacy_tail11",
    )
    parser.add_argument(
        "--raise-on-failure",
        action="store_true",
        help="surface non-200 AppWorld API calls as explicit execution errors",
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--memory-view",
        type=Path,
        help="private, preselected train-only exemplar hints keyed by task hash",
    )
    parser.add_argument(
        "--doc-view",
        type=Path,
        help="private, preselected train-only API documentation keyed by task hash",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error("pilot result must stay under data/private_corem")
    if not root.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error("AppWorld root must be private")
    if output.exists():
        parser.error("refusing to overwrite pilot result")
    if not args.experiment_name.isidentifier():
        parser.error("experiment name must be a Python identifier")
    parsed_endpoint = urllib.parse.urlparse(args.endpoint)
    if (
        parsed_endpoint.scheme != "http"
        or parsed_endpoint.hostname != "127.0.0.1"
        or parsed_endpoint.path != "/api/chat"
        or parsed_endpoint.port is None
    ):
        parser.error("model endpoint must be a 127.0.0.1 /api/chat service")
    if (root / "experiments" / "outputs" / args.experiment_name).exists():
        parser.error("refusing to reuse an AppWorld experiment directory")
    if not 1 <= args.num_tasks <= 30 or not 1 <= args.max_steps <= 50:
        parser.error("invalid task count or step limit")
    if not 512 <= args.context_tokens <= 8192 or not 32 <= args.predict_tokens <= 1024:
        parser.error("invalid context or generation token budget")
    memory_view = args.memory_view.resolve() if args.memory_view else None
    if memory_view and not memory_view.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error("memory view must stay under data/private_corem")
    doc_view = args.doc_view.resolve() if args.doc_view else None
    if doc_view and not doc_view.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error("doc view must stay under data/private_corem")

    from appworld import load_task_ids, update_root

    _require_appworld_version(importlib.metadata.version("appworld"))
    update_root(str(root))
    selected = _sample_task_ids(load_task_ids("train"), args.seed, args.num_tasks)
    memory_payload = json.loads(memory_view.read_text()) if memory_view else None
    if memory_payload is not None:
        if (
            memory_payload.get("split") != "train"
            or memory_payload.get("seed") != args.seed
        ):
            parser.error("memory view split or seed does not match")
        expected_hashes = {
            hashlib.sha256(task_id.encode()).hexdigest() for task_id in selected
        }
        if set(memory_payload.get("hints", {})) != expected_hashes:
            parser.error("memory view must contain exactly the selected train tasks")
    doc_payload = json.loads(doc_view.read_text()) if doc_view else None
    if doc_payload is not None:
        if doc_payload.get("split") != "train" or doc_payload.get("seed") != args.seed:
            parser.error("doc view split or seed does not match")
        expected_hashes = {
            hashlib.sha256(task_id.encode()).hexdigest() for task_id in selected
        }
        if set(doc_payload.get("hints", {})) != expected_hashes:
            parser.error("doc view must contain exactly the selected train tasks")
    model_digest = _model_digest(args.endpoint, args.model)
    outcomes = [
        _run_task(
            task_id,
            args.experiment_name,
            args.endpoint,
            args.model,
            args.max_steps,
            args.seed,
            memory_payload["hints"][hashlib.sha256(task_id.encode()).hexdigest()]
            if memory_payload is not None
            else None,
            args.context_tokens,
            args.predict_tokens,
            args.prompt_variant,
            args.history_strategy,
            doc_payload["hints"][hashlib.sha256(task_id.encode()).hexdigest()]
            if doc_payload is not None
            else None,
            args.raise_on_failure,
        )
        for task_id in selected
    ]
    result = {
        "status": "train_only_state_based_agent_pilot_not_corem_result",
        "benchmark": f"AppWorld {APPWORLD_VERSION}",
        "split": "train",
        "task_sampling": "seeded_distinct_generators_first_variant",
        "model": args.model,
        "model_endpoint": args.endpoint,
        "model_digest": model_digest,
        "seed": args.seed,
        "appworld_random_seed": APPWORLD_RANDOM_SEED,
        "max_steps": args.max_steps,
        "context_tokens": args.context_tokens,
        "predict_tokens": args.predict_tokens,
        "prompt_variant": args.prompt_variant,
        "history_strategy": args.history_strategy,
        "raise_on_failure": args.raise_on_failure,
        "num_tasks": len(outcomes),
        "memory_condition": memory_payload.get("method") if memory_payload else "none",
        "memory_view_sha256": hashlib.sha256(memory_view.read_bytes()).hexdigest()
        if memory_view
        else None,
        "doc_condition": doc_payload.get("method") if doc_payload else "none",
        "doc_view_sha256": hashlib.sha256(doc_view.read_bytes()).hexdigest()
        if doc_view
        else None,
        "successes": sum(bool(row["success"]) for row in outcomes),
        "completed_calls": sum(bool(row["task_completed"]) for row in outcomes),
        "evaluation_errors": sum(bool(row["evaluation_error"]) for row in outcomes),
        "model_errors": sum(int(row["model_errors"]) for row in outcomes),
        "execution_errors": sum(int(row["execution_errors"]) for row in outcomes),
        "execution_failure_messages": sum(
            int(row["execution_failure_messages"]) for row in outcomes
        ),
        "syntax_failure_messages": sum(
            int(row["syntax_failure_messages"]) for row in outcomes
        ),
        "safety_failure_messages": sum(
            int(row["safety_failure_messages"]) for row in outcomes
        ),
        "runtime_failure_messages": sum(
            int(row["runtime_failure_messages"]) for row in outcomes
        ),
        "generated_syntax_errors": sum(
            int(row["generated_syntax_errors"]) for row in outcomes
        ),
        "markdown_fence_steps": sum(
            int(row["markdown_fence_steps"]) for row in outcomes
        ),
        "empty_code_messages": sum(int(row["empty_code_messages"]) for row in outcomes),
        "api_doc_steps": sum(int(row["api_doc_steps"]) for row in outcomes),
        "api_reference_steps": sum(int(row["api_reference_steps"]) for row in outcomes),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "train_split_sha256": hashlib.sha256(
            (root / "data/datasets/train.txt").read_bytes()
        ).hexdigest(),
        "private_task_outcomes": outcomes,
        "protected_text_exported": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"train pilot: {result['successes']}/{result['num_tasks']} state successes, "
        f"{result['evaluation_errors']} evaluation errors, "
        f"{result['model_errors']} model errors, "
        f"{result['execution_failure_messages']} execution failure messages"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
