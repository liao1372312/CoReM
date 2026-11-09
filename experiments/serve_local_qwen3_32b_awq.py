#!/usr/bin/env python3
"""Local-only Qwen3-32B-AWQ server; prompts and code are never logged."""

from __future__ import annotations

import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def sufficient_vram(free_bytes: int, reserve_gib: float) -> bool:
    return free_bytes >= reserve_gib * 1024**3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen3-32B-AWQ")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--min-free-gib", type=float, default=25.0)
    parser.add_argument("--startup-headroom-gib", type=float, default=19.0)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("invalid port")
    if args.min_free_gib < 0 or args.startup_headroom_gib < 0:
        parser.error("VRAM thresholds must be nonnegative")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    free_before, _ = torch.cuda.mem_get_info()
    if not sufficient_vram(
        free_before, args.min_free_gib + args.startup_headroom_gib
    ):
        parser.error("insufficient pre-load VRAM reserve")
    model_dir = args.model_dir.resolve()
    model_digest = hashlib.sha256(
        (model_dir / "model.safetensors.index.json").read_bytes()
    ).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        device_map="cuda:0",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    free_after, _ = torch.cuda.mem_get_info()
    if not sufficient_vram(free_after, args.min_free_gib):
        parser.error("insufficient post-load VRAM reserve")
    stats = {
        "requests": 0,
        "accepted_requests": 0,
        "truncated_requests": 0,
        "rejected_oversize_requests": 0,
        "max_prompt_tokens": 0,
        "prompt_tokens_sum": 0,
        "completion_tokens_sum": 0,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            return

        def reply(self, status: int, body: dict[str, object]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/stats":
                self.reply(200, stats)
            elif self.path == "/api/tags":
                self.reply(200, {"models": [{"name": args.model_name,
                                              "digest": model_digest}]})
            else:
                self.reply(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/chat":
                self.reply(404, {"error": "not_found"})
                return
            if not sufficient_vram(torch.cuda.mem_get_info()[0], args.min_free_gib):
                self.reply(503, {"error": "insufficient_free_vram"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2_000_000:
                    raise ValueError("invalid request size")
                request = json.loads(self.rfile.read(length))
                if (request.get("model") != args.model_name
                        or request.get("stream") is not False):
                    raise ValueError("unsupported model or streaming mode")
                messages = request["messages"]
                if not isinstance(messages, list) or not all(
                    isinstance(item, dict)
                    and item.get("role") in {"system", "user", "assistant"}
                    and isinstance(item.get("content"), str)
                    for item in messages
                ):
                    raise ValueError("invalid messages")
                options = request.get("options", {})
                max_new = min(512, max(1, int(options.get("num_predict", 256))))
                max_context = min(8192, max(128, int(options.get("num_ctx", 4096))))
                torch.manual_seed(int(options.get("seed", 2027)))
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                encoded = tokenizer(prompt, return_tensors="pt")
                prompt_tokens = int(encoded["input_ids"].shape[1])
                stats["requests"] += 1
                stats["max_prompt_tokens"] = max(
                    stats["max_prompt_tokens"], prompt_tokens
                )
                if prompt_tokens > max_context:
                    stats["rejected_oversize_requests"] += 1
                    self.reply(413, {"error": "context_limit_exceeded"})
                    return
                stats["accepted_requests"] += 1
                stats["prompt_tokens_sum"] += prompt_tokens
                input_ids = encoded["input_ids"].to(model.device)
                attention_mask = encoded["attention_mask"].to(model.device)
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                answer = tokenizer.decode(
                    generated[0, input_ids.shape[1]:], skip_special_tokens=True
                ).strip()
                stats["completion_tokens_sum"] += int(
                    generated.shape[1] - input_ids.shape[1]
                )
                self.reply(200, {"message": {"content": answer}, "done": True})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                self.reply(400, {"error": "invalid_request"})
            except Exception as error:
                print(f"Qwen AWQ inference error class: {type(error).__name__}",
                      flush=True)
                self.reply(500, {"error": "inference_failed"})

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Qwen AWQ local server ready on 127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
