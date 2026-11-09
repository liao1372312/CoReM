#!/usr/bin/env python3
"""One-shot, generic Qwen code-compliance and GPU-capacity check.

This uses no AppWorld task content and does not produce a benchmark score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--min-free-gib", type=float, default=9.0)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    free, _ = torch.cuda.mem_get_info()
    if free < args.min_free_gib * 1024**3:
        print(f"insufficient free VRAM: {free / 1024**3:.2f} GiB")
        return 2
    started = time.perf_counter()
    model_dir = args.model_dir.resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        device_map="cuda:0",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ),
    )
    messages = [
        {
            "role": "system",
            "content": "Output exactly one executable Python line; no Markdown or explanation.",
        },
        {
            "role": "user",
            "content": "First AppWorld step: print the available app descriptions using "
            "apis.api_docs.show_app_descriptions().",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(
        generated[0, encoded.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()
    output = {
        "status": "generic_code_compliance_check_not_benchmark",
        "model_weights_sha256": hashlib.sha256(
            (model_dir / "model-00001-of-00004.safetensors").read_bytes()
        ).hexdigest(),
        "free_vram_before_gib": round(free / 1024**3, 3),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "exact_expected_code": answer == "print(apis.api_docs.show_app_descriptions())",
        "response": answer,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
