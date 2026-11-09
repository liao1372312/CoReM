#!/usr/bin/env python3
"""Create a fresh binary salt without printing it."""

from __future__ import annotations

import argparse
import secrets
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bytes", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() or args.bytes < 16:
        parser.error("output must be new and salt length at least 16 bytes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(secrets.token_bytes(args.bytes))
    args.output.chmod(0o600)
    print(f"created {args.bytes}-byte private salt at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
