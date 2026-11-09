"""Frozen v23 preflight rejects altered manifests and insufficient GPU reserve."""

from __future__ import annotations

import unittest

from experiments.preflight_appworld_v23 import (
    MODEL_INDEX_SHA256,
    STARTUP_MIN_FREE_MIB,
    parse_manifest,
)


class V23PreflightTests(unittest.TestCase):
    def test_manifest_rejects_missing_or_unsafe_entries(self) -> None:
        valid = [
            f"{'a' * 64}  model.safetensors-0000{index}-of-00004.safetensors"
            for index in range(1, 5)
        ] + [
            f"{MODEL_INDEX_SHA256}  model.safetensors.index.json",
            f"{'b' * 64}  tokenizer.json",
        ]
        self.assertEqual(len(parse_manifest("\n".join(valid))), 6)
        with self.assertRaises(ValueError):
            parse_manifest("\n".join(valid[:-1]))
        with self.assertRaises(ValueError):
            parse_manifest("\n".join(valid[:-1] + [f"{'b' * 64}  ../tokenizer.json"]))
        with self.assertRaises(ValueError):
            parse_manifest("\n".join(valid[:-1] + [valid[0]]))

    def test_startup_requirement_is_not_postload_only(self) -> None:
        self.assertEqual(STARTUP_MIN_FREE_MIB, 34 * 1024)


if __name__ == "__main__":
    unittest.main()
