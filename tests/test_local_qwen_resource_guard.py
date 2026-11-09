"""GPU-reserve checks must fail closed when another process grows."""

from __future__ import annotations

import unittest

from experiments.serve_local_qwen_agent import (
    _can_load_with_reserve,
    _fits_context,
    _has_vram_reserve,
)


class VramReserveTests(unittest.TestCase):
    def test_threshold_inclusive(self) -> None:
        self.assertTrue(_has_vram_reserve(25 * 1024**3, 25.0))
        self.assertFalse(_has_vram_reserve(25 * 1024**3 - 1, 25.0))

    def test_context_limit_is_inclusive(self) -> None:
        self.assertTrue(_fits_context(6144, 6144))
        self.assertFalse(_fits_context(6145, 6144))

    def test_startup_requires_load_headroom_and_serving_reserve(self) -> None:
        self.assertTrue(_can_load_with_reserve(34 * 1024**3, 25.0, 9.0))
        self.assertFalse(_can_load_with_reserve(34 * 1024**3 - 1, 25.0, 9.0))
        with self.assertRaises(ValueError):
            _can_load_with_reserve(34 * 1024**3, 25.0, -1.0)


if __name__ == "__main__":
    unittest.main()
