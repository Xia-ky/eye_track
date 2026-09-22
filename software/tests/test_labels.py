"""未来监督只能引用真实存在的 100 Hz 标签。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE_ROOT))

from see_v1.labels import build_target_index_pairs


class TargetIndexTests(unittest.TestCase):
    def test_fixed_50ms_current_and_20ms_future_indices(self) -> None:
        self.assertEqual(
            ((0, 2), (5, 7), (10, 12), (15, 17)),
            build_target_index_pairs(
                row_count=20,
                label_period_ms=10,
                clip_ms=50,
                future_ms=20,
            ),
        )

    def test_rows_without_real_future_are_omitted(self) -> None:
        self.assertEqual(
            ((0, 2), (5, 7), (10, 12)),
            build_target_index_pairs(17, 10, 50, 20),
        )

    def test_non_integral_periods_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "clip_ms"):
            build_target_index_pairs(20, 10, 45, 20)
        with self.assertRaisesRegex(ValueError, "future_ms"):
            build_target_index_pairs(20, 10, 50, 15)


if __name__ == "__main__":
    unittest.main()
