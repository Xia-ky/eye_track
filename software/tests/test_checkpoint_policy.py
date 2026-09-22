"""最佳检查点优先优化 20 ms 未来坐标。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE_ROOT))

from see_v1.checkpoints import CheckpointScore, is_better_score


class CheckpointPolicyTests(unittest.TestCase):
    def test_future_distance_has_priority(self) -> None:
        best = CheckpointScore(future_distance=5.0, current_distance=4.0)
        candidate = CheckpointScore(
            future_distance=4.9,
            current_distance=100.0,
        )
        self.assertTrue(is_better_score(candidate, best))

    def test_current_distance_breaks_future_tie(self) -> None:
        best = CheckpointScore(future_distance=5.0, current_distance=4.0)
        self.assertTrue(
            is_better_score(
                CheckpointScore(5.0, 3.9),
                best,
            )
        )
        self.assertFalse(
            is_better_score(
                CheckpointScore(5.0, 4.1),
                best,
            )
        )

    def test_first_finite_score_is_accepted(self) -> None:
        self.assertTrue(
            is_better_score(
                CheckpointScore(5.0, 4.0),
                None,
            )
        )


if __name__ == "__main__":
    unittest.main()
