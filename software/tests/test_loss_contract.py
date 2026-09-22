"""四项损失权重不能在训练代码中隐式改变。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE_ROOT))

from see_v1.losses import combine_loss_values


class LossContractTests(unittest.TestCase):
    def test_agreed_weighted_sum(self) -> None:
        self.assertAlmostEqual(
            1.0 + 2.0 + 0.2 * 3.0 + 0.1 * 4.0,
            combine_loss_values(
                current=1.0,
                future=2.0,
                output_distill=3.0,
                feature_distill=4.0,
                weights=(1.0, 1.0, 0.2, 0.1),
            ),
        )


if __name__ == "__main__":
    unittest.main()
