"""当前与未来坐标指标必须分开，并在 640×480 像素空间计算。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE_ROOT))

from see_v1.metrics import CoordinateMetricAccumulator


class CoordinateMetricTests(unittest.TestCase):
    def test_current_and_future_metrics_are_independent(self) -> None:
        metrics = CoordinateMetricAccumulator(
            sensor_width=640,
            sensor_height=480,
            tolerances=(5, 10),
        )
        metrics.update(
            current_predictions=[(0.0, 0.0)],
            current_targets=[(3 / 640, 4 / 480)],
            future_predictions=[(0.0, 0.0)],
            future_targets=[(6 / 640, 8 / 480)],
        )
        result = metrics.compute()
        self.assertEqual(1, result["sample_count"])
        self.assertAlmostEqual(5.0, result["current_distance"])
        self.assertAlmostEqual(10.0, result["future_distance"])
        self.assertAlmostEqual(1.0, result["current_p5"])
        self.assertAlmostEqual(0.0, result["future_p5"])
        self.assertAlmostEqual(1.0, result["future_p10"])
        self.assertAlmostEqual(3.0, result["current_x_abs"])
        self.assertAlmostEqual(4.0, result["current_y_abs"])

    def test_empty_metrics_are_rejected(self) -> None:
        metrics = CoordinateMetricAccumulator(640, 480, (5, 10))
        with self.assertRaisesRegex(ValueError, "no samples"):
            metrics.compute()


if __name__ == "__main__":
    unittest.main()
