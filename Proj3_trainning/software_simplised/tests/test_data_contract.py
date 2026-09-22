"""SEE-D 数据几何和时间单位契约。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = SOFTWARE_ROOT / "configs"
sys.path.insert(0, str(SOFTWARE_ROOT))

from esda.config import load_config
from esda.data.factory import derive_data_geometry


class DataGeometryTests(unittest.TestCase):
    def test_paper_geometry_is_80_by_60_and_thirty_steps(self) -> None:
        config = load_config(CONFIG_ROOT / "see_d_float32.json")
        geometry = derive_data_geometry(config.data)

        self.assertEqual(80, geometry.width)
        self.assertEqual(60, geometry.height)
        self.assertEqual(50_000, geometry.short_window_us)
        self.assertEqual(1_500_000, geometry.long_window_us)
        self.assertEqual(30, geometry.sequence_length)
        self.assertEqual((30, 3, 60, 80), geometry.sample_shape)


@unittest.skipUnless(
    importlib.util.find_spec("numpy") is not None,
    "NumPy is installed in the AutoDL environment",
)
class LabelTransformTests(unittest.TestCase):
    def test_scale_subsample_normalize_matches_original_order(self) -> None:
        import numpy as np

        from esda.data.transforms import (
            NormalizeLabel,
            ScaleLabel,
            TemporalSubsample,
        )

        labels = np.array(
            [[80.0 + index, 40.0 + index, float(index)] for index in range(10)],
            dtype=np.float32,
        )
        transformed = ScaleLabel(0.125)(labels.copy())
        transformed = TemporalSubsample(0.2)(transformed)
        transformed = NormalizeLabel(80, 60)(transformed)

        self.assertEqual((2, 3), transformed.shape)
        np.testing.assert_allclose(
            transformed[:, :2],
            np.array([[0.125, 1.0 / 12.0], [0.1328125, 0.09375]]),
        )


if __name__ == "__main__":
    unittest.main()
