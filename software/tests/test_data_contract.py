"""固定阶段的数据几何必须保持论文的 80×60×3 表示。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE_ROOT))

from see_v1.config import load_config
from see_v1.data import (
    FutureRegressionDataset,
    build_cache_tag,
    build_record_paths,
    derive_geometry,
)


class DataGeometryTests(unittest.TestCase):
    def test_fixed_baseline_geometry(self) -> None:
        config = load_config(
            SOFTWARE_ROOT / "configs" / "version1_fixed50.json"
        )
        geometry = derive_geometry(config)
        self.assertEqual(80, geometry.width)
        self.assertEqual(60, geometry.height)
        self.assertEqual(50_000, geometry.clip_us)
        self.assertEqual(1_500_000, geometry.sequence_us)
        self.assertEqual((30, 3, 60, 80), geometry.sample_shape)

    def test_cache_tag_separates_future_supervision(self) -> None:
        config = load_config(
            SOFTWARE_ROOT / "configs" / "version1_fixed50.json"
        )
        self.assertEqual(
            "fixed50_seq30_bins3_future20",
            build_cache_tag(config),
        )

    def test_explicit_split_list_controls_record_paths(self) -> None:
        root = Path("event_data")
        records = build_record_paths(root, ("1_2",))
        self.assertEqual(
            (
                (
                    root / "train" / "1_2" / "1_2.h5",
                    root / "train" / "1_2" / "label.txt",
                ),
            ),
            records,
        )


class DataAugmentationTests(unittest.TestCase):
    def test_translation_uses_yx_array_axes_for_xy_label_offsets(self) -> None:
        """图像 `(y,x)` 轴必须与标签 `(x,y)` 位移保持一致。"""

        import numpy as np

        dataset = FutureRegressionDataset(
            cached_dataset=(),
            training=True,
            flip_probability=0.0,
            shift_probability=1.0,
            max_shift=2,
        )
        frames = np.zeros((1, 1, 10, 12), dtype=np.float32)
        frames[0, 0, 4, 5] = 1.0
        targets = np.asarray(
            [[5 / 12, 4 / 10, 1.0, 5 / 12, 4 / 10]],
            dtype=np.float32,
        )

        # x 向右移动 2，y 向上移动 1；固定随机值使测试可重复。
        with patch(
            "see_v1.data.random.random",
            side_effect=(1.0, 0.0),
        ), patch(
            "see_v1.data.random.randint",
            side_effect=(2, -1),
        ):
            shifted_frames, shifted_targets = dataset._augment(
                frames,
                targets,
            )

        self.assertEqual(1.0, shifted_frames[0, 0, 3, 7])
        self.assertAlmostEqual(7 / 12, shifted_targets[0, 0])
        self.assertAlmostEqual(3 / 10, shifted_targets[0, 1])
        self.assertAlmostEqual(7 / 12, shifted_targets[0, 3])
        self.assertAlmostEqual(3 / 10, shifted_targets[0, 4])


if __name__ == "__main__":
    unittest.main()
