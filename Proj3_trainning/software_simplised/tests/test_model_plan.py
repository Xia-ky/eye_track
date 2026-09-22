"""不依赖 PyTorch 的逐层结构和参数量检查。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = SOFTWARE_ROOT / "configs"
sys.path.insert(0, str(SOFTWARE_ROOT))

from esda.config import load_config
from esda.models.plan import (
    build_layer_plan,
    count_trainable_parameters,
    format_layer_plan,
)


class ModelPlanTests(unittest.TestCase):
    def test_paper_model_plan_matches_original(self) -> None:
        config = load_config(CONFIG_ROOT / "see_d_float32.json")
        plan = build_layer_plan(config)

        blocks = [layer for layer in plan if layer.kind == "inverted_residual"]
        self.assertEqual(9, len(blocks))
        self.assertEqual(16, plan[-1].cumulative_stride)
        self.assertEqual(177_970, count_trainable_parameters(config))
        self.assertEqual(
            [24, 32, 48, 48, 64, 64, 72, 72, 72, 256, 64],
            [layer.out_channels for layer in plan],
        )

    def test_small_model_plan_has_five_selected_blocks(self) -> None:
        config = load_config(CONFIG_ROOT / "see_d_small_float32.json")
        plan = build_layer_plan(config)
        blocks = [layer for layer in plan if layer.kind == "inverted_residual"]

        self.assertEqual(
            ["block_0", "block_1", "block_3", "block_5", "block_8"],
            [block.name for block in blocks],
        )
        self.assertEqual(16, plan[-1].cumulative_stride)
        self.assertLess(count_trainable_parameters(config), 177_970)

    def test_formatted_plan_explains_residual_and_summary(self) -> None:
        config = load_config(CONFIG_ROOT / "see_d_small_float32.json")
        rendered = format_layer_plan(
            build_layer_plan(config),
            count_trainable_parameters(config),
        )

        self.assertIn("block_3", rendered)
        self.assertIn("residual=no", rendered)
        self.assertIn("blocks=5", rendered)
        self.assertIn("cumulative_stride=16", rendered)


if __name__ == "__main__":
    unittest.main()
