"""动态 Float32/QAT 模型工厂及 AutoDL 张量结构测试。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = SOFTWARE_ROOT / "configs"
sys.path.insert(0, str(SOFTWARE_ROOT))

from esda.config import load_config
from esda.models.factory import select_model_kind


class ModelFactoryTests(unittest.TestCase):
    def test_training_mode_selects_explicit_model_kind(self) -> None:
        float_config = load_config(CONFIG_ROOT / "see_d_float32.json")
        qat_config = load_config(CONFIG_ROOT / "see_d_int8_qat.json")
        self.assertEqual("float32", select_model_kind(float_config))
        self.assertEqual("int8_qat", select_model_kind(qat_config))


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("MinkowskiEngine") is not None,
    "PyTorch and MinkowskiEngine are installed on AutoDL",
)
class AutoDLModelStructureTests(unittest.TestCase):
    def test_float_models_support_dynamic_depth_and_backward(self) -> None:
        import torch

        from esda.models.factory import build_model

        for filename, expected_blocks in (
            ("see_d_float32.json", 9),
            ("see_d_small_float32.json", 5),
        ):
            with self.subTest(config=filename):
                config = load_config(CONFIG_ROOT / filename)
                model = build_model(config)
                self.assertEqual(expected_blocks, len(model.block_order))
                inputs = torch.zeros(2, 30, 3, 60, 80, device=config.runtime.device)
                inputs[:, :, :, 10, 10] = 1
                output = model(inputs)
                self.assertEqual((2, 30, 2), tuple(output.shape))
                output.sum().backward()


if __name__ == "__main__":
    unittest.main()
