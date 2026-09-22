"""七块学生的纯配置映射和 AutoDL 张量接口。"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE_ROOT))

from see_v1.architecture import (
    apply_explicit_residual_policy,
    build_legacy_student_model_config,
    serialize_legacy_student_model_config,
)
from see_v1.config import load_config


class StudentArchitectureTests(unittest.TestCase):
    def test_coordinate_heads_have_stable_explicit_initialization(self) -> None:
        # 源码契约防止上传旧版随机输出头后重新出现数百像素初始误差。
        source = (
            SOFTWARE_ROOT / "see_v1" / "student.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "nn.init.normal_(self.coarse_head.weight, mean=0.0, std=0.01)",
            source,
        )
        self.assertIn(
            "nn.init.constant_(self.coarse_head.bias, 0.5)",
            source,
        )
        self.assertIn(
            "nn.init.zeros_(self.current_correction_head.weight)",
            source,
        )
        self.assertIn(
            "nn.init.zeros_(self.current_correction_head.bias)",
            source,
        )
        self.assertIn(
            "nn.init.zeros_(self.future_delta_head.weight)",
            source,
        )
        self.assertIn(
            "nn.init.zeros_(self.future_delta_head.bias)",
            source,
        )

    def test_explicit_blocks_map_to_seven_legacy_stages(self) -> None:
        config = load_config(
            SOFTWARE_ROOT / "configs" / "version1_fixed50.json"
        )
        legacy = build_legacy_student_model_config(config)
        self.assertEqual(
            [
                [8, 32, 1, 2],
                [6, 48, 1, 2],
                [4, 48, 1, 1],
                [1, 64, 1, 1],
                [1, 72, 1, 2],
                [1, 72, 1, 1],
                [1, 128, 1, 1],
            ],
            legacy["backbone"],
        )
        on_disk = json.loads(
            config.student.model_cfg.read_text(encoding="utf-8")
        )
        self.assertEqual(on_disk, legacy)

    def test_runtime_model_config_is_generated_from_main_json(self) -> None:
        config = load_config(
            SOFTWARE_ROOT / "configs" / "version1_fixed50.json"
        )
        self.assertEqual(
            build_legacy_student_model_config(config),
            json.loads(serialize_legacy_student_model_config(config)),
        )

    def test_residual_flags_are_not_inferred_by_upstream_model(self) -> None:
        config = load_config(
            SOFTWARE_ROOT / "configs" / "version1_fixed50.json"
        )

        class DummyBlock:
            use_residual = None

        modules = [DummyBlock() for _ in config.student.blocks]
        apply_explicit_residual_policy(modules, config.student.blocks)
        self.assertEqual(
            [block.residual for block in config.student.blocks],
            [module.use_residual for module in modules],
        )


# 稀疏 CUDA 前向只在 AutoDL 环境执行；本地仍运行上方纯配置测试。
@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("MinkowskiEngine") is not None,
    "PyTorch and MinkowskiEngine integration runs on AutoDL",
)
class AutoDLStudentTests(unittest.TestCase):
    def test_forward_shapes_and_backward(self) -> None:
        import torch

        from see_v1.student import TokenTransformerStudent

        config = load_config(
            SOFTWARE_ROOT / "configs" / "version1_fixed50.json"
        )
        model = TokenTransformerStudent(config).to(config.runtime.device)
        inputs = torch.zeros(
            2, 30, 3, 60, 80, device=config.runtime.device
        )
        inputs[:, :, :, 10, 10] = 1
        outputs = model(inputs)
        self.assertEqual((2, 30, 2), tuple(outputs.current.shape))
        self.assertEqual((2, 30, 2), tuple(outputs.future.shape))
        self.assertEqual((2, 30, 64), tuple(outputs.spatial_features.shape))
        self.assertEqual((2, 30, 4, 16), tuple(outputs.tokens.shape))
        outputs.current.sum().backward()


if __name__ == "__main__":
    unittest.main()
