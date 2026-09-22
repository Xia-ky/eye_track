"""官方量化教师必须冻结并暴露当前坐标与池化 Tail 特征。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE_ROOT))

from see_v1.config import load_config


# 教师构造依赖 PyTorch、MinkowskiEngine 和 CUDA，轻量本地环境自动跳过。
@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("MinkowskiEngine") is not None,
    "Official quantized teacher integration runs on AutoDL",
)
class AutoDLTeacherTests(unittest.TestCase):
    def test_teacher_is_frozen_and_exposes_tail_features(self) -> None:
        import torch

        from see_v1.teacher import OfficialSeeDTeacher

        config = load_config(
            SOFTWARE_ROOT / "configs" / "version1_fixed50.json"
        )
        teacher = OfficialSeeDTeacher(config).to(config.runtime.device)
        teacher.train(True)
        self.assertFalse(teacher.training)
        self.assertFalse(any(p.requires_grad for p in teacher.parameters()))
        inputs = torch.zeros(
            2, 30, 3, 60, 80, device=config.runtime.device
        )
        inputs[:, :, :, 10, 10] = 1
        outputs = teacher(inputs)
        self.assertEqual((2, 30, 2), tuple(outputs.current.shape))
        self.assertEqual((2, 30, 64), tuple(outputs.tail_features.shape))
        self.assertFalse(outputs.current.requires_grad)
        self.assertFalse(outputs.tail_features.requires_grad)


if __name__ == "__main__":
    unittest.main()
