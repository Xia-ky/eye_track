"""官方教师资产必须完整、可追溯且没有在复制时发生变化。"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = (
    "2e23b7f717bac1c8801ddc969d2e3ade48b1a4facc780c0b7ac48fc5e0be8be6"
)
# 该列表同时保护训练入口、官方数据实现、split 清单和教师权重的完整上传。
WEIGHT = (
    SOFTWARE_ROOT
    / "weights"
    / "Table2"
    / "SEE-D"
    / "model_best_p10_acc.pth"
)
REQUIRED = (
    SOFTWARE_ROOT / "main.py",
    SOFTWARE_ROOT / "dataset" / "ThreeET_plus.py",
    SOFTWARE_ROOT / "dataset_lists" / "train_files.txt",
    SOFTWARE_ROOT / "dataset_lists" / "val_files.txt",
    SOFTWARE_ROOT / "model" / "HAWQ_mobilenetv2.py",
    SOFTWARE_ROOT / "MinkowskiEngine" / "__init__.py",
    SOFTWARE_ROOT / "src" / "coordinate_map_manager.cpp",
    SOFTWARE_ROOT / "pybind" / "minkowski.cpp",
    SOFTWARE_ROOT / "setup.py",
    SOFTWARE_ROOT / "configs" / "model_cfg" / "SEE-D_teacher.json",
    SOFTWARE_ROOT / "weights" / "Table2" / "SEE-D" / "cfg.json",
    WEIGHT,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OfficialAssetTests(unittest.TestCase):
    def test_official_teacher_hash_and_required_assets(self) -> None:
        for path in REQUIRED:
            self.assertTrue(path.is_file(), path)
        self.assertEqual(EXPECTED_SHA256, sha256(WEIGHT))


if __name__ == "__main__":
    unittest.main()
