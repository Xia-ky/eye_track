"""确保重构期间不会修改两份作为行为基准的原始 software 目录。"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = Path(__file__).with_name("original_source_manifest.json")


def sha256(path: Path) -> str:
    """分块计算文件哈希，避免把大型检查点一次读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OriginalSourceProtectionTests(unittest.TestCase):
    """原始源码的任何字节变化都应立即使测试失败。"""

    def test_original_files_still_match_baseline(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        original_files = manifest["original_files"]
        self.assertGreater(len(original_files), 0)

        for relative_path, expected_hash in original_files.items():
            path = PROJECT_ROOT / relative_path
            with self.subTest(path=relative_path):
                self.assertTrue(path.is_file(), f"原始文件缺失：{relative_path}")
                self.assertEqual(
                    expected_hash,
                    sha256(path),
                    f"原始文件被修改：{relative_path}",
                )


if __name__ == "__main__":
    unittest.main()
