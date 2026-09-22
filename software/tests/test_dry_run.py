"""dry-run 只能做静态检查，不能导入 CUDA 模型。"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]


class DryRunTests(unittest.TestCase):
    def test_dry_run_validates_complete_contract_without_cuda(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--config",
                "configs/version1_fixed50.json",
                "--dry-run",
            ],
            cwd=SOFTWARE_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("STATIC_CHECK_PASSED", result.stdout)
        self.assertIn("student_blocks=7", result.stdout)
        self.assertIn("teacher=official_int8", result.stdout)


if __name__ == "__main__":
    unittest.main()
