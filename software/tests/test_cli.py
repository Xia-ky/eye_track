"""CLI 覆盖只改变运行参数，不篡改原始 JSON。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE_ROOT))

from see_v1.cli import apply_overrides, build_parser
from see_v1.config import ConfigError, load_config


class CliTests(unittest.TestCase):
    def test_training_overrides_are_explicit(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--config",
                str(SOFTWARE_ROOT / "configs" / "version1_fixed50.json"),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--workers",
                "0",
                "--data-root",
                "/tmp/data",
                "--train-list",
                "/tmp/lists/train_files.txt",
                "--val-list",
                "/tmp/lists/val_files.txt",
                "--metadata-root",
                "/tmp/metadata",
                "--cache-root",
                "/tmp/cache",
                "--output",
                "/tmp/output",
                "--dry-run",
            ]
        )
        original = load_config(args.config)
        updated = apply_overrides(original, args)
        self.assertEqual(1, updated.training.epochs)
        self.assertEqual(2, updated.training.batch_size)
        self.assertEqual(0, updated.training.workers)
        self.assertEqual(Path("/tmp/data"), updated.data.root)
        self.assertEqual(
            Path("/tmp/lists/train_files.txt"),
            updated.data.train_list,
        )
        self.assertEqual(
            Path("/tmp/lists/val_files.txt"),
            updated.data.val_list,
        )
        self.assertEqual(Path("/tmp/metadata"), updated.data.metadata_root)
        self.assertEqual(Path("/tmp/cache"), updated.data.cache_root)
        self.assertEqual(Path("/tmp/output"), updated.output.root)
        self.assertTrue(args.dry_run)
        self.assertEqual(100, original.training.epochs)

    def test_invalid_numeric_override_is_rejected(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--config",
                str(SOFTWARE_ROOT / "configs" / "version1_fixed50.json"),
                "--epochs",
                "0",
            ]
        )
        with self.assertRaises(ConfigError):
            apply_overrides(load_config(args.config), args)


if __name__ == "__main__":
    unittest.main()
