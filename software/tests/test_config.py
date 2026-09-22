"""Version 1 单文件 JSON 的结构和交叉约束。"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = SOFTWARE_ROOT / "configs" / "version1_fixed50.json"
sys.path.insert(0, str(SOFTWARE_ROOT))

from see_v1.config import ConfigError, load_config, parse_config


class ConfigTests(unittest.TestCase):
    def _raw(self) -> dict:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def _parse_modified(self, mutate):
        raw = copy.deepcopy(self._raw())
        mutate(raw)
        return parse_config(raw, CONFIG_PATH)

    def test_agreed_fixed_baseline_contract(self) -> None:
        config = load_config(CONFIG_PATH)
        self.assertEqual(7, len(config.student.blocks))
        self.assertEqual(
            [0, 1, 2, 3, 5, 6, 8],
            [block.source_index for block in config.student.blocks],
        )
        self.assertEqual(
            [192, 192, 192, 48, 64, 72, 72],
            [block.expand_channels for block in config.student.blocks],
        )
        self.assertEqual(128, config.student.blocks[-1].out_channels)
        self.assertEqual(64, config.student.tail_channels)
        self.assertEqual(4, config.tokens.per_clip)
        self.assertEqual(16, config.tokens.dimension)
        self.assertEqual(5, config.transformer.history_clips)
        self.assertEqual(20, config.targets.future_ms)
        self.assertEqual(4, config.data.augmentation.max_shift)
        self.assertEqual("official_int8", config.teacher.kind)
        self.assertEqual(32, config.teacher.shift_bit)
        self.assertEqual(32, config.teacher.bias_bit)
        self.assertEqual(
            (1.0, 1.0, 0.2, 0.1),
            config.loss.weights,
        )
        self.assertEqual(5, config.training.early_stopping_patience)
        self.assertAlmostEqual(
            0.1,
            config.training.early_stopping_min_delta,
        )

    def test_unknown_root_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unknown fields"):
            self._parse_modified(lambda raw: raw.update({"mystery": 1}))

    def test_adjacent_channel_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigError, "in_channels"):
            self._parse_modified(
                lambda raw: raw["student"]["blocks"][1].update(
                    {"in_channels": 31}
                )
            )

    def test_illegal_residual_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigError, "residual"):
            self._parse_modified(
                lambda raw: raw["student"]["blocks"][0].update(
                    {"residual": True}
                )
            )

    def test_future_horizon_must_follow_label_period(self) -> None:
        with self.assertRaisesRegex(ConfigError, "future_ms"):
            self._parse_modified(
                lambda raw: raw["targets"].update({"future_ms": 15})
            )

    def test_only_fixed_clip_mode_is_accepted_in_first_milestone(self) -> None:
        with self.assertRaisesRegex(ConfigError, "clip.mode"):
            self._parse_modified(
                lambda raw: raw["data"]["clip"].update({"mode": "adaptive"})
            )

    def test_teacher_hash_must_be_sha256(self) -> None:
        with self.assertRaisesRegex(ConfigError, "sha256"):
            self._parse_modified(
                lambda raw: raw["teacher"].update({"sha256": "abc"})
            )

    def test_student_block_count_and_tail_channels_are_dynamic(self) -> None:
        def mutate(raw):
            # 删除 block_6 后，block_5 仍能直接连接 block_8。
            del raw["student"]["blocks"][5]
            raw["student"]["blocks"][-1]["out_channels"] = 96
            raw["student"]["tail_channels"] = 32

        config = self._parse_modified(mutate)
        self.assertEqual(6, len(config.student.blocks))
        self.assertEqual(96, config.student.blocks[-1].out_channels)
        self.assertEqual(32, config.student.tail_channels)


if __name__ == "__main__":
    unittest.main()
