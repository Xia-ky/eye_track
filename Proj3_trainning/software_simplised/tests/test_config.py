"""JSON 配置契约：网络深度可变，但连接和残差条件必须严格有效。"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = SOFTWARE_ROOT / "configs"
sys.path.insert(0, str(SOFTWARE_ROOT))

from esda.config import ConfigError, load_config, parse_experiment_config


class ConfigTests(unittest.TestCase):
    def _paper_raw(self) -> dict:
        return json.loads(
            (CONFIG_ROOT / "see_d_float32.json").read_text(encoding="utf-8")
        )

    def _parse_modified(self, mutate) -> None:
        raw = copy.deepcopy(self._paper_raw())
        mutate(raw)
        parse_experiment_config(
            raw,
            source_path=CONFIG_ROOT / "modified-for-test.json",
        )

    def test_paper_config_has_expected_explicit_blocks(self) -> None:
        config = load_config(CONFIG_ROOT / "see_d_float32.json")
        blocks = config.model.inverted_residual_blocks
        self.assertEqual(9, len(blocks))
        self.assertEqual(
            [f"block_{index}" for index in range(9)],
            [block.name for block in blocks],
        )
        self.assertEqual(
            [192, 192, 288, 48, 64, 64, 72, 72, 72],
            [block.expand_channels for block in blocks],
        )

    def test_five_block_small_config_is_valid(self) -> None:
        config = load_config(CONFIG_ROOT / "see_d_small_float32.json")
        blocks = config.model.inverted_residual_blocks
        self.assertEqual(
            ["block_0", "block_1", "block_3", "block_5", "block_8"],
            [block.name for block in blocks],
        )
        self.assertEqual(
            ["block_0", "block_1", "block_3", "block_5", "block_8"],
            [block.checkpoint_source for block in blocks],
        )

    def test_empty_block_list_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ConfigError,
            "model.inverted_residual_blocks must contain at least one block",
        ):
            self._parse_modified(
                lambda raw: raw["model"].update(
                    {"inverted_residual_blocks": []}
                )
            )

    def test_adjacent_channel_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ConfigError,
            r"model\.inverted_residual_blocks\[1\]\.in_channels",
        ):
            self._parse_modified(
                lambda raw: raw["model"]["inverted_residual_blocks"][1].update(
                    {"in_channels": 33}
                )
            )

    def test_invalid_residual_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigError, "use_residual"):
            self._parse_modified(
                lambda raw: raw["model"]["inverted_residual_blocks"][0].update(
                    {"use_residual": True}
                )
            )

    def test_unknown_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigError, "root has unknown fields"):
            self._parse_modified(
                lambda raw: raw.update({"mystery_setting": 123})
            )

    def test_duplicate_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigError, "duplicate block name"):
            self._parse_modified(
                lambda raw: raw["model"]["inverted_residual_blocks"][1].update(
                    {"name": "block_0"}
                )
            )

    def test_duplicate_checkpoint_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ConfigError,
            "duplicate checkpoint_source",
        ):
            self._parse_modified(
                lambda raw: raw["model"]["inverted_residual_blocks"][1].update(
                    {"checkpoint_source": "block_0"}
                )
            )


if __name__ == "__main__":
    unittest.main()
