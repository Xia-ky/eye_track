"""Transformer 只能读取当前及最近四个历史 clip。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE_ROOT))

from see_v1.masking import build_attention_permissions


class AttentionPermissionTests(unittest.TestCase):
    def test_future_clips_are_blocked(self) -> None:
        permissions = build_attention_permissions(6, 4, 5)
        query = 2 * 4
        for key in range(0, 3 * 4):
            expected = key // 4 <= 2
            self.assertEqual(expected, permissions[query][key])

    def test_same_clip_tokens_can_share_information(self) -> None:
        permissions = build_attention_permissions(3, 4, 5)
        query = 2 * 4
        self.assertTrue(all(permissions[query][2 * 4 : 3 * 4]))

    def test_history_is_limited_to_five_clips_including_current(self) -> None:
        permissions = build_attention_permissions(7, 4, 5)
        query = 6 * 4
        self.assertFalse(any(permissions[query][0 * 4 : 2 * 4]))
        self.assertTrue(all(permissions[query][2 * 4 : 7 * 4]))

    def test_invalid_dimensions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_attention_permissions(0, 4, 5)


if __name__ == "__main__":
    unittest.main()
