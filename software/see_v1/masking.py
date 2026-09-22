"""四 Token 序列的 clip 级因果与有限历史掩码。"""

from __future__ import annotations

from typing import Tuple


def build_attention_permissions(
    clip_count: int,
    tokens_per_clip: int,
    history_clips: int,
) -> Tuple[Tuple[bool, ...], ...]:
    """返回 `[query_token][key_token]` 是否允许注意的布尔矩阵。"""

    if clip_count <= 0 or tokens_per_clip <= 0 or history_clips <= 0:
        raise ValueError("clip_count, tokens_per_clip and history_clips must be positive")
    token_count = clip_count * tokens_per_clip
    rows = []
    for query_token in range(token_count):
        # Token 索引整除每 clip Token 数即可恢复所属 clip。
        query_clip = query_token // tokens_per_clip
        earliest_clip = max(0, query_clip - history_clips + 1)
        # 当前 clip 的所有 Token 可互相访问；未来和过旧 clip 被屏蔽。
        rows.append(
            tuple(
                earliest_clip <= key_token // tokens_per_clip <= query_clip
                for key_token in range(token_count)
            )
        )
    return tuple(rows)


def build_torch_attention_mask(
    clip_count: int,
    tokens_per_clip: int,
    history_clips: int,
    device,
):
    """转换为 PyTorch Transformer 接受的 0/-inf 浮点掩码。"""

    import torch

    permissions = build_attention_permissions(
        clip_count,
        tokens_per_clip,
        history_clips,
    )
    # PyTorch Transformer 使用加性浮点掩码：0 保留，-inf 屏蔽。
    allowed = torch.tensor(permissions, dtype=torch.bool, device=device)
    mask = torch.zeros(allowed.shape, dtype=torch.float32, device=device)
    return mask.masked_fill(~allowed, float("-inf"))
