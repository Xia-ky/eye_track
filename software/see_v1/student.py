"""七块稀疏 SEE-D 学生与四 Token 因果 Transformer。"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from MinkowskiEngine.MinkowskiSparseTensor import SparseTensor
from model.mobilenet_submanifold import MobileNetSubmanifold
from model.utils import dense_to_sparse

from .architecture import (
    apply_explicit_residual_policy,
    materialize_legacy_student_model_config,
)
from .config import ExperimentConfig
from .masking import build_torch_attention_mask


class StudentOutputs(NamedTuple):
    """学生训练所需的预测、中间特征和可解释输出。"""

    current: torch.Tensor
    future: torch.Tensor
    coarse_current: torch.Tensor
    spatial_features: torch.Tensor
    tokens: torch.Tensor
    temporal_features: torch.Tensor


class _LegacyArgs(dict):
    """同时支持上游代码使用的属性访问和 `name in args`。"""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


class SparseSpatialEncoder(nn.Module):
    """复用官方稀疏算子，只移除原 GRU 和坐标分类头。"""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        # 通过临时兼容 JSON 复用官方稀疏卷积构造器。
        args = _LegacyArgs(
            device=config.runtime.device,
            model_cfg=str(materialize_legacy_student_model_config(config)),
        )
        upstream = MobileNetSubmanifold(
            args,
            sample_channel=config.student.input_channels,
            num_classes=2,
        )
        # 官方模块会自动推断残差；这里用主 JSON 的显式选择覆盖它。
        apply_explicit_residual_policy(
            upstream.features[1:-1],
            config.student.blocks,
        )
        self.device_name = config.runtime.device
        self.features = upstream.features
        self.pool = upstream.pool
        self.output_channels = config.student.tail_channels
        self.block_order = tuple(block.name for block in config.student.blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """将 `[B,T,C,H,W]` 体素序列编码为 `[B,T,64]`。"""

        if inputs.ndim != 5:
            raise ValueError("student input must have shape [B,T,C,H,W]")
        batch_size, sequence_length, channels, height, width = inputs.shape
        # 每个 clip 独立通过空间主干，因此先合并 batch 与时间维。
        dense = inputs.reshape(
            batch_size * sequence_length,
            channels,
            height,
            width,
        ).permute(0, 2, 3, 1)
        # dense_to_sparse 只保留活跃体素，返回坐标和对应通道特征。
        coordinates, features = dense_to_sparse(dense)
        sparse = SparseTensor(
            features=features.contiguous(),
            coordinates=coordinates.int().contiguous(),
            device=inputs.device,
        )
        # 七块 SCNN 和 Tail 后执行全局稀疏平均池化。
        sparse = self.features(sparse)
        pooled = self.pool(sparse)
        return pooled.F.reshape(
            batch_size,
            sequence_length,
            self.output_channels,
        )


class TokenTransformerStudent(nn.Module):
    """从每个 clip 的 64D 特征生成四个 16D Token。"""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.spatial = SparseSpatialEncoder(config)
        feature_dim = config.student.tail_channels
        token_count = config.tokens.per_clip
        token_dim = config.tokens.dimension
        self.tokens_per_clip = token_count
        self.history_clips = config.transformer.history_clips
        # 每个 64D clip 特征一次性投影为 4×16D Token。
        self.token_projection = nn.Linear(
            feature_dim,
            token_count * token_dim,
        )
        self.token_type_embedding = nn.Embedding(token_count, token_dim)
        self.time_embedding = nn.Embedding(
            config.data.sequence_length,
            token_dim,
        )
        self.auxiliary_projection = nn.Linear(
            config.tokens.auxiliary_dimension,
            token_dim,
        )
        # Transformer 只处理时间上下文；空间关系已由 SCNN 编码。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=config.transformer.heads,
            dim_feedforward=config.transformer.feedforward_dimension,
            dropout=config.transformer.dropout,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer.layers,
        )
        self.coarse_head = nn.Linear(feature_dim, 2)
        self.current_correction_head = nn.Linear(token_dim, 2)
        self.future_delta_head = nn.Linear(token_dim, 2)

        # 给坐标分支一个有意义且稳定的初始状态：
        # - 粗坐标从归一化屏幕中心 (0.5, 0.5) 附近开始；
        # - 当前修正量从 0 开始；
        # - 未来位移从 0 开始，因此初始 future == current。
        #
        # PyTorch 的 Linear 默认初始化会让两个残差头在训练前产生较大的随机
        # 坐标偏移。对像素坐标回归而言，这会使冒烟训练从数百像素误差开始，
        # 同时破坏本来很强的 “20 ms 后位置约等于当前位置” persistence 基线。
        nn.init.normal_(self.coarse_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.coarse_head.bias, 0.5)
        nn.init.zeros_(self.current_correction_head.weight)
        nn.init.zeros_(self.current_correction_head.bias)
        nn.init.zeros_(self.future_delta_head.weight)
        nn.init.zeros_(self.future_delta_head.bias)

    def _event_density(self, inputs: torch.Tensor) -> torch.Tensor:
        """返回每个 clip 在 60×80 网格上的活跃位置比例 `[B,T,1]`。"""

        occupied = inputs.abs().sum(dim=2).ne(0)
        return occupied.float().mean(dim=(-1, -2), keepdim=False).unsqueeze(-1)

    def forward(self, inputs: torch.Tensor) -> StudentOutputs:
        """执行空间编码、Token 时序建模以及当前/未来坐标回归。"""

        spatial_features = self.spatial(inputs)
        batch_size, clip_count, feature_dim = spatial_features.shape
        # 粗坐标直接来自空间特征，为时序分支提供稳定的残差基线。
        coarse_current = self.coarse_head(spatial_features)
        tokens = self.token_projection(spatial_features).reshape(
            batch_size,
            clip_count,
            self.tokens_per_clip,
            self.config.tokens.dimension,
        )

        # 类型嵌入区分同一 clip 内四个 Token，时间嵌入区分 30 个 clip。
        token_ids = torch.arange(
            self.tokens_per_clip,
            device=inputs.device,
        )
        clip_ids = torch.arange(clip_count, device=inputs.device)
        tokens = tokens + self.token_type_embedding(token_ids)[None, None, :, :]
        tokens = tokens + self.time_embedding(clip_ids)[None, :, None, :]

        # 固定 50 ms 基线中时长比恒为 1；该通道为可变 clip 预留接口。
        duration_ratio = torch.ones(
            batch_size,
            clip_count,
            1,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        auxiliary = torch.cat(
            (
                coarse_current,
                self._event_density(inputs),
                duration_ratio,
            ),
            dim=-1,
        )
        tokens = tokens + self.auxiliary_projection(auxiliary)[:, :, None, :]

        # Transformer 期望一维序列，按 clip 优先顺序展开为 120 个 Token。
        flattened = tokens.reshape(
            batch_size,
            clip_count * self.tokens_per_clip,
            self.config.tokens.dimension,
        )
        mask = build_torch_attention_mask(
            clip_count,
            self.tokens_per_clip,
            self.history_clips,
            inputs.device,
        )
        # 掩码只允许访问当前及前四个 clip，阻止未来信息泄漏。
        encoded = self.transformer(
            flattened.transpose(0, 1),
            mask=mask,
        ).transpose(0, 1)
        encoded = encoded.reshape(
            batch_size,
            clip_count,
            self.tokens_per_clip,
            self.config.tokens.dimension,
        )
        # 同一 clip 的四个 Token 均值形成一个 16D 时序特征。
        temporal_features = encoded.mean(dim=2)
        current = coarse_current + self.current_correction_head(
            temporal_features
        )
        # 未来头预测相对当前位置的位移，而不是重新回归绝对坐标。
        future = current + self.future_delta_head(temporal_features)
        return StudentOutputs(
            current=current,
            future=future,
            coarse_current=coarse_current,
            spatial_features=spatial_features,
            tokens=tokens,
            temporal_features=temporal_features,
        )
