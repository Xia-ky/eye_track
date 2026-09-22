"""论文官方 INT8 SEE-D 的冻结教师包装器。"""

from __future__ import annotations

import hashlib
from typing import NamedTuple

import torch
from torch import nn

from model.HAWQ_mobilenetv2 import MobileNetSubmanifoldQuant

from .config import ExperimentConfig


class TeacherOutputs(NamedTuple):
    """教师用于蒸馏的当前坐标和池化后 Tail 特征。"""

    current: torch.Tensor
    tail_features: torch.Tensor


class _LegacyArgs(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def _sha256(path) -> str:
    """流式计算官方检查点摘要。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OfficialSeeDTeacher(nn.Module):
    """保持 eval/no_grad，并从全局池化后量化激活提取 64D 特征。"""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        teacher = config.teacher
        # 在 torch.load 前验证来源，避免把错误权重当作固定教师。
        actual_hash = _sha256(teacher.checkpoint)
        if actual_hash != teacher.sha256:
            raise ValueError(
                "official teacher SHA-256 mismatch: "
                f"expected {teacher.sha256}, got {actual_hash}"
            )
        # 旧模型同时使用属性访问和成员查询，包装参数以兼容其接口。
        args = _LegacyArgs(
            device=config.runtime.device,
            model_cfg=str(teacher.model_cfg),
            evaluate=True,
            checkpoint=str(teacher.checkpoint),
            shift_bit=teacher.shift_bit,
            bias_bit=teacher.bias_bit,
            conv1_bit=teacher.conv1_bit,
            fixBN_ratio=teacher.fix_bn_ratio,
        )
        self.teacher = MobileNetSubmanifoldQuant(args)
        # strict=True 保证官方结构与权重键完全一致。
        state_dict = torch.load(
            str(teacher.checkpoint),
            map_location=config.runtime.device,
        )
        self.teacher.load_state_dict(state_dict, strict=True)
        self._tail_features = None
        # 官方 forward 只返回坐标；hook 捕获全局池化后的 64D 蒸馏特征。
        self._hook = self.teacher.model.quant_act_output.register_forward_hook(
            self._capture_tail
        )
        # 冻结参数并锁定 eval，教师不会进入优化器或更新 BatchNorm 状态。
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        super().train(False)

    def _capture_tail(self, _module, _inputs, output):
        """保存当前前向的稀疏 Tail 特征矩阵。"""

        sparse = output[0] if isinstance(output, tuple) else output
        self._tail_features = sparse.F

    def train(self, mode: bool = True):
        """忽略外部模式切换请求，始终保持教师为推理模式。"""

        super().train(False)
        self.teacher.eval()
        return self

    def forward(self, inputs: torch.Tensor) -> TeacherOutputs:
        """运行冻结教师。

        Args:
            inputs: 体素序列，形状为 ``[B,T,3,60,80]``。

        Returns:
            当前坐标 ``[B,T,2]`` 和 Tail 特征 ``[B,T,64]``。
        """

        batch_size, clip_count = inputs.shape[:2]
        # 每次前向先清空 hook 缓存，防止异常路径复用上一 batch 特征。
        self._tail_features = None
        with torch.no_grad():
            current = self.teacher(inputs)
            features = self._tail_features
            if features is None:
                raise RuntimeError("official teacher Tail hook did not run")
            features = features.reshape(batch_size, clip_count, -1)
        return TeacherOutputs(
            current=current.detach(),
            tail_features=features.detach(),
        )
