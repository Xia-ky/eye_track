"""真实标签监督为主、官方教师蒸馏为辅的四项损失。"""

from __future__ import annotations

from typing import NamedTuple, Sequence


def combine_loss_values(
    current,
    future,
    output_distill,
    feature_distill,
    weights: Sequence[float],
):
    """按固定顺序加权合并当前、未来和两项蒸馏损失。"""

    if len(weights) != 4:
        raise ValueError("weights must contain four values")
    return (
        weights[0] * current
        + weights[1] * future
        + weights[2] * output_distill
        + weights[3] * feature_distill
    )


try:
    import torch
    import torch.nn.functional as functional
    from torch import nn
except ImportError:  # 本地静态检查环境不安装 PyTorch。
    torch = None
    functional = None
    nn = None


if torch is not None:

    class LossBreakdown(NamedTuple):
        """供日志和反向传播共同使用的损失分解。"""

        total: torch.Tensor
        current: torch.Tensor
        future: torch.Tensor
        output_distill: torch.Tensor
        feature_distill: torch.Tensor


    class DistillationCriterion(nn.Module):
        """Adapter 属于训练辅助模块，不放进学生部署权重。"""

        def __init__(self, config):
            super().__init__()
            self.weights = config.loss.weights
            self.register_buffer(
                "coordinate_weights",
                torch.tensor(
                    config.loss.coordinate_weights,
                    dtype=torch.float32,
                ),
            )
            # Adapter 只参与训练，用于把可变学生 Tail 对齐到教师 64D 特征。
            self.feature_adapter = nn.Linear(
                config.student.tail_channels,
                64,
            )

        def _coordinate_loss(self, prediction, target):
            """计算考虑 640:480 宽高比例的归一化坐标 MSE。"""

            error = (prediction - target).pow(2)
            return (error * self.coordinate_weights).mean()

        def forward(
            self,
            student_outputs,
            teacher_outputs,
            targets,
        ):
            """计算四项训练目标并返回总损失及其分量。"""

            current_target = targets[..., :2]
            future_target = targets[..., 3:5]
            current = self._coordinate_loss(
                student_outputs.current,
                current_target,
            )
            future = self._coordinate_loss(
                student_outputs.future,
                future_target,
            )
            # 教师只监督当前位置；20 ms 未来位置始终来自真实数据标签。
            output_distill = functional.smooth_l1_loss(
                student_outputs.current,
                teacher_outputs.current.detach(),
            )
            # detach 明确切断教师梯度，反向只更新学生和 Adapter。
            aligned_features = self.feature_adapter(
                student_outputs.spatial_features
            )
            feature_distill = functional.smooth_l1_loss(
                aligned_features,
                teacher_outputs.tail_features.detach(),
            )
            total = combine_loss_values(
                current,
                future,
                output_distill,
                feature_distill,
                self.weights,
            )
            return LossBreakdown(
                total=total,
                current=current,
                future=future,
                output_distill=output_distill,
                feature_distill=feature_distill,
            )

else:

    class DistillationCriterion:
        def __init__(self, _config):
            raise RuntimeError(
                "DistillationCriterion requires PyTorch on AutoDL"
            )
