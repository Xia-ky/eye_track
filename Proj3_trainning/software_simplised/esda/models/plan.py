"""在不加载深度学习框架时推导 SEE-D 的逐层结构和参数量。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from esda.config import ExperimentConfig


@dataclass(frozen=True)
class LayerPlan:
    """一个空间层的可阅读描述。

    `cumulative_stride` 是相对于 80x60 体素输入的累计空间步幅。
    """

    name: str
    kind: str
    in_channels: int
    hidden_channels: Optional[int]
    out_channels: int
    stride: int
    use_residual: bool
    cumulative_stride: int


def build_layer_plan(config: ExperimentConfig) -> List[LayerPlan]:
    """按 JSON 顺序生成 Stem、N 个倒残差块和 Tail。"""

    model = config.model
    cumulative_stride = model.stem.stride
    layers = [
        LayerPlan(
            name="stem",
            kind="stem",
            in_channels=model.input_channels,
            hidden_channels=None,
            out_channels=model.stem.out_channels,
            stride=model.stem.stride,
            use_residual=False,
            cumulative_stride=cumulative_stride,
        )
    ]
    for block in model.inverted_residual_blocks:
        cumulative_stride *= block.stride
        layers.append(
            LayerPlan(
                name=block.name,
                kind="inverted_residual",
                in_channels=block.in_channels,
                hidden_channels=block.expand_channels,
                out_channels=block.out_channels,
                stride=block.stride,
                use_residual=block.use_residual,
                cumulative_stride=cumulative_stride,
            )
        )
    layers.append(
        LayerPlan(
            name="tail",
            kind="tail",
            in_channels=model.tail.in_channels,
            hidden_channels=None,
            out_channels=model.tail.out_channels,
            stride=1,
            use_residual=False,
            cumulative_stride=cumulative_stride,
        )
    )
    return layers


def _sparse_conv_parameters(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
) -> int:
    return in_channels * out_channels * kernel_size * kernel_size


def count_trainable_parameters(config: ExperimentConfig) -> int:
    """按原实现的无 bias 稀疏卷积、BN、GRU 和 Linear 计算参数量。"""

    model = config.model
    total = _sparse_conv_parameters(
        model.input_channels,
        model.stem.out_channels,
        model.stem.kernel_size,
    )
    total += 2 * model.stem.out_channels

    for block in model.inverted_residual_blocks:
        hidden = block.expand_channels
        total += block.in_channels * hidden
        total += 2 * hidden
        total += hidden * 3 * 3
        total += 2 * hidden
        total += hidden * block.out_channels
        total += 2 * block.out_channels

    total += _sparse_conv_parameters(
        model.tail.in_channels,
        model.tail.out_channels,
        model.tail.kernel_size,
    )
    total += 2 * model.tail.out_channels

    gru = model.temporal
    layer_input = gru.input_size
    for _ in range(gru.num_layers):
        total += 3 * gru.hidden_size * layer_input
        total += 3 * gru.hidden_size * gru.hidden_size
        total += 6 * gru.hidden_size
        layer_input = gru.hidden_size

    head = model.regression_head
    total += head.in_features * head.out_features + head.out_features
    return total


def format_layer_plan(
    plan: Sequence[LayerPlan],
    parameter_count: int,
) -> str:
    """生成适合终端和日志保存的逐层中文说明。"""

    lines = [
        "name       kind                channels             stride  residual  total_stride"
    ]
    for layer in plan:
        if layer.hidden_channels is None:
            channels = f"{layer.in_channels}->{layer.out_channels}"
        else:
            channels = (
                f"{layer.in_channels}->{layer.hidden_channels}"
                f"->{layer.out_channels}"
            )
        lines.append(
            f"{layer.name:<10} {layer.kind:<19} {channels:<20} "
            f"{layer.stride:<7} residual={'yes' if layer.use_residual else 'no':<3} "
            f"{layer.cumulative_stride}"
        )
    block_count = sum(
        layer.kind == "inverted_residual" for layer in plan
    )
    lines.append(
        f"summary: blocks={block_count}, "
        f"cumulative_stride={plan[-1].cumulative_stride}, "
        f"trainable_parameters={parameter_count}"
    )
    return "\n".join(lines)
