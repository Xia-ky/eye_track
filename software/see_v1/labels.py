"""从 100 Hz 原始标签构造当前坐标和真实的未来坐标。"""

from __future__ import annotations

from typing import Sequence, Tuple


def build_target_index_pairs(
    row_count: int,
    label_period_ms: int,
    clip_ms: int,
    future_ms: int,
) -> Tuple[Tuple[int, int], ...]:
    """返回每个 clip 的当前标签行和未来标签行。

    标签没有时间戳，因此按数据集约定的固定 100 Hz 周期换算。不存在
    真实未来行的尾部样本被丢弃，绝不复制最后一行作为伪未来标签。
    """

    if row_count < 0:
        raise ValueError("row_count must be non-negative")
    if label_period_ms <= 0:
        raise ValueError("label_period_ms must be positive")
    if clip_ms <= 0 or clip_ms % label_period_ms:
        raise ValueError("clip_ms must be a positive label-period multiple")
    if future_ms <= 0 or future_ms % label_period_ms:
        raise ValueError("future_ms must be a positive label-period multiple")

    # 固定 50 ms clip 在 100 Hz 标签中每隔 5 行取一个当前目标；
    # 20 ms 未来目标则相对当前行前进 2 行。
    current_step = clip_ms // label_period_ms
    future_step = future_ms // label_period_ms
    return tuple(
        (current, current + future_step)
        for current in range(0, row_count, current_step)
        if current + future_step < row_count
    )


def build_current_future_labels(
    labels,
    label_period_ms: int,
    clip_ms: int,
    future_ms: int,
    sensor_width: int,
    sensor_height: int,
):
    """把原始 `[x,y,state,...]` 标签转换为归一化五列监督。

    返回形状为 ``[clip_count, 5]``，列顺序固定为
    ``[current_x, current_y, state, future_x, future_y]``。
    """

    import numpy as np

    source = np.asarray(labels, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] < 2:
        raise ValueError("labels must have shape [rows, >=2]")
    pairs = build_target_index_pairs(
        source.shape[0],
        label_period_ms,
        clip_ms,
        future_ms,
    )
    # 坐标除以原始传感器宽高，而不是 1/8 下采样后的体素尺寸。
    result = np.empty((len(pairs), 5), dtype=np.float32)
    for output_index, (current_index, future_index) in enumerate(pairs):
        current = source[current_index]
        future = source[future_index]
        result[output_index, 0] = current[0] / sensor_width
        result[output_index, 1] = current[1] / sensor_height
        result[output_index, 2] = current[2] if current.shape[0] > 2 else 1.0
        result[output_index, 3] = future[0] / sensor_width
        result[output_index, 4] = future[1] / sensor_height
    return result
