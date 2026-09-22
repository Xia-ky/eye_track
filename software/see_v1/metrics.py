"""当前位置和 20 ms 未来位置的流式像素指标。"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _to_rows(value):
    """将 Tensor 或嵌套序列统一转成 CPU 行列表。"""

    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return value


class CoordinateMetricAccumulator:
    """流式累计当前和未来坐标的原始传感器像素指标。"""

    def __init__(
        self,
        sensor_width: int,
        sensor_height: int,
        tolerances: Sequence[int],
    ):
        self.width = sensor_width
        self.height = sensor_height
        self.tolerances = tuple(tolerances)
        self.count = 0
        self.distance = {"current": 0.0, "future": 0.0}
        self.x_abs = {"current": 0.0, "future": 0.0}
        self.y_abs = {"current": 0.0, "future": 0.0}
        self.correct = {
            phase: {tolerance: 0 for tolerance in self.tolerances}
            for phase in ("current", "future")
        }

    def _update_phase(self, phase, predictions, targets):
        """累计单个预测阶段的欧氏距离、轴向误差和命中率。"""

        for prediction, target in zip(
            _to_rows(predictions),
            _to_rows(targets),
        ):
            # 模型输出是归一化坐标；指标恢复到 640×480 原始像素尺度。
            x_error = abs(float(prediction[0]) - float(target[0])) * self.width
            y_error = abs(float(prediction[1]) - float(target[1])) * self.height
            distance = math.hypot(x_error, y_error)
            self.distance[phase] += distance
            self.x_abs[phase] += x_error
            self.y_abs[phase] += y_error
            # P5/P10/P15 使用二维欧氏距离，而非单轴误差。
            for tolerance in self.tolerances:
                if distance <= tolerance:
                    self.correct[phase][tolerance] += 1

    def update(
        self,
        current_predictions,
        current_targets,
        future_predictions,
        future_targets,
    ) -> None:
        """追加一个 batch；四组输入都应展平为 `[B×T,2]`。"""

        current_rows = _to_rows(current_predictions)
        current_target_rows = _to_rows(current_targets)
        future_rows = _to_rows(future_predictions)
        future_target_rows = _to_rows(future_targets)
        size = len(current_rows)
        if not (
            len(current_target_rows)
            == len(future_rows)
            == len(future_target_rows)
            == size
        ):
            raise ValueError("prediction and target lengths must match")
        self._update_phase("current", current_rows, current_target_rows)
        self._update_phase("future", future_rows, future_target_rows)
        self.count += size

    def compute(self):
        """返回截至当前 batch 的平均指标，不清空累计状态。"""

        if not self.count:
            raise ValueError("no samples were accumulated")
        result = {"sample_count": self.count}
        for phase in ("current", "future"):
            result[f"{phase}_distance"] = self.distance[phase] / self.count
            result[f"{phase}_x_abs"] = self.x_abs[phase] / self.count
            result[f"{phase}_y_abs"] = self.y_abs[phase] / self.count
            for tolerance in self.tolerances:
                result[f"{phase}_p{tolerance}"] = (
                    self.correct[phase][tolerance] / self.count
                )
        return result
