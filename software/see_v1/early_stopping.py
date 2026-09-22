"""基于验证集未来像素距离的 Early Stopping 状态机。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class FutureDistanceEarlyStopping:
    """跟踪具有最小改善阈值的连续未改善 epoch。

    ``best_distance`` 只记录达到 ``min_delta`` 的有效改善基线。检查点仍可
    独立保存更小的微量改善，因此停止策略不会改变模型选择策略。
    """

    patience: int
    min_delta: float
    best_distance: Optional[float] = None
    epochs_without_improvement: int = 0

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be a positive integer")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and non-negative")

    def update(self, distance: float) -> bool:
        """记录一个验证距离，并返回训练是否应当停止。"""

        value = float(distance)
        improved = math.isfinite(value) and (
            self.best_distance is None
            or value < self.best_distance - self.min_delta
        )
        if improved:
            self.best_distance = value
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience
