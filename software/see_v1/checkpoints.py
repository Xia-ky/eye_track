"""Version 1 检查点选择和原子保存。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CheckpointScore:
    """用于选择部署模型的验证集主、次排序指标。"""

    future_distance: float
    current_distance: float


def is_better_score(
    candidate: CheckpointScore,
    best: Optional[CheckpointScore],
) -> bool:
    """按未来误差优先、当前误差次优的字典序判断候选模型。"""

    values = (candidate.future_distance, candidate.current_distance)
    # NaN/Inf 不得替换任何可用检查点。
    if not all(math.isfinite(value) for value in values):
        return False
    if best is None:
        return True
    return values < (best.future_distance, best.current_distance)


def save_training_checkpoint(
    path: Path,
    epoch: int,
    student,
    criterion,
    optimizer,
    score: CheckpointScore,
    config_snapshot: dict,
) -> None:
    """保存可恢复训练状态，先写临时文件再原子替换。"""

    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    # 完整检查点包含恢复训练所需的学生、蒸馏适配器和优化器状态。
    torch.save(
        {
            "epoch": epoch,
            "student_state_dict": student.state_dict(),
            "distillation_state_dict": criterion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "score": {
                "future_distance": score.future_distance,
                "current_distance": score.current_distance,
            },
            "config": config_snapshot,
        },
        str(temporary),
    )
    temporary.replace(destination)


def save_deployment_student(path: Path, student) -> None:
    """只保存部署所需学生；不含教师和蒸馏 Adapter。"""

    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    # 部署文件保持最小化，推理端无需构建教师、损失或优化器。
    torch.save(student.state_dict(), str(temporary))
    temporary.replace(destination)
