"""官方教师冻结、七块学生更新的完整训练与验证循环。"""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

from .checkpoints import (
    CheckpointScore,
    is_better_score,
    save_deployment_student,
    save_training_checkpoint,
)
from .config import ExperimentConfig
from .data import build_dataloaders
from .early_stopping import FutureDistanceEarlyStopping
from .metrics import CoordinateMetricAccumulator


def _utc_now() -> str:
    """生成可排序、与服务器时区无关的运行标识时间戳。"""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _jsonable(value):
    """递归转换 Path/dataclass，使配置和报告可直接序列化。"""

    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value) -> None:
    """通过同目录临时文件原子更新 JSON 状态。"""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _set_seed(seed: int) -> None:
    """同步所有随机源，并启用确定性的 cuDNN 算法选择。"""

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _environment_summary():
    """收集复现实验所需的运行时和 GPU 版本信息。"""

    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }


def _run_epoch(
    config: ExperimentConfig,
    student,
    teacher,
    criterion,
    loader,
    optimizer,
    training: bool,
):
    """执行一次训练或验证 epoch。

    Loader 输入为体素 ``[B,T,3,60,80]`` 和标签 ``[B,T,5]``。
    返回值包含四项 batch 平均损失，以及按全部 ``B×T`` clip
    累计的当前/未来像素指标。
    """

    import torch
    import tqdm

    # 学生和蒸馏 Adapter 遵循阶段模式；教师始终保持 eval。
    if training:
        student.train()
        criterion.train()
    else:
        student.eval()
        criterion.eval()
    teacher.train(False)

    metrics = CoordinateMetricAccumulator(
        config.data.sensor_width,
        config.data.sensor_height,
        config.training.pixel_tolerances,
    )
    totals = {
        "loss": 0.0,
        "loss_current": 0.0,
        "loss_future": 0.0,
        "loss_output_distill": 0.0,
        "loss_feature_distill": 0.0,
    }
    batch_count = 0
    description = "Train" if training else "Valid"
    iterator = tqdm.tqdm(loader, desc=description)
    # 验证阶段关闭整条学生图的梯度，减少显存和计算开销。
    grad_context = torch.enable_grad if training else torch.no_grad
    with grad_context():
        for inputs, targets, _masked_lengths in iterator:
            # non_blocking 与训练 DataLoader 的 pinned memory 配合使用。
            inputs = inputs.to(
                config.runtime.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            targets = targets.to(
                config.runtime.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            if optimizer is not None:
                optimizer.zero_grad()
            # 教师与学生查看完全相同的体素，确保蒸馏目标时间对齐。
            teacher_outputs = teacher(inputs)
            student_outputs = student(inputs)
            losses = criterion(
                student_outputs,
                teacher_outputs,
                targets,
            )
            # 尽早中止 NaN/Inf，避免损坏后续检查点和指标文件。
            if not torch.isfinite(losses.total):
                raise FloatingPointError("non-finite total loss")
            if training:
                losses.total.backward()
                optimizer.step()

            totals["loss"] += float(losses.total.detach().cpu())
            totals["loss_current"] += float(losses.current.detach().cpu())
            totals["loss_future"] += float(losses.future.detach().cpu())
            totals["loss_output_distill"] += float(
                losses.output_distill.detach().cpu()
            )
            totals["loss_feature_distill"] += float(
                losses.feature_distill.detach().cpu()
            )
            batch_count += 1

            # 指标按 clip 计算，因此将 batch 和 30 个时间步合并。
            metrics.update(
                student_outputs.current.reshape(-1, 2),
                targets[..., :2].reshape(-1, 2),
                student_outputs.future.reshape(-1, 2),
                targets[..., 3:5].reshape(-1, 2),
            )
            # 进度条显示的是本 epoch 截至当前 batch 的累计均值。
            partial = metrics.compute()
            iterator.set_postfix(
                loss=f"{totals['loss'] / batch_count:.4f}",
                current=f"{partial['current_distance']:.3f}",
                future=f"{partial['future_distance']:.3f}",
            )
    if not batch_count:
        raise RuntimeError(f"{description} loader produced zero batches")
    result = {
        key: value / batch_count for key, value in totals.items()
    }
    result.update(metrics.compute())
    return result


def _append_jsonl(path: Path, record) -> None:
    """追加一条 epoch 记录，保留训练中断前已完成的历史。"""

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")


def _write_epoch_csv(path: Path, records) -> None:
    """原子重写完整 CSV，使列集合可随指标扩展。"""

    if not records:
        return
    fields = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def train_experiment(config: ExperimentConfig) -> Path:
    """训练并返回本次不可覆盖的运行目录。"""

    import torch

    from .losses import DistillationCriterion
    from .student import TokenTransformerStudent
    from .teacher import OfficialSeeDTeacher

    _set_seed(config.training.seed)
    # exist_ok=False 防止时间戳冲突时覆盖历史实验。
    run_id = f"{_utc_now()}-{config.output.run_name}"
    run_dir = config.output.root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    # 在构建数据和模型前写出配置/环境，失败运行同样可追溯。
    config_snapshot = _jsonable(config)
    _write_json(run_dir / "config.json", config_snapshot)
    _write_json(run_dir / "environment.json", _environment_summary())
    report_path = run_dir / "report.json"
    _write_json(
        report_path,
        {
            "status": "running",
            "run_id": run_id,
            "teacher_sha256": config.teacher.sha256,
        },
    )

    try:
        # 所有重依赖均在训练入口内构建；dry-run 不会走到这里。
        train_loader, val_loader = build_dataloaders(config)
        student = TokenTransformerStudent(config).to(config.runtime.device)
        teacher = OfficialSeeDTeacher(config).to(config.runtime.device)
        criterion = DistillationCriterion(config).to(config.runtime.device)
        # 优化器包含学生和损失中的特征 Adapter，不包含冻结教师。
        optimized_parameters = list(student.parameters()) + list(
            criterion.parameters()
        )
        optimizer = torch.optim.Adam(
            optimized_parameters,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        if any(parameter.requires_grad for parameter in teacher.parameters()):
            raise RuntimeError("teacher unexpectedly has trainable parameters")

        best_score = None
        epoch_records = []
        jsonl_path = run_dir / "metrics.jsonl"
        stopper = FutureDistanceEarlyStopping(
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta,
        )
        epochs_completed = 0
        stopped_early = False
        stop_reason = None
        for epoch in range(1, config.training.epochs + 1):
            # 每个 epoch 先训练再在固定验证顺序上评估。
            train_metrics = _run_epoch(
                config,
                student,
                teacher,
                criterion,
                train_loader,
                optimizer,
                training=True,
            )
            val_metrics = _run_epoch(
                config,
                student,
                teacher,
                criterion,
                val_loader,
                optimizer=None,
                training=False,
            )
            # train_/val_ 前缀保持 CSV 和 JSONL 字段无歧义。
            record = {"epoch": epoch}
            record.update(
                {f"train_{key}": value for key, value in train_metrics.items()}
            )
            record.update(
                {f"val_{key}": value for key, value in val_metrics.items()}
            )
            epoch_records.append(record)
            _append_jsonl(jsonl_path, record)
            _write_epoch_csv(run_dir / "metrics.csv", epoch_records)

            # 模型选择以未来 20 ms 误差为主，当前误差只用于打破平局。
            score = CheckpointScore(
                future_distance=val_metrics["future_distance"],
                current_distance=val_metrics["current_distance"],
            )
            # last 每轮覆盖；best 和部署权重只在验证指标改善时更新。
            save_training_checkpoint(
                run_dir / "last.pth",
                epoch,
                student,
                criterion,
                optimizer,
                score,
                config_snapshot,
            )
            if is_better_score(score, best_score):
                best_score = score
                save_training_checkpoint(
                    run_dir / "best_future_distance.pth",
                    epoch,
                    student,
                    criterion,
                    optimizer,
                    score,
                    config_snapshot,
                )
                save_deployment_student(
                    run_dir / "student_deploy.pth",
                    student,
                )
            print(
                f"Epoch {epoch}/{config.training.epochs} "
                f"current={score.current_distance:.4f}px "
                f"future20={score.future_distance:.4f}px"
            )
            epochs_completed = epoch

            # 停止判断位于指标和检查点持久化之后，因此触发轮是完整 epoch。
            if stopper.update(score.future_distance):
                stopped_early = True
                stop_reason = (
                    "val_future_distance did not improve by at least "
                    f"{stopper.min_delta:.4f}px for "
                    f"{stopper.patience} consecutive epochs"
                )
                print(
                    f"Early stopping at epoch {epoch}/"
                    f"{config.training.epochs}: {stop_reason}"
                )
                break

        # 正常跑满和 early stopping 都属于成功完成。
        _write_json(
            report_path,
            {
                "status": "passed",
                "run_id": run_id,
                "epochs": config.training.epochs,
                "configured_epochs": config.training.epochs,
                "epochs_completed": epochs_completed,
                "stopped_early": stopped_early,
                "stop_reason": stop_reason,
                "teacher_sha256": config.teacher.sha256,
                "best_future_distance": best_score.future_distance,
                "best_current_distance": best_score.current_distance,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return run_dir
    except Exception as error:
        # 将完整 traceback 持久化后重新抛出，让 Shell 获得非零退出码。
        _write_json(
            report_path,
            {
                "status": "failed",
                "run_id": run_id,
                "teacher_sha256": config.teacher.sha256,
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
