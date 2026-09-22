"""Version 1 命令行解析和不可变配置覆盖。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import ConfigError, ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    """创建训练入口解析器；所有可选项均覆盖 JSON 中的同名运行参数。"""

    parser = argparse.ArgumentParser(
        description="Train the Version 1 teacher-student eye tracker"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--train-list", type=Path)
    parser.add_argument("--val-list", type=Path)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def apply_overrides(
    config: ExperimentConfig,
    args: argparse.Namespace,
) -> ExperimentConfig:
    """返回应用 CLI 覆盖后的新配置，不修改原始冻结 dataclass。

    Proj2 和 Proj3 通过该接口替换数据、缓存与输出路径，同时共享同一个
    网络和损失 JSON。数值覆盖在构造新对象前完成边界校验。
    """

    # argparse 的 None 表示调用者未覆盖，应保留 JSON 原值。
    if args.epochs is not None and args.epochs < 1:
        raise ConfigError("--epochs must be a positive integer")
    if args.batch_size is not None and args.batch_size < 1:
        raise ConfigError("--batch-size must be a positive integer")
    if args.workers is not None and args.workers < 0:
        raise ConfigError("--workers must be a non-negative integer")
    # 按配置子域分别 replace，避免手工重建大型 ExperimentConfig。
    training = replace(
        config.training,
        epochs=(
            args.epochs
            if args.epochs is not None
            else config.training.epochs
        ),
        batch_size=(
            args.batch_size
            if args.batch_size is not None
            else config.training.batch_size
        ),
        workers=(
            args.workers
            if args.workers is not None
            else config.training.workers
        ),
    )
    data = replace(
        config.data,
        root=(
            args.data_root
            if args.data_root is not None
            else config.data.root
        ),
        train_list=(
            args.train_list
            if args.train_list is not None
            else config.data.train_list
        ),
        val_list=(
            args.val_list
            if args.val_list is not None
            else config.data.val_list
        ),
        metadata_root=(
            args.metadata_root
            if args.metadata_root is not None
            else config.data.metadata_root
        ),
        cache_root=(
            args.cache_root
            if args.cache_root is not None
            else config.data.cache_root
        ),
    )
    output = replace(
        config.output,
        root=(
            args.output
            if args.output is not None
            else config.output.root
        ),
    )
    runtime = replace(
        config.runtime,
        device=(
            args.device
            if args.device is not None
            else config.runtime.device
        ),
    )
    # 顶层对象同样保持不可变语义，便于保存可靠的配置快照。
    return replace(
        config,
        training=training,
        data=data,
        output=output,
        runtime=runtime,
    )
