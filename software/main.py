#!/usr/bin/env python3
"""Version 1 固定 50 ms 教师—学生训练入口。"""

from __future__ import annotations

import hashlib
from pathlib import Path

from see_v1.cli import apply_overrides, build_parser
from see_v1.config import load_config


def _sha256(path: Path) -> str:
    """以流式方式计算文件摘要，避免一次性把大型检查点读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def static_check(config) -> None:
    """在导入 GPU 训练依赖前验证本次运行所需的静态资产。

    本检查同时保护教师权重的来源完整性。数据根目录允许在本地缺失，
    因为 Windows 端主要执行 dry-run，真实数据只存在于 AutoDL。
    """

    # 这些文件均应随代码发布，缺失意味着上传或目录结构不完整。
    required = (
        config.teacher.model_cfg,
        config.teacher.checkpoint,
        config.data.train_list,
        config.data.val_list,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required files: " + ", ".join(missing))
    # 配置中的固定摘要用于阻止误用个人训练权重或损坏的官方检查点。
    actual_hash = _sha256(config.teacher.checkpoint)
    if actual_hash != config.teacher.sha256:
        raise ValueError(
            "official teacher SHA-256 mismatch: "
            f"expected {config.teacher.sha256}, got {actual_hash}"
        )
    print(
        "STATIC_CHECK_PASSED "
        f"student_blocks={len(config.student.blocks)} "
        f"teacher={config.teacher.kind} "
        f"tokens={config.tokens.per_clip}x{config.tokens.dimension} "
        f"future_ms={config.targets.future_ms}"
    )
    if not config.data.root.exists():
        print(
            "DATA_ROOT_NOT_PRESENT_LOCALLY "
            f"path={config.data.root} "
            "(expected before AutoDL training)"
        )


def main() -> int:
    """解析配置和覆盖项，并在通过静态检查后进入训练主循环。"""

    parser = build_parser()
    args = parser.parse_args()
    # 命令行覆盖优先于 JSON，Proj2/Proj3 因而可复用同一个基准配置。
    config = apply_overrides(load_config(args.config), args)
    static_check(config)
    if args.dry_run:
        return 0
    # 延迟导入使本地无 PyTorch/MinkowskiEngine 环境仍可执行配置检查。
    from see_v1.trainer import train_experiment

    train_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
