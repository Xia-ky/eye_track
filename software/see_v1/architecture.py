"""把可读的逐块学生配置转换为上游 SEE-D 兼容模型 JSON。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from .config import ExperimentConfig


def apply_explicit_residual_policy(modules, block_configs) -> None:
    """把主 JSON 的残差开关覆盖到官方倒残差块实例。

    官方实现会仅根据步幅与通道自动推断残差；显式覆盖后，JSON 中的
    `residual=false` 也能真正关闭一个原本满足残差条件的块。
    """

    modules = tuple(modules)
    block_configs = tuple(block_configs)
    if len(modules) != len(block_configs):
        raise ValueError("runtime block count does not match student.blocks")
    for module, block in zip(modules, block_configs):
        if not hasattr(module, "use_residual"):
            raise TypeError("upstream block has no use_residual attribute")
        module.use_residual = block.residual


def build_legacy_student_model_config(
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """把逐块通道配置压缩为官方构造器使用的 backbone 格式。

    官方格式使用 ``[expand_ratio, out_channels, repeats, stride]``，
    因此这里要求扩展通道可被输入通道整除。每个显式块的 repeats 固定为 1，
    块数量完全由主 JSON 的数组长度决定。
    """

    backbone = []
    for block in config.student.blocks:
        expand_ratio = block.expand_channels // block.in_channels
        backbone.append(
            [
                expand_ratio,
                block.out_channels,
                1,
                block.stride,
            ]
        )
    # 只复用官方 features/pool；rnn 字段仅满足上游配置解析要求。
    return {
        "input_channel": config.student.stem_channels,
        "last_channel": config.student.tail_channels,
        "backbone": backbone,
        "pool": {"type": "global_avg"},
        # 上游构造函数要求该字段；学生只复用其 features 和 pool。
        "rnn": {"type": "gru", "units": 64, "num_layers": 1},
    }


def materialize_legacy_student_model_config(
    config: ExperimentConfig,
    directory: Path = None,
) -> Path:
    """把主 JSON 的块数组写成官方构造器可读取的临时 JSON。

    官方 `MobileNetSubmanifold` 只接受文件路径。运行时生成该兼容文件可保证真正
    构建的网络始终来自主配置，不要求使用者同步维护第二份网络 JSON。
    """

    # 内容摘要使相同结构复用同一配置文件，不同结构自然隔离缓存。
    serialized = serialize_legacy_student_model_config(config)
    identity = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    root = (
        Path(directory)
        if directory is not None
        else Path(config.data.cache_root) / "_model_configs"
    )
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"student-{identity}.json"
    # 采用同目录临时文件，避免并发启动读取到部分 JSON。
    if not destination.exists():
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(serialized + "\n", encoding="utf-8")
        temporary.replace(destination)
    return destination


def serialize_legacy_student_model_config(
    config: ExperimentConfig,
) -> str:
    """返回确定性的官方兼容 JSON，便于测试和生成内容哈希。"""

    return json.dumps(
        build_legacy_student_model_config(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
