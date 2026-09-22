"""根据已验证 JSON 显式选择模型，不使用 `eval()` 或通配符导入。"""

from __future__ import annotations

from esda.config import ExperimentConfig


def select_model_kind(config: ExperimentConfig) -> str:
    """返回建模分支；配置校验已保证只会出现这两种模式。"""

    return config.training.mode


def build_model(config: ExperimentConfig):
    """延迟导入 CUDA 相关模块，使本地仍可读取和检查 JSON。"""

    kind = select_model_kind(config)
    if kind == "float32":
        from .see_d import SeeDModel

        return SeeDModel(config)
    if kind == "int8_qat":
        from .quantization import QuantizedSeeDModel

        return QuantizedSeeDModel(config)
    raise ValueError(f"unsupported model kind: {kind}")
