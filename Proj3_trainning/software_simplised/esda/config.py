"""读取并严格校验 SEE-D 的单文件 JSON 配置。

本模块只依赖 Python 标准库，因此在没有 CUDA、PyTorch 和
MinkowskiEngine 的电脑上也能先检查网络连接是否正确。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


class ConfigError(ValueError):
    """配置缺失、包含未知字段或网络连接无效。"""


@dataclass(frozen=True)
class DataConfig:
    data_root: str
    train_list: str
    val_list: str
    metadata_root: str
    cache_root: str
    sensor_width: int
    sensor_height: int
    spatial_factor: float
    temporal_subsample_factor: float
    sequence_length: int
    train_stride: int
    val_stride: int
    n_time_bins: int
    voxel_grid_channel_normalization: bool
    flip_probability: float
    shift_probability: float
    max_shift: int


@dataclass(frozen=True)
class StemConfig:
    out_channels: int
    kernel_size: int
    stride: int


@dataclass(frozen=True)
class BlockConfig:
    name: str
    checkpoint_source: Optional[str]
    in_channels: int
    expand_channels: int
    out_channels: int
    stride: int
    use_residual: bool


@dataclass(frozen=True)
class TailConfig:
    in_channels: int
    out_channels: int
    kernel_size: int


@dataclass(frozen=True)
class TemporalConfig:
    kind: str
    input_size: int
    hidden_size: int
    num_layers: int


@dataclass(frozen=True)
class HeadConfig:
    in_features: int
    out_features: int


@dataclass(frozen=True)
class ModelConfig:
    name: str
    input_channels: int
    stem: StemConfig
    inverted_residual_blocks: Tuple[BlockConfig, ...]
    tail: TailConfig
    temporal: TemporalConfig
    regression_head: HeadConfig


@dataclass(frozen=True)
class OptimizerConfig:
    kind: str
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class LossConfig:
    kind: str
    coordinate_weights: Tuple[float, float]


@dataclass(frozen=True)
class TrainingConfig:
    mode: str
    seed: int
    epochs: int
    batch_size: int
    num_workers: int
    optimizer: OptimizerConfig
    loss: LossConfig
    pixel_tolerances: Tuple[int, ...]


@dataclass(frozen=True)
class QuantizationConfig:
    enabled: bool
    shift_bit: int
    bias_bit: int
    conv1_bit: int
    fix_bn_ratio: float
    initialize_from: str


@dataclass(frozen=True)
class CheckpointConfig:
    path: str
    mode: str


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    checkpoint: CheckpointConfig
    debug: bool


@dataclass(frozen=True)
class OutputConfig:
    root: str
    run_name: str


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    quantization: QuantizationConfig
    runtime: RuntimeConfig
    output: OutputConfig
    source_path: Path


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a JSON object")
    return value


def _reject_unknown(
    path: str,
    value: Mapping[str, Any],
    allowed: Sequence[str],
) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ConfigError(f"{path} has unknown fields: {', '.join(unknown)}")
    missing = sorted(set(allowed) - set(value))
    if missing:
        raise ConfigError(f"{path} is missing fields: {', '.join(missing)}")


def _integer(value: Any, path: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{path} must be an integer >= {minimum}")
    return value


def _number(value: Any, path: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ConfigError(f"{path} must be finite and >= {minimum}")
    return result


def _probability(value: Any, path: str) -> float:
    result = _number(value, path)
    if result > 1.0:
        raise ConfigError(f"{path} must be between 0 and 1")
    return result


def _string(value: Any, path: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        suffix = "" if allow_empty else " and non-empty"
        raise ConfigError(f"{path} must be a string{suffix}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be a boolean")
    return value


def _parse_data(raw_value: Any) -> DataConfig:
    raw = _mapping(raw_value, "data")
    allowed = (
        "data_root", "train_list", "val_list", "metadata_root", "cache_root",
        "sensor_width", "sensor_height", "spatial_factor",
        "temporal_subsample_factor", "sequence_length", "train_stride",
        "val_stride", "n_time_bins", "voxel_grid_channel_normalization",
        "augmentation",
    )
    _reject_unknown("data", raw, allowed)
    augmentation = _mapping(raw["augmentation"], "data.augmentation")
    _reject_unknown(
        "data.augmentation",
        augmentation,
        ("flip_probability", "shift_probability", "max_shift"),
    )
    return DataConfig(
        data_root=_string(raw["data_root"], "data.data_root"),
        train_list=_string(raw["train_list"], "data.train_list"),
        val_list=_string(raw["val_list"], "data.val_list"),
        metadata_root=_string(raw["metadata_root"], "data.metadata_root"),
        cache_root=_string(raw["cache_root"], "data.cache_root"),
        sensor_width=_integer(raw["sensor_width"], "data.sensor_width"),
        sensor_height=_integer(raw["sensor_height"], "data.sensor_height"),
        spatial_factor=_number(raw["spatial_factor"], "data.spatial_factor"),
        temporal_subsample_factor=_number(
            raw["temporal_subsample_factor"],
            "data.temporal_subsample_factor",
        ),
        sequence_length=_integer(raw["sequence_length"], "data.sequence_length"),
        train_stride=_integer(raw["train_stride"], "data.train_stride"),
        val_stride=_integer(raw["val_stride"], "data.val_stride"),
        n_time_bins=_integer(raw["n_time_bins"], "data.n_time_bins"),
        voxel_grid_channel_normalization=_boolean(
            raw["voxel_grid_channel_normalization"],
            "data.voxel_grid_channel_normalization",
        ),
        flip_probability=_probability(
            augmentation["flip_probability"],
            "data.augmentation.flip_probability",
        ),
        shift_probability=_probability(
            augmentation["shift_probability"],
            "data.augmentation.shift_probability",
        ),
        max_shift=_integer(
            augmentation["max_shift"],
            "data.augmentation.max_shift",
            minimum=0,
        ),
    )


def _parse_model(raw_value: Any) -> ModelConfig:
    raw = _mapping(raw_value, "model")
    _reject_unknown(
        "model",
        raw,
        (
            "name", "input_channels", "stem", "inverted_residual_blocks",
            "tail", "temporal", "regression_head",
        ),
    )
    stem_raw = _mapping(raw["stem"], "model.stem")
    _reject_unknown(
        "model.stem",
        stem_raw,
        ("out_channels", "kernel_size", "stride"),
    )
    tail_raw = _mapping(raw["tail"], "model.tail")
    _reject_unknown(
        "model.tail",
        tail_raw,
        ("in_channels", "out_channels", "kernel_size"),
    )
    temporal_raw = _mapping(raw["temporal"], "model.temporal")
    _reject_unknown(
        "model.temporal",
        temporal_raw,
        ("type", "input_size", "hidden_size", "num_layers"),
    )
    head_raw = _mapping(raw["regression_head"], "model.regression_head")
    _reject_unknown(
        "model.regression_head",
        head_raw,
        ("in_features", "out_features"),
    )

    blocks_value = raw["inverted_residual_blocks"]
    if not isinstance(blocks_value, list):
        raise ConfigError("model.inverted_residual_blocks must be a JSON array")
    blocks = []
    for index, block_value in enumerate(blocks_value):
        path = f"model.inverted_residual_blocks[{index}]"
        block = _mapping(block_value, path)
        _reject_unknown(
            path,
            block,
            (
                "name", "checkpoint_source", "in_channels",
                "expand_channels", "out_channels", "stride", "use_residual",
            ),
        )
        checkpoint_source = block["checkpoint_source"]
        if checkpoint_source is not None:
            checkpoint_source = _string(
                checkpoint_source,
                f"{path}.checkpoint_source",
            )
        blocks.append(
            BlockConfig(
                name=_string(block["name"], f"{path}.name"),
                checkpoint_source=checkpoint_source,
                in_channels=_integer(
                    block["in_channels"],
                    f"{path}.in_channels",
                ),
                expand_channels=_integer(
                    block["expand_channels"],
                    f"{path}.expand_channels",
                ),
                out_channels=_integer(
                    block["out_channels"],
                    f"{path}.out_channels",
                ),
                stride=_integer(block["stride"], f"{path}.stride"),
                use_residual=_boolean(
                    block["use_residual"],
                    f"{path}.use_residual",
                ),
            )
        )

    return ModelConfig(
        name=_string(raw["name"], "model.name"),
        input_channels=_integer(raw["input_channels"], "model.input_channels"),
        stem=StemConfig(
            out_channels=_integer(
                stem_raw["out_channels"],
                "model.stem.out_channels",
            ),
            kernel_size=_integer(
                stem_raw["kernel_size"],
                "model.stem.kernel_size",
            ),
            stride=_integer(stem_raw["stride"], "model.stem.stride"),
        ),
        inverted_residual_blocks=tuple(blocks),
        tail=TailConfig(
            in_channels=_integer(
                tail_raw["in_channels"],
                "model.tail.in_channels",
            ),
            out_channels=_integer(
                tail_raw["out_channels"],
                "model.tail.out_channels",
            ),
            kernel_size=_integer(
                tail_raw["kernel_size"],
                "model.tail.kernel_size",
            ),
        ),
        temporal=TemporalConfig(
            kind=_string(temporal_raw["type"], "model.temporal.type"),
            input_size=_integer(
                temporal_raw["input_size"],
                "model.temporal.input_size",
            ),
            hidden_size=_integer(
                temporal_raw["hidden_size"],
                "model.temporal.hidden_size",
            ),
            num_layers=_integer(
                temporal_raw["num_layers"],
                "model.temporal.num_layers",
            ),
        ),
        regression_head=HeadConfig(
            in_features=_integer(
                head_raw["in_features"],
                "model.regression_head.in_features",
            ),
            out_features=_integer(
                head_raw["out_features"],
                "model.regression_head.out_features",
            ),
        ),
    )


def _parse_training(raw_value: Any) -> TrainingConfig:
    raw = _mapping(raw_value, "training")
    _reject_unknown(
        "training",
        raw,
        (
            "mode", "seed", "epochs", "batch_size", "num_workers",
            "optimizer", "loss", "pixel_tolerances",
        ),
    )
    optimizer = _mapping(raw["optimizer"], "training.optimizer")
    _reject_unknown(
        "training.optimizer",
        optimizer,
        ("type", "learning_rate", "weight_decay"),
    )
    loss = _mapping(raw["loss"], "training.loss")
    _reject_unknown(
        "training.loss",
        loss,
        ("type", "coordinate_weights"),
    )
    weights = loss["coordinate_weights"]
    if not isinstance(weights, list) or len(weights) != 2:
        raise ConfigError(
            "training.loss.coordinate_weights must contain two numbers"
        )
    tolerances = raw["pixel_tolerances"]
    if not isinstance(tolerances, list) or not tolerances:
        raise ConfigError("training.pixel_tolerances must be a non-empty array")
    return TrainingConfig(
        mode=_string(raw["mode"], "training.mode"),
        seed=_integer(raw["seed"], "training.seed", minimum=0),
        epochs=_integer(raw["epochs"], "training.epochs"),
        batch_size=_integer(raw["batch_size"], "training.batch_size"),
        num_workers=_integer(
            raw["num_workers"],
            "training.num_workers",
            minimum=0,
        ),
        optimizer=OptimizerConfig(
            kind=_string(optimizer["type"], "training.optimizer.type"),
            learning_rate=_number(
                optimizer["learning_rate"],
                "training.optimizer.learning_rate",
            ),
            weight_decay=_number(
                optimizer["weight_decay"],
                "training.optimizer.weight_decay",
            ),
        ),
        loss=LossConfig(
            kind=_string(loss["type"], "training.loss.type"),
            coordinate_weights=(
                _number(weights[0], "training.loss.coordinate_weights[0]"),
                _number(weights[1], "training.loss.coordinate_weights[1]"),
            ),
        ),
        pixel_tolerances=tuple(
            _integer(value, f"training.pixel_tolerances[{index}]")
            for index, value in enumerate(tolerances)
        ),
    )


def _parse_quantization(raw_value: Any) -> QuantizationConfig:
    raw = _mapping(raw_value, "quantization")
    _reject_unknown(
        "quantization",
        raw,
        (
            "enabled", "shift_bit", "bias_bit", "conv1_bit",
            "fix_bn_ratio", "initialize_from",
        ),
    )
    return QuantizationConfig(
        enabled=_boolean(raw["enabled"], "quantization.enabled"),
        shift_bit=_integer(raw["shift_bit"], "quantization.shift_bit"),
        bias_bit=_integer(raw["bias_bit"], "quantization.bias_bit"),
        conv1_bit=_integer(raw["conv1_bit"], "quantization.conv1_bit"),
        fix_bn_ratio=_probability(
            raw["fix_bn_ratio"],
            "quantization.fix_bn_ratio",
        ),
        initialize_from=_string(
            raw["initialize_from"],
            "quantization.initialize_from",
            allow_empty=True,
        ),
    )


def _parse_runtime(raw_value: Any) -> RuntimeConfig:
    raw = _mapping(raw_value, "runtime")
    _reject_unknown("runtime", raw, ("device", "checkpoint", "debug"))
    checkpoint = _mapping(raw["checkpoint"], "runtime.checkpoint")
    _reject_unknown("runtime.checkpoint", checkpoint, ("path", "mode"))
    return RuntimeConfig(
        device=_string(raw["device"], "runtime.device"),
        checkpoint=CheckpointConfig(
            path=_string(
                checkpoint["path"],
                "runtime.checkpoint.path",
                allow_empty=True,
            ),
            mode=_string(checkpoint["mode"], "runtime.checkpoint.mode"),
        ),
        debug=_boolean(raw["debug"], "runtime.debug"),
    )


def _parse_output(raw_value: Any) -> OutputConfig:
    raw = _mapping(raw_value, "output")
    _reject_unknown("output", raw, ("root", "run_name"))
    return OutputConfig(
        root=_string(raw["root"], "output.root"),
        run_name=_string(raw["run_name"], "output.run_name"),
    )


def parse_experiment_config(
    raw_value: Any,
    source_path: Path,
) -> ExperimentConfig:
    """把已解析 JSON 转为不可变配置，并立即执行所有交叉校验。"""

    raw = _mapping(raw_value, "root")
    _reject_unknown(
        "root",
        raw,
        ("data", "model", "training", "quantization", "runtime", "output"),
    )
    config = ExperimentConfig(
        data=_parse_data(raw["data"]),
        model=_parse_model(raw["model"]),
        training=_parse_training(raw["training"]),
        quantization=_parse_quantization(raw["quantization"]),
        runtime=_parse_runtime(raw["runtime"]),
        output=_parse_output(raw["output"]),
        source_path=Path(source_path).resolve(),
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    """校验不能由单个 JSON 字段类型检查覆盖的结构关系。"""

    data = config.data
    if not 0.0 < data.spatial_factor <= 1.0:
        raise ConfigError("data.spatial_factor must be in (0, 1]")
    if not 0.0 < data.temporal_subsample_factor <= 1.0:
        raise ConfigError("data.temporal_subsample_factor must be in (0, 1]")
    reciprocal = 1.0 / data.temporal_subsample_factor
    if not math.isclose(reciprocal, round(reciprocal)):
        raise ConfigError(
            "data.temporal_subsample_factor reciprocal must be an integer"
        )
    for dimension_name, dimension in (
        ("width", data.sensor_width * data.spatial_factor),
        ("height", data.sensor_height * data.spatial_factor),
    ):
        if not math.isclose(dimension, round(dimension)):
            raise ConfigError(
                f"scaled sensor {dimension_name} must be an integer"
            )

    model = config.model
    if model.name != "see_d":
        raise ConfigError("model.name must be 'see_d'")
    blocks = model.inverted_residual_blocks
    if not blocks:
        raise ConfigError(
            "model.inverted_residual_blocks must contain at least one block"
        )
    if model.stem.stride not in (1, 2):
        raise ConfigError("model.stem.stride must be 1 or 2")
    if model.stem.out_channels != blocks[0].in_channels:
        raise ConfigError(
            "model.stem.out_channels must equal "
            "model.inverted_residual_blocks[0].in_channels"
        )

    names = set()
    sources = set()
    valid_name = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
    for index, block in enumerate(blocks):
        path = f"model.inverted_residual_blocks[{index}]"
        if not valid_name.fullmatch(block.name):
            raise ConfigError(
                f"{path}.name must contain only letters, digits, and underscores"
            )
        if block.name in names:
            raise ConfigError(f"duplicate block name: {block.name}")
        names.add(block.name)
        if block.checkpoint_source is not None:
            if not valid_name.fullmatch(block.checkpoint_source):
                raise ConfigError(
                    f"{path}.checkpoint_source has an invalid identifier"
                )
            if block.checkpoint_source in sources:
                raise ConfigError(
                    "duplicate checkpoint_source: "
                    f"{block.checkpoint_source}"
                )
            sources.add(block.checkpoint_source)
        if block.stride not in (1, 2):
            raise ConfigError(f"{path}.stride must be 1 or 2")
        if block.use_residual and not (
            block.stride == 1 and block.in_channels == block.out_channels
        ):
            raise ConfigError(
                f"{path}.use_residual requires stride == 1 and equal channels"
            )
        if index and blocks[index - 1].out_channels != block.in_channels:
            raise ConfigError(
                f"{path}.in_channels must equal previous block out_channels "
                f"({blocks[index - 1].out_channels})"
            )

    if blocks[-1].out_channels != model.tail.in_channels:
        raise ConfigError(
            "model.tail.in_channels must equal the final block out_channels"
        )
    if model.tail.out_channels != model.temporal.input_size:
        raise ConfigError(
            "model.temporal.input_size must equal model.tail.out_channels"
        )
    if model.temporal.kind != "gru":
        raise ConfigError("model.temporal.type must be 'gru'")
    if model.temporal.hidden_size != model.regression_head.in_features:
        raise ConfigError(
            "model.regression_head.in_features must equal GRU hidden_size"
        )
    if model.regression_head.out_features != 2:
        raise ConfigError("model.regression_head.out_features must equal 2")

    training = config.training
    if training.mode not in ("float32", "int8_qat"):
        raise ConfigError("training.mode must be 'float32' or 'int8_qat'")
    if training.optimizer.kind != "adam":
        raise ConfigError("training.optimizer.type must be 'adam'")
    if training.loss.kind != "weighted_mse":
        raise ConfigError("training.loss.type must be 'weighted_mse'")
    if training.mode == "int8_qat" and not config.quantization.enabled:
        raise ConfigError("int8_qat mode requires quantization.enabled=true")
    if training.mode == "float32" and config.quantization.enabled:
        raise ConfigError("float32 mode requires quantization.enabled=false")
    if config.runtime.checkpoint.mode not in ("none", "resume", "initialize"):
        raise ConfigError(
            "runtime.checkpoint.mode must be none, resume, or initialize"
        )
    if config.runtime.checkpoint.mode != "none" and not (
        config.runtime.checkpoint.path
    ):
        raise ConfigError(
            "runtime.checkpoint.path is required for resume or initialize"
        )


def load_config(path: Path) -> ExperimentConfig:
    """从 UTF-8 JSON 文件加载配置；错误信息保留完整字段路径。"""

    source_path = Path(path)
    try:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"cannot load config {source_path}: {error}") from error
    return parse_experiment_config(raw, source_path=source_path)


def config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    """返回后续写入检查点的 JSON 兼容配置副本。"""

    return json.loads(config.source_path.read_text(encoding="utf-8"))
