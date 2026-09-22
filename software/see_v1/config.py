"""Version 1 配置读取与无需 CUDA 的严格结构校验。"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple


class ConfigError(ValueError):
    """JSON 字段或跨组件连接不符合 Version 1 契约。"""


# 以下冻结 dataclass 是运行期唯一配置表示。冻结可防止训练过程中某个模块
# 原地修改参数，确保写出的 config.json 与实际运行保持一致。
@dataclass(frozen=True)
class ClipConfig:
    mode: str
    duration_ms: int


@dataclass(frozen=True)
class AugmentationConfig:
    flip_probability: float
    shift_probability: float
    max_shift: int


@dataclass(frozen=True)
class DataConfig:
    root: Path
    train_list: Path
    val_list: Path
    metadata_root: Path
    cache_root: Path
    sensor_width: int
    sensor_height: int
    spatial_factor: float
    label_period_ms: int
    sequence_length: int
    train_stride: int
    val_stride: int
    n_time_bins: int
    normalize_voxel_channels: bool
    clip: ClipConfig
    augmentation: AugmentationConfig


@dataclass(frozen=True)
class BlockConfig:
    name: str
    source_index: int
    in_channels: int
    expand_channels: int
    out_channels: int
    stride: int
    residual: bool


@dataclass(frozen=True)
class StudentConfig:
    model_cfg: Path
    input_channels: int
    stem_channels: int
    blocks: Tuple[BlockConfig, ...]
    tail_channels: int


@dataclass(frozen=True)
class TeacherConfig:
    kind: str
    model_cfg: Path
    checkpoint: Path
    sha256: str
    shift_bit: int
    bias_bit: int
    conv1_bit: int
    fix_bn_ratio: float


@dataclass(frozen=True)
class TokenConfig:
    per_clip: int
    dimension: int
    auxiliary_dimension: int


@dataclass(frozen=True)
class TransformerConfig:
    history_clips: int
    layers: int
    heads: int
    feedforward_dimension: int
    dropout: float


@dataclass(frozen=True)
class TargetConfig:
    future_ms: int


@dataclass(frozen=True)
class LossConfig:
    current: float
    future: float
    output_distill: float
    feature_distill: float
    coordinate_weights: Tuple[float, float]

    @property
    def weights(self) -> Tuple[float, float, float, float]:
        """按损失实现要求返回四项权重的固定顺序。"""

        return (
            self.current,
            self.future,
            self.output_distill,
            self.feature_distill,
        )


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    epochs: int
    batch_size: int
    workers: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    early_stopping_min_delta: float
    pixel_tolerances: Tuple[int, ...]


@dataclass(frozen=True)
class RuntimeConfig:
    device: str


@dataclass(frozen=True)
class OutputConfig:
    root: Path
    run_name: str


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    student: StudentConfig
    teacher: TeacherConfig
    tokens: TokenConfig
    transformer: TransformerConfig
    targets: TargetConfig
    loss: LossConfig
    training: TrainingConfig
    runtime: RuntimeConfig
    output: OutputConfig
    source_path: Path


def _object(value: Any, path: str) -> Mapping[str, Any]:
    """验证 JSON 对象类型，并在错误中保留完整字段路径。"""

    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be an object")
    return value


def _exact(
    value: Mapping[str, Any],
    path: str,
    fields: Sequence[str],
) -> None:
    """实施严格 schema：未知字段和缺失字段均立即报错。"""

    expected = set(fields)
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise ConfigError(f"{path} has unknown fields: {', '.join(unknown)}")
    if missing:
        raise ConfigError(f"{path} is missing fields: {', '.join(missing)}")


def _integer(value: Any, path: str, minimum: int = 1) -> int:
    """解析有下界的整数，同时排除 Python 中属于 int 子类的 bool。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{path} must be an integer >= {minimum}")
    return value


def _number(value: Any, path: str, minimum: float = 0.0) -> float:
    """解析有限浮点数，拒绝 NaN、Inf 和小于下界的值。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ConfigError(f"{path} must be finite and >= {minimum}")
    return result


def _probability(value: Any, path: str) -> float:
    """解析闭区间 `[0,1]` 内的概率。"""

    result = _number(value, path)
    if result > 1:
        raise ConfigError(f"{path} must be between 0 and 1")
    return result


def _string(value: Any, path: str) -> str:
    """解析非空字符串。"""

    if not isinstance(value, str) or not value:
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def _boolean(value: Any, path: str) -> bool:
    """严格解析 JSON 布尔值，不接受 0/1 等替代表示。"""

    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be boolean")
    return value


def _resolve(source: Path, value: Any, path: str) -> Path:
    """相对路径以配置文件所在目录为基准，避免依赖当前工作目录。"""

    raw = Path(_string(value, path))
    if raw.is_absolute():
        return raw
    return (source.parent / raw).resolve()


def _parse_data(raw_value: Any, source: Path) -> DataConfig:
    """解析数据几何、切片、缓存和增强配置。"""

    raw = _object(raw_value, "data")
    _exact(
        raw,
        "data",
        (
            "root",
            "train_list",
            "val_list",
            "metadata_root",
            "cache_root",
            "sensor_width",
            "sensor_height",
            "spatial_factor",
            "label_period_ms",
            "sequence_length",
            "train_stride",
            "val_stride",
            "n_time_bins",
            "normalize_voxel_channels",
            "clip",
            "augmentation",
        ),
    )
    clip = _object(raw["clip"], "data.clip")
    _exact(clip, "data.clip", ("mode", "duration_ms"))
    augmentation = _object(raw["augmentation"], "data.augmentation")
    _exact(
        augmentation,
        "data.augmentation",
        ("flip_probability", "shift_probability", "max_shift"),
    )
    return DataConfig(
        root=_resolve(source, raw["root"], "data.root"),
        train_list=_resolve(source, raw["train_list"], "data.train_list"),
        val_list=_resolve(source, raw["val_list"], "data.val_list"),
        metadata_root=_resolve(
            source, raw["metadata_root"], "data.metadata_root"
        ),
        cache_root=_resolve(source, raw["cache_root"], "data.cache_root"),
        sensor_width=_integer(raw["sensor_width"], "data.sensor_width"),
        sensor_height=_integer(raw["sensor_height"], "data.sensor_height"),
        spatial_factor=_number(raw["spatial_factor"], "data.spatial_factor"),
        label_period_ms=_integer(
            raw["label_period_ms"], "data.label_period_ms"
        ),
        sequence_length=_integer(
            raw["sequence_length"], "data.sequence_length"
        ),
        train_stride=_integer(raw["train_stride"], "data.train_stride"),
        val_stride=_integer(raw["val_stride"], "data.val_stride"),
        n_time_bins=_integer(raw["n_time_bins"], "data.n_time_bins"),
        normalize_voxel_channels=_boolean(
            raw["normalize_voxel_channels"],
            "data.normalize_voxel_channels",
        ),
        clip=ClipConfig(
            mode=_string(clip["mode"], "data.clip.mode"),
            duration_ms=_integer(
                clip["duration_ms"], "data.clip.duration_ms"
            ),
        ),
        augmentation=AugmentationConfig(
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
        ),
    )


def _parse_student(raw_value: Any, source: Path) -> StudentConfig:
    """解析可变长度的学生倒残差块数组。"""

    raw = _object(raw_value, "student")
    _exact(
        raw,
        "student",
        (
            "model_cfg",
            "input_channels",
            "stem_channels",
            "blocks",
            "tail_channels",
        ),
    )
    block_values = raw["blocks"]
    if not isinstance(block_values, list) or not block_values:
        raise ConfigError("student.blocks must be a non-empty array")
    # 保留 JSON 顺序；该顺序就是学生主干的真实执行顺序。
    blocks = []
    for index, value in enumerate(block_values):
        path = f"student.blocks[{index}]"
        block = _object(value, path)
        _exact(
            block,
            path,
            (
                "name",
                "source_index",
                "in_channels",
                "expand_channels",
                "out_channels",
                "stride",
                "residual",
            ),
        )
        blocks.append(
            BlockConfig(
                name=_string(block["name"], f"{path}.name"),
                source_index=_integer(
                    block["source_index"], f"{path}.source_index", minimum=0
                ),
                in_channels=_integer(
                    block["in_channels"], f"{path}.in_channels"
                ),
                expand_channels=_integer(
                    block["expand_channels"], f"{path}.expand_channels"
                ),
                out_channels=_integer(
                    block["out_channels"], f"{path}.out_channels"
                ),
                stride=_integer(block["stride"], f"{path}.stride"),
                residual=_boolean(block["residual"], f"{path}.residual"),
            )
        )
    return StudentConfig(
        model_cfg=_resolve(source, raw["model_cfg"], "student.model_cfg"),
        input_channels=_integer(
            raw["input_channels"], "student.input_channels"
        ),
        stem_channels=_integer(raw["stem_channels"], "student.stem_channels"),
        blocks=tuple(blocks),
        tail_channels=_integer(raw["tail_channels"], "student.tail_channels"),
    )


def _parse_teacher(raw_value: Any, source: Path) -> TeacherConfig:
    """解析官方教师资产路径和量化构造参数。"""

    raw = _object(raw_value, "teacher")
    _exact(
        raw,
        "teacher",
        (
            "kind",
            "model_cfg",
            "checkpoint",
            "sha256",
            "shift_bit",
            "bias_bit",
            "conv1_bit",
            "fix_bn_ratio",
        ),
    )
    return TeacherConfig(
        kind=_string(raw["kind"], "teacher.kind"),
        model_cfg=_resolve(source, raw["model_cfg"], "teacher.model_cfg"),
        checkpoint=_resolve(source, raw["checkpoint"], "teacher.checkpoint"),
        sha256=_string(raw["sha256"], "teacher.sha256").lower(),
        shift_bit=_integer(raw["shift_bit"], "teacher.shift_bit"),
        bias_bit=_integer(raw["bias_bit"], "teacher.bias_bit"),
        conv1_bit=_integer(raw["conv1_bit"], "teacher.conv1_bit"),
        fix_bn_ratio=_probability(
            raw["fix_bn_ratio"], "teacher.fix_bn_ratio"
        ),
    )


def parse_config(raw_value: Any, source_path: Path) -> ExperimentConfig:
    """把已解码 JSON 转为强类型配置并执行跨字段校验。"""

    source = Path(source_path).resolve()
    raw = _object(raw_value, "root")
    fields = (
        "data",
        "student",
        "teacher",
        "tokens",
        "transformer",
        "targets",
        "loss",
        "training",
        "runtime",
        "output",
    )
    _exact(raw, "root", fields)

    tokens = _object(raw["tokens"], "tokens")
    _exact(tokens, "tokens", ("per_clip", "dimension", "auxiliary_dimension"))
    transformer = _object(raw["transformer"], "transformer")
    _exact(
        transformer,
        "transformer",
        (
            "history_clips",
            "layers",
            "heads",
            "feedforward_dimension",
            "dropout",
        ),
    )
    targets = _object(raw["targets"], "targets")
    _exact(targets, "targets", ("future_ms",))
    loss = _object(raw["loss"], "loss")
    _exact(
        loss,
        "loss",
        (
            "current",
            "future",
            "output_distill",
            "feature_distill",
            "coordinate_weights",
        ),
    )
    coordinate_weights = loss["coordinate_weights"]
    if not isinstance(coordinate_weights, list) or len(coordinate_weights) != 2:
        raise ConfigError("loss.coordinate_weights must contain two values")
    training = _object(raw["training"], "training")
    _exact(
        training,
        "training",
        (
            "seed",
            "epochs",
            "batch_size",
            "workers",
            "learning_rate",
            "weight_decay",
            "early_stopping_patience",
            "early_stopping_min_delta",
            "pixel_tolerances",
        ),
    )
    tolerances = training["pixel_tolerances"]
    if not isinstance(tolerances, list) or not tolerances:
        raise ConfigError("training.pixel_tolerances must be non-empty")
    runtime = _object(raw["runtime"], "runtime")
    _exact(runtime, "runtime", ("device",))
    output = _object(raw["output"], "output")
    _exact(output, "output", ("root", "run_name"))

    # 先完成逐字段类型校验，再由 validate_config 检查组件连接关系。
    config = ExperimentConfig(
        data=_parse_data(raw["data"], source),
        student=_parse_student(raw["student"], source),
        teacher=_parse_teacher(raw["teacher"], source),
        tokens=TokenConfig(
            per_clip=_integer(tokens["per_clip"], "tokens.per_clip"),
            dimension=_integer(tokens["dimension"], "tokens.dimension"),
            auxiliary_dimension=_integer(
                tokens["auxiliary_dimension"],
                "tokens.auxiliary_dimension",
                minimum=0,
            ),
        ),
        transformer=TransformerConfig(
            history_clips=_integer(
                transformer["history_clips"], "transformer.history_clips"
            ),
            layers=_integer(transformer["layers"], "transformer.layers"),
            heads=_integer(transformer["heads"], "transformer.heads"),
            feedforward_dimension=_integer(
                transformer["feedforward_dimension"],
                "transformer.feedforward_dimension",
            ),
            dropout=_probability(
                transformer["dropout"], "transformer.dropout"
            ),
        ),
        targets=TargetConfig(
            future_ms=_integer(targets["future_ms"], "targets.future_ms")
        ),
        loss=LossConfig(
            current=_number(loss["current"], "loss.current"),
            future=_number(loss["future"], "loss.future"),
            output_distill=_number(
                loss["output_distill"], "loss.output_distill"
            ),
            feature_distill=_number(
                loss["feature_distill"], "loss.feature_distill"
            ),
            coordinate_weights=(
                _number(coordinate_weights[0], "loss.coordinate_weights[0]"),
                _number(coordinate_weights[1], "loss.coordinate_weights[1]"),
            ),
        ),
        training=TrainingConfig(
            seed=_integer(training["seed"], "training.seed", minimum=0),
            epochs=_integer(training["epochs"], "training.epochs"),
            batch_size=_integer(
                training["batch_size"], "training.batch_size"
            ),
            workers=_integer(
                training["workers"], "training.workers", minimum=0
            ),
            learning_rate=_number(
                training["learning_rate"], "training.learning_rate"
            ),
            weight_decay=_number(
                training["weight_decay"], "training.weight_decay"
            ),
            early_stopping_patience=_integer(
                training["early_stopping_patience"],
                "training.early_stopping_patience",
            ),
            early_stopping_min_delta=_number(
                training["early_stopping_min_delta"],
                "training.early_stopping_min_delta",
            ),
            pixel_tolerances=tuple(
                _integer(value, f"training.pixel_tolerances[{index}]")
                for index, value in enumerate(tolerances)
            ),
        ),
        runtime=RuntimeConfig(
            device=_string(runtime["device"], "runtime.device")
        ),
        output=OutputConfig(
            root=_resolve(source, output["root"], "output.root"),
            run_name=_string(output["run_name"], "output.run_name"),
        ),
        source_path=source,
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    """验证无法由单字段类型表达的网络与数据管线不变量。"""

    data = config.data
    if data.clip.mode != "fixed":
        raise ConfigError(
            "data.clip.mode must be 'fixed' in the first milestone"
        )
    if not 0 < data.spatial_factor <= 1:
        raise ConfigError("data.spatial_factor must be in (0, 1]")
    if data.clip.duration_ms % data.label_period_ms:
        raise ConfigError("data.clip.duration_ms must follow label_period_ms")
    if config.targets.future_ms % data.label_period_ms:
        raise ConfigError("targets.future_ms must follow data.label_period_ms")

    # 学生块必须形成连续的通道链，且显式残差策略必须在结构上可执行。
    student = config.student
    if student.stem_channels != student.blocks[0].in_channels:
        raise ConfigError(
            "student.stem_channels must equal first block in_channels"
        )
    names = set()
    source_indices = set()
    name_pattern = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
    for index, block in enumerate(student.blocks):
        path = f"student.blocks[{index}]"
        if not name_pattern.fullmatch(block.name):
            raise ConfigError(f"{path}.name is invalid")
        if block.name in names:
            raise ConfigError(f"duplicate student block name: {block.name}")
        if block.source_index in source_indices:
            raise ConfigError(
                f"duplicate student block source_index: {block.source_index}"
            )
        names.add(block.name)
        source_indices.add(block.source_index)
        if block.stride not in (1, 2):
            raise ConfigError(f"{path}.stride must be 1 or 2")
        if block.residual and not (
            block.stride == 1 and block.in_channels == block.out_channels
        ):
            raise ConfigError(
                f"{path}.residual requires stride 1 and equal channels"
            )
        if index and student.blocks[index - 1].out_channels != block.in_channels:
            raise ConfigError(
                f"{path}.in_channels must equal previous out_channels"
            )
        if block.expand_channels < block.in_channels:
            raise ConfigError(
                f"{path}.expand_channels must be >= in_channels"
            )
        if block.expand_channels % block.in_channels:
            raise ConfigError(
                f"{path}.expand_channels must be divisible by in_channels"
            )
    # 教师种类、权重摘要和 Transformer 维度属于当前实现的固定契约。
    if config.teacher.kind != "official_int8":
        raise ConfigError("teacher.kind must be 'official_int8'")
    if not re.fullmatch(r"[0-9a-f]{64}", config.teacher.sha256):
        raise ConfigError("teacher.sha256 must contain 64 hexadecimal digits")
    if config.tokens.per_clip * config.tokens.dimension != 64:
        raise ConfigError(
            "tokens.per_clip * tokens.dimension must equal 64"
        )
    if config.tokens.dimension % config.transformer.heads:
        raise ConfigError(
            "tokens.dimension must be divisible by transformer.heads"
        )


def load_config(path: Path) -> ExperimentConfig:
    """从 UTF-8 JSON 文件加载并返回已完全校验的实验配置。"""

    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        return parse_config(json.load(handle), source)
