"""固定 50 ms 第一里程碑的数据几何与 Tonic 缓存数据管线。"""

from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
from typing import Tuple

from .config import ExperimentConfig
from .labels import build_current_future_labels


@dataclass(frozen=True)
class DataGeometry:
    """由配置派生、供切片和体素化共同使用的数据尺寸。"""

    width: int
    height: int
    clip_us: int
    sequence_us: int
    sample_shape: Tuple[int, int, int, int]


def derive_geometry(config: ExperimentConfig) -> DataGeometry:
    """计算下采样网格、微秒时间窗口和单样本张量形状。"""

    data = config.data
    width = int(data.sensor_width * data.spatial_factor)
    height = int(data.sensor_height * data.spatial_factor)
    clip_us = data.clip.duration_ms * 1000
    return DataGeometry(
        width=width,
        height=height,
        clip_us=clip_us,
        sequence_us=clip_us * data.sequence_length,
        sample_shape=(
            data.sequence_length,
            data.n_time_bins,
            height,
            width,
        ),
    )


def build_cache_tag(config: ExperimentConfig) -> str:
    """生成包含所有影响缓存内容的关键参数标签。"""

    return (
        f"fixed{config.data.clip.duration_ms}"
        f"_seq{config.data.sequence_length}"
        f"_bins{config.data.n_time_bins}"
        f"_future{config.targets.future_ms}"
    )


def build_record_paths(data_root: Path, records):
    """把记录编号转换为事件和标签路径。"""

    records = tuple(record for record in records if record)
    if not records:
        raise ValueError("record list is empty")
    root = Path(data_root)
    # 3ET 的 train/val 清单都引用 root/train 下的记录目录。
    return tuple(
        (
            root / "train" / record / f"{record}.h5",
            root / "train" / record / "label.txt",
        )
        for record in records
    )


def resolve_record_paths(data_root: Path, list_path: Path):
    """读取显式 split 清单，供冒烟训练覆盖完整清单。"""

    records = tuple(
        line.strip()
        for line in Path(list_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return build_record_paths(data_root, records)


class CurrentFutureTargetTransform:
    """构造 `[current_x,current_y,state,future_x,future_y]`。"""

    def __init__(self, config: ExperimentConfig):
        self.data = config.data
        self.future_ms = config.targets.future_ms

    def __call__(self, labels):
        # 标签变换在时间切片前执行，输出行与 50 ms clip 一一对应。
        return build_current_future_labels(
            labels,
            label_period_ms=self.data.label_period_ms,
            clip_ms=self.data.clip.duration_ms,
            future_ms=self.future_ms,
            sensor_width=self.data.sensor_width,
            sensor_height=self.data.sensor_height,
        )


class FutureRegressionDataset:
    """缓存保持确定性；随机增强同时作用于当前和未来坐标。"""

    def __init__(
        self,
        cached_dataset,
        training: bool,
        flip_probability: float,
        shift_probability: float,
        max_shift: int,
    ):
        self.cached_dataset = cached_dataset
        self.training = training
        self.flip_probability = flip_probability
        self.shift_probability = shift_probability
        self.max_shift = max_shift

    def __len__(self):
        return len(self.cached_dataset)

    @staticmethod
    def _coordinates_valid(targets) -> bool:
        """检查增强后的当前和未来归一化坐标均未越界。"""

        current = targets[:, (0, 1)]
        future = targets[:, (3, 4)]
        return bool(
            ((current >= 0) & (current <= 1)).all()
            and ((future >= 0) & (future <= 1)).all()
        )

    def _augment(self, frames, targets):
        """对体素和两组坐标同步执行随机翻转与平移。"""

        import numpy as np

        # 缓存内容必须保持只读语义；标签复制后才能安全原地更新。
        frames = np.asarray(frames)
        targets = np.asarray(targets).copy()
        # 最后一维是 x 方向，翻转后当前/未来横坐标必须同步变换。
        if random.random() < self.flip_probability:
            frames = np.flip(frames, axis=-1)
            targets[:, 0] = 1 - targets[:, 0]
            targets[:, 3] = 1 - targets[:, 3]

        # 随机平移最多重采样 5 次；越界候选不会污染原始目标。
        if random.random() < self.shift_probability and self.max_shift:
            _, _, height, width = frames.shape
            original_targets = targets.copy()
            for _ in range(5):
                shift_x = random.randint(-self.max_shift, self.max_shift)
                shift_y = random.randint(-self.max_shift, self.max_shift)
                targets[:, 0] = original_targets[:, 0] + shift_x / width
                targets[:, 3] = original_targets[:, 3] + shift_x / width
                targets[:, 1] = original_targets[:, 1] + shift_y / height
                targets[:, 4] = original_targets[:, 4] + shift_y / height
                if self._coordinates_valid(targets):
                    padding = self.max_shift
                    padded = np.pad(
                        frames,
                        (
                            (0, 0),
                            (0, 0),
                            (padding, padding),
                            (padding, padding),
                        ),
                    )
                    padded = np.roll(
                        padded,
                        # NumPy 空间轴顺序是 (height, width)=(y, x)。
                        (shift_y, shift_x),
                        axis=(2, 3),
                    )
                    frames = padded[
                        :,
                        :,
                        padding : height + padding,
                        padding : width + padding,
                    ]
                    break
                targets = original_targets.copy()
        return np.ascontiguousarray(frames), targets

    def __getitem__(self, index):
        """返回 `(frames, targets, masked_length)` 以兼容训练循环接口。"""

        import torch

        frames, targets = self.cached_dataset[index]
        if self.training:
            frames, targets = self._augment(frames, targets)
        return frames, targets, torch.tensor(0)


def _build_split(config: ExperimentConfig, training: bool):
    """构建单个 split 的读取、切片、体素化、缓存和增强流水线。"""

    # 大型 GPU/数据依赖延迟导入，使配置和标签单测可在轻量环境运行。
    import h5py
    import numpy as np
    import tonic.transforms as tonic_transforms
    from tonic import DiskCachedDataset, SlicedDataset
    from tonic.dataset import Dataset as TonicDataset

    from dataset import (
        EventSlicesToVoxelGrid,
        SliceByTimeEventsTargets,
        SliceLongEventsToShort,
    )

    data = config.data
    geometry = derive_geometry(config)
    split = "train" if training else "val"
    list_path = data.train_list if training else data.val_list

    class ConfiguredThreeETRecording(TonicDataset):
        """按显式记录清单读取 H5 事件和 100 Hz 文本标签。"""

        sensor_size = (data.sensor_width, data.sensor_height, 2)
        dtype = np.dtype(
            [("t", int), ("x", int), ("y", int), ("p", int)]
        )
        ordering = dtype.names

        def __init__(self):
            super().__init__(
                str(data.root),
                transform=tonic_transforms.Downsample(
                    spatial_factor=data.spatial_factor
                ),
                target_transform=CurrentFutureTargetTransform(config),
            )
            paths = resolve_record_paths(data.root, list_path)
            self.data = [str(event_path) for event_path, _ in paths]
            self.targets = [str(label_path) for _, label_path in paths]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            # H5 中的 p 为 0/1；稀疏体素化前转换为对称的 -1/+1。
            with h5py.File(self.data[index], "r") as handle:
                events = handle["events"][:].astype(self.dtype)
            events["p"] = events["p"] * 2 - 1
            with open(self.targets[index], "r", encoding="utf-8") as handle:
                labels = np.asarray(
                    [
                        list(
                            map(
                                float,
                                line.strip("()\n").split(", "),
                            )
                        )
                        for line in handle
                    ],
                    dtype=np.float32,
                )
            # 空间下采样只作用于事件；标签仍按原始 640×480 归一化。
            if self.transform is not None:
                events = self.transform(events)
            if self.target_transform is not None:
                labels = self.target_transform(labels)
            return events, labels

    recording = ConfiguredThreeETRecording()
    stride = data.train_stride if training else data.val_stride
    # 每个长样本覆盖 30 个 clip；stride=1 时相邻样本仅前进 50 ms。
    slicer = SliceByTimeEventsTargets(
        geometry.sequence_us,
        overlap=geometry.sequence_us - geometry.clip_us * stride,
        seq_length=data.sequence_length,
        seq_stride=stride,
        include_incomplete=False,
    )
    # 长样本先切成 30 个短 clip，再各自编码为 3×60×80 体素。
    post_transform = tonic_transforms.Compose(
        [
            SliceLongEventsToShort(
                geometry.clip_us,
                overlap=0,
                include_incomplete=True,
            ),
            EventSlicesToVoxelGrid(
                sensor_size=(geometry.width, geometry.height, 2),
                n_time_bins=data.n_time_bins,
                per_channel_normalize=data.normalize_voxel_channels,
            ),
        ]
    )
    tag = build_cache_tag(config)
    metadata_path = Path(data.metadata_root) / f"{split}_{tag}"
    cache_path = Path(data.cache_root) / f"{split}_{tag}"
    # 元数据描述切片索引；DiskCachedDataset 保存确定性的体素结果。
    sliced = SlicedDataset(
        recording,
        slicer,
        transform=post_transform,
        metadata_path=str(metadata_path),
    )
    cached = DiskCachedDataset(sliced, cache_path=str(cache_path))
    augmentation = data.augmentation
    # 随机增强位于缓存之外，因此不会把某次随机结果固化到磁盘。
    return FutureRegressionDataset(
        cached,
        training=training,
        flip_probability=augmentation.flip_probability,
        shift_probability=augmentation.shift_probability,
        max_shift=augmentation.max_shift,
    )


def build_dataloaders(config: ExperimentConfig):
    """构建训练和验证 DataLoader；只在 AutoDL 环境延迟导入 PyTorch。"""

    from torch.utils.data import DataLoader

    train_dataset = _build_split(config, training=True)
    val_dataset = _build_split(config, training=False)
    training = config.training
    # 训练集打乱并启用 pinned memory；验证集保持确定顺序。
    train_loader = DataLoader(
        train_dataset,
        batch_size=training.batch_size,
        shuffle=True,
        num_workers=training.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training.batch_size,
        shuffle=False,
        num_workers=training.workers,
    )
    return train_loader, val_loader
