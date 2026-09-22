# Version 1 设计与实现说明

## 1. 文档范围

本文严格记录 `code/version1` 当前已经实现并已在 AutoDL 上运行的固定 50 ms 版本。其目标是：

1. 使用论文官方 INT8 SEE-D 作为冻结教师网络；
2. 训练由 JSON 描述的七块缩窄稀疏卷积学生网络；
3. 把每个事件 clip 编码为四个 Token；
4. 使用带有限历史窗口的因果 Transformer；
5. 同时回归当前位置和 20 ms 后的位置；
6. 使用真实标签监督和教师知识蒸馏共同训练学生。

本文中的默认参数来自：

```text
software/configs/version1_fixed50.json
```

## 2. 代码入口与模块职责

```text
version1/
├── DESIGN.md
├── Proj2_test_train/             # 裁剪小数据并执行冒烟训练
├── Proj3_trainning/              # 完整数据训练、日志持久化和自动关机入口
└── software/
    ├── main.py                   # 命令行入口：静态检查或启动训练
    ├── configs/
    │   ├── version1_fixed50.json # 完整实验配置
    │   └── model_cfg/
    │       ├── SEE-D_teacher.json
    │       └── SEE-D_student7.json
    ├── dataset/                  # 官方事件切片和体素化实现
    ├── model/                    # 官方稀疏 SEE-D 与量化 SEE-D 实现
    ├── weights/Table2/SEE-D/     # 官方教师配置与检查点
    └── see_v1/
        ├── architecture.py       # JSON 块配置转换和残差策略
        ├── checkpoints.py        # 训练/部署检查点保存与最优模型选择
        ├── cli.py                # 参数解析
        ├── config.py             # JSON 读取、类型转换和一致性校验
        ├── data.py               # 数据读取、切片、体素化、缓存和增强
        ├── early_stopping.py     # 验证集未来误差的提前停止状态机
        ├── labels.py             # 当前/未来标签配对
        ├── losses.py             # 四项损失
        ├── masking.py            # Transformer 因果有限历史掩码
        ├── metrics.py            # 像素距离与 P5/P10/P15
        ├── student.py            # 七块学生、四 Token 和 Transformer
        ├── teacher.py            # 官方 INT8 教师包装器
        └── trainer.py            # 训练、验证、日志和检查点主循环
```

## 3. 端到端数据流

```text
H5 事件 + label.txt
        │
        ├─ 事件极性转换、空间下采样
        ├─ 当前标签与 +20 ms 标签配对
        ▼
连续 1.5 s 长序列
        │
        ├─ 划分为 30 个 50 ms clip
        ├─ 每个 clip 划分为 3 个时间 bin
        ▼
体素张量 [B, 30, 3, 60, 80]
        │
        ├───────────────┬────────────────────┐
        ▼               ▼                    │
冻结九块教师       七块学生 SCNN             │
        │               │                    │
教师坐标 [B,30,2]  学生空间特征 [B,30,64]   │
教师特征 [B,30,64]      │                    │
        │          粗当前位置 [B,30,2]       │
        │          四 Token [B,30,4,16]      │
        │               │                    │
        │          因果 Transformer          │
        │               │                    │
        │          时序特征 [B,30,16]        │
        │               ├─ 当前修正头         │
        │               └─ 未来位移头         │
        │                                    │
        └────────────── 四项损失 ◀────────────┘
                             │
                             ▼
                    仅更新学生和特征适配器
```

符号约定：

| 符号 | 含义 | 当前值 |
|---|---|---:|
| `B` | batch size | 20 |
| `T` | 一个训练样本包含的 clip 数 | 30 |
| `C` | 每个 clip 的时间 bin/输入通道数 | 3 |
| `H × W` | 下采样后的体素网格 | `60 × 80` |
| `K` | 每个 clip 的 Token 数 | 4 |
| `D` | 每个 Token 的维度 | 16 |
| `N` | MinkowskiEngine 中当前层活跃稀疏坐标数 | 随输入变化 |

## 4. 原始输入与标签

### 4.1 事件文件

每条记录的输入路径为：

```text
event_data/train/<record_id>/<record_id>.h5
```

H5 文件中的 `events` 数据集包含四个字段：

| 字段 | 含义 |
|---|---|
| `t` | 事件时间戳 |
| `x` | 原始横坐标 |
| `y` | 原始纵坐标 |
| `p` | 事件极性 |

读取后执行：

```text
p = p × 2 - 1
```

因此极性变为 `-1/+1`。随后使用 `spatial_factor=0.125` 将原始 `640 × 480`
坐标映射到 `80 × 60`。

### 4.2 标签文件

标签路径为：

```text
event_data/train/<record_id>/label.txt
```

标签频率为 100 Hz，即相邻标签间隔 10 ms。每个 50 ms clip 对应每隔 5 行
选取一次当前标签：

```text
current_step = 50 ms / 10 ms = 5
```

未来目标为 20 ms 后的位置：

```text
future_step = 20 ms / 10 ms = 2
```

标签配对为：

```text
当前行 i  ──配对──> 未来行 i + 2
```

只有确实存在未来行的样本才会进入输出。每个 clip 的目标形状为 `[5]`：

```text
[current_x / 640,
 current_y / 480,
 state,
 future_x / 640,
 future_y / 480]
```

所以坐标监督使用 `[0,1]` 范围内的归一化坐标，而评估时再恢复为原始
`640 × 480` 像素坐标。

## 5. 切片、体素化与训练样本形状

### 5.1 长序列切片

一个训练样本包含 30 个连续 clip：

```text
单 clip 时长      = 50 ms
sequence_length   = 30
一个样本覆盖时长 = 30 × 50 ms = 1500 ms
```

训练集和验证集的 `stride` 均为 1 个 clip，因此相邻长样本相差 50 ms，
重叠 1450 ms。

### 5.2 clip 内体素化

长序列先由 `SliceLongEventsToShort` 切成 30 个 50 ms clip。每个 clip 再由
官方 `EventSlicesToVoxelGrid` 转为 3 个时间 bin：

```text
单个 clip  : [3, 60, 80]
单个样本   : [30, 3, 60, 80]
一个 batch : [B, 30, 3, 60, 80]
```

当前配置：

```text
n_time_bins = 3
normalize_voxel_channels = false
```

### 5.3 缓存与增强

确定性的切片和体素化结果由 Tonic `DiskCachedDataset` 缓存。训练增强在读取缓存
之后执行：

| 增强 | 概率/范围 | 标签同步处理 |
|---|---|---|
| 水平翻转 | `0.5` | 当前和未来的 `x` 同时变为 `1-x` |
| 平移 | 概率 `0.5`，最大 `±4` 个体素像素 | 当前和未来 `x/y` 同步平移 |

平移最多尝试 5 次，只接受当前和未来坐标都仍位于 `[0,1]` 内的结果。验证集直接
使用缓存结果。DataLoader 最终返回：

```text
frames  : [B, 30, 3, 60, 80]
targets : [B, 30, 5]
```

## 6. 稀疏卷积中的张量表示

学生空间编码器把 `B` 和 `T` 合并：

```text
[B, T, C, H, W]
    -> [B×T, C, H, W]
    -> [B×T, H, W, C]
```

`dense_to_sparse` 只收集活跃位置，构造 MinkowskiEngine `SparseTensor`：

```text
coordinates : [N, 3]  # batch/clip 索引、y、x
features    : [N, C]
```

因此稀疏卷积层的真实特征形状是 `[N,C]`。后文的 `60×80`、`30×40` 等空间
尺寸是对应的名义稠密包络；每层真实活跃点数 `N` 随事件分布和步幅变化。

## 7. 单个倒残差稀疏块的内部结构

教师和学生的主干都使用 MobileNetV2 风格的倒残差块，稀疏算子来自原项目。
每个块依次执行：

```text
输入 SparseTensor [Nin, Cin]
  │
  ├─ 1×1 稀疏逐点卷积：Cin -> Cexpand
  ├─ BatchNorm
  ├─ ReLU6
  │
  ├─ 3×3 稀疏逐通道卷积：Cexpand -> Cexpand
  │    stride = 1 或 2
  ├─ BatchNorm
  ├─ ReLU6
  │
  ├─ 1×1 稀疏投影卷积：Cexpand -> Cout
  ├─ BatchNorm
  │
  └─ 当 residual=true 时：输出 + 输入
```

投影卷积后不再接 ReLU6。残差只用于 `stride=1` 且输入/输出通道相同的块。
`stride=2` 会降低稀疏坐标分辨率并改变活跃点集合。

## 8. 官方九块教师网络

### 8.1 教师来源和运行方式

教师模型为官方 `MobileNetSubmanifoldQuant` INT8 SEE-D，使用：

```text
software/weights/Table2/SEE-D/model_best_p10_acc.pth
```

检查点 SHA-256：

```text
2e23b7f717bac1c8801ddc969d2e3ade48b1a4facc780c0b7ac48fc5e0be8be6
```

构建教师前先验证 SHA-256，之后使用 `strict=True` 加载状态字典。量化参数为：

| 参数 | 值 |
|---|---:|
| `shift_bit` | 32 |
| `bias_bit` | 32 |
| `conv1_bit` | 8 |
| `fixBN_ratio` | 0.3 |

教师始终处于 `eval()`，所有参数 `requires_grad=False`，前向传播在
`torch.no_grad()` 中执行。

### 8.2 教师逐层结构

教师接收 `[B,30,3,60,80]`，在空间主干中将其视为 `B×30` 个稀疏帧。下表的
包络尺寸按累计步幅给出。

| 层 | 输入通道 | 扩展通道 | 输出通道 | 步幅 | 残差 | 累计步幅 | 名义空间包络 |
|---|---:|---:|---:|---:|---|---:|---|
| Stem | 3 | — | 24 | 2 | 否 | 2 | `30×40` |
| Block 0 | 24 | 192 | 32 | 2 | 否 | 4 | `15×20` |
| Block 1 | 32 | 192 | 48 | 2 | 否 | 8 | 约 `8×10` |
| Block 2 | 48 | 288 | 48 | 1 | 是 | 8 | 约 `8×10` |
| Block 3 | 48 | 48 | 64 | 1 | 否 | 8 | 约 `8×10` |
| Block 4 | 64 | 64 | 64 | 1 | 是 | 8 | 约 `8×10` |
| Block 5 | 64 | 64 | 72 | 2 | 否 | 16 | 约 `4×5` |
| Block 6 | 72 | 72 | 72 | 1 | 是 | 16 | 约 `4×5` |
| Block 7 | 72 | 72 | 72 | 1 | 是 | 16 | 约 `4×5` |
| Block 8 | 72 | 72 | 256 | 1 | 否 | 16 | 约 `4×5` |
| Tail | 256 | — | 64 | 1 | 否 | 16 | 约 `4×5` |
| GlobalAvgPool | 64 | — | 64 | — | — | — | `1×1` |

官方教师在池化后的序列结构为：

```text
[B,30,64]
  -> GRU(input=64, hidden=64, layers=1)
  -> Linear(64,2)
  -> teacher.current [B,30,2]
```

教师还通过 `quant_act_output` 的 forward hook 提取池化后的 64 维 Tail 特征：

```text
teacher.tail_features : [B,30,64]
```

这两个输出分别用于输出蒸馏和特征蒸馏。

## 9. 七块缩窄学生 SCNN

### 9.1 保留的块

学生按照 JSON 逐项构造，当前顺序为：

```text
Block 0 -> Block 1 -> Block 2 -> Block 3
        -> Block 5 -> Block 6 -> Block 8
```

`source_index` 保留了每个学生块在官方九块教师中的来源编号。

### 9.2 学生逐层结构

| 层 | 教师来源 | 输入通道 | 扩展通道 | 输出通道 | 步幅 | 残差 | 累计步幅 | 名义空间包络 |
|---|---:|---:|---:|---:|---:|---|---:|---|
| Stem | — | 3 | — | 24 | 2 | 否 | 2 | `30×40` |
| Block 0 | 0 | 24 | 192 | 32 | 2 | 否 | 4 | `15×20` |
| Block 1 | 1 | 32 | 192 | 48 | 2 | 否 | 8 | 约 `8×10` |
| Block 2 | 2 | 48 | 192 | 48 | 1 | 是 | 8 | 约 `8×10` |
| Block 3 | 3 | 48 | 48 | 64 | 1 | 否 | 8 | 约 `8×10` |
| Block 5 | 5 | 64 | 64 | 72 | 2 | 否 | 16 | 约 `4×5` |
| Block 6 | 6 | 72 | 72 | 72 | 1 | 是 | 16 | 约 `4×5` |
| Block 8 | 8 | 72 | 72 | 128 | 1 | 否 | 16 | 约 `4×5` |
| Tail | — | 128 | — | 64 | 1 | 否 | 16 | 约 `4×5` |
| GlobalAvgPool | — | 64 | — | 64 | — | — | — | `1×1` |

相对教师，当前学生的结构变化为：

| 位置 | 教师 | 学生 |
|---|---|---|
| Block 2 扩展通道 | 288 | 192 |
| Block 4 | 64 通道残差块 | 删除 |
| Block 7 | 72 通道残差块 | 删除 |
| Block 8 输出通道 | 256 | 128 |
| 块总数 | 9 | 7 |

GlobalAvgPool 后输出：

```text
student.spatial_features : [B,30,64]
```

该 64 维特征同时送往粗坐标头、Token 投影层和特征蒸馏适配器。

## 10. 四 Token 表示

### 10.1 粗坐标

学生空间特征先通过：

```text
Linear(64,2)
```

得到：

```text
coarse_current : [B,30,2]
```

粗坐标头权重使用均值 0、标准差 0.01 的正态分布初始化，偏置初始化为
`[0.5,0.5]`，使训练开始时预测位于归一化屏幕中心附近。

### 10.2 特征拆分为四个 Token

每个 clip 的 64 维空间特征执行：

```text
Linear(64, 4×16)
  -> reshape
  -> [B,30,4,16]
```

四个 Token 都由同一个 64 维 clip 特征投影产生。随后叠加三类信息：

1. Token 类型嵌入：`Embedding(4,16)`；
2. clip 时间位置嵌入：`Embedding(30,16)`；
3. 辅助量嵌入：`Linear(4,16)`。

### 10.3 四维辅助量

每个 clip 的辅助向量为：

```text
[coarse_x, coarse_y, event_density, duration_ratio]
```

形状变化：

```text
[B,30,4]
  -> Linear(4,16)
  -> [B,30,16]
  -> 在该 clip 的四个 Token 上广播
```

`event_density` 的计算方法是：先沿 3 个输入通道取绝对值并求和，判断每个
`60×80` 位置是否活跃，再计算活跃位置占整个网格的比例。固定 50 ms 版本的
`duration_ratio` 恒为 1。

叠加完成后的 `student.tokens` 形状为：

```text
[B,30,4,16]
```

## 11. 因果 Transformer

### 11.1 Transformer 参数

| 参数 | 值 |
|---|---:|
| `d_model` | 16 |
| `nhead` | 1 |
| Encoder 层数 | 2 |
| 前馈层维度 | 32 |
| 激活函数 | ReLU |
| Dropout | 0 |
| 每个 clip 的 Token 数 | 4 |
| 历史窗口 | 5 个 clip |

单层 `TransformerEncoderLayer` 的逻辑为：

```text
16D Token
  -> 单头自注意力
  -> 残差连接与 LayerNorm
  -> Linear(16,32)
  -> ReLU
  -> Linear(32,16)
  -> 残差连接与 LayerNorm
  -> 16D Token
```

### 11.2 序列展开

Transformer 前把 clip 和 Token 两个维度合并：

```text
[B,30,4,16]
  -> [B,120,16]
  -> transpose
  -> [120,B,16]
```

### 11.3 因果有限历史掩码

对第 `q` 个 clip 中的查询 Token，只允许访问：

```text
max(0, q-4) ... q
```

即当前 clip 和最多前 4 个 clip。每个允许的 clip 中，四个 Token 都可被访问。
掩码矩阵形状为 `[120,120]`：

```text
允许位置 = 0
禁止位置 = -inf
```

这使第 `q` 个 clip 的输出不依赖 `q+1` 及之后的 clip。

### 11.4 Token 聚合与两个输出头

两层 Transformer 输出恢复为：

```text
[B,120,16]
  -> [B,30,4,16]
```

四个 Token 取均值：

```text
temporal_features : [B,30,16]
```

然后使用两个线性头：

```text
current_correction = Linear(16,2)(temporal_features)
current = coarse_current + current_correction

future_delta = Linear(16,2)(temporal_features)
future = current + future_delta
```

最终：

```text
student.current : [B,30,2]
student.future  : [B,30,2]
```

当前修正头和未来位移头的权重、偏置均初始化为 0。因此初始化时：

```text
current = coarse_current
future  = current
```

## 12. 四项损失函数

### 12.1 坐标加权均方误差

由于原始图像宽高为 `640×480`，归一化横坐标的像素尺度更大。坐标权重设置为：

```text
wxy = [4/3, 1]
```

对预测 `p` 和目标 `y`：

```text
Lcoord(p,y) = mean((p-y)² × wxy)
```

### 12.2 当前真实标签损失

```text
Lcurrent = Lcoord(student.current, target.current)
```

输入与目标形状均为 `[B,30,2]`。

### 12.3 未来真实标签损失

```text
Lfuture = Lcoord(student.future, target.future20)
```

输入与目标形状均为 `[B,30,2]`。这里使用数据集中真实的 20 ms 后标签。

### 12.4 输出蒸馏损失

```text
Loutput = SmoothL1(student.current, teacher.current.detach())
```

教师当前坐标为 `[B,30,2]`。该项把官方教师的当前定位知识传给学生。

### 12.5 特征蒸馏损失

学生特征先通过训练辅助适配器：

```text
feature_adapter = Linear(64,64)
aligned_student = feature_adapter(student.spatial_features)
Lfeature = SmoothL1(aligned_student, teacher.tail_features.detach())
```

输入和对齐后的特征形状均为 `[B,30,64]`。

### 12.6 总损失

当前配置的总损失为：

```text
Ltotal =
    1.0 × Lcurrent
  + 1.0 × Lfuture
  + 0.2 × Loutput
  + 0.1 × Lfeature
```

反向传播更新：

```text
学生网络参数 + feature_adapter 参数
```

## 13. 指标定义

### 13.1 从归一化误差恢复像素误差

对每一个 clip：

```text
ex = |pred_x - target_x| × 640
ey = |pred_y - target_y| × 480
distance = sqrt(ex² + ey²)
```

当前坐标和未来 20 ms 坐标分别统计。

### 13.2 输出指标

| 指标 | 含义 |
|---|---|
| `current_distance` | 当前坐标平均欧氏像素误差 |
| `current_x_abs` | 当前横坐标平均绝对像素误差 |
| `current_y_abs` | 当前纵坐标平均绝对像素误差 |
| `current_p5/p10/p15` | 当前误差不超过 5/10/15 px 的比例 |
| `future_distance` | 20 ms 后坐标平均欧氏像素误差 |
| `future_x_abs` | 20 ms 后横坐标平均绝对像素误差 |
| `future_y_abs` | 20 ms 后纵坐标平均绝对像素误差 |
| `future_p5/p10/p15` | 20 ms 后误差不超过 5/10/15 px 的比例 |
| `sample_count` | 本轮累计的 clip 数，即各 batch 的 `B×T` 总和 |

例如：

```text
future_p10 = 误差 ≤ 10 px 的未来预测数 / 全部未来预测数
```

## 14. 完整训练流程

### 14.1 启动阶段

1. `main.py` 读取 `version1_fixed50.json`；
2. `config.py` 解析路径、数据、网络、损失和训练参数；
3. 校验块通道连续性、残差条件、未来时长与标签周期等约束；
4. 设置 Python、NumPy、PyTorch 和 CUDA 随机种子为 20；
5. 创建带 UTC 时间戳且不可覆盖的运行目录；
6. 保存 `config.json`、`environment.json` 和初始 `report.json`；
7. 构建训练集、验证集及其缓存；
8. 创建七块学生；
9. 校验官方权重 SHA-256，创建并冻结九块教师；
10. 创建四项损失及其 64→64 特征适配器；
11. 创建 Adam 优化器。

### 14.2 单个 batch

每个 batch 严格执行：

1. 将 `frames [B,30,3,60,80]` 移到 GPU；
2. 将 `targets [B,30,5]` 移到 GPU；
3. 教师前向得到当前坐标 `[B,30,2]` 和 Tail 特征 `[B,30,64]`；
4. 学生 SCNN 得到空间特征 `[B,30,64]`；
5. 粗坐标头得到 `[B,30,2]`；
6. 生成并增强四 Token，得到 `[B,30,4,16]`；
7. 展开为 120 Token，通过两层因果 Transformer；
8. 四 Token 均值得到 `[B,30,16]`；
9. 当前修正头输出 `student.current [B,30,2]`；
10. 未来位移头输出 `student.future [B,30,2]`；
11. 计算当前、未来、输出蒸馏、特征蒸馏四项损失；
12. 检查总损失为有限值；
13. 训练阶段执行反向传播和 Adam 更新；
14. 累计损失、平均像素距离及 P5/P10/P15。

验证阶段使用同样的教师、学生、损失和指标计算，但不更新参数。

### 14.3 当前训练参数

| 参数 | 值 |
|---|---:|
| Epoch | 100 |
| Batch size | 20 |
| DataLoader workers | 4 |
| 优化器 | Adam |
| Learning rate | 0.001 |
| Weight decay | 0 |
| Early stopping patience | 5 个 epoch |
| Early stopping min delta | 0.1 px |
| 设备 | `cuda:0` |

每个 epoch 完成指标、`last.pth` 和最优检查点写出后，使用
`val_future_distance` 更新 Early Stopping 状态。只有未来平均误差至少改善
0.1 px 才重置计数；连续 5 个 epoch 没有有效改善时正常结束训练。提前停止的运行
仍标记为 `passed`，因此 Proj3 会正常持久化并按 `AUTO_SHUTDOWN` 设置关机。

## 15. 日志、模型选择与输出文件

每个 epoch 同时写入：

```text
metrics.jsonl  # 逐行追加的 epoch 记录
metrics.csv    # 完整的表格记录
```

控制台显示当前累计均值：

```text
loss=<总损失>
current=<当前平均像素距离>
future=<20 ms 后平均像素距离>
```

最优检查点按以下二元组从小到大选择：

```text
(val_future_distance, val_current_distance)
```

也就是优先选择未来 20 ms 平均误差最小的 epoch；未来误差相同才比较当前误差。

运行目录中的主要文件：

| 文件 | 内容 |
|---|---|
| `config.json` | 实际解析后的配置快照 |
| `environment.json` | Python、PyTorch、CUDA 和 GPU 信息 |
| `metrics.jsonl` | 每个 epoch 的训练/验证指标 |
| `metrics.csv` | 同一批指标的 CSV 版本 |
| `last.pth` | 最近 epoch 的完整训练状态 |
| `best_future_distance.pth` | 最优完整训练状态 |
| `student_deploy.pth` | 仅学生 `state_dict` |
| `report.json` | 运行状态、配置/实际 epoch、提前停止原因、最佳误差、教师哈希和完成时间 |

完整训练检查点包含：

```text
epoch
student_state_dict
distillation_state_dict
optimizer_state_dict
score
config
```

`student_deploy.pth` 只包含学生网络参数，用于后续推理和部署。

## 16. Proj2 与 Proj3 的执行关系

### 16.1 Proj2 冒烟训练

`Proj2_test_train` 从完整训练集裁剪少量 train/val 数据，生成临时数据子集，然后调用
同一份 `software`。它验证：

- 数据读取和标签配对；
- 元数据与缓存生成；
- 教师权重加载；
- MinkowskiEngine 稀疏前向；
- 四 Token Transformer；
- 四项损失、反向传播和验证；
- 指标、报告和检查点写出。

### 16.2 Proj3 完整训练

`Proj3_trainning` 使用完整 split 清单和完整数据集，默认执行 100 个 epoch。运行中的
缓存和元数据位于临时盘，日志、报告和模型结果持久化到 `/root/autodl-fs/results`。
训练脚本只在训练命令成功且结果持久化成功后执行自动关机。

## 17. JSON 调整网络结构

学生结构以 `student.blocks` 数组为准，块数由数组长度动态决定。每个块必须显式写出：

```json
{
  "name": "block_2",
  "source_index": 2,
  "in_channels": 48,
  "expand_channels": 192,
  "out_channels": 48,
  "stride": 1,
  "residual": true
}
```

字段含义：

| 字段 | 含义 |
|---|---|
| `name` | 学生块名称 |
| `source_index` | 对应官方教师的块编号 |
| `in_channels` | 块输入通道 |
| `expand_channels` | 1×1 扩展后的通道 |
| `out_channels` | 投影后的输出通道 |
| `stride` | 逐通道稀疏卷积步幅 |
| `residual` | 是否执行输入与输出相加 |

调整数组时，前一块 `out_channels` 必须等于后一块 `in_channels`；首块输入必须等于
`stem_channels`；末块输出必须与 Tail 输入一致；残差块必须满足步幅为 1 且输入输出
通道相同。

## 18. 待实现的想法

以下内容是 Version 1 训练基线之后的候选方向。

### 18.1 可变时长 clip

将当前固定 50 ms clip 扩展为事件量驱动的自适应切片：

```text
最短时长 Tmin = 10 ms
最长时长 Tmax = 50 ms
目标事件数 Ntarget = 4096
事件硬上限 Nmax = 8192
```

当事件密集时，达到最短时长并累计足够事件即可输出；凝视等事件稀疏阶段则等待到
最长时长。真实 clip 时长作为 `duration_ratio` 输入四维辅助量。

### 18.2 固定与可变时长混合训练

先使用固定 50 ms 数据稳定训练，再在同一学生模型中混入可变时长 clip，使模型同时
适应规则采样与事件量驱动采样，并对不同切片策略进行独立验证。

### 18.3 在线 Transformer KV Cache

推理时缓存历史四个 clip 的 Key/Value，只计算新到 clip 的四个 Token，避免每次重新
计算完整五 clip 窗口。

### 18.4 Kalman 平滑

在学生的当前位置和未来位置之后增加轻量 Kalman 状态估计，对预测轨迹进行平滑，并
单独评估静止凝视、扫视和快速转向场景。

### 18.5 结构搜索与消融

围绕 JSON 配置进行：

- 5/6/7 块学生对比；
- 各倒残差块扩展通道和输出通道搜索；
- 1/2/4 Token 对比；
- Transformer 历史窗口、层数和前馈维度对比；
- 当前监督、未来监督、输出蒸馏和特征蒸馏的逐项消融；
- 多随机种子训练与均值、方差报告。

### 18.6 动态调度策略

后续可用规则控制或强化学习选择 clip 时长、事件阈值和计算模式。奖励可综合未来预测
误差、当前误差、端到端延迟、事件数量、计算量以及超时/资源约束惩罚。

### 18.7 Zynq-7010 部署

在模型精度和结构确定后，对学生 SCNN、Token 投影、Transformer 和输出头进行定点
量化，测量参数量、稀疏激活量、片上缓存需求、DSP/BRAM/LUT 使用率和端到端延迟，
再决定各模块由 PL、PS 或软硬件流水共同承担。
