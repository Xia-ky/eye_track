# SEE-D 等效清晰重构设计

## 1. 背景与目标

`Proj3_trainning/software` 来自论文项目，能够完成 SEE-D 的训练与评估，但数据处理、
模型构建、训练循环、MLflow、检查点和量化逻辑耦合较深，阅读和修改网络参数的成本较高。

本次工作允许重构：

- `Proj3_trainning/software_simplised`
- `Proj2_test_train/software_simplised`

原始目录保持不变：

- `Proj3_trainning/software`
- `Proj2_test_train/software`

重构目标是：

1. 严格保留论文和原代码的 SEE-D 数据处理、网络运算、损失和指标定义。
2. 使用一份完整 JSON 配置描述运行参数和网络结构。
3. 允许从 JSON 增减倒残差块，并调整每个块的输入通道、扩展通道、输出通道、步幅和残差开关。
4. 保留 Float32 训练、Int8 QAT 和论文官方检查点评估。
5. 让代码职责清楚、输入输出形状明确，便于阅读和继续修改。
6. 提供一个使用完整训练记录和完整验证记录的 1-epoch AutoDL 测试项目。

## 2. 非目标

本次不包括：

- 保持旧 `main.py`、`int_inference.py` 或旧命令行参数兼容。
- NAS 搜索和 NAS 配置生成。
- SEE-A、SEE-B、SEE-C 或普通 MobileNetV2 训练配置。
- FPGA 工程、硬件 JSON 或整数硬件导出链的重构。
- 新增并行分支、特征融合或与论文不同的网络模块。
- 更换 MinkowskiEngine 稀疏卷积后端。
- 为错误的网络配置自动推断或修复通道。

## 3. 总体目录

正式训练版本：

```text
Proj3_trainning/
├── software/                         # 原始基准，保持不变
└── software_simplised/
    ├── train.py                      # Float32 / Int8 QAT 统一训练入口
    ├── evaluate.py                   # 检查点评估入口
    ├── inspect_model.py              # 输出逐层结构、形状和参数量
    ├── sync_proj2.py                 # 将核心代码同步到 Proj2 并校验哈希
    ├── README.md                     # 中文使用说明和阅读顺序
    ├── configs/
    │   ├── see_d_float32.json
    │   └── see_d_int8_qat.json
    ├── esda/
    │   ├── config.py
    │   ├── data/
    │   │   ├── dataset.py
    │   │   ├── transforms.py
    │   │   ├── voxel.py
    │   │   └── factory.py
    │   ├── models/
    │   │   ├── sparse_ops.py
    │   │   ├── blocks.py
    │   │   ├── see_d.py
    │   │   ├── quantization.py
    │   │   └── factory.py
    │   ├── engine/
    │   │   ├── losses.py
    │   │   ├── metrics.py
    │   │   ├── trainer.py
    │   │   └── checkpoint.py
    │   └── utils/
    │       ├── logging.py
    │       └── reproducibility.py
    ├── third_party/
    │   └── minkowski_engine/         # 原项目随附的修改版稀疏卷积后端
    └── tests/
        ├── test_config.py
        ├── test_data_contract.py
        ├── test_model_structure.py
        ├── test_checkpoint_mapping.py
        └── compare_with_original.py
```

AutoDL 单记录完整测试版本：

```text
Proj2_test_train/
├── software/                         # 原始测试代码，保持不变
└── software_simplised/
    ├── run.sh
    ├── setup_env.sh
    ├── prepare_one_record.py
    ├── configs/
    │   └── see_d_one_record.json
    ├── README.md
    └── ...                           # 与 Proj3 相同的重构核心代码
```

`Proj3_trainning/software_simplised` 是重构代码的设计基准。Proj2 中的核心 Python 文件
由它同步得到，并通过清单和 SHA-256 检查防止两份核心代码意外不一致。Proj2 保持可独立
上传和运行，不依赖 AutoDL 上同时存在 Proj3。

原项目随附的 MinkowskiEngine 源码按原始内容放入 `third_party/minkowski_engine`。
环境安装脚本从这个明确路径编译安装，避免它与 SEE-D 自己的模型代码混在同一级目录。

## 4. 配置设计

每次运行读取一份完整 JSON，不使用多层配置继承。配置分为：

- `data`
- `model`
- `training`
- `quantization`
- `runtime`
- `output`

网络部分使用一个有序列表完整列出本次网络包含的倒残差块。列表长度可变，每个块包含：

```json
{
  "name": "block_2",
  "checkpoint_source": "block_2",
  "in_channels": 48,
  "expand_channels": 288,
  "out_channels": 48,
  "stride": 1,
  "use_residual": true
}
```

严格校验以下条件：

1. Stem 输出等于第一个块输入。
2. 前一个块输出等于后一个块输入。
3. 最后一个块输出等于 Tail 输入。
4. Tail 输出等于 GRU 输入。
5. GRU 隐藏维度等于回归头输入。
6. 启用残差时，必须满足步幅为 1 且输入输出通道相同。
7. 通道、步幅和层数必须为正整数。
8. 块名称不得重复。
9. 未知字段和缺失字段均立即报错。
10. 倒残差块列表至少包含 1 个块；允许通过 JSON 增加、删除和重排块。
11. 倒残差块步幅只允许 1 或 2，与当前 SEE-D 使用的算子范围一致。
12. 非空的 `checkpoint_source` 必须是唯一、稳定的字符串标识，不使用当前列表下标猜测来源。

校验器不自动修改配置，也不隐式关闭残差。

论文基准配置默认使用以下 9 个块，严格对应当前 SEE-D：

| 块 | 输入 | 扩展 | 输出 | 步幅 | 残差 |
|---|---:|---:|---:|---:|---|
| 0 | 24 | 192 | 32 | 2 | 否 |
| 1 | 32 | 192 | 48 | 2 | 否 |
| 2 | 48 | 288 | 48 | 1 | 是 |
| 3 | 48 | 48 | 64 | 1 | 否 |
| 4 | 64 | 64 | 64 | 1 | 是 |
| 5 | 64 | 64 | 72 | 2 | 否 |
| 6 | 72 | 72 | 72 | 1 | 是 |
| 7 | 72 | 72 | 72 | 1 | 是 |
| 8 | 72 | 72 | 256 | 1 | 否 |

## 5. 数据等效契约

输入：

- H5 `events`，字段为 `(t, x, y, p)`。
- `label.txt`，标签采样率为 100 Hz。
- 时间戳单位为微秒。

处理顺序保持：

1. 极性 `{0, 1}` 转为 `{-1, +1}`。
2. 空间缩放因子 `0.125`，得到宽 80、高 60。
3. 标签每 5 个采样一次，100 Hz 变为 20 Hz。
4. 标签按宽 80、高 60 归一化。
5. 长窗口长度 1.5 秒，步长 50 ms。
6. 每个长窗口分成 30 个 50 ms 时间片。
7. 每个时间片转换为 3-bin voxel grid。

单样本输出：

- 事件 `float32 [30, 3, 60, 80]`
- 标签 `float32 [30, 2]`

批量输出：

- 事件 `[B, 30, 3, 60, 80]`
- 标签 `[B, 30, 2]`

训练增强保持原顺序和概率：

- 水平翻转，概率 0.5。
- 随机平移，概率 0.5，最大位移 20。

验证集不执行随机增强。磁盘缓存只包装一次；必须通过数值对比证明该工程修正不会改变
送入模型的张量。

## 6. SEE-D Float32 模型契约

前向路径保持：

```text
[B,T,3,60,80]
  -> 合并 B 和 T
  -> dense_to_sparse
  -> Sparse Stem: 3x3, stride 2, 3->24, BN, ReLU6
  -> N 个 JSON 定义的倒残差块（N >= 1，论文基准 N = 9）
  -> Sparse Tail: 1x1, 256->64, BN, ReLU6
  -> MinkowskiGlobalAvgPooling
  -> [B,T,64]
  -> GRU(64, 64, 1 layer, batch_first=True)
  -> Linear(64, 2)
  -> [B,T,2]
```

倒残差块必须保留原项目的卷积类型、BN、ReLU6 和残差相加位置。重构只改变职责划分、
命名、注释和配置方式，不改变数学运算。

`inspect_model.py` 输出：

- 层名。
- 输入、扩展和输出通道。
- 步幅。
- 是否启用残差。
- 推导出的张量形状。
- 各层及总参数量。

## 7. 训练、损失与指标契约

默认行为保持：

- Adam，学习率 0.001。
- SEE-D 不启用学习率调度器。
- 默认训练 100 epochs。
- 随机种子 20。
- 加权 MSE，坐标权重 `[4/3, 1]`。
- Loss 使用全部 30 个时间步。
- P5 和 P10 使用最后一个时间步。
- Distance 使用全部时间步，并按宽 80、高 60 恢复到像素坐标。

保存内容：

- 最优检查点。
- 最后一轮检查点。
- 本次完整 JSON。
- 控制台日志。
- 每轮训练和验证指标。
- 机器可读的最终 `report.json`。

## 8. 动态检查点与 Int8 QAT 契约

- Float32 与 Int8 QAT 共用同一个 `model` 结构。
- QAT 保留原项目使用的量化/HAWQ 模块。
- 论文默认 `shift_bit=16`、`bias_bit=16`。
- 支持从 Float32 检查点初始化 QAT。
- 不擅自把 GRU 和最终 FC 改为稀疏整数硬件算子。
- 支持评估论文官方检查点。

检查点加载器不写死倒残差块数量。它先从检查点参数键中发现源网络包含的块和参数，再按
当前 JSON 中每个块的 `checkpoint_source` 显式映射。

新版保存的检查点同时包含完整生效 JSON 和块清单。检查点中的块数量由保存时的模型决定，
可以少于、等于或多于论文基准结构，不存在固定块数常量。旧版官方检查点通过单独的兼容
适配器解析出同样的源块清单。

提供两种互不混用的加载模式：

1. `resume`：恢复同一网络继续训练。模型结构、所有参数键和形状必须完全一致。
2. `initialize`：使用基准检查点初始化删减或调整后的网络。

`initialize` 模式规则：

- `checkpoint_source` 指向检查点中的源块，例如当前 `small_block_1` 可以显式继承
  `block_2`。
- 删除某个源块时，其他块仍按稳定来源标识加载，不会因为列表下标移动而错配。
- 来源参数形状完全一致才可加载；形状不一致时列出具体参数并停止。
- 新增且不继承旧权重的块必须显式设置 `checkpoint_source: null`，随后使用模型默认初始化。
- Stem、Tail、GRU 和回归头也按名称和形状检查；不兼容时停止，不能静默跳过。
- 每次初始化生成机器可读报告，列出已加载块、已删除源块、全新初始化块和不兼容项。
- 禁止没有报告的部分加载。

## 9. Proj2 单记录完整训练测试

默认选择：

- 训练记录：`train/1_2`
- 验证记录：`train/1_6`
- 两条记录均使用完整文件，不裁剪为 2 秒片段。
- 训练 1 epoch。
- 执行完整的数据切片、缓存、前向、反向、验证、检查点保存和结果报告。

记录 ID 可在 JSON 中修改。测试输出写入持久盘的独立结果目录。`run.sh`：

1. 检查数据文件、Python 环境、CUDA、MinkowskiEngine 和输出目录。
2. 生成只包含指定完整记录的数据列表。
3. 保存本次生效配置和环境摘要。
4. 运行 1-epoch Float32 训练与验证。
5. 检查检查点和 `report.json` 是否生成。
6. 将所有控制台输出写入日志。
7. `AUTO_SHUTDOWN=0` 时不关机。
8. 仅当训练成功且 `AUTO_SHUTDOWN=1` 时请求 AutoDL 关机。
9. 训练失败时保留实例，避免关机后无法检查错误。

## 10. 验证策略

### 10.1 本地静态验证

- `compileall` 检查全部 Python 文件。
- JSON 正确和错误用例。
- 可变数量倒残差块的连接和残差约束，包括 1 块、删减块和默认 9 块用例。
- 默认配置与原 SEE-D 参数逐项对比。
- 模型形状静态推导。
- 动态检查点块发现、稳定来源映射和张量形状检查。
- 删除中间块后，后续块仍从正确 `checkpoint_source` 加载的测试。
- Proj2 与 Proj3 核心文件 SHA-256 一致性。

### 10.2 AutoDL 数值等效验证

固定相同记录、切片、检查点和随机种子，对比原版与重构版：

- 样本数量和切片时间边界。
- voxel grid 与标签。
- sparse input。
- Stem 输出。
- 默认 9 块基准配置中所有倒残差块的逐块输出。
- Tail 和全局池化输出。
- GRU 输出和最终坐标。
- Loss、P5、P10 和 Distance。

Float32 前向使用明确的绝对/相对容差。GPU 非确定性允许合理浮点误差，但不允许通过放宽
容差掩盖结构差异。原版逐层数值等效验证使用默认 9 块论文配置；删减网络没有原版对应物，
改用形状推导、前向、反向、保存和恢复测试验证其可运行性。

### 10.3 运行验证

1. 先运行 Proj2 的完整单记录 1-epoch 测试。
2. 再运行官方 Float32 检查点评估。
3. 再运行 Int8 QAT 检查点评估。
4. 全部通过后才启动 Proj3 全数据 100-epoch 训练。

## 11. 可解释性要求

所有人工重写的核心模块必须包含：

- 模块用途。
- 输入和输出结构。
- 张量各维含义。
- 单位和归一化范围。
- 与论文流程的对应位置。
- 为什么使用稀疏张量、倒残差、全局池化和 GRU。
- 修改哪些 JSON 字段会影响该模块。
- 常见配置错误及报错原因。

README 提供两条阅读路线：

1. 只想改倒残差参数的最短路线。
2. 想理解完整数据和模型流程的详细路线。

## 12. 实施顺序

1. 编写行为锁定测试和原版基准提取工具。
2. 实现 JSON 配置和严格校验。
3. 重构数据流水线。
4. 重构 Float32 SEE-D。
5. 重构损失、指标、训练和评估。
6. 接入 Int8 QAT 和官方检查点映射。
7. 增加模型结构检查工具。
8. 生成 Proj2 自包含测试副本、配置和脚本。
9. 完成本地静态验证。
10. 输出 AutoDL 数值等效和完整单记录测试命令。

## 13. 验收标准

满足以下条件才可宣称完成：

- 原始 `software` 目录未被修改。
- 新 JSON 默认配置完整表达原 SEE-D 结构。
- JSON 可以合法删减倒残差块，并能据此构建更小的网络。
- 删减网络可以通过稳定来源标识继承检查点中形状兼容的倒残差块，并输出完整加载报告。
- 非法通道或残差设置能在建模前给出明确错误。
- 本地静态测试全部通过。
- Proj2 和 Proj3 的核心实现哈希一致。
- Proj2 可独立上传到 AutoDL，并对完整训练/验证记录完成 1 epoch。
- 相同输入和检查点下，原版与重构版的逐阶段输出在规定容差内一致。
- Float32 与 Int8 QAT 能正常训练、保存、恢复和评估。
- README 足以指导用户修改倒残差块并运行 Proj2/Proj3。
