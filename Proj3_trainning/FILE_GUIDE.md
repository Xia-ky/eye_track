# Proj3_trainning 文件作用说明

这份说明用于回答两个问题：

1. 完整训练从哪个文件开始，数据经过哪些步骤，结果保存到哪里？
2. `Proj3_trainning` 中每个文件或同类文件组分别负责什么？

项目由两层组成：

- **AutoDL 运行封装**：根目录的 `run.sh`、`setup_env.sh`、依赖表和说明文档，是本项目为自动训练与持久化增加的代码。
- **ESDA 上游软件**：`software/` 中的训练、数据集、模型和内置 MinkowskiEngine 实现。为方便与原项目对照，除 `main.py` 的少量 AutoDL 适配外不修改上游实现。

## 代码是怎么来的

### 1. 论文作者发布的眼动追踪代码

主要来源是论文作者的仓库：

```text
https://github.com/CASR-HKU/ESDA.git
branch: eye_tracking
local commit: 48f24c7a32d4231e00689a17a8a51ce1dbb2929b
```

原始 `eye_tracking/software/` 有68个文件。本地逐文件 SHA-256 对比结果：

- 66个文件与作者仓库逐字节一致；
- `software/main.py` 做了 AutoDL/MLflow/输出目录适配；
- `software/dataset/ThreeET_plus.py` 增加了可配置的 `data_list_dir`；
- 作者分支中的文件没有从 Proj3 删除。

作者代码包括：

- `configs/`：论文模型和训练参数；
- `dataset/`：ThreeET+ 数据读取、切片和事件表示；
- `model/`：SEE、MobileNet、GRU和量化模型；
- `utils/`（除 `mlflow_paths.py`）：训练、指标、优化器和记录工具；
- 顶层训练、评估、可视化和整数导出脚本；
- `weights/Table2/`：作者为论文 Table 2 发布的配置和检查点。

### 2. 内嵌的 MinkowskiEngine

`MinkowskiEngine/`、`pybind/`、`src/` 和 `setup.py` 是为了让旧版 ESDA
在固定 PyTorch/CUDA 环境下可重复编译而放入项目的稀疏卷积后端。版本由
`MinkowskiEngine/__init__.py` 明确记录为：

```text
MinkowskiEngine 0.5.4
```

其源码版权头标明来源于 Chris Choy 和 NVIDIA。Python API 位于
`MinkowskiEngine/`，Python/C++绑定位于 `pybind/`，CPU/CUDA算子位于
`src/`。这些文件来自本地 `ESDA_main` 保存的上游软件版本，没有由本项目
重新实现。

### 3. MinkowskiEngine 随附的第三方代码

`src/3rdparty/` 是 MinkowskiEngine 为实现GPU哈希表、坐标映射和性能分析
而随源码携带的依赖：

- `robin_hood.h`：Martin Ankerl 的 robin-hood hashing，MIT许可证；
- `cudf/`、`hash/`、`concurrent_unordered_map.cuh`：NVIDIA/cuDF来源，
  文件版权头采用Apache 2.0等相应许可证。

它们不是眼动追踪算法，不建议当作项目业务代码修改。

### 4. Tonic 和 ThreeET+ 的关系

Tonic本身通过 `requirements-esda-software.txt` 安装，并没有把整个Tonic
库复制进来。`ThreeET_plus.py` 继承 Tonic 的 `Dataset`；
`custom_transforms.py` 明确注明部分切片逻辑修改自
`tonic.slicers.SliceByTimeEventsTargets`。因此 `dataset/` 是作者针对
ThreeET+ 数据集编写的适配层，底层事件处理调用外部 Tonic 库。

### 5. HAWQ量化代码

`model/HAWQ_quant_module/` 和 `HAWQ_mobilenetv2.py` 随论文作者的
`eye_tracking` 分支发布，用于构建 `MobileNetSubmanifoldQuant` 和模拟
低比特量化。本项目没有重写其量化数学；它们是 ESDA 量化训练流程的一部分。

### 6. 本项目新增或修改的文件

在 `software/` 内，本项目实际新增/修改的范围很小：

| 文件 | 本项目改动 |
|---|---|
| `software/main.py` | MLflow新实验创建、artifact URI转本地路径、可配置数据清单/metadata/cache、可选增强参数修复及中文说明。 |
| `software/dataset/ThreeET_plus.py` | 把硬编码的 `./dataset` 改为构造参数 `data_list_dir`，使远端运行可显式传入清单目录。 |
| `software/utils/mlflow_paths.py` | 新增：把本地路径或 `file://` MLflow artifact URI安全转成 `pathlib.Path`。 |

`software/` 外的 `run.sh`、`setup_env.sh`、依赖文件和根目录文档则是为
Proj1/Proj2/Proj3及AutoDL运行流程编写的封装。

### 7. 运行时生成、不是源码的文件

下列内容不是手写源码：

- `weights/**/*.pth`：论文作者训练生成的二进制权重；
- `Proj3_trainning_result/**/artifacts/*.pth`：本次训练生成的权重；
- `metrics/`、`params/`、`tags/`、`meta.yaml`：MLflow生成；
- `metadata/*.h5`：Tonic生成的切片索引；
- `cache/*.hdf5`：首次训练生成的事件体素缓存；
- `.DS_Store`：macOS Finder自动生成，可忽略。

## 一次训练的数据流

```text
run.sh
  │ 选择 MODEL、epoch、batch size、数据与输出路径
  ▼
software/main.py
  │ 读取 configs/float32/SEE-D.json
  │ 创建模型、优化器和 MLflow 实验
  ▼
dataset/ThreeET_plus.py
  │ 按 dataset/train_files.txt、val_files.txt 读取 HDF5 事件和标签
  ▼
Tonic SlicedDataset + dataset/* transform
  │ 时间切片、空间降采样、体素化
  ├── metadata/：保存切片索引
  └── cache/：保存体素化结果，后续 epoch 复用
  ▼
model/submanifoldModel.py 等
  │ 调用 MinkowskiEngine 稀疏卷积完成前向与反向传播
  ▼
utils/training_utils.py
  │ 训练、验证、计算 loss / p5 / p10 / distance
  ▼
MLflow artifacts
  │ 保存参数、指标、args.json、cfg.json 和 .pth 检查点
  ▼
run.sh
  │ 生成 report.json，复制到 /root/autodl-fs，写 PASSED/FAILED
  └── AUTO_SHUTDOWN=1 时请求 AutoDL 关机
```

## 根目录：AutoDL 运行封装

| 文件 | 作用 |
|---|---|
| `README.md` | 面向使用者的入口：安装、启动、后台运行、自动关机和结果检查。 |
| `FILE_GUIDE.md` | 当前文件：介绍数据流和项目内各文件职责。 |
| `run.sh` | 完整训练编排器；校验参数/GPU/数据，调用 `main.py`，收集日志和模型，原子持久化结果，最后可自动关机。 |
| `setup_env.sh` | 创建 Python 3.8 Conda 环境，安装固定版本依赖，并针对 CUDA 11.1/Compute Capability 8.6 编译 MinkowskiEngine。 |
| `requirements-esda-software.txt` | ESDA Python 依赖清单；部分版本为旧版 PyTorch、MLflow 和 MinkowskiEngine 的兼容性锁定。 |
| `docs/superpowers/specs/2026-07-26-proj3-documentation-design.md` | 本次文档与注释工作的范围设计，记录哪些代码允许修改。 |
| `docs/superpowers/plans/2026-07-26-proj3-documentation.md` | 本次文档与注释工作的实施和静态验证清单。 |

## `software/` 顶层：ESDA 命令入口

| 文件 | 作用 |
|---|---|
| `software/README.md` | ESDA 上游说明，包含数据集布局、float32/int8 训练、检查点评估和整数模型导出方法。 |
| `software/main.py` | float32/int8 训练与评估主入口；合并 JSON/命令行参数，创建数据管线、模型和 MLflow 运行，逐 epoch 训练验证并保存检查点。 |
| `software/test.py` | 面向测试集的推理/评估脚本。 |
| `software/visualize.py` | 把模型输出、事件表示或预测结果可视化。 |
| `software/sparsity_analysis.py` | 分析中间特征或网络层的稀疏率，为硬件和计算量评估提供依据。 |
| `software/int_inference.py` | 从量化训练检查点生成/执行整数推理，并导出后续硬件流程所需参数。 |
| `software/generate_search_cfg.py` | 生成网络结构搜索使用的候选配置。 |
| `software/gen_NAS_json.py` | 把神经架构搜索结果整理成 JSON 模型配置。 |
| `software/setup.py` | 构建仓库内附带的 MinkowskiEngine Python、C++ 和 CUDA 扩展；`setup_env.sh` 会调用它。 |
| `software/.DS_Store`（若存在） | macOS Finder 自动生成的目录元数据；训练不使用，可以忽略。 |

## `software/configs/`：模型与训练配置

JSON 配置由 `main.py` 读入，然后被命令行中非空参数覆盖。

| 文件/文件组 | 作用 |
|---|---|
| `configs/augment.json` | 数据增强参数，例如随机变换范围或增强策略。 |
| `configs/float32/SEE-A.json` | SEE-A 的 float32 训练配置。 |
| `configs/float32/SEE-B.json` | SEE-B 的 float32 训练配置。 |
| `configs/float32/SEE-C.json` | SEE-C 的 float32 训练配置。 |
| `configs/float32/SEE-D.json` | SEE-D 的 float32 训练配置；`run.sh` 默认使用此文件。 |
| `configs/float32/MobileNetV2.json` | MobileNetV2 基线的 float32 训练配置。 |
| `configs/int8/SEE-A.json` | SEE-A 的 int8/量化训练配置。 |
| `configs/int8/SEE-B.json` | SEE-B 的 int8/量化训练配置。 |
| `configs/int8/SEE-C.json` | SEE-C 的 int8/量化训练配置。 |
| `configs/int8/SEE-D.json` | SEE-D 的 int8/量化训练配置。 |
| `configs/int8/MobileNetV2.json` | MobileNetV2 的 int8/量化训练配置。 |
| `configs/model_cfg/SEE-A_model.json` | SEE-A 的逐层网络结构描述，供 NAS/整数导出/硬件流程读取。 |
| `configs/model_cfg/SEE-B_model.json` | SEE-B 的逐层网络结构描述。 |
| `configs/model_cfg/SEE-C_model.json` | SEE-C 的逐层网络结构描述。 |
| `configs/model_cfg/SEE-D_model.json` | SEE-D 的逐层网络结构描述。 |
| `configs/model_cfg/mobilenetv2_model.json` | MobileNetV2 的逐层网络结构描述。 |

## `software/dataset/`：事件数据读取与表示

| 文件 | 作用 |
|---|---|
| `dataset/__init__.py` | 汇总并导出数据集类和变换，使 `main.py` 可以从 `dataset` 直接导入。 |
| `dataset/ThreeET_plus.py` | ThreeET+ 眼动事件数据集入口；按照 split 和清单定位记录、HDF5 事件及标签。 |
| `dataset/train_files.txt` | 正式训练记录 ID 清单。 |
| `dataset/val_files.txt` | 验证记录 ID 清单。 |
| `dataset/test_files.txt` | 测试记录 ID 清单。 |
| `dataset/sample.py` | 单条样本/事件与标签的数据结构和基础处理。 |
| `dataset/regression_dataset.py` | 把切片后的事件表示包装成眼睛中心坐标回归样本，并处理缓存和可选增强。 |
| `dataset/histogram.py` | 将事件累积为直方图类表示。 |
| `dataset/custom_transforms.py` | 项目自定义变换，例如标签缩放、归一化、时间子采样和事件体素化。 |
| `dataset/augmentation.py` | 训练阶段的数据增强实现。 |
| `dataset/visualizations.py` | 数据集样本、事件和标签的辅助可视化函数。 |

## `software/model/`：眼动追踪网络

| 文件 | 作用 |
|---|---|
| `model/__init__.py` | 导出配置中可按名称创建的模型类；`main.py` 通过 architecture 名称实例化模型。 |
| `model/BaselineEyeTrackingModel.py` | 眼动坐标回归的基线模型和通用输出头。 |
| `model/submanifoldModel.py` | SEE 系列的核心子流形稀疏卷积模型组装逻辑。 |
| `model/UNetSubmanifold.py` | 使用 MinkowskiEngine 子流形稀疏卷积的 U-Net。 |
| `model/UNetStandard.py` | 对照用的标准稠密/常规卷积 U-Net。 |
| `model/resnet_submanifold.py` | 子流形稀疏卷积版 ResNet 模块。 |
| `model/resnet_standard.py` | 常规卷积版 ResNet 对照模块。 |
| `model/mobilenet_submanifold.py` | 子流形稀疏卷积版 MobileNet 模块。 |
| `model/mobilenet_standard.py` | 常规卷积版 MobileNet 模块。 |
| `model/ConvGRU.py` | 用卷积门控循环单元建模事件序列的时间信息。 |
| `model/sparsity.py` | 稀疏特征/层的稀疏率统计与辅助操作。 |
| `model/benchmark.py` | 统计模型参数量、FLOPs 或运行特征。 |
| `model/utils.py` | 模型构建所需的通用层、尺寸或配置辅助函数。 |
| `model/HAWQ_mobilenetv2.py` | 基于 HAWQ 方法的量化 MobileNetV2。 |
| `model/HAWQ_quant_module/quant_modules.py` | 量化卷积/线性层、模型冻结和解冻逻辑。 |
| `model/HAWQ_quant_module/quant_utils.py` | HAWQ 量化所需的尺度、舍入、位宽等工具。 |

## `software/utils/`：训练公共工具

| 文件 | 作用 |
|---|---|
| `utils/utils.py` | 配置命令处理、预训练权重加载、事件后处理变换和 epoch 状态更新等通用函数。 |
| `utils/training_utils.py` | 单个 epoch 的训练/验证循环及检查点辅助逻辑。 |
| `utils/optimizer.py` | 封装优化器和学习率调度器的创建与更新。 |
| `utils/metrics.py` | 加权 MSE、p5/p10 命中率、像素距离等损失和指标。 |
| `utils/recorder.py` | 累积每轮指标、写日志并提取最佳结果。 |
| `utils/mlflow_paths.py` | 将 MLflow 的 artifact URI 安全转换为本地文件路径。 |

## `software/weights/`：论文预训练权重

`weights/Table2/` 下五个模型目录结构相同：

| 文件/文件组 | 作用 |
|---|---|
| `weights/Table2/SEE-A/cfg.json` | 论文 Table 2 中 SEE-A 权重对应的推理配置。 |
| `weights/Table2/SEE-A/model_best_p10_acc.pth` | SEE-A 的预训练 PyTorch state dict。 |
| `weights/Table2/SEE-B/cfg.json` | SEE-B 权重对应的推理配置。 |
| `weights/Table2/SEE-B/model_best_p10_acc.pth` | SEE-B 的预训练 state dict。 |
| `weights/Table2/SEE-C/cfg.json` | SEE-C 权重对应的推理配置。 |
| `weights/Table2/SEE-C/model_best_p10_acc.pth` | SEE-C 的预训练 state dict。 |
| `weights/Table2/SEE-D/cfg.json` | SEE-D 权重对应的推理配置。 |
| `weights/Table2/SEE-D/model_best_p10_acc.pth` | SEE-D 的预训练 state dict。 |
| `weights/Table2/MobileNetV2/cfg.json` | MobileNetV2 权重对应的推理配置。 |
| `weights/Table2/MobileNetV2/model_best_p10_acc.pth` | MobileNetV2 的预训练 state dict。 |

`.pth` 是二进制模型参数，不是源代码；不要用文本编辑器修改。

## `software/MinkowskiEngine/`：稀疏卷积 Python API

| 文件 | 作用 |
|---|---|
| `MinkowskiEngine/__init__.py` | MinkowskiEngine Python 包入口，导出主要张量、层和后端扩展。 |
| `MinkowskiEngine/MinkowskiCommon.py` | 维度、卷积模式、区域类型等共享定义。 |
| `MinkowskiEngine/MinkowskiTensor.py` | 稀疏张量公共基类及坐标管理连接。 |
| `MinkowskiEngine/MinkowskiSparseTensor.py` | 稀疏张量主体：只保存活跃坐标及其特征。 |
| `MinkowskiEngine/MinkowskiTensorField.py` | 连续坐标/张量场表示及其到稀疏张量的转换。 |
| `MinkowskiEngine/MinkowskiCoordinateManager.py` | Python 层坐标映射与 kernel map 管理接口。 |
| `MinkowskiEngine/MinkowskiKernelGenerator.py` | 根据卷积核尺寸、步幅、膨胀率生成稀疏邻域。 |
| `MinkowskiEngine/MinkowskiConvolution.py` | 稀疏卷积、转置卷积和子流形卷积 Python 层。 |
| `MinkowskiEngine/MinkowskiChannelwiseConvolution.py` | 逐通道稀疏卷积。 |
| `MinkowskiEngine/MinkowskiPooling.py` | 局部/全局稀疏池化与反池化层。 |
| `MinkowskiEngine/MinkowskiNormalization.py` | 稀疏 BatchNorm、InstanceNorm 等归一化层。 |
| `MinkowskiEngine/MinkowskiNonlinearity.py` | ReLU、PReLU、Sigmoid 等稀疏特征激活层。 |
| `MinkowskiEngine/MinkowskiBroadcast.py` | 在稀疏特征与全局特征间执行广播运算。 |
| `MinkowskiEngine/MinkowskiPruning.py` | 根据布尔掩码删除稀疏坐标/特征。 |
| `MinkowskiEngine/MinkowskiInterpolation.py` | 稀疏坐标之间的特征插值。 |
| `MinkowskiEngine/MinkowskiUnion.py` | 合并多组不同坐标映射的稀疏张量。 |
| `MinkowskiEngine/MinkowskiFunctional.py` | 激活等操作的函数式 API。 |
| `MinkowskiEngine/MinkowskiOps.py` | cat、sum、mean 等多稀疏张量组合操作。 |
| `MinkowskiEngine/MinkowskiNetwork.py` | 稀疏网络基类和维度管理。 |
| `MinkowskiEngine/sparse_matrix_functions.py` | 稀疏矩阵相关 autograd 函数。 |
| `MinkowskiEngine/diagnostics.py` | 输出编译器、CUDA、BLAS 等安装诊断信息。 |
| `MinkowskiEngine/utils/__init__.py` | 汇总工具函数。 |
| `MinkowskiEngine/utils/collation.py` | 将不同样本的坐标/特征合并为稀疏 batch。 |
| `MinkowskiEngine/utils/coords.py` | 坐标批处理与兼容辅助函数。 |
| `MinkowskiEngine/utils/gradcheck.py` | 稀疏算子的梯度检查工具。 |
| `MinkowskiEngine/utils/init.py` | 稀疏卷积权重初始化函数。 |
| `MinkowskiEngine/utils/quantization.py` | 坐标量化、去重与索引映射。 |
| `MinkowskiEngine/utils/summary.py` | 类似 PyTorch summary 的模型结构/参数摘要。 |
| `MinkowskiEngine/modules/__init__.py` | 导出附加网络模块。 |
| `MinkowskiEngine/modules/resnet_block.py` | MinkowskiEngine 的 ResNet BasicBlock/Bottleneck。 |
| `MinkowskiEngine/modules/senet_block.py` | 稀疏 Squeeze-and-Excitation 网络块。 |

## `software/pybind/`：Python 与 C++/CUDA 的边界

| 文件 | 作用 |
|---|---|
| `pybind/minkowski.cpp` | 注册 CPU 侧 C++ 函数、类型和 Python 模块入口。 |
| `pybind/minkowski.cu` | 注册 CUDA 侧算子接口。 |
| `pybind/extern.hpp` | 声明 pybind 层需要调用的后端函数。 |

## `software/src/`：MinkowskiEngine C++/CUDA 后端

扩展名含义：`.hpp` 是 C++ 头文件，`.cpp` 是 CPU 实现，`.cuh` 是 CUDA 头文件，`.cu` 是 CUDA 实现。

### 基础设施与数据结构

| 文件 | 作用 |
|---|---|
| `src/common.hpp` | 后端各模块共享的类型、宏和工具。 |
| `src/types.hpp` | 坐标、索引、数值类型等核心别名。 |
| `src/utils.hpp` | C++ 后端通用辅助函数。 |
| `src/errors.hpp` | CPU/CUDA 错误检查与异常宏。 |
| `src/dispatcher.hpp` | 按数据类型、设备或维度分派模板实现。 |
| `src/coordinate.hpp` | 单个稀疏坐标的数据结构和哈希/比较。 |
| `src/coordinate_map_key.hpp` | 标识一组坐标映射的 key。 |
| `src/coordinate_map.hpp` | 坐标到行索引的通用映射抽象。 |
| `src/coordinate_map_cpu.hpp` | CPU 坐标映射实现。 |
| `src/coordinate_map_gpu.cuh` | GPU 坐标映射声明和设备端辅助。 |
| `src/coordinate_map_gpu.cu` | GPU 坐标映射实现。 |
| `src/coordinate_map_functors.cuh` | GPU 哈希表使用的坐标函子。 |
| `src/coordinate_map_manager.hpp` | 管理多层坐标映射和 kernel map 的接口。 |
| `src/coordinate_map_manager.cpp` | 坐标管理器 CPU 实现与显式实例化。 |
| `src/coordinate_map_manager.cu` | 坐标管理器 CUDA 实现。 |
| `src/kernel_region.hpp` | 定义卷积核覆盖的空间区域和偏移。 |
| `src/kernel_map.hpp` | 保存输入/输出活跃坐标之间的邻接索引。 |
| `src/kernel_map.cuh` | kernel map 的 CUDA 辅助实现。 |
| `src/storage.cuh` | CUDA 临时存储和内存管理辅助。 |
| `src/allocators.cuh` | GPU 数据结构使用的分配器。 |
| `src/gpu.cuh` | CUDA 通用声明、设备属性和启动辅助。 |
| `src/gpu.cu` | CUDA 通用函数实现。 |
| `src/sharedmem.cuh` | CUDA kernel 动态共享内存类型辅助。 |
| `src/mkl_alternate.hpp` | 没有 Intel MKL 时使用的 BLAS 兼容替代层。 |
| `src/primitives/small_vector.hpp` | 针对少量元素优化的小型向量容器。 |

### 卷积、矩阵乘法与数学

| 文件 | 作用 |
|---|---|
| `src/convolution_kernel.hpp` | 稀疏卷积 CPU kernel 声明/模板。 |
| `src/convolution_kernel.cuh` | 稀疏卷积 CUDA kernel 声明。 |
| `src/convolution_kernel.cu` | 稀疏卷积 CUDA kernel 实现。 |
| `src/convolution_cpu.cpp` | 稀疏卷积 CPU 前向/反向实现。 |
| `src/convolution_gpu.cu` | 稀疏卷积 GPU 前向/反向实现。 |
| `src/convolution_transpose_cpu.cpp` | 稀疏转置卷积 CPU 实现。 |
| `src/convolution_transpose_gpu.cu` | 稀疏转置卷积 GPU 实现。 |
| `src/spmm.cuh` | 稀疏-稠密矩阵乘 CUDA 声明。 |
| `src/spmm.cu` | 稀疏-稠密矩阵乘 CUDA 实现。 |
| `src/math_functions.hpp` | 后端数学函数公共接口。 |
| `src/math_functions.cuh` | CUDA 数学函数声明。 |
| `src/math_functions_cpu.cpp` | CPU 数学函数实现。 |
| `src/math_functions_gpu.cu` | GPU 数学函数实现。 |

### 池化、广播、插值和剪枝

| 文件 | 作用 |
|---|---|
| `src/pooling_max_kernel.hpp` | 最大池化 CPU kernel 声明。 |
| `src/pooling_max_kernel.cuh` | 最大池化 CUDA kernel 声明。 |
| `src/pooling_max_kernel.cu` | 最大池化 CUDA kernel 实现。 |
| `src/pooling_avg_kernel.hpp` | 平均池化 CPU kernel 声明。 |
| `src/pooling_avg_kernel.cuh` | 平均池化 CUDA kernel 声明。 |
| `src/pooling_avg_kernel.cu` | 平均池化 CUDA kernel 实现。 |
| `src/local_pooling_cpu.cpp` | 局部池化 CPU 前向/反向。 |
| `src/local_pooling_gpu.cu` | 局部池化 GPU 前向/反向。 |
| `src/local_pooling_transpose_cpu.cpp` | 局部转置池化 CPU 实现。 |
| `src/local_pooling_transpose_gpu.cu` | 局部转置池化 GPU 实现。 |
| `src/global_pooling_cpu.cpp` | 全局池化 CPU 实现。 |
| `src/global_pooling_gpu.cu` | 全局池化 GPU 实现。 |
| `src/direct_max_pool.cpp` | 直接最大池化的专用实现。 |
| `src/broadcast_kernel.hpp` | 广播运算 CPU kernel 声明。 |
| `src/broadcast_kernel.cuh` | 广播运算 CUDA kernel 声明。 |
| `src/broadcast_kernel.cu` | 广播运算 CUDA kernel 实现。 |
| `src/broadcast_cpu.cpp` | 广播运算 CPU 前向/反向。 |
| `src/broadcast_gpu.cu` | 广播运算 GPU 前向/反向。 |
| `src/interpolation_kernel.hpp` | 稀疏插值 kernel 公共声明。 |
| `src/interpolation_cpu.cpp` | 稀疏插值 CPU 实现。 |
| `src/interpolation_gpu.cu` | 稀疏插值 GPU 实现。 |
| `src/pruning.hpp` | 稀疏坐标剪枝接口。 |
| `src/pruning_cpu.cpp` | 剪枝 CPU 实现。 |
| `src/pruning_gpu.cu` | 剪枝 GPU 实现。 |
| `src/quantization.cpp` | 坐标量化与重复坐标合并的 C++ 实现。 |

### `software/src/3rdparty/`：随源码附带的第三方组件

这些文件不是 ESDA 业务代码，不建议修改：

| 文件 | 作用 |
|---|---|
| `src/3rdparty/robin_hood.h` | 高性能 robin-hood 哈希表。 |
| `src/3rdparty/concurrent_unordered_map.cuh` | CUDA 并发无序映射实现。 |
| `src/3rdparty/hash/managed.cuh` | GPU 哈希表托管内存辅助。 |
| `src/3rdparty/hash/helper_functions.cuh` | GPU 哈希表公共设备函数。 |
| `src/3rdparty/hash/hash_allocator.cuh` | GPU 哈希表内存分配器。 |
| `src/3rdparty/cudf/types.h` | cuDF 兼容的 C 类型定义。 |
| `src/3rdparty/cudf/types.hpp` | cuDF 兼容的 C++ 类型定义。 |
| `src/3rdparty/cudf/utilities/error.hpp` | cuDF 风格 CUDA 错误检查。 |
| `src/3rdparty/cudf/utilities/legacy/wrapper_types.hpp` | 旧版 cuDF 强类型包装器。 |
| `src/3rdparty/cudf/detail/utilities/device_atomics.cuh` | CUDA 原子操作辅助。 |
| `src/3rdparty/cudf/detail/utilities/device_operators.cuh` | CUDA 设备端比较/算术操作器。 |
| `src/3rdparty/cudf/detail/utilities/hash_functions.cuh` | CUDA 哈希函数。 |
| `src/3rdparty/cudf/detail/nvtx/nvtx3.hpp` | NVTX 性能分析标记接口。 |
| `src/3rdparty/cudf/detail/nvtx/ranges.hpp` | NVTX 区间标记封装。 |

## 哪些文件适合你优先修改

如果目标只是调整训练，通常只需要：

1. 改 `run.sh` 的环境变量，或启动命令中的 `MODEL/EPOCHS/WORKERS/BATCH_SIZE`。
2. 改 `configs/float32/<MODEL>.json` 中的超参数。
3. 需要改变训练/验证划分时改 `dataset/train_files.txt` 和 `dataset/val_files.txt`。
4. 需要改变网络结构时再改 `model/`。

除非正在研究稀疏算子本身，否则不要修改 `MinkowskiEngine/`、`pybind/`、`src/` 和 `src/3rdparty/`。这些目录与特定 PyTorch/CUDA ABI 紧密耦合，轻微修改也可能导致重新编译失败或数值行为变化。
