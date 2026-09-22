# SEE-D software_simplised 设计

## 目标

在不修改任何既有代码或配置内容的前提下，从 `Proj3_trainning/software`
复制出一个只面向 SEE-D 的最小可运行目录 `software_simplised`。

## 保留能力

- SEE-D float32 训练与验证。
- SEE-D int8 量化感知训练与验证。
- 论文 Table 2 SEE-D 官方 checkpoint 评估。
- int8 中间模型导出。
- 测试集坐标输出。
- 仓库内修改版 MinkowskiEngine 的原地编译。

## 删除范围

- NAS 搜索配置生成。
- NAS JSON 生成。
- 稀疏度分析。
- 预测可视化。
- SEE-A、SEE-B、SEE-C、MobileNetV2 配置。
- 除 SEE-D 之外的论文 checkpoint。
- 训练主路径不导入的数据集直方图和可视化模块。

## 为什么保留整个 model 目录

`model/__init__.py` 在模块导入时会导入标准 MobileNet、ResNet、UNet、
稀疏模型和量化模型。用户要求不能修改代码，因此不能通过裁剪
`model/__init__.py` 来删除这些模块；缺少其中任意关键文件可能使
`from model import *` 失败或者使 SEE-D 类不可见。

## 内容完整性

- 原 `software` 目录保持不变。
- 所有复制的 `.py`、`.json`、`.txt`、C/C++/CUDA 和 checkpoint 文件保持
  字节级一致。
- 生成 `SOURCE_MANIFEST.csv`，记录来源、目标、大小和 SHA-256。
- 新增的 `README.md`、清单和设计文档属于说明文件，不冒充上游代码。

## 验证

- 目标目录只包含设计规定的入口、配置、源码与权重。
- 对清单中的每个文件重新计算 SHA-256，并与来源比较。
- Python 文件通过 `compileall` 静态语法检查。
- 使用 AST/文件存在性检查本地导入依赖。
- 验证 float32、int8 和官方权重配置都指向存在的 SEE-D 文件。

