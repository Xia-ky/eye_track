# SEE-D software_simplised Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建一个代码和配置内容不变、只保留 SEE-D 完整训练/量化/评估链路的 `software_simplised`。

**Architecture:** 从原 `software` 按白名单复制，不直接修改复制文件。整体保留被 `model/__init__.py` 静态导入的 `model` 包，裁剪顶层辅助入口、无关配置、无关权重与未导入的数据集可视化文件；用哈希清单证明来源一致。

**Tech Stack:** PowerShell、Python AST/compileall、SHA-256、PyTorch/MinkowskiEngine 源码。

---

### Task 1: 创建目标目录和复制白名单

**Files:**
- Create: `Proj3_trainning/software_simplised/`
- Copy unchanged: `Proj3_trainning/software/main.py`
- Copy unchanged: `Proj3_trainning/software/int_inference.py`
- Copy unchanged: `Proj3_trainning/software/test.py`
- Copy unchanged: `Proj3_trainning/software/setup.py`
- Copy unchanged: `Proj3_trainning/software/model/`
- Copy unchanged: `Proj3_trainning/software/utils/`
- Copy unchanged: `Proj3_trainning/software/MinkowskiEngine/`
- Copy unchanged: `Proj3_trainning/software/pybind/`
- Copy unchanged: `Proj3_trainning/software/src/`

- [ ] **Step 1:** 确认目标目录不存在，防止覆盖用户文件。
- [ ] **Step 2:** 创建目标目录。
- [ ] **Step 3:** 按白名单复制入口和源码目录。
- [ ] **Step 4:** 统计原文件与目标文件数量和大小。

### Task 2: 复制 SEE-D 数据、配置和论文权重

**Files:**
- Copy unchanged: `dataset/__init__.py`
- Copy unchanged: `dataset/ThreeET_plus.py`
- Copy unchanged: `dataset/custom_transforms.py`
- Copy unchanged: `dataset/augmentation.py`
- Copy unchanged: `dataset/regression_dataset.py`
- Copy unchanged: `dataset/sample.py`
- Copy unchanged: `dataset/train_files.txt`
- Copy unchanged: `dataset/val_files.txt`
- Copy unchanged: `dataset/test_files.txt`
- Copy unchanged: `configs/augment.json`
- Copy unchanged: `configs/float32/SEE-D.json`
- Copy unchanged: `configs/int8/SEE-D.json`
- Copy unchanged: `configs/model_cfg/SEE-D_model.json`
- Copy unchanged: `weights/Table2/SEE-D/`
- Copy unchanged: `Proj3_trainning/requirements-esda-software.txt`

- [ ] **Step 1:** 创建与原项目一致的相对目录。
- [ ] **Step 2:** 复制数据集主路径文件。
- [ ] **Step 3:** 只复制 SEE-D 配置。
- [ ] **Step 4:** 只复制 Table 2 的 SEE-D 权重。
- [ ] **Step 5:** 复制已验证的 Python 依赖清单。

### Task 3: 添加说明和来源清单

**Files:**
- Create: `Proj3_trainning/software_simplised/README.md`
- Create: `Proj3_trainning/software_simplised/UPSTREAM_README.md`
- Create: `Proj3_trainning/software_simplised/SOURCE_MANIFEST.csv`

- [ ] **Step 1:** 原样复制上游 `software/README.md` 为 `UPSTREAM_README.md`。
- [ ] **Step 2:** 编写简化版目录说明，明确哪些文件被保留及运行入口。
- [ ] **Step 3:** 为所有原样复制文件记录来源、目标、字节数和 SHA-256。
- [ ] **Step 4:** 标记新增说明文件不属于上游代码哈希比较范围。

### Task 4: 静态验证

**Files:**
- Verify: `Proj3_trainning/software_simplised/`

- [ ] **Step 1:** 对清单中每个来源和目标重新计算 SHA-256。
- [ ] **Step 2:** 验证不存在 SEE-A/B/C 或 MobileNetV2 配置和权重。
- [ ] **Step 3:** 使用 `python -m compileall` 检查所有 Python 文件语法。
- [ ] **Step 4:** 验证三个入口需要的本地模块和配置文件均存在。
- [ ] **Step 5:** 删除静态检查产生的 `__pycache__`。
- [ ] **Step 6:** 输出最终文件树、文件数量、总大小和验证结论。

