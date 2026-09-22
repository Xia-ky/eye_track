# software 文件说明

## 人工编写的 Version 1 代码

| 路径 | 作用 |
|---|---|
| `main.py` | 命令行入口；先检查配置、文件和官方权重哈希，再开始训练 |
| `configs/version1_fixed50.json` | 唯一的实验主配置，可调整学生块、Transformer、损失和训练参数 |
| `configs/model_cfg/SEE-D_student7.json` | 当前七块默认值的参考快照，仅用于人工核对 |
| `see_v1/config.py` | 严格解析 JSON；未知字段、断裂通道、非法残差立即报错 |
| `see_v1/labels.py` | 从 100 Hz 标签配对当前坐标和真实 `t+20 ms` 坐标 |
| `see_v1/data.py` | H5 读取、50 ms 切片、稀疏体素缓存、同步坐标增强 |
| `see_v1/student.py` | 七块学生 SCNN、4×16D Token、因果 Transformer 和两个输出头 |
| `see_v1/teacher.py` | 官方 INT8 SEE-D 加载、哈希校验、冻结和 64D Tail hook |
| `see_v1/masking.py` | 只允许注意当前及最近四个 clip，禁止读取未来 |
| `see_v1/losses.py` | 当前、未来、输出蒸馏、特征蒸馏四项损失 |
| `see_v1/metrics.py` | 分别统计当前和未来的距离、P5/P10/P15、横纵误差 |
| `see_v1/checkpoints.py` | 按未来距离优先选择最佳模型，原子保存训练/部署权重 |
| `see_v1/trainer.py` | epoch 循环、日志、验证、检查点和最终报告 |
| `tests/` | 不需要 GPU 的单元/静态测试，以及 AutoDL 上才运行的集成测试 |

## 从官方 ESDA 原样复用的代码/资产

| 路径 | 来源与原因 |
|---|---|
| `model/` | 官方 ESDA 稀疏 MobileNet、量化算子；保持论文运算路径 |
| `dataset/` | 官方 3ET 切片与体素转换；保持论文数据表示 |
| `dataset_lists/` | 官方训练/验证记录划分 |
| `configs/model_cfg/SEE-D_teacher.json` | 官方 SEE-D 网络结构 |
| `weights/Table2/SEE-D/model_best_p10_acc.pth` | 官方 Table 2 INT8 SEE-D 教师权重 |
| `MinkowskiEngine/`、`src/`、`pybind/`、`setup.py` | 官方项目携带的 MinkowskiEngine 构建源码 |

官方权重的 SHA-256 为：

```text
2e23b7f717bac1c8801ddc969d2e3ade48b1a4facc780c0b7ac48fc5e0be8be6
```

## 设计边界

- 教师只参与训练，不写入 `student_deploy.pth`。
- 未来目标来自数据集真实标签，不来自教师伪标签。
- 教师只对齐当前输出与池化 Tail 特征，不对齐教师 GRU 和学生 Transformer 隐状态。
- 网络块数由 JSON 数组动态决定，不把七块或官方九块写死为检查条件。

学生运行时的兼容模型 JSON 会直接由 `version1_fixed50.json` 自动生成到缓存目录，
因此增删块、修改通道或关闭残差时不需要同步编辑 `SEE-D_student7.json`。
