# SEE-D 数据流与网络结构图

## 文件

- `SEE-D-dataflow.drawio`：四页、可编辑的 Draw.io 源文件。
- `SEE-D-dataflow-page-01.png`：端到端总览。
- `SEE-D-dataflow-page-02.png`：数据预处理与中间格式。
- `SEE-D-dataflow-page-03.png`：逐倒残差块的 SEE-D 稀疏主干。
- `SEE-D-dataflow-page-04.png`：GRU、训练指标、量化和异构部署。
- `generate_see_d_flowchart.py`：从结构化节点定义重新生成图和预览。

## 最重要的形状

| 阶段 | 格式 |
|---|---|
| 原始事件 | structured array `[Ne]`，字段 `(t,x,y,p)`，`t` 为 μs |
| 事件序列表示 | `float32 [30,3,60,80]` |
| DataLoader 输入 | `float32 [B,30,3,60,80]` |
| 稀疏坐标 / 特征 | `coord [N,3]=(bt,y,x)` / `feat [N,C]` |
| SCNN 池化输出 | `[B×30,64]` |
| GRU 输入 / 输出 | `[B,30,64]` / `[B,30,64]` |
| FC 输出 | `[B,30,2]`，归一化 `(x,y)` |

`3` 个 voxel 通道是一个 50 ms 事件片内的三个时间 bin；极性以有符号值参与
时间插值和累加，并不是 `2 polarity × 3 time bins = 6` 个通道。

## 倒残差块

`SEE-D_model.json` 的 backbone 五个 stage 展开为 9 个 block：

1. `24→192→32`, stride 2
2. `32→192→48`, stride 2
3. `48→288→48`, stride 1，残差
4. `48→48→64`, stride 1
5. `64→64→64`, stride 1，残差
6. `64→64→72`, stride 2
7. `72→72→72`, stride 1，残差
8. `72→72→72`, stride 1，残差
9. `72→72→256`, stride 1

随后 `1×1 256→64`、稀疏全局平均池化、标准 `nn.GRU(64,64)` 和
`Linear(64,2)`。

## 代码审计结论

1. `SEE-D.json` 中 `train_stride=1`、`val_stride=1`，所以两个 split 都采用
   50 ms 步长和 1.45 s 重叠。`main.py` 附近“validation non-overlapping”的注释
   不符合当前配置。
2. `Shift.shift_array()` 把 `(shift_x, shift_y)` 应用到数组 `(H,W)` 两轴，
   但标签更新写作 `x += shift_x/W`、`y += shift_y/H`。图中按代码事实标注；
   如果后续追求严格几何一致性，建议单独做可视化单元测试后再修改。
3. `p5/p10/p15` 只评价序列最后一个时间步；`Dist` 汇总全部时间步。
4. Proj3 当前训练是 `MobileNetSubmanifold` float32；论文 Table 2 的部署结果来自
   `MobileNetSubmanifoldQuant` 的 int8 SCNN + ARM 浮点 GRU/FC。
5. 第 3 页的 H×W 是稀疏坐标逻辑包围尺寸。MinkowskiEngine 实际保存活跃坐标，
   不会为每层分配完整的稠密特征图。

## 来源映射

- 数据读取：`software/dataset/ThreeET_plus.py`
- 切片与 voxel：`software/dataset/custom_transforms.py`
- 缓存与增强：`software/dataset/regression_dataset.py`、
  `software/dataset/augmentation.py`
- 训练组装：`software/main.py`
- 浮点模型：`software/model/mobilenet_submanifold.py`
- 稠密转稀疏：`software/model/utils.py`
- 量化模型：`software/model/HAWQ_mobilenetv2.py`
- loss 与指标：`software/utils/metrics.py`、`software/utils/training_utils.py`
- 配置：`software/configs/float32/SEE-D.json`、
  `software/configs/model_cfg/SEE-D_model.json`

## 重新生成

使用 Codex 工作区自带的 Python（需要 Pillow）运行：

```powershell
python generate_see_d_flowchart.py
```
