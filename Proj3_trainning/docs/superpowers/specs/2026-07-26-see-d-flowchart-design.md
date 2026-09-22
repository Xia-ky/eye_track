# SEE-D 数据流与网络结构流程图设计

## 目标

为 `Proj3_trainning` 生成一份可编辑的多页 Draw.io 流程图，帮助读者从原始
3ET 事件数据一路追踪到 SEE-D 的眼动坐标输出，并能逐块核对代码、论文和部署
结构。

## 信息来源

- 论文：`Paper/Zhang_Co-designing_a_Sub-millisecond_Latency_Event-based_Eye_Tracking_System_with_Submanifold_CVPRW_2024_paper.pdf`
- 浮点训练配置：`software/configs/float32/SEE-D.json`
- 网络配置：`software/configs/model_cfg/SEE-D_model.json`
- 数据集、切片、表示、增强、训练和模型源代码
- 论文中的软件架构、SCNN/GRU 异构部署图和 Table 2

## 页面结构

1. **端到端总览**：原始事件、体素网格、稀疏化、SCNN、GRU、FC 与输出。
2. **数据预处理与中间格式**：精确展示采样率、窗口、步长、缓存和张量格式。
3. **SEE-D 逐倒残差块**：精确到初始卷积、9 个倒残差块、最终卷积和全局池化。
4. **训练、指标、量化和部署**：区分当前 float32 训练路径、论文 int8 SCNN 路径，
   并标明 FPGA PL 与 ARM CPU 的边界。

## 图形约定

- 蓝色：事件与稀疏数据。
- 绿色：SCNN 主干。
- 紫色：时序网络与回归头。
- 橙色：训练、损失与指标。
- 红色：代码审计提示或容易误解之处。
- 所有形状均为 Draw.io 原生节点和连线，便于二次编辑。

## 尺寸口径

- 原始分辨率：`640×480`。
- 空间下采样后：`80×60`。
- 标签：100 Hz 经 `0.2` 时间抽样后为 20 Hz。
- 一个训练序列：`30` 个时间步，每步 `50 ms`，合计 `1.5 s`。
- 体素网格：`[T=30, C=3, H=60, W=80]`，`float32`。
- 稀疏坐标：`[N,3] = (bt,y,x)`，稀疏特征：`[N,C]`。
- 稀疏卷积后的空间尺寸用逻辑包围尺寸表示；stride=2 时按坐标量化取上界。

## 审核重点

- 配置和代码中的实际验证步长都是 `50 ms`，与源码附近“验证不重叠”的注释不一致。
- `Shift` 将 `(shift_x, shift_y)` 传给数组 `(H,W)` 轴，同时按 `(x/W,y/H)` 更新标签；
  图中按代码原样标注，不将变量名误当成严格的几何轴语义。
- `p5/p10` 只比较序列最后一个时间步；`Dist` 汇总全部 30 个时间步。
- 当前 `Proj3_trainning` 使用浮点 `MobileNetSubmanifold`；论文部署结果对应
  `MobileNetSubmanifoldQuant` 的 int8 SCNN 与浮点 GRU+FC 异构路径。

