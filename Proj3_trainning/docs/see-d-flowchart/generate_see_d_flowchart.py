"""Generate the editable SEE-D Draw.io diagram and matching PNG previews.

The diagram contents are derived from the local paper and Proj3 source code.
The same node geometry drives both outputs, so the preview is a faithful audit
view of the editable Draw.io source rather than a separately redrawn picture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape, unescape
from pathlib import Path
import re
import textwrap
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DRAWIO_PATH = ROOT / "SEE-D-dataflow.drawio"

PAGE_W = 1600
PAGE_H = 1000
SCALE = 1.5

COLORS = {
    "bg": "#F7F9FC",
    "ink": "#18212F",
    "muted": "#526174",
    "line": "#7C8DA3",
    "blue": "#DCEBFF",
    "blue_stroke": "#3973B9",
    "green": "#E0F3E8",
    "green_stroke": "#2E8B57",
    "purple": "#ECE5FA",
    "purple_stroke": "#7356A8",
    "orange": "#FFF0D6",
    "orange_stroke": "#C97816",
    "red": "#FDE5E2",
    "red_stroke": "#C7463A",
    "gray": "#EEF2F6",
    "gray_stroke": "#7B8795",
    "white": "#FFFFFF",
}


@dataclass
class Node:
    id: str
    x: int
    y: int
    w: int
    h: int
    text: str
    fill: str = COLORS["white"]
    stroke: str = COLORS["line"]
    font_size: int = 13
    bold: bool = False
    rounded: bool = True
    dashed: bool = False
    align: str = "center"
    valign: str = "middle"
    stroke_width: int = 2


@dataclass
class Edge:
    source: str
    target: str
    label: str = ""
    color: str = COLORS["line"]
    dashed: bool = False
    width: int = 2


@dataclass
class Page:
    name: str
    title: str
    subtitle: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def add(self, *nodes: Node) -> None:
        self.nodes.extend(nodes)

    def connect(self, source: str, target: str, label: str = "", color: str = COLORS["line"],
                dashed: bool = False, width: int = 2) -> None:
        self.edges.append(Edge(source, target, label, color, dashed, width))


def clean_text(value: str) -> str:
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    return unescape(value)


def title_node(page: Page) -> None:
    page.add(
        Node("title", 40, 24, 1520, 44, page.title, COLORS["bg"], COLORS["bg"], 25, True,
             rounded=False, align="left", stroke_width=0),
        Node("subtitle", 40, 70, 1520, 34, page.subtitle, COLORS["bg"], COLORS["bg"], 12,
             align="left", stroke_width=0),
    )


def build_pages() -> list[Page]:
    pages: list[Page] = []

    # ------------------------------------------------------------------ Page 1
    p = Page(
        "01-端到端总览",
        "SEE-D：从事件流到眼中心坐标的端到端数据流",
        "依据本地论文与 Proj3_trainning/software；蓝=数据/稀疏表示，绿=SCNN，紫=GRU/FC，橙=训练",
    )
    title_node(p)
    p.add(
        Node("raw", 50, 155, 205, 150,
             "<b>原始 3ET 记录</b><br>HDF5: events<br>[t,x,y,p], t=μs<br>640×480；p∈{0,1}<br><br>label.txt<br>float32；100 Hz",
             COLORS["blue"], COLORS["blue_stroke"], 13),
        Node("transform", 300, 155, 240, 150,
             "<b>空间/时间变换</b><br>事件坐标 ×0.125<br>标签每 5 个取 1 个<br>标签坐标归一化<br><br>640×480 → 80×60<br>100 Hz → 20 Hz",
             COLORS["blue"], COLORS["blue_stroke"], 13),
        Node("slice", 585, 155, 245, 150,
             "<b>序列切片与表示</b><br>窗口 1.5 s，步长 50 ms<br>30 × 50 ms 事件片<br>每片 3-bin voxel grid<br><br>float32 [30,3,60,80]",
             COLORS["blue"], COLORS["blue_stroke"], 13),
        Node("sparse", 875, 155, 245, 150,
             "<b>稠密 → 稀疏</b><br>合并 B 与 T<br>[BT,60,80,3]<br>仅保留三通道绝对值和≠0<br><br>coord [N,3]=(bt,y,x)<br>feat [N,3]",
             COLORS["blue"], COLORS["blue_stroke"], 13),
        Node("scnn", 1165, 140, 370, 180,
             "<b>SEE-D 稀疏 MobileNetV2 主干</b><br>初始 3×3 s2: 3→24<br>9 个倒残差块<br>末端 1×1: 256→64<br>稀疏全局平均池化<br><br>每个时间片 → 64-D embedding",
             COLORS["green"], COLORS["green_stroke"], 14),
        Node("seq", 260, 405, 310, 145,
             "<b>恢复时间维</b><br>pool 后 x.F: [B×T,64]<br>view → [B,30,64]<br><br>每个 50 ms 片一个特征向量",
             COLORS["green"], COLORS["green_stroke"], 13),
        Node("gru", 650, 405, 300, 145,
             "<b>GRU（1 层）</b><br>input_size=64<br>hidden_size=64<br>batch_first=True<br><br>输出 [B,30,64]",
             COLORS["purple"], COLORS["purple_stroke"], 14),
        Node("fc", 1030, 405, 255, 145,
             "<b>逐时间步 FC</b><br>Linear 64→2<br><br>输出 [B,30,2]<br>归一化 (x,y)",
             COLORS["purple"], COLORS["purple_stroke"], 14),
        Node("xy", 1360, 405, 175, 145,
             "<b>眼中心</b><br>x_px = x·80<br>y_px = y·60<br><br>论文部署：<br>每片持续输出",
             COLORS["purple"], COLORS["purple_stroke"], 13),
        Node("trainlane", 50, 650, 1485, 235,
             "", "#FCFDFE", COLORS["gray_stroke"], 12, rounded=True, dashed=True,
             align="left"),
        Node("cfg", 80, 690, 285, 150,
             "<b>当前 Proj3 float32 训练</b><br>MobileNetSubmanifold<br>Adam，lr=0.001，100 epochs<br>batch=20，weighted MSE<br>参数量：177,970",
             COLORS["orange"], COLORS["orange_stroke"], 13),
        Node("loss", 430, 690, 310, 150,
             "<b>监督与指标</b><br>loss：全部 30 步 (x,y)<br>权重 [4/3, 1]<br>p5/p10：仅最后一步<br>Dist：全部 30 步，80×60 尺度",
             COLORS["orange"], COLORS["orange_stroke"], 13),
        Node("qat", 805, 690, 310, 150,
             "<b>论文 int8 路径</b><br>float32 预训练 → HAWQv3 QAT<br>SCNN 输入与权重 int8<br>GRU+FC 保持 CPU 浮点 SIMD<br><br>不是本次 float32 run.sh 的同一路径",
             COLORS["orange"], COLORS["orange_stroke"], 13),
        Node("paper", 1180, 690, 325, 150,
             "<b>论文 Table 2：SEE-D</b><br>p5 81.37%｜p10 99.53%｜Dist 3.71 px<br>SCNN 0.59 ms｜GRU+FC 0.11 ms<br>总延迟 0.70 ms｜功耗 3.86 W<br>单次能耗 2.29 mJ",
             COLORS["gray"], COLORS["gray_stroke"], 13),
    )
    for a, b in [("raw", "transform"), ("transform", "slice"), ("slice", "sparse"), ("sparse", "scnn"),
                 ("scnn", "seq"), ("seq", "gru"), ("gru", "fc"), ("fc", "xy")]:
        p.connect(a, b, color=COLORS["blue_stroke"] if b in {"transform", "slice", "sparse"} else COLORS["green_stroke"])
    p.connect("cfg", "loss", color=COLORS["orange_stroke"])
    p.connect("loss", "qat", "部署前另行量化", COLORS["orange_stroke"])
    p.connect("qat", "paper", color=COLORS["orange_stroke"])
    pages.append(p)

    # ------------------------------------------------------------------ Page 2
    p = Page(
        "02-数据预处理与格式",
        "SEE-D 数据预处理：时间、空间、缓存与批处理格式",
        "本页严格按 ThreeET_plus.py、custom_transforms.py、regression_dataset.py、main.py 和 SEE-D.json 展开",
    )
    title_node(p)
    p.add(
        Node("event_h5", 45, 145, 240, 170,
             "<b>HDF5 /events</b><br>原始 dtype: 4×uint64<br>字段 [t,x,y,p]<br>t：微秒；x∈[0,639]<br>y∈[0,479]；p∈{0,1}",
             COLORS["blue"], COLORS["blue_stroke"], 13),
        Node("event_load", 340, 145, 245, 170,
             "<b>ThreeETplus.__getitem__</b><br>astype structured int<br>p = p×2−1<br>得到 p∈{−1,+1}<br><br>事件数组 [Ne]，Ne 可变",
             COLORS["blue"], COLORS["blue_stroke"], 13),
        Node("event_down", 640, 145, 250, 170,
             "<b>Tonic Downsample</b><br>spatial_factor=0.125<br>x,y 坐标缩至 1/8<br><br>有效传感器尺寸<br>(W,H,P)=(80,60,2)",
             COLORS["blue"], COLORS["blue_stroke"], 13),
        Node("label_txt", 45, 380, 240, 170,
             "<b>label.txt</b><br>每行 '(...)' → float32<br>原始频率 100 Hz<br>至少含 [x,y,eye_state]<br><br>x,y 为 640×480 像素坐标",
             COLORS["blue"], COLORS["blue_stroke"], 13),
        Node("label_scale", 340, 380, 245, 170,
             "<b>标签变换</b><br>ScaleLabel(0.125)<br>TemporalSubsample(0.2)<br>即 labels[::5]<br><br>100 Hz → 20 Hz",
             COLORS["blue"], COLORS["blue_stroke"], 13),
        Node("label_norm", 640, 380, 250, 170,
             "<b>NormalizeLabel</b><br>x÷80，y÷60<br>合并前一步等价于<br>x_original÷640<br>y_original÷480<br><br>target float32 [L,≥3]",
             COLORS["blue"], COLORS["blue_stroke"], 13),
        Node("long_slice", 960, 145, 275, 235,
             "<b>长序列滑窗</b><br>SliceByTimeEventsTargets<br>window = 30×50 ms = 1.5 s<br>stride = 50 ms<br>overlap = 1.45 s<br>include_incomplete=False<br><br>事件与 30 个标签同步切片",
             COLORS["blue"], COLORS["blue_stroke"], 13),
        Node("audit_stride", 1270, 145, 280, 235,
             "<b>审计提示：验证步长</b><br>train_stride=1<br>val_stride=1<br>所以训练和验证都每 50 ms 滑动。<br><br>main.py 中“validation non-overlapping”注释与当前配置不一致。",
             COLORS["red"], COLORS["red_stroke"], 13),
        Node("short_slice", 960, 430, 275, 190,
             "<b>序列内二次切片</b><br>SliceLongEventsToShort<br>time_window=50,000 μs<br>overlap=0<br><br>1.5 s → 30 个事件片<br>每片对应一个 20 Hz 标签",
             COLORS["blue"], COLORS["blue_stroke"], 13),
        Node("voxel", 1270, 430, 280, 190,
             "<b>3-bin Voxel Grid</b><br>每个 50 ms 事件片<br>to_voxel_grid_numpy<br>有符号极性按时间插值累加<br>squeeze polarity 轴<br><br>每序列 [30,3,60,80] float32",
             COLORS["blue"], COLORS["blue_stroke"], 13),
        Node("cache", 960, 680, 275, 205,
             "<b>SlicedDataset + DiskCachedDataset</b><br>metadata：切片索引<br>cache：voxel grid + target<br>首次 epoch 生成较慢<br>后续 epoch 从磁盘缓存读取",
             COLORS["gray"], COLORS["gray_stroke"], 13),
        Node("augment", 1270, 680, 280, 205,
             "<b>仅训练集增强</b><br>Flip p=0.5：W 轴翻转，x→1−x<br>Shift p=0.5，max_shift=20<br>同时移动 voxel 与归一化标签<br><br><font color='#C7463A'>轴审计：代码把 shift_x/shift_y<br>传给数组 H/W 轴，命名需谨慎解释</font>",
             COLORS["orange"], COLORS["orange_stroke"], 12),
        Node("loader", 340, 680, 550, 205,
             "<b>DataLoader 输出（送入模型）</b><br>inputs: torch.float32 [B,30,3,60,80]<br>targets: torch.float32 [B,30,≥3]<br>masked_lengths: int tensor；当前 flip+shift 配置通常全 0<br><br>B=20（默认全训练）；shuffle：train=True，val=False<br>pin_memory：train=True",
             COLORS["purple"], COLORS["purple_stroke"], 14),
    )
    for a, b in [("event_h5", "event_load"), ("event_load", "event_down"), ("label_txt", "label_scale"),
                 ("label_scale", "label_norm"), ("event_down", "long_slice"), ("label_norm", "long_slice"),
                 ("long_slice", "short_slice"), ("short_slice", "voxel"), ("voxel", "cache"),
                 ("cache", "augment"), ("augment", "loader")]:
        p.connect(a, b, color=COLORS["blue_stroke"])
    p.connect("long_slice", "audit_stride", "当前 cfg", COLORS["red_stroke"], dashed=True)
    pages.append(p)

    # ------------------------------------------------------------------ Page 3
    p = Page(
        "03-SEE-D逐倒残差块",
        "SEE-D 稀疏主干：精确到每一个倒残差块",
        "每块均为 1×1 expand → 3×3 depthwise sparse conv → 1×1 project；残差仅在 stride=1 且 Cin=Cout 时启用",
    )
    title_node(p)
    p.add(
        Node("dense", 40, 130, 265, 135,
             "<b>模型输入</b><br>[B,30,3,60,80]<br>view → [B×30,3,60,80]<br>permute → [B×30,60,80,3]",
             COLORS["blue"], COLORS["blue_stroke"], 12),
        Node("to_sparse", 350, 130, 295, 135,
             "<b>dense_to_sparse</b><br>mask = abs(C).sum≠0<br>coord [N,3]=(bt,y,x), int32<br>feat [N,3], float32<br>Minkowski SparseTensor",
             COLORS["blue"], COLORS["blue_stroke"], 12),
        Node("stem", 690, 130, 300, 135,
             "<b>Stem：ConvBNReLU</b><br>Sparse Conv 3×3, s=2, 3→24<br>BN + ReLU6<br>逻辑 H×W: 60×80 → 30×40<br>tensor_stride: 1 → 2",
             COLORS["green"], COLORS["green_stroke"], 12),
        Node("legend", 1040, 125, 515, 145,
             "<b>读法</b><br>E = 1×1 sparse conv + BN + ReLU6<br>DW = 3×3 channelwise + BN + ReLU6；P = 1×1 sparse conv + BN（无激活）<br>C @ H×W；N 随稀疏度变化；stride=1 不主动扩张活跃坐标",
             COLORS["gray"], COLORS["gray_stroke"], 11),
    )
    p.connect("dense", "to_sparse", color=COLORS["blue_stroke"])
    p.connect("to_sparse", "stem", color=COLORS["green_stroke"])

    blocks = [
        ("b0", 45, 340, "Block 0｜stage [8,32,1,2]", "24 @ 30×40", "E 24→192<br>DW 192, s2<br>P 192→32", "32 @ 15×20", "否", 4),
        ("b1", 355, 340, "Block 1｜stage [6,48,2,2] #1", "32 @ 15×20", "E 32→192<br>DW 192, s2<br>P 192→48", "48 @ 8×10", "否", 8),
        ("b2", 665, 340, "Block 2｜stage [6,48,2,2] #2", "48 @ 8×10", "E 48→288<br>DW 288, s1<br>P 288→48", "48 @ 8×10", "是", 8),
        ("b3", 975, 340, "Block 3｜stage [1,64,2,1] #1", "48 @ 8×10", "E 48→48<br>DW 48, s1<br>P 48→64", "64 @ 8×10", "否", 8),
        ("b4", 1285, 340, "Block 4｜stage [1,64,2,1] #2", "64 @ 8×10", "E 64→64<br>DW 64, s1<br>P 64→64", "64 @ 8×10", "是", 8),
        ("b5", 45, 610, "Block 5｜stage [1,72,3,2] #1", "64 @ 8×10", "E 64→64<br>DW 64, s2<br>P 64→72", "72 @ 4×5", "否", 16),
        ("b6", 355, 610, "Block 6｜stage [1,72,3,2] #2", "72 @ 4×5", "E 72→72<br>DW 72, s1<br>P 72→72", "72 @ 4×5", "是", 16),
        ("b7", 665, 610, "Block 7｜stage [1,72,3,2] #3", "72 @ 4×5", "E 72→72<br>DW 72, s1<br>P 72→72", "72 @ 4×5", "是", 16),
        ("b8", 975, 610, "Block 8｜stage [1,256,1,1]", "72 @ 4×5", "E 72→72<br>DW 72, s1<br>P 72→256", "256 @ 4×5", "否", 16),
    ]
    for bid, x, y, heading, inp, ops, out, residual, ts in blocks:
        p.add(Node(
            bid, x, y, 270, 205,
            f"<b>{heading}</b><br><br>输入：{inp}<br>{ops}<br>输出：{out}<br>残差：{residual}｜tensor_stride={ts}",
            COLORS["green"], COLORS["green_stroke"], 11,
        ))
    for a, b in [("stem", "b0"), ("b0", "b1"), ("b1", "b2"), ("b2", "b3"), ("b3", "b4"),
                 ("b4", "b5"), ("b5", "b6"), ("b6", "b7"), ("b7", "b8")]:
        p.connect(a, b, color=COLORS["green_stroke"])
    p.add(
        Node("tail", 1285, 610, 270, 95,
             "<b>Tail 1×1 ConvBNReLU</b><br>256→64，s=1，BN+ReLU6<br>64 @ 4×5；tensor_stride=16",
             COLORS["green"], COLORS["green_stroke"], 11),
        Node("gap", 1285, 735, 270, 80,
             "<b>Minkowski GlobalAvgPool</b><br>每个 (batch,time) 的活跃点均值<br>[B×30,64]",
             COLORS["green"], COLORS["green_stroke"], 11),
        Node("note", 45, 870, 1510, 70,
             "<b>尺寸口径：</b>H×W 为稀疏坐标的逻辑包围尺寸；60×80 经 s2 → 30×40 → 15×20 → 8×10 → 4×5。"
             " MinkowskiEngine 实际存储的是活跃坐标与特征，内存并不分配完整稠密网格。",
             COLORS["gray"], COLORS["gray_stroke"], 12),
    )
    p.connect("b8", "tail", color=COLORS["green_stroke"])
    p.connect("tail", "gap", color=COLORS["green_stroke"])
    pages.append(p)

    # ------------------------------------------------------------------ Page 4
    p = Page(
        "04-训练量化与异构部署",
        "SEE-D：GRU 回归、训练指标、int8 量化与 Zynq 异构部署边界",
        "左侧是当前 Proj3 float32 训练；右侧是论文部署链。两者共享 SEE-D 拓扑，但数值格式和执行设备不同",
    )
    title_node(p)
    p.add(
        Node("embedding", 55, 150, 245, 135,
             "<b>SCNN embedding</b><br>GlobalAvgPool 后<br>x.F [B×30,64]<br>view → [B,30,64]",
             COLORS["green"], COLORS["green_stroke"], 13),
        Node("gru", 355, 150, 245, 135,
             "<b>标准 PyTorch GRU</b><br>1 层，64→64<br>batch_first=True<br>输出 [B,30,64]<br><font color='#526174'>不是 ConvGRU.py</font>",
             COLORS["purple"], COLORS["purple_stroke"], 13),
        Node("fc", 655, 150, 220, 135,
             "<b>Linear 回归头</b><br>64→2<br>逐时间步执行<br>output [B,30,2]<br>归一化 (x,y)",
             COLORS["purple"], COLORS["purple_stroke"], 13),
        Node("target", 55, 375, 245, 165,
             "<b>监督 target</b><br>[B,30,≥3]<br>训练仅取 [:,:,:2]<br>eye_state 列不进入坐标 loss<br><br>单位：归一化坐标",
             COLORS["orange"], COLORS["orange_stroke"], 13),
        Node("loss", 355, 375, 245, 165,
             "<b>Weighted MSE</b><br>全部 30 个时间步<br>weights=[640/480,1]<br>即 x 误差权重 4/3<br><br>反向传播到 SCNN+GRU+FC",
             COLORS["orange"], COLORS["orange_stroke"], 13),
        Node("metrics", 655, 375, 300, 165,
             "<b>验证指标</b><br>p5/p10/p15：只用最后一步<br>误差换算尺度 80×60<br>Dist：全部 30 步欧氏距离均值<br><br>保存 best val loss / p5 / p10",
             COLORS["orange"], COLORS["orange_stroke"], 13),
        Node("artifacts", 55, 650, 545, 190,
             "<b>当前 Proj3 训练产物</b><br>model_epoch*.pth｜model_last_epoch100.pth<br>model_best_val_loss.pth｜model_best_p5_acc.pth｜model_best_p10_acc.pth<br>MLflow：args.json、cfg.json、log.txt、metrics/artifacts<br><br>当前配置 architecture=MobileNetSubmanifold（float32）",
             COLORS["gray"], COLORS["gray_stroke"], 13),
        Node("quant", 1015, 150, 515, 170,
             "<b>论文部署前：HAWQv3 量化感知微调</b><br>float32 checkpoint → MobileNetSubmanifoldQuant<br>SCNN 的输入激活 X 与权重 W 量化为 int8<br>乘法、加法、移位构成整数图；比例除法变为额外整数乘法+shift<br><br>GRU+FC 未并入 FPGA int8 SCNN",
             COLORS["orange"], COLORS["orange_stroke"], 13),
        Node("fpga", 1015, 390, 310, 230,
             "<b>FPGA Programmable Logic</b><br>输入：binary bitmap + sparse feature<br>token=(x,y,end) + feature stream<br><br>Tokenize → sparse conv blocks<br>1×1 / SLB / DW3×3 / residual<br>→ sparse global pooling<br><br>SCNN：int8｜论文 0.59 ms",
             COLORS["green"], COLORS["green_stroke"], 13),
        Node("cpu", 1370, 390, 160, 230,
             "<b>ARM Cortex-A53</b><br>接收 64-D embedding<br><br>GRU + FC<br>Eigen C++<br>NEON SIMD<br>浮点<br><br>论文 0.11 ms",
             COLORS["purple"], COLORS["purple_stroke"], 12),
        Node("result", 1015, 690, 515, 170,
             "<b>论文 Table 2：SEE-D 端到端结果</b><br>p5=81.37%｜p10=99.53%｜Dist=3.71 px｜参数≈178K<br>总延迟=0.70 ms｜功耗=3.86 W｜能耗=2.29 mJ/inf<br>资源：DSP 1606｜BRAM 1092｜FF 90K｜LUT 130K<br><br><font color='#C7463A'>这些是论文量化异构系统结果，不可直接等同于本次 GPU float32 训练日志。</font>",
             COLORS["gray"], COLORS["gray_stroke"], 13),
        Node("boundary", 980, 350, 585, 300, "", "#FCFDFE", COLORS["red_stroke"], 12,
             rounded=True, dashed=True, align="left"),
    )
    for a, b in [("embedding", "gru"), ("gru", "fc")]:
        p.connect(a, b, color=COLORS["purple_stroke"])
    p.connect("target", "loss", color=COLORS["orange_stroke"])
    p.connect("fc", "loss", "prediction", COLORS["orange_stroke"])
    p.connect("fc", "metrics", color=COLORS["orange_stroke"])
    p.connect("loss", "artifacts", "optimizer.step / checkpoints", COLORS["orange_stroke"])
    p.connect("quant", "fpga", "导出 int8 SCNN", COLORS["orange_stroke"])
    p.connect("fpga", "cpu", "64-D embedding", COLORS["purple_stroke"])
    p.connect("cpu", "result", "归一化 (x,y)", COLORS["purple_stroke"])
    pages.append(p)

    return pages


def node_style(n: Node) -> str:
    return (
        f"rounded={1 if n.rounded else 0};whiteSpace=wrap;html=1;"
        f"fillColor={n.fill};strokeColor={n.stroke};strokeWidth={n.stroke_width};"
        f"fontColor={COLORS['ink']};fontSize={n.font_size};"
        f"fontStyle={1 if n.bold else 0};align={n.align};verticalAlign={n.valign};"
        f"spacing=8;dashed={1 if n.dashed else 0};shadow=0;"
    )


def to_drawio(pages: list[Page]) -> None:
    mxfile = ET.Element("mxfile", {
        "host": "Electron",
        "agent": "Codex",
        "version": "24.7.17",
        "type": "device",
    })
    for page_index, page in enumerate(pages):
        diagram = ET.SubElement(mxfile, "diagram", {
            "id": f"see-d-page-{page_index + 1}",
            "name": page.name,
        })
        model = ET.SubElement(diagram, "mxGraphModel", {
            "dx": "1422", "dy": "794", "grid": "1", "gridSize": "10",
            "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
            "fold": "1", "page": "1", "pageScale": "1",
            "pageWidth": str(PAGE_W), "pageHeight": str(PAGE_H),
            "math": "0", "shadow": "0",
        })
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", {"id": "0"})
        ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
        for node in page.nodes:
            cell = ET.SubElement(root, "mxCell", {
                "id": node.id,
                "value": node.text,
                "style": node_style(node),
                "vertex": "1",
                "parent": "1",
            })
            ET.SubElement(cell, "mxGeometry", {
                "x": str(node.x), "y": str(node.y), "width": str(node.w), "height": str(node.h),
                "as": "geometry",
            })
        for edge_index, edge in enumerate(page.edges):
            style = (
                f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
                f"html=1;strokeColor={edge.color};strokeWidth={edge.width};"
                f"dashed={1 if edge.dashed else 0};endArrow=block;endFill=1;"
                "labelBackgroundColor=#FFFFFF;fontSize=11;"
            )
            cell = ET.SubElement(root, "mxCell", {
                "id": f"e{edge_index}", "value": edge.label, "style": style,
                "edge": "1", "parent": "1", "source": edge.source, "target": edge.target,
            })
            ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    ET.indent(mxfile, space="  ")
    ET.ElementTree(mxfile).write(DRAWIO_PATH, encoding="utf-8", xml_declaration=True)


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf" if bold else "C:/Windows/Fonts/simsun.ttc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), max(10, int(size * SCALE)))
    return ImageFont.load_default()


def wrap_line(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    if not text:
        return [""]
    parts: list[str] = []
    current = ""
    for char in text:
        proposal = current + char
        if draw.textlength(proposal, font=font) <= max_width or not current:
            current = proposal
        else:
            parts.append(current)
            current = char
    if current:
        parts.append(current)
    return parts


def draw_arrow(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color: str, width: int, dashed: bool) -> None:
    scaled = [(int(x * SCALE), int(y * SCALE)) for x, y in points]
    line_width = max(2, int(width * SCALE))
    if dashed:
        for a, b in zip(scaled[:-1], scaled[1:]):
            x1, y1 = a
            x2, y2 = b
            length = max(abs(x2 - x1), abs(y2 - y1))
            if length == 0:
                continue
            for i in range(0, length, 18):
                j = min(i + 10, length)
                t1, t2 = i / length, j / length
                draw.line((x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1,
                           x1 + (x2 - x1) * t2, y1 + (y2 - y1) * t2),
                          fill=color, width=line_width)
    else:
        draw.line(scaled, fill=color, width=line_width, joint="curve")
    if len(scaled) >= 2:
        x2, y2 = scaled[-1]
        x1, y1 = scaled[-2]
        if abs(x2 - x1) >= abs(y2 - y1):
            sign = 1 if x2 > x1 else -1
            tip = [(x2, y2), (x2 - sign * 12, y2 - 7), (x2 - sign * 12, y2 + 7)]
        else:
            sign = 1 if y2 > y1 else -1
            tip = [(x2, y2), (x2 - 7, y2 - sign * 12), (x2 + 7, y2 - sign * 12)]
        draw.polygon(tip, fill=color)


def edge_points(source: Node, target: Node) -> list[tuple[int, int]]:
    sx, sy = source.x + source.w, source.y + source.h // 2
    tx, ty = target.x, target.y + target.h // 2
    if target.x >= source.x + source.w + 20:
        mid = (sx + tx) // 2
        return [(sx, sy), (mid, sy), (mid, ty), (tx, ty)]
    # Vertical or wrapped connection.
    sx = source.x + source.w // 2
    sy = source.y + source.h
    tx = target.x + target.w // 2
    ty = target.y
    mid = (sy + ty) // 2
    return [(sx, sy), (sx, mid), (tx, mid), (tx, ty)]


def draw_node(draw: ImageDraw.ImageDraw, node: Node) -> None:
    box = tuple(int(v * SCALE) for v in (node.x, node.y, node.x + node.w, node.y + node.h))
    if node.stroke_width > 0:
        if node.dashed:
            draw.rounded_rectangle(box, radius=int(12 * SCALE), fill=node.fill, outline=node.stroke,
                                   width=max(1, int(node.stroke_width * SCALE)))
            # Pillow has no stable dashed rounded outline; corner styling still identifies the group.
        else:
            radius = int(12 * SCALE) if node.rounded else 0
            draw.rounded_rectangle(box, radius=radius, fill=node.fill, outline=node.stroke,
                                   width=max(1, int(node.stroke_width * SCALE)))
    else:
        draw.rectangle(box, fill=node.fill)
    raw = clean_text(node.text)
    font = get_font(node.font_size, node.bold or "<b>" in node.text)
    lines: list[str] = []
    max_width = int((node.w - 18) * SCALE)
    for logical_line in raw.splitlines():
        lines.extend(wrap_line(draw, logical_line, font, max_width))
    spacing = max(2, int(3 * SCALE))
    line_heights = [draw.textbbox((0, 0), line or " ", font=font)[3] for line in lines]
    total_h = sum(line_heights) + spacing * max(0, len(lines) - 1)
    y = int(node.y * SCALE + 10 * SCALE)
    if node.valign == "middle":
        y = int(node.y * SCALE + max(6 * SCALE, (node.h * SCALE - total_h) / 2))
    for line, line_h in zip(lines, line_heights):
        line_w = draw.textlength(line, font=font)
        if node.align == "left":
            x = int((node.x + 10) * SCALE)
        else:
            x = int(node.x * SCALE + (node.w * SCALE - line_w) / 2)
        draw.text((x, y), line, fill=COLORS["ink"], font=font)
        y += line_h + spacing


def render_preview(page: Page, index: int) -> Path:
    image = Image.new("RGB", (int(PAGE_W * SCALE), int(PAGE_H * SCALE)), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    nodes = {n.id: n for n in page.nodes}
    # Group boundaries first so they do not cover contained nodes.
    group_nodes = [n for n in page.nodes if n.dashed and not clean_text(n.text).strip()]
    for node in group_nodes:
        draw_node(draw, node)
    for edge in page.edges:
        if edge.source not in nodes or edge.target not in nodes:
            continue
        points = edge_points(nodes[edge.source], nodes[edge.target])
        draw_arrow(draw, points, edge.color, edge.width, edge.dashed)
        if edge.label:
            mid = points[len(points) // 2]
            font = get_font(10)
            label = clean_text(edge.label)
            bbox = draw.textbbox((0, 0), label, font=font)
            x, y = int(mid[0] * SCALE), int(mid[1] * SCALE)
            pad = int(3 * SCALE)
            draw.rectangle((x - pad, y - (bbox[3] - bbox[1]) - pad,
                            x + (bbox[2] - bbox[0]) + pad, y + pad), fill=COLORS["white"])
            draw.text((x, y - (bbox[3] - bbox[1])), label, fill=edge.color, font=font)
    for node in page.nodes:
        if node not in group_nodes:
            draw_node(draw, node)
    output = ROOT / f"SEE-D-dataflow-page-{index:02d}.png"
    image.save(output, optimize=True)
    return output


def write_readme() -> None:
    text = """# SEE-D 数据流与网络结构图

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
"""
    (ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    pages = build_pages()
    to_drawio(pages)
    outputs = [render_preview(page, i + 1) for i, page in enumerate(pages)]
    write_readme()
    print(f"Draw.io: {DRAWIO_PATH}")
    for output in outputs:
        print(f"Preview: {output}")


if __name__ == "__main__":
    main()
