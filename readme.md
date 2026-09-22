# Version 1：官方教师 + 七块学生 + 20 ms 预测

本目录已经实现设计文档中的第一阶段（固定 50 ms clip）。训练时使用论文作者发布的
INT8 SEE-D 权重作为冻结教师；部署文件只包含七块学生 SCNN、四 Token Transformer
和当前/未来坐标头。

## 目录

```text
version1/
├── DESIGN.md                  # 完整设计、数据流、损失和后续阶段
├── docs/                      # 实施计划
├── software/                  # 共用训练代码、官方教师资产、JSON 配置和测试
├── Proj2_test_train/          # 真实数据裁剪 + 1 epoch 冒烟训练
└── Proj3_trainning/           # 100 epoch 完整训练
```

`software` 只需上传一次，Proj2 与 Proj3 都从这里调用同一份代码，因此不会出现
“冒烟测试一套代码、正式训练另一套代码”的漂移。

## 已实现的数据流

```text
3ET H5 events + 100 Hz labels
  -> 50 ms clip
  -> 3 通道稀疏体素 (3, 60, 80)
  -> 七个可由 JSON 增删/改通道的倒残差块
  -> GlobalAvgPool -> 64D
  -> 4 × 16D Token
  -> 最近 5 个 clip 的因果 Transformer
  -> 当前坐标 + 20 ms 后坐标
```

教师使用同一个输入，提供当前坐标和池化后的 64D Tail 特征。教师始终
`eval + no_grad + requires_grad=False`，且权重 SHA-256 在启动时校验。

## AutoDL 放置方式

把整个 `version1` 目录放到：

```text
/root/autodl-tmp/zynq_cnn/code/version1
```

数据保持在持久盘：

```text
/root/autodl-fs/event_data/train/<record_id>/<record_id>.h5
/root/autodl-fs/event_data/train/<record_id>/label.txt
```

环境继续使用已经跑通 Version 0 的 `esda` Conda 环境。

## 运行顺序

先做不占 GPU 的静态契约检查：

```bash
cd /root/autodl-tmp/zynq_cnn/code/version1/software
conda run --no-capture-output -n esda \
  python main.py --config configs/version1_fixed50.json --dry-run
```

然后运行真实小样本冒烟训练：

```bash
cd /root/autodl-tmp/zynq_cnn/code/version1/Proj2_test_train
DATA_ROOT=/root/autodl-fs/event_data AUTO_SHUTDOWN=0 bash run.sh
```

只有出现以下标记才进入完整训练：

```text
/root/autodl-fs/results/Proj2_test_train_v1/PASSED
```

完整训练（训练成功或失败后都保存日志，结束后请求关机）：

```bash
cd /root/autodl-tmp/zynq_cnn/code/version1/Proj3_trainning
DATA_ROOT=/root/autodl-fs/event_data \
AUTO_SHUTDOWN=1 \
EPOCHS=100 \
BATCH_SIZE=20 \
WORKERS=4 \
bash run.sh
```

结果默认写到 `/root/autodl-fs/results/Proj3_trainning_v1/`。关键文件：

- `console.log`：完整控制台输出；
- `report.json`：运行状态和最优当前/未来平均距离；
- `metrics.csv`、`metrics.jsonl`：逐 epoch 分项损失、P5/P10/P15 和距离；
- `best_future_distance.pth`：可恢复训练的最佳完整检查点；
- `student_deploy.pth`：不含官方教师与蒸馏 Adapter 的学生部署权重。

## 修改网络

主要编辑
[`configs/version1_fixed50.json`](software/configs/version1_fixed50.json)
中的 `student.blocks`。块数从数组长度动态获得；每块显式指定输入、扩展、输出通道、
步幅和残差。配置加载器会拒绝通道断裂和非法残差，避免错误拖到 CUDA 前向阶段才出现。

当前只实现固定 50 ms 阶段。混合/动态 clip、Kalman、强化学习和 FPGA 部署不在本轮代码内。
