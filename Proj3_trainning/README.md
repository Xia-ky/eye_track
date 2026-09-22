# Proj3_trainning：ESDA 完整训练

本目录用于在 AutoDL GPU 实例上执行完整的 ESDA 眼动追踪训练。它是在 ESDA 上游 `software/` 代码外增加的一层运行封装，负责环境安装、运行前检查、日志与模型持久化，以及训练结束后的自动关机。

> 文件较多时，先阅读 [FILE_GUIDE.md](FILE_GUIDE.md)。其中解释了目录结构、训练数据流和各文件（或同类文件组）的作用。

## 推荐阅读顺序

1. 本文件：知道怎样安装、启动和查看结果。
2. `run.sh`：理解一次完整训练的生命周期。
3. `software/main.py`：理解模型、数据集、缓存、训练和验证如何连接。
4. `software/configs/float32/SEE-D.json`：查看本次 SEE-D 的具体超参数。
5. `software/dataset/`、`software/model/`、`software/utils/`：分别深入数据、网络和训练工具。
6. `software/MinkowskiEngine/`、`software/src/`：需要研究稀疏卷积底层实现时再读。

## 默认训练参数

`run.sh` 支持使用环境变量覆盖参数；不设置时使用下列默认值：

```text
MODEL=SEE-D
EPOCHS=100
WORKERS=4
BATCH_SIZE=20
AUTO_SHUTDOWN=1
```

`MODEL` 可取 `SEE-A`、`SEE-B`、`SEE-C`、`SEE-D` 或 `MobileNetV2`。

## 运行前提

1. `Proj2_test_train` 冒烟训练已经通过，并存在：

   ```text
   /root/autodl-fs/results/Proj2_test_train/PASSED
   ```

2. 完整数据集位于 `/root/autodl-fs/event_data`，至少包含 `train/`。
3. 当前实例具有可用 NVIDIA GPU。
4. 项目推荐放在高速临时盘，例如：

   ```text
   /root/autodl-tmp/zynq_cnn/Proj3_trainning
   ```

   训练结果写入持久化数据盘 `/root/autodl-fs/results/Proj3_trainning`，关机后仍保留。

## 安装环境

如果 Proj2 与 Proj3 使用同一实例、同一个 `esda` Conda 环境，不必重新安装。换了实例或系统镜像时执行：

```bash
cd /root/autodl-tmp/zynq_cnn/Proj3_trainning
chmod +x setup_env.sh run.sh
bash setup_env.sh 2>&1 | tee setup_env.log
```

安装结束后会输出 PyTorch、CUDA、GPU 可用性和 MinkowskiEngine 版本。

## 执行完整训练

前台运行：

```bash
cd /root/autodl-tmp/zynq_cnn/Proj3_trainning
bash run.sh
```

覆盖默认参数：

```bash
MODEL=SEE-D EPOCHS=100 WORKERS=4 BATCH_SIZE=20 bash run.sh
```

后台运行并把控制台输出直接保存到持久盘：

```bash
mkdir -p /root/autodl-fs/results/Proj3_trainning
RUN_LOG=/root/autodl-fs/results/Proj3_trainning/console-$(date -u +%Y%m%dT%H%M%SZ).log

nohup env \
  DATA_ROOT=/root/autodl-fs/event_data \
  MODEL=SEE-D \
  EPOCHS=100 \
  WORKERS=4 \
  BATCH_SIZE=20 \
  AUTO_SHUTDOWN=1 \
  bash run.sh >"$RUN_LOG" 2>&1 &

echo "PID=$!"
echo "日志=$RUN_LOG"
```

查看训练进度：

```bash
tail -f "$RUN_LOG"
```

按 `Ctrl+C` 只会退出 `tail`，不会终止由 `nohup` 启动的训练。

## 自动关机

`AUTO_SHUTDOWN=1` 时，无论训练成功还是失败，`run.sh` 都会先尝试保存报告与日志，再调用 AutoDL 的 `/usr/bin/shutdown`。调试脚本时应关闭自动关机：

```bash
AUTO_SHUTDOWN=0 bash run.sh
```

注意：脚本只能保证“已尝试持久化并请求关机”。若 AutoDL 拒绝关机命令，日志会打印警告，需要在网页手动关机。

## 结果目录

```text
/root/autodl-fs/results/Proj3_trainning/
├── PASSED 或 FAILED                 # 最近一次运行的最终状态
├── report.json                      # 最近一次运行的机器可读摘要
├── latest_run_id                    # 最近一次运行目录名
├── console-<timestamp>.log          # nohup 命令产生的完整控制台日志
└── <run_id>/
    ├── report.json                  # 此次运行的固定报告
    ├── training.log                 # main.py 的训练输出
    ├── mlflow/                      # MLflow 参数、指标和 .pth 检查点
    ├── metadata/                    # Tonic 切片索引
    └── cache/                       # 体素化后的磁盘缓存
```

检查结果：

```bash
cat /root/autodl-fs/results/Proj3_trainning/report.json
find /root/autodl-fs/results/Proj3_trainning -type f \
  \( -name "*.pth" -o -name "*.log" \)
```

## 训练过程为什么第一轮慢

`software/main.py` 使用 Tonic 的 `DiskCachedDataset`。第一个 epoch 需要读取 HDF5 事件、切片并生成体素表示，同时写入磁盘缓存；后续 epoch 直接读取缓存，因此通常快很多。这是预期行为。

