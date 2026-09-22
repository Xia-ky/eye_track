# Proj3_trainning

该入口使用完整官方 train/val 列表，并从头训练修复初始化后的学生网络。它不会加载
Proj2 的冒烟权重；Proj2 的 `PASSED` 文件只作为环境和代码链路已经通过的门禁。

Proj3 与 Proj2 共用同级目录的 `software`：

```text
/root/autodl-tmp/zynq_cnn/
├── Proj2_test_train/
├── Proj3_trainning/
├── software/
└── event_data/
```

脚本启动前会确认：

- `/root/autodl-fs` 已正确挂载；
- Proj2 已生成 `PASSED`；
- NVIDIA GPU 可见；
- `student.py` 包含未来位移头零初始化修复；
- epoch、batch、worker 和 OpenMP 线程数有效。

后台运行 100 epoch：

```bash
cd /root/autodl-tmp/zynq_cnn/Proj3_trainning
mkdir -p /root/autodl-fs/results/Proj3_trainning_v1

RUN_LOG=/root/autodl-fs/results/Proj3_trainning_v1/launcher-$(date -u +%Y%m%dT%H%M%SZ).log

nohup env \
OPENMP_THREADS=1 \
DATA_ROOT=/root/autodl-tmp/zynq_cnn/event_data \
AUTO_SHUTDOWN=1 \
EPOCHS=100 \
BATCH_SIZE=20 \
WORKERS=4 \
bash run.sh > "$RUN_LOG" 2>&1 &

echo "PID=$!"
echo "日志=$RUN_LOG"
```

结果位于：

```text
/root/autodl-fs/results/Proj3_trainning_v1/<run_id>/
```

训练中的当前指标以 `current_*` 开头，20 ms 预测指标以 `future_*` 开头。最佳模型
首先按 `val_future_distance` 最小选择，相同时再比较 `val_current_distance`。
