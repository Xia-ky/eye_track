# Proj2_test_train：真实数据小规模训练

本目录直接执行 Shell 和 Python 文件，不需要本地 CLI。它从真实数据中确定性选择：

- `train_files.txt` 第一条训练记录；
- `val_files.txt` 第一条验证记录；
- 每条记录前 2 秒事件（半开区间）；
- 每条记录前 200 行标签。

随后以 `SEE-D`、`batch_size=2`、`workers=0`、`epochs=1` 完成训练、反向传播、验证和检查点写入。

## 前提

1. `Proj1_testio` 已成功，存在：

   ```text
   /root/autodl-fs/results/Proj1_testio/PASSED
   ```

2. 已切换到带 NVIDIA GPU 的实例。
3. 数据仍位于 `/root/autodl-fs/event_data`。

## 第一次安装环境

推荐镜像：Miniconda、Python 3.8、Ubuntu 20.04、CUDA 11.3（cudagl）。

```bash
cd /root/autodl-tmp/Proj2_test_train
chmod +x setup_env.sh run.sh
bash setup_env.sh
```

安装脚本创建 `esda` 环境，安装 PyTorch 1.8.2 + cu111，并编译项目自带的 MinkowskiEngine。

## 执行冒烟训练

脚本默认在结束后自动关机：

```bash
cd /root/autodl-tmp/Proj2_test_train
bash run.sh
```

若正在排错、不希望自动关机：

```bash
AUTO_SHUTDOWN=0 bash run.sh
```

成功标记与结果：

```text
/root/autodl-fs/results/Proj2_test_train/PASSED
/root/autodl-fs/results/Proj2_test_train/report.json
/root/autodl-fs/results/Proj2_test_train/<run_id>/training.log
```

只有看到 `PASSED` 和至少一个 `.pth` 文件后，才进入完整训练。

