# Proj1_testio：无卡读写测试

本目录不依赖 CUDA、PyTorch、Conda 或 h5py，可在 AutoDL 的无卡模式执行。它会：

1. 读取 `dataset/train_files.txt` 的第一条记录；
2. 从真实数据集读取对应的 HDF5 文件和标签文件；
3. 分别测试 `/root/autodl-tmp` 与 `/root/autodl-fs` 的写入、读回、SHA-256 校验和删除；
4. 将报告写入 `/root/autodl-fs/results/Proj1_testio`。

## 数据布局

先把数据上传到：

```text
/root/autodl-fs/event_data/
└── train/
    └── 1_2/
        ├── 1_2.h5
        └── label.txt
```

`1_2` 只是清单第一条记录的示例，实际以 `dataset/train_files.txt` 为准。

## 执行

```bash
cd /root/autodl-tmp/Proj1_testio
chmod +x run.sh
bash run.sh
```

成功标记：

```text
/root/autodl-fs/results/Proj1_testio/PASSED
```

失败时查看：

```bash
cat /root/autodl-fs/results/Proj1_testio/report.json
```

若只是临时在其他 Linux 目录测试脚本，可显式覆盖路径：

```bash
DATA_ROOT=/path/to/event_data \
RESULT_DIR=/tmp/esda-results/Proj1_testio \
WORK_DIR=/tmp/esda-work/Proj1_testio \
REQUIRE_AUTODL_FS_MOUNT=0 \
bash run.sh
```

