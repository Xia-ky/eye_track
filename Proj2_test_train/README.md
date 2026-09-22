# Proj2_test_train

该目录从官方训练/验证列表各取第一条记录，分别裁成 2 秒真实样本，然后用
`../software` 的完整 Version 1 流程训练 1 个 epoch。

若当前实例还没有已跑通的 `esda` 环境，先执行：

```bash
bash setup_env.sh
```

已有 Version 0 的 `esda` 环境时不要重复安装，直接运行：

```bash
cd /root/autodl-tmp/zynq_cnn/code/version1/Proj2_test_train
DATA_ROOT=/root/autodl-fs/event_data AUTO_SHUTDOWN=0 bash run.sh
```

通过条件不是“进程没有报错”，而是同时找到：

- `best_future_distance.pth`
- `student_deploy.pth`
- `/root/autodl-fs/results/Proj2_test_train_v1/PASSED`

结果和失败日志都保存在 `/root/autodl-fs/results/Proj2_test_train_v1/`。
首次测试建议 `AUTO_SHUTDOWN=0`；确认自动关机功能时再设为 `1`。
