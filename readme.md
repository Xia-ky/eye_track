# Version 0

本目录是 2026-07-29 归档的已跑通基线，不在这里继续加入 Version 1 的教师—学生、
Transformer 或动态 clip 改动。

```text
version0/
├── Proj1_testio/          # 数据读取和持久盘读写验证
├── Proj2_test_train/      # 训练集小样本裁剪与冒烟训练
└── Proj3_trainning/       # 完整 SEE-D 训练、训练结果和简化代码
```

Version 1 的设计见 [`../version1/DESIGN.md`](../version1/DESIGN.md)。

注意：部分历史报告保存了移动前的绝对 Windows 路径。这些路径是生成报告时的记录，
不影响报告中已经保存的指标和哈希；若重新运行验证脚本，应改用本目录下的新路径。
