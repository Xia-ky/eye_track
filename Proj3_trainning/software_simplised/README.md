# software_simplised：SEE-D 最小代码目录

这个目录从 `../software` 按白名单复制而来，目的是只保留 SEE-D 的训练、
量化、评估、测试和 MinkowskiEngine 编译链路。

## 重要约束

- 原始目录 `../software` 没有被修改。
- 本目录中的既有 `.py`、JSON、数据清单、C/C++/CUDA 源码和 checkpoint
  都是原样复制，没有重写代码。
- `SOURCE_MANIFEST.csv` 记录每个复制文件的来源、大小和 SHA-256。
- 本文件是新增的阅读说明，不属于上游源码。

## 最短阅读顺序

```text
configs/float32/SEE-D.json
→ configs/model_cfg/SEE-D_model.json
→ dataset/ThreeET_plus.py
→ dataset/custom_transforms.py
→ model/utils.py
→ model/mobilenet_submanifold.py
→ utils/metrics.py
→ utils/training_utils.py
→ main.py
→ configs/int8/SEE-D.json
→ model/HAWQ_mobilenetv2.py
→ int_inference.py
```

## 目录作用

```text
software_simplised/
├── main.py                    float32/int8训练和验证主入口
├── int_inference.py           导出整数推理所需的中间数据和结构
├── test.py                    测试集坐标预测
├── setup.py                   编译仓库内修改版MinkowskiEngine
├── requirements-esda-software.txt
├── configs/
│   ├── augment.json
│   ├── float32/SEE-D.json
│   ├── int8/SEE-D.json
│   └── model_cfg/SEE-D_model.json
├── dataset/                   3ET读取、切片、voxel、增强和数据清单
├── model/                     SEE-D及model/__init__.py导入所需模块
├── utils/                     loss、指标、训练循环、优化器和MLflow辅助
├── weights/Table2/SEE-D/      论文官方SEE-D量化checkpoint
├── MinkowskiEngine/           修改版Python接口
├── pybind/                    MinkowskiEngine Python绑定
└── src/                       MinkowskiEngine C++/CUDA源码
```

## 为什么 model 中还有其他网络

原始 `model/__init__.py` 会在导入时加载标准 MobileNet、ResNet、UNet、
稀疏模型和量化模型。因为本次不允许修改代码，所以这些文件必须保留，否则
`from model import *` 可能失败。这些文件不在 SEE-D 第一遍阅读范围内。

当前保留的正式配置只会创建：

```text
MobileNetSubmanifold
MobileNetSubmanifoldQuant
```

## 已删除的内容

```text
generate_search_cfg.py        NAS搜索配置生成
gen_NAS_json.py               NAS/硬件结构JSON生成入口
sparsity_analysis.py          稀疏度分析
visualize.py                  可视化工具
configs中的SEE-A/B/C和MobileNetV2
weights中除Table2/SEE-D外的模型
dataset/histogram.py
dataset/visualizations.py
```

## 使用前提

在已经完成原 `Proj3_trainning/setup_env.sh` 安装的 `esda` Conda 环境中，
本目录可以复用已安装的 PyTorch、Tonic 和 MinkowskiEngine。

所有命令应从本目录执行，保证配置中的相对路径能够正确解析：

```bash
cd /root/autodl-tmp/zynq_cnn/Proj3_trainning/software_simplised
```

## float32训练

下面展示直接调用 `main.py` 时的参数关系。运行目录可以按实际路径调整：

```bash
conda run --no-capture-output -n esda python main.py \
  --config_file=configs/float32/SEE-D.json \
  --mlflow_path=/root/autodl-tmp/see-d-float32/mlflow \
  --data_dir=/root/autodl-tmp/zynq_cnn/event_data \
  --data_list_dir=dataset \
  --metadata_root=/root/autodl-tmp/see-d-float32/metadata \
  --cache_root=/root/autodl-tmp/see-d-float32/cache \
  --num_epochs=100 \
  --num_workers=4 \
  --batch_size=20
```

## 论文官方checkpoint评估

官方 checkpoint 是量化模型，因此使用它同目录下的 `cfg.json`：

```bash
conda run --no-capture-output -n esda python main.py -e \
  --config_file=weights/Table2/SEE-D/cfg.json \
  --checkpoint=weights/Table2/SEE-D/model_best_p10_acc.pth \
  --shift_bit=16 \
  --bias_bit=16 \
  --mlflow_path=/root/autodl-tmp/see-d-official-eval/mlflow \
  --data_dir=/root/autodl-tmp/zynq_cnn/event_data \
  --data_list_dir=dataset \
  --metadata_root=/root/autodl-tmp/see-d-official-eval/metadata \
  --cache_root=/root/autodl-tmp/see-d-official-eval/cache \
  --num_workers=4 \
  --batch_size=20
```

## int8量化训练

当前代码在创建 `MobileNetSubmanifoldQuant` 时实际读取的是
`args.checkpoint`，所以应传入 float32 的最佳 checkpoint：

```bash
conda run --no-capture-output -n esda python main.py \
  --config_file=configs/int8/SEE-D.json \
  --checkpoint=/path/to/float32/model_best_p10_acc.pth \
  --shift_bit=16 \
  --bias_bit=16 \
  --mlflow_path=/root/autodl-tmp/see-d-int8/mlflow \
  --data_dir=/root/autodl-tmp/zynq_cnn/event_data \
  --data_list_dir=dataset \
  --metadata_root=/root/autodl-tmp/see-d-int8/metadata \
  --cache_root=/root/autodl-tmp/see-d-int8/cache \
  --num_epochs=100 \
  --num_workers=4 \
  --batch_size=20
```

上游 `UPSTREAM_README.md` 的量化示例使用 `--load`，但本地实际
`HAWQ_mobilenetv2.py` 构造函数检查的是 `--checkpoint`。本说明按当前代码
事实记录，没有修改该行为。

## 不包含什么

本目录不是一个重新设计的框架，也没有修复原项目中的重复缓存、通配符导入、
MLflow耦合或旧命令生成问题。它只是在保持代码不变的前提下减少阅读范围。

