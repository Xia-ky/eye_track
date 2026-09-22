#!/usr/bin/env bash
# 在新的 AutoDL Miniconda 镜像中创建 ESDA 训练环境并编译 MinkowskiEngine。
# 同一实例已经成功执行过一次后通常无需重复运行。
set -Eeuo pipefail

# 使用脚本所在位置定位 software/，避免依赖调用者当前所在目录。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-esda}"
# MinkowskiEngine 的 C++/CUDA 扩展并行编译任务数；内存不足时可调小。
MAX_JOBS="${MAX_JOBS:-8}"

# 在下载任何依赖前先确认镜像和源码目录正确。
command -v conda >/dev/null 2>&1 || {
  echo "错误：没有找到 conda，请选择 AutoDL Miniconda 镜像。" >&2
  exit 2
}
[[ -f "${PROJECT_ROOT}/software/setup.py" ]] || {
  echo "错误：software/setup.py 不存在。" >&2
  exit 2
}

CONDA_BASE="$(conda info --base)"
# 非交互脚本需要加载 conda.sh，才能可靠访问 Conda 环境管理功能。
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# 旧版 ESDA/MinkowskiEngine 与 Python 3.8、PyTorch 1.8 组合已经过冒烟测试。
if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.8
fi

# cudatoolkit-dev 提供编译 CUDA 扩展所需工具；ninja 加速编译；OpenBLAS
# 是 MinkowskiEngine CPU 侧线性代数依赖。
conda install -y -n "${ENV_NAME}" \
  -c conda-forge \
  cudatoolkit-dev=11.1 \
  ninja \
  openblas

# 固定 pip/setuptools 是兼容旧版 setup.py 与 numpy.distutils 的关键。
# 较新的 setuptools 可能导致 distutils.msvccompiler 缺失。
conda run -n "${ENV_NAME}" \
  python -m pip install --upgrade pip==23.3.2 setuptools==59.5.0 wheel
conda run -n "${ENV_NAME}" \
  python -m pip install \
  torch==1.8.2 \
  torchvision==0.9.2 \
  torchaudio==0.8.2 \
  --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
# 安装 ESDA 的 Python 运行依赖，版本锁定原因见 requirements 文件。
conda run -n "${ENV_NAME}" \
  python -m pip install \
  -r "${PROJECT_ROOT}/requirements-esda-software.txt"

# software/setup.py 编译仓库内附带的 MinkowskiEngine C++/CUDA 扩展。
# RTX 3060 Ti/3090 都是 Ampere，CUDA Compute Capability 为 8.6。
(
  cd "${PROJECT_ROOT}/software"
  export MAX_JOBS
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
  conda run --no-capture-output -n "${ENV_NAME}" \
    python setup.py install --force_cuda
)

# 安装后立即做最小导入检查，避免到正式训练才发现 CUDA 扩展不可用。
conda run -n "${ENV_NAME}" python -c \
  'import torch, MinkowskiEngine; print("torch", torch.__version__); print("CUDA", torch.version.cuda); print("GPU available", torch.cuda.is_available()); print("MinkowskiEngine", MinkowskiEngine.__version__)'
