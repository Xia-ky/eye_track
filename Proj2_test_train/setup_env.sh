#!/usr/bin/env bash
# 仅在新实例尚无 esda 环境时执行。已经通过 Version 0 冒烟训练的环境无需重装。

set -Eeuo pipefail

# 环境安装始终以同级 software 为源码，支持项目目录整体移动。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOFTWARE_DIR="$(cd "${PROJECT_DIR}/../software" && pwd)"
CONDA_ENV="${CONDA_ENV:-esda}"
MAX_JOBS="${MAX_JOBS:-8}"

# 在执行任何安装前验证基础镜像和本地 MinkowskiEngine 源码。
command -v conda >/dev/null 2>&1 || {
    echo "错误：未找到 conda，请选择 AutoDL Miniconda 镜像。" >&2
    exit 2
}
[[ -f "${SOFTWARE_DIR}/setup.py" ]] || {
    echo "错误：MinkowskiEngine setup.py 不存在。" >&2
    exit 2
}

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# 已存在的环境原地补齐依赖；新实例才创建 Python 3.8 环境。
if ! conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV}"; then
    conda create -y -n "${CONDA_ENV}" python=3.8
fi

# 编译依赖与 CUDA 11.1 保持一致，匹配项目使用的 PyTorch 1.8.2。
conda install -y -n "${CONDA_ENV}" \
    -c conda-forge \
    cudatoolkit-dev=11.1 \
    ninja \
    openblas

# 固定 setuptools，避免旧版 MinkowskiEngine 经 numpy.distutils 导入
# distutils.msvccompiler 时被新版 setuptools 移除。
conda run -n "${CONDA_ENV}" \
    python -m pip install --upgrade \
    pip==23.3.2 setuptools==59.5.0 wheel
conda run -n "${CONDA_ENV}" \
    python -m pip install \
    torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 \
    --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
conda run -n "${CONDA_ENV}" \
    python -m pip install -r "${PROJECT_DIR}/requirements-version1.txt"

# MinkowskiEngine 必须在目标 conda 环境内强制启用 CUDA 编译。
(
    cd "${SOFTWARE_DIR}"
    export MAX_JOBS
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
    conda run --no-capture-output -n "${CONDA_ENV}" \
        python setup.py install --force_cuda
)

# 最后执行一次导入与 GPU 探测，安装成功但动态库不可用时立即失败。
conda run --no-capture-output -n "${CONDA_ENV}" python -c \
    'import h5py, torch, MinkowskiEngine; print("torch", torch.__version__, torch.version.cuda); print("GPU", torch.cuda.is_available()); print("MinkowskiEngine OK")'
