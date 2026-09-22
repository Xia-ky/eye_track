#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-esda}"
MAX_JOBS="${MAX_JOBS:-8}"

command -v conda >/dev/null 2>&1 || {
  echo "错误：没有找到 conda，请选择 AutoDL Miniconda 镜像。" >&2
  exit 2
}
[[ -f "${PROJECT_ROOT}/software/setup.py" ]] || {
  echo "错误：software/setup.py 不存在。" >&2
  exit 2
}

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.8
fi

conda install -y -n "${ENV_NAME}" \
  -c conda-forge \
  cudatoolkit-dev=11.1 \
  ninja \
  openblas

conda run -n "${ENV_NAME}" \
  python -m pip install --upgrade pip==23.3.2 setuptools==59.5.0 wheel
conda run -n "${ENV_NAME}" \
  python -m pip install \
  torch==1.8.2 \
  torchvision==0.9.2 \
  torchaudio==0.8.2 \
  --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
conda run -n "${ENV_NAME}" \
  python -m pip install \
  -r "${PROJECT_ROOT}/requirements-esda-software.txt"

(
  cd "${PROJECT_ROOT}/software"
  export MAX_JOBS
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
  conda run --no-capture-output -n "${ENV_NAME}" \
    python setup.py install --force_cuda
)

conda run -n "${ENV_NAME}" python -c \
  'import torch, MinkowskiEngine; print("torch", torch.__version__); print("CUDA", torch.version.cuda); print("GPU available", torch.cuda.is_available()); print("MinkowskiEngine", MinkowskiEngine.__version__)'
