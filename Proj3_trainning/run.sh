#!/usr/bin/env bash
# Proj3 的完整训练入口。
#
# 这个脚本不实现神经网络本身；它负责把 AutoDL 上的路径和参数传给
# software/main.py，并保证日志、检查点和最终状态被复制到持久化数据盘。
# 用法示例：
#   AUTO_SHUTDOWN=0 MODEL=SEE-D EPOCHS=1 bash run.sh
set -Eeuo pipefail
# -E：ERR trap 可被函数继承；-e：未处理的失败立即退出；
# -u：读取未定义变量时报错；pipefail：管道中任一命令失败即算失败。

# ---------- 正式训练参数（均可用同名环境变量覆盖） ----------
# BASH_SOURCE[0] 是当前脚本本身；由它计算根目录后，从任何工作目录启动都可以。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 选择 float32 配置文件 software/configs/float32/<MODEL>.json。
MODEL="${MODEL:-SEE-D}"
EPOCHS="${EPOCHS:-100}"
# WORKERS 是 PyTorch DataLoader 子进程数，不是 GPU 数量。
WORKERS="${WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-20}"
ENV_NAME="${ENV_NAME:-esda}"
# DATA_ROOT 指向完整事件数据；LIST_ROOT 指向训练/验证记录清单。
DATA_ROOT="${DATA_ROOT:-/root/autodl-fs/event_data}"
LIST_ROOT="${LIST_ROOT:-${PROJECT_ROOT}/software/dataset}"
# RESULT_DIR 位于持久盘；实例关机或释放后仍保留。
RESULT_DIR="${RESULT_DIR:-/root/autodl-fs/results/Proj3_trainning}"
# 正式训练前要求冒烟测试成功，避免用昂贵 GPU 时间发现基础环境错误。
PREVIOUS_MARKER="${PREVIOUS_MARKER:-/root/autodl-fs/results/Proj2_test_train/PASSED}"
# 训练中的大量缓存先写 AutoDL 高速临时盘，结束后再整体复制到持久盘。
WORK_BASE="${WORK_BASE:-/root/autodl-tmp/esda-manual-runs}"
# 1 表示训练成功或失败后都请求关机；调试时务必设为 0。
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"
# 防止 /root/autodl-fs 未挂载时误把结果写入容器自身文件系统。
REQUIRE_AUTODL_FS_MOUNT="${REQUIRE_AUTODL_FS_MOUNT:-1}"
# -------------------------------------------------

# 尽早验证用户输入。这里失败时还没有开始训练，不会产生昂贵计算。
case "${MODEL}" in
  SEE-A|SEE-B|SEE-C|SEE-D|MobileNetV2) ;;
  *)
    echo "错误：不支持的模型：${MODEL}" >&2
    exit 2
    ;;
esac
[[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "错误：EPOCHS 必须为正整数。" >&2
  exit 2
}
[[ "${WORKERS}" =~ ^[0-9]+$ ]] || {
  echo "错误：WORKERS 必须为非负整数。" >&2
  exit 2
}
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "错误：BATCH_SIZE 必须为正整数。" >&2
  exit 2
}

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-full-${MODEL}"
# LOCAL_RUN：训练期间的高速工作目录；RUN_DIR：训练结束后的持久化副本。
LOCAL_RUN="${WORK_BASE}/Proj3_trainning/${RUN_ID}"
RUN_DIR="${RESULT_DIR}/${RUN_ID}"
REPORT="${LOCAL_RUN}/report.json"
# 防止正常完成后，EXIT trap 再次生成一次失败报告。
FINALIZED=0

# 先写同目录临时文件，再用 mv 原子替换。
# 这样即使进程在写入中途退出，也不会留下半截 JSON 或状态文件。
atomic_write() {
  local destination="$1"
  local temporary="${destination}.tmp.$$"
  cat >"${temporary}"
  mv "${temporary}" "${destination}"
}

# PASSED 与 FAILED 互斥：每次运行结束只保留一个状态标记。
replace_marker() {
  local marker="$1"
  local temporary="${RESULT_DIR}/.${marker}.tmp.$$"
  rm -f "${RESULT_DIR}/PASSED" "${RESULT_DIR}/FAILED"
  : >"${temporary}"
  mv "${temporary}" "${RESULT_DIR}/${marker}"
}

# 将整个本地运行目录复制成隐藏临时目录，复制完成后再改成正式目录名。
# 消费结果的程序因此不会看到只复制了一部分的运行目录。
persist_run() {
  local temporary="${RESULT_DIR}/.${RUN_ID}.tmp.$$"
  [[ ! -e "${temporary}" && ! -e "${RUN_DIR}" ]] || {
    echo "错误：结果目录已存在：${RUN_DIR}" >&2
    return 3
  }
  mkdir -p "${RESULT_DIR}"
  cp -a "${LOCAL_RUN}" "${temporary}"
  mv "${temporary}" "${RUN_DIR}"
  cp "${REPORT}" "${RESULT_DIR}/report.json.tmp.$$"
  mv "${RESULT_DIR}/report.json.tmp.$$" "${RESULT_DIR}/report.json"
  printf '%s\n' "${RUN_ID}" >"${RESULT_DIR}/latest_run_id.tmp.$$"
  mv "${RESULT_DIR}/latest_run_id.tmp.$$" "${RESULT_DIR}/latest_run_id"
}

# 生成机器可读报告、持久化本次运行，并更新顶层状态标记。
finish_run() {
  local status="$1"
  local exit_code="$2"
  mkdir -p "${LOCAL_RUN}"
  cat <<EOF | atomic_write "${REPORT}"
{"stage":"Proj3_trainning","run_id":"${RUN_ID}","model":"${MODEL}","epochs":${EPOCHS},"workers":${WORKERS},"batch_size":${BATCH_SIZE},"status":"${status}","exit_code":${exit_code},"finished_at_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
  persist_run
  if [[ "${status}" == "passed" ]]; then
    replace_marker PASSED
  else
    replace_marker FAILED
  fi
  FINALIZED=1
}

shutdown_instance() {
  [[ "${AUTO_SHUTDOWN}" == "1" ]] || return 0
  echo "训练阶段结束，正在请求 AutoDL 关机。"
  "${SHUTDOWN_COMMAND:-/usr/bin/shutdown}" || {
    echo "警告：自动关机失败，请立即在 AutoDL 网页手动关机。" >&2
  }
}

# EXIT trap 是最后一道保险：
# - 正常路径已调用 finish_run 时，只负责关机；
# - 任意未预期错误导致退出时，补写 failed 报告后再关机。
on_exit() {
  local exit_code="$?"
  trap - EXIT
  if [[ "${FINALIZED}" -ne 1 ]]; then
    finish_run failed "${exit_code}" || true
  fi
  shutdown_instance
  exit "${exit_code}"
}
trap on_exit EXIT

# 以下是训练前置检查。任何一项失败都会通过 EXIT trap 留下失败报告。
if [[ "${REQUIRE_AUTODL_FS_MOUNT}" == "1" ]]; then
  mountpoint -q /root/autodl-fs || {
    echo "错误：/root/autodl-fs 没有挂载。" >&2
    exit 2
  }
fi
[[ -f "${PREVIOUS_MARKER}" ]] || {
  echo "错误：必须先通过 Proj2_test_train：${PREVIOUS_MARKER}" >&2
  exit 5
}
[[ -d "${DATA_ROOT}/train" ]] || {
  echo "错误：数据目录不存在：${DATA_ROOT}/train" >&2
  exit 3
}
[[ -f "${LIST_ROOT}/train_files.txt" && -f "${LIST_ROOT}/val_files.txt" ]] || {
  echo "错误：训练/验证清单不存在：${LIST_ROOT}" >&2
  exit 3
}
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null || {
  echo "错误：当前实例没有可用的 NVIDIA GPU。" >&2
  exit 2
}

mkdir -p "${LOCAL_RUN}/metadata" "${LOCAL_RUN}/cache" \
  "${LOCAL_RUN}/mlflow" "${RESULT_DIR}"
rm -f "${RESULT_DIR}/PASSED" "${RESULT_DIR}/FAILED"

CONFIG_FILE="${PROJECT_ROOT}/software/configs/float32/${MODEL}.json"
[[ -f "${CONFIG_FILE}" ]] || {
  echo "错误：模型配置不存在：${CONFIG_FILE}" >&2
  exit 3
}

# main.py 的输出同时显示在终端并由 tee 写入 training.log。
# 临时关闭 `set -e`，是为了在训练失败后先取得 Python 的真实退出码，
# 再调用 finish_run 保存失败原因；PIPESTATUS[0] 对应管道左侧的 conda 命令。
cd "${PROJECT_ROOT}/software"
set +e
conda run --no-capture-output -n "${ENV_NAME}" python main.py \
  --config_file="${CONFIG_FILE}" \
  --mlflow_path="${LOCAL_RUN}/mlflow" \
  --num_epochs="${EPOCHS}" \
  --num_workers="${WORKERS}" \
  --batch_size="${BATCH_SIZE}" \
  --data_dir="${DATA_ROOT}" \
  --data_list_dir="${LIST_ROOT}" \
  --metadata_root="${LOCAL_RUN}/metadata" \
  --cache_root="${LOCAL_RUN}/cache" 2>&1 |
  tee "${LOCAL_RUN}/training.log"
TRAIN_EXIT="${PIPESTATUS[0]}"
set -e

# 训练失败：保留 Python 退出码，生成 FAILED，再交给 EXIT trap 请求关机。
if [[ "${TRAIN_EXIT}" -ne 0 ]]; then
  finish_run failed "${TRAIN_EXIT}"
  exit "${TRAIN_EXIT}"
fi
# 退出码为 0 还不够；至少存在一个 .pth 才认为训练产物完整。
find "${LOCAL_RUN}/mlflow" -type f -name '*.pth' -print -quit |
  grep -q . || {
    echo "错误：训练完成但没有生成 .pth 检查点。" >&2
    exit 6
  }

# 所有检查均通过后写 PASSED。随后脚本自然退出，EXIT trap 执行自动关机。
finish_run passed 0
echo "完整训练通过：${RESULT_DIR}"
