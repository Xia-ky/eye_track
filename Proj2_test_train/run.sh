#!/usr/bin/env bash
set -Eeuo pipefail

# ---------- 通常只需检查这一段 ----------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-SEE-D}"
ENV_NAME="${ENV_NAME:-esda}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-fs/event_data}"
LIST_ROOT="${LIST_ROOT:-${PROJECT_ROOT}/software/dataset}"
RESULT_DIR="${RESULT_DIR:-/root/autodl-fs/results/Proj2_test_train}"
PREVIOUS_MARKER="${PREVIOUS_MARKER:-/root/autodl-fs/results/Proj1_testio/PASSED}"
WORK_BASE="${WORK_BASE:-/root/autodl-tmp/esda-manual-runs}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"
REQUIRE_AUTODL_FS_MOUNT="${REQUIRE_AUTODL_FS_MOUNT:-1}"
# ----------------------------------------

case "${MODEL}" in
  SEE-A|SEE-B|SEE-C|SEE-D|MobileNetV2) ;;
  *)
    echo "错误：不支持的模型：${MODEL}" >&2
    exit 2
    ;;
esac

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-smoke-${MODEL}"
LOCAL_RUN="${WORK_BASE}/Proj2_test_train/${RUN_ID}"
RUN_DIR="${RESULT_DIR}/${RUN_ID}"
REPORT="${LOCAL_RUN}/report.json"
FINALIZED=0

atomic_write() {
  local destination="$1"
  local temporary="${destination}.tmp.$$"
  cat >"${temporary}"
  mv "${temporary}" "${destination}"
}

replace_marker() {
  local marker="$1"
  local temporary="${RESULT_DIR}/.${marker}.tmp.$$"
  rm -f "${RESULT_DIR}/PASSED" "${RESULT_DIR}/FAILED"
  : >"${temporary}"
  mv "${temporary}" "${RESULT_DIR}/${marker}"
}

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

finish_run() {
  local status="$1"
  local exit_code="$2"
  mkdir -p "${LOCAL_RUN}"
  cat <<EOF | atomic_write "${REPORT}"
{"stage":"Proj2_test_train","run_id":"${RUN_ID}","model":"${MODEL}","status":"${status}","exit_code":${exit_code},"finished_at_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
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

if [[ "${REQUIRE_AUTODL_FS_MOUNT}" == "1" ]]; then
  mountpoint -q /root/autodl-fs || {
    echo "错误：/root/autodl-fs 没有挂载。" >&2
    exit 2
  }
fi
[[ -f "${PREVIOUS_MARKER}" ]] || {
  echo "错误：必须先通过 Proj1_testio：${PREVIOUS_MARKER}" >&2
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

mkdir -p "${LOCAL_RUN}" "${RESULT_DIR}"
rm -f "${RESULT_DIR}/PASSED" "${RESULT_DIR}/FAILED"

SUBSET_ROOT="${LOCAL_RUN}/subset"
RUNTIME_CONFIG="${LOCAL_RUN}/smoke_runtime_config.json"
BASE_CONFIG="${PROJECT_ROOT}/software/configs/float32/${MODEL}.json"

conda run --no-capture-output -n "${ENV_NAME}" \
  python "${PROJECT_ROOT}/make_subset.py" \
  --source-data "${DATA_ROOT}" \
  --source-lists "${LIST_ROOT}" \
  --output-root "${SUBSET_ROOT}" \
  --base-config "${BASE_CONFIG}" \
  --smoke-overrides "${PROJECT_ROOT}/smoke_config.json" \
  --runtime-config "${RUNTIME_CONFIG}"

mkdir -p "${LOCAL_RUN}/metadata" "${LOCAL_RUN}/cache" "${LOCAL_RUN}/mlflow"
cd "${PROJECT_ROOT}/software"

set +e
conda run --no-capture-output -n "${ENV_NAME}" python main.py \
  --config_file="${RUNTIME_CONFIG}" \
  --mlflow_path="${LOCAL_RUN}/mlflow" \
  --num_epochs=1 \
  --num_workers=0 \
  --batch_size=2 \
  --data_dir="${SUBSET_ROOT}/event_data" \
  --data_list_dir="${SUBSET_ROOT}/dataset" \
  --metadata_root="${LOCAL_RUN}/metadata" \
  --cache_root="${LOCAL_RUN}/cache" 2>&1 |
  tee "${LOCAL_RUN}/training.log"
TRAIN_EXIT="${PIPESTATUS[0]}"
set -e

if [[ "${TRAIN_EXIT}" -ne 0 ]]; then
  finish_run failed "${TRAIN_EXIT}"
  exit "${TRAIN_EXIT}"
fi
find "${LOCAL_RUN}/mlflow" -type f -name '*.pth' -print -quit |
  grep -q . || {
    echo "错误：训练完成但没有生成 .pth 检查点。" >&2
    exit 6
  }

finish_run passed 0
echo "冒烟训练通过：${RESULT_DIR}"
