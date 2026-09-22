#!/usr/bin/env bash
set -Eeuo pipefail

# ---------- 按实际上传位置修改这些变量 ----------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/root/autodl-fs/event_data}"
LIST_FILE="${LIST_FILE:-${PROJECT_ROOT}/dataset/train_files.txt}"
RESULT_DIR="${RESULT_DIR:-/root/autodl-fs/results/Proj1_testio}"
WORK_DIR="${WORK_DIR:-/root/autodl-tmp/Proj1_testio}"
REQUIRE_AUTODL_FS_MOUNT="${REQUIRE_AUTODL_FS_MOUNT:-1}"
# ----------------------------------------------

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-io"
RUN_DIR="${RESULT_DIR}/${RUN_ID}"
REPORT_TMP="${WORK_DIR}/report-${RUN_ID}.json"
FINISHED=0

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

write_failed_report() {
  local exit_code="$1"
  mkdir -p "${WORK_DIR}" "${RESULT_DIR}" "${RUN_DIR}"
  cat <<EOF | atomic_write "${REPORT_TMP}"
{"stage":"Proj1_testio","run_id":"${RUN_ID}","status":"failed","exit_code":${exit_code},"finished_at_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
  cp "${REPORT_TMP}" "${RUN_DIR}/report.json"
  cp "${REPORT_TMP}" "${RESULT_DIR}/report.json"
  replace_marker FAILED
}

on_exit() {
  local exit_code="$?"
  trap - EXIT
  if [[ "${FINISHED}" -ne 1 ]]; then
    write_failed_report "${exit_code}" || true
  fi
  exit "${exit_code}"
}
trap on_exit EXIT

if [[ "${REQUIRE_AUTODL_FS_MOUNT}" == "1" ]]; then
  mountpoint -q /root/autodl-fs || {
    echo "错误：/root/autodl-fs 没有挂载，请先在 AutoDL 挂载文件存储。" >&2
    exit 2
  }
fi

mkdir -p "${WORK_DIR}" "${RESULT_DIR}" "${RUN_DIR}"
rm -f "${RESULT_DIR}/PASSED" "${RESULT_DIR}/FAILED"

[[ -r "${LIST_FILE}" ]] || {
  echo "错误：训练清单不可读：${LIST_FILE}" >&2
  exit 3
}
RECORD_ID="$(sed -n '/[^[:space:]]/{s/\r$//;p;q;}' "${LIST_FILE}")"
[[ -n "${RECORD_ID}" ]] || {
  echo "错误：训练清单为空。" >&2
  exit 3
}

H5_FILE="${DATA_ROOT}/train/${RECORD_ID}/${RECORD_ID}.h5"
LABEL_FILE="${DATA_ROOT}/train/${RECORD_ID}/label.txt"
[[ -r "${H5_FILE}" && -r "${LABEL_FILE}" ]] || {
  echo "错误：找不到第一条训练样本：${RECORD_ID}" >&2
  echo "期望：${H5_FILE}" >&2
  echo "期望：${LABEL_FILE}" >&2
  exit 3
}

H5_BYTES="$(wc -c <"${H5_FILE}")"
LABEL_BYTES="$(wc -c <"${LABEL_FILE}")"
head -c 4096 "${H5_FILE}" >/dev/null
head -n 1 "${LABEL_FILE}" >/dev/null

check_write_read_delete() {
  local root="$1"
  local probe="${root}/.esda-io-${RUN_ID}-$$"
  local expected actual
  mkdir -p "${root}"
  printf 'esda-io-check:%s\n' "${RUN_ID}" >"${probe}"
  expected="$(sha256sum "${probe}" | awk '{print $1}')"
  actual="$(sha256sum "${probe}" | awk '{print $1}')"
  rm -f "${probe}"
  [[ "${expected}" == "${actual}" ]]
  printf '%s\n' "${actual}"
}

TMP_SHA256="$(check_write_read_delete "${WORK_DIR}")"
FS_SHA256="$(check_write_read_delete "${RESULT_DIR}")"

cat <<EOF | atomic_write "${REPORT_TMP}"
{"stage":"Proj1_testio","run_id":"${RUN_ID}","status":"passed","exit_code":0,"record_id":"${RECORD_ID}","h5_file":"${H5_FILE}","h5_bytes":${H5_BYTES},"label_file":"${LABEL_FILE}","label_bytes":${LABEL_BYTES},"temporary_sha256":"${TMP_SHA256}","persistent_sha256":"${FS_SHA256}","finished_at_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
cp "${REPORT_TMP}" "${RUN_DIR}/report.json"
cp "${REPORT_TMP}" "${RESULT_DIR}/report.json"
printf '%s\n' "${RUN_ID}" >"${RESULT_DIR}/latest_run_id"
replace_marker PASSED
FINISHED=1

echo "无卡读写测试通过。"
echo "报告：${RESULT_DIR}/report.json"

