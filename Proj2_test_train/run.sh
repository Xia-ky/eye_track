#!/usr/bin/env bash
# Version1 冒烟训练：
# 1. 从完整 3ET 数据中各取一条 train/val 记录并裁成 2 秒；
# 2. 使用官方 SEE-D 作为冻结教师，训练七块学生网络 1 个 epoch；
# 3. 将控制台日志、报告和权重复制到持久盘；
# 4. 仅当 AUTO_SHUTDOWN=1 时在结束后请求 AutoDL 关机。

set -Eeuo pipefail

# 路径均由脚本位置推导，避免依赖调用者当前工作目录。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
SOFTWARE_DIR="${VERSION_DIR}/software"

# 所有可覆盖运行参数集中在此处；默认值对应 AutoDL 标准目录。
CONDA_ENV="${CONDA_ENV:-esda}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-fs/event_data}"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/esda-version1-runs}"
RESULT_DIR="${RESULT_DIR:-/root/autodl-fs/results/Proj2_test_train_v1}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"
SHUTDOWN_COMMAND="${SHUTDOWN_COMMAND:-/usr/bin/shutdown}"
REQUIRE_AUTODL_FS_MOUNT="${REQUIRE_AUTODL_FS_MOUNT:-1}"

# 每次运行使用不可复用的 UTC 标识，结果目录不会覆盖历史实验。
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-smoke-v1"
LOCAL_RUN="${WORK_ROOT}/Proj2_test_train/${RUN_ID}"
SUBSET_ROOT="${LOCAL_RUN}/subset"
OUTPUT_ROOT="${LOCAL_RUN}/output"
LOG_FILE="${LOCAL_RUN}/console.log"
STATUS="failed"
EXIT_CODE=1

# 在产生临时数据前完成持久盘、数据和 GPU 的前置校验。
if [[ "${REQUIRE_AUTODL_FS_MOUNT}" == "1" ]]; then
    mountpoint -q /root/autodl-fs || {
        echo "错误：/root/autodl-fs 未挂载，拒绝训练。" >&2
        exit 2
    }
fi
[[ -d "${DATA_ROOT}/train" ]] || {
    echo "错误：数据目录不存在：${DATA_ROOT}/train" >&2
    exit 3
}
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null || {
    echo "错误：当前实例没有可用 NVIDIA GPU。" >&2
    exit 4
}

mkdir -p "${LOCAL_RUN}" "${RESULT_DIR}"

persist_result() {
    # 即使训练失败，也保存最后的日志，避免实例关机后丢失排错信息。
    # 先复制到隐藏临时目录，再原子改名，读取者不会看到半写入结果。
    local destination="${RESULT_DIR}/${RUN_ID}"
    local temporary="${RESULT_DIR}/.${RUN_ID}.tmp.$$"
    [[ ! -e "${destination}" && ! -e "${temporary}" ]] || return 3
    mkdir -p "${temporary}"
    [[ ! -f "${LOG_FILE}" ]] || cp "${LOG_FILE}" "${temporary}/console.log"
    [[ ! -f "${SUBSET_ROOT}/subset_report.json" ]] ||
        cp "${SUBSET_ROOT}/subset_report.json" "${temporary}/"
    [[ ! -d "${OUTPUT_ROOT}" ]] ||
        cp -a "${OUTPUT_ROOT}" "${temporary}/output"
    printf '%s\n' "${STATUS}" > "${temporary}/STATUS"
    printf '%s\n' "${EXIT_CODE}" > "${temporary}/EXIT_CODE"
    mv "${temporary}" "${destination}"
    printf '%s\n' "${RUN_ID}" > "${RESULT_DIR}/latest_run_id.tmp.$$"
    mv "${RESULT_DIR}/latest_run_id.tmp.$$" "${RESULT_DIR}/latest_run_id"
    if [[ "${STATUS}" == "passed" ]]; then
        touch "${RESULT_DIR}/PASSED"
        rm -f "${RESULT_DIR}/FAILED"
    else
        touch "${RESULT_DIR}/FAILED"
        rm -f "${RESULT_DIR}/PASSED"
    fi
}

finish() {
    # EXIT trap 保留训练命令原始退出码；持久化失败使用专用退出码 70。
    EXIT_CODE=$?
    trap - EXIT
    if [[ "${EXIT_CODE}" -eq 0 ]]; then
        STATUS="passed"
    fi
    if ! persist_result; then
        echo "错误：结果未能完整写入持久盘，取消自动关机。" >&2
        exit 70
    fi
    echo "Proj2 ${STATUS}: ${RESULT_DIR}/${RUN_ID}"
    # 关机发生在结果确认落盘之后，且只能由显式环境变量启用。
    if [[ "${AUTO_SHUTDOWN}" == "1" ]]; then
        echo "AUTO_SHUTDOWN=1，正在请求关闭 AutoDL 实例。"
        "${SHUTDOWN_COMMAND}" -h now || true
    fi
    exit "${EXIT_CODE}"
}
trap finish EXIT

{
    # 第一阶段生成真实但很小的数据；第二阶段运行与正式训练相同的软件。
    echo "run_id=${RUN_ID}"
    echo "data_root=${DATA_ROOT}"
    echo "software_dir=${SOFTWARE_DIR}"

    conda run --no-capture-output -n "${CONDA_ENV}" \
        python "${PROJECT_DIR}/make_subset.py" \
        --source-data "${DATA_ROOT}" \
        --source-lists "${SOFTWARE_DIR}/dataset_lists" \
        --output-root "${SUBSET_ROOT}" \
        --labels 200 \
        --duration-us 2000000

    cd "${SOFTWARE_DIR}"
    conda run --no-capture-output -n "${CONDA_ENV}" \
        python main.py \
        --config configs/version1_fixed50.json \
        --epochs 1 \
        --batch-size 2 \
        --workers 0 \
        --data-root "${SUBSET_ROOT}/event_data" \
        --train-list "${SUBSET_ROOT}/dataset/train_files.txt" \
        --val-list "${SUBSET_ROOT}/dataset/val_files.txt" \
        --metadata-root "${LOCAL_RUN}/metadata" \
        --cache-root "${LOCAL_RUN}/cache" \
        --output "${OUTPUT_ROOT}"

    # 主流程返回 0 仍不够：必须确认真正生成了最佳模型和部署权重。
    test -n "$(find "${OUTPUT_ROOT}" -name best_future_distance.pth -print -quit)"
    test -n "$(find "${OUTPUT_ROOT}" -name student_deploy.pth -print -quit)"
# 合并 stdout/stderr，同时保留终端可见输出和完整日志文件。
} 2>&1 | tee "${LOG_FILE}"
