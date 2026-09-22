#!/usr/bin/env bash
# Version1 完整训练入口。Proj2 冒烟训练通过后才允许启动，防止在明显的
# 环境、数据或网络错误上浪费长时间 GPU 租赁费用。

set -Eeuo pipefail

# 从脚本位置推导共享 software，调用者可从任意目录启动。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
SOFTWARE_DIR="${VERSION_DIR}/software"

# 可覆盖参数集中定义；默认值对应经过 Proj2 验证的 AutoDL 布局。
CONDA_ENV="${CONDA_ENV:-esda}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-fs/event_data}"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/esda-version1-runs}"
RESULT_DIR="${RESULT_DIR:-/root/autodl-fs/results/Proj3_trainning_v1}"
SMOKE_MARKER="${SMOKE_MARKER:-/root/autodl-fs/results/Proj2_test_train_v1/PASSED}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"
SHUTDOWN_COMMAND="${SHUTDOWN_COMMAND:-/usr/bin/shutdown}"
REQUIRE_AUTODL_FS_MOUNT="${REQUIRE_AUTODL_FS_MOUNT:-1}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-20}"
WORKERS="${WORKERS:-4}"
# 不直接信任外部 OMP_NUM_THREADS：AutoDL 镜像可能继承空格或其他非法值，
# 从而触发 libgomp 警告。使用单独且经过校验的变量显式覆盖它。
OPENMP_THREADS="${OPENMP_THREADS:-1}"

# 独立运行目录隔离缓存、日志和检查点，禁止覆盖既有实验。
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-full-v1"
LOCAL_RUN="${WORK_ROOT}/Proj3_trainning/${RUN_ID}"
OUTPUT_ROOT="${LOCAL_RUN}/output"
LOG_FILE="${LOCAL_RUN}/console.log"
STATUS="failed"
EXIT_CODE=1

# 在启动昂贵的 GPU 任务前拒绝空值、负数和非法线程配置。
[[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "错误：EPOCHS 必须是正整数。" >&2
    exit 2
}
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "错误：BATCH_SIZE 必须是正整数。" >&2
    exit 2
}
[[ "${WORKERS}" =~ ^[0-9]+$ ]] || {
    echo "错误：WORKERS 必须是非负整数。" >&2
    exit 2
}
[[ "${OPENMP_THREADS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "错误：OPENMP_THREADS 必须是正整数。" >&2
    exit 2
}
export OMP_NUM_THREADS="${OPENMP_THREADS}"

# 持久盘、冒烟通过标记、数据和 GPU 是正式训练的硬前置条件。
if [[ "${REQUIRE_AUTODL_FS_MOUNT}" == "1" ]]; then
    mountpoint -q /root/autodl-fs || {
        echo "错误：/root/autodl-fs 未挂载，拒绝训练。" >&2
        exit 2
    }
fi
test -f "${SMOKE_MARKER}" || {
    echo "未找到 ${SMOKE_MARKER}；请先成功运行 Proj2_test_train。"
    exit 2
}
[[ -d "${DATA_ROOT}/train" ]] || {
    echo "错误：数据目录不存在：${DATA_ROOT}/train" >&2
    exit 3
}
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null || {
    echo "错误：当前实例没有可用 NVIDIA GPU。" >&2
    exit 4
}
# 正式训练必须使用已经通过 10 epoch 冒烟验证的稳定输出头初始化。
grep -q "nn.init.zeros_(self.future_delta_head.weight)" \
    "${SOFTWARE_DIR}/see_v1/student.py" || {
    echo "错误：共享 software 仍是旧版随机未来头，请先上传修复后的 student.py。" >&2
    exit 5
}

mkdir -p "${LOCAL_RUN}" "${RESULT_DIR}"

persist_result() {
    # 结果先写隐藏临时目录，再原子发布，防止自动关机留下半份模型。
    local destination="${RESULT_DIR}/${RUN_ID}"
    local temporary="${RESULT_DIR}/.${RUN_ID}.tmp.$$"
    [[ ! -e "${destination}" && ! -e "${temporary}" ]] || return 3
    mkdir -p "${temporary}"
    [[ ! -f "${LOG_FILE}" ]] || cp "${LOG_FILE}" "${temporary}/console.log"
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
    # 捕获主流程退出码；无论成功失败都尝试持久化日志和状态。
    EXIT_CODE=$?
    trap - EXIT
    if [[ "${EXIT_CODE}" -eq 0 ]]; then
        STATUS="passed"
    fi
    if ! persist_result; then
        echo "错误：结果未能完整写入持久盘，取消自动关机。" >&2
        exit 70
    fi
    echo "Proj3 ${STATUS}: ${RESULT_DIR}/${RUN_ID}"
    # 仅显式启用时关机，并确保该分支位于持久化成功之后。
    if [[ "${AUTO_SHUTDOWN}" == "1" ]]; then
        echo "AUTO_SHUTDOWN=1，正在请求关闭 AutoDL 实例。"
        "${SHUTDOWN_COMMAND}" -h now || true
    fi
    exit "${EXIT_CODE}"
}
trap finish EXIT

{
    # 正式训练使用完整 split 清单，缓存留在临时盘，输出稍后持久化。
    echo "run_id=${RUN_ID}"
    echo "epochs=${EPOCHS} batch_size=${BATCH_SIZE} workers=${WORKERS}"
    echo "openmp_threads=${OPENMP_THREADS}"
    echo "data_root=${DATA_ROOT}"

    cd "${SOFTWARE_DIR}"
    conda run --no-capture-output -n "${CONDA_ENV}" \
        python main.py \
        --config configs/version1_fixed50.json \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --workers "${WORKERS}" \
        --data-root "${DATA_ROOT}" \
        --train-list "${SOFTWARE_DIR}/dataset_lists/train_files.txt" \
        --val-list "${SOFTWARE_DIR}/dataset_lists/val_files.txt" \
        --metadata-root "${LOCAL_RUN}/metadata" \
        --cache-root "${LOCAL_RUN}/cache" \
        --output "${OUTPUT_ROOT}"

    # 成功退出前再次验证关键模型产物，避免空运行被标记为通过。
    test -n "$(find "${OUTPUT_ROOT}" -name best_future_distance.pth -print -quit)"
    test -n "$(find "${OUTPUT_ROOT}" -name student_deploy.pth -print -quit)"
# 合并标准输出和错误输出，使后台运行日志包含完整诊断信息。
} 2>&1 | tee "${LOG_FILE}"
