#!/usr/bin/env bash
set -Eeuo pipefail

# Parameter points run sequentially.
# Each parameter point uses MPI internally with MPI_RANKS processes.

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly RUN_DIR="${SCRIPT_DIR}/DaMaSCUS-CRUST/bin"
readonly EXECUTABLE="${RUN_DIR}/DaMaSCUS-CRUST"
readonly CONFIG="${RUN_DIR}/config.cfg"
readonly MPI_RANKS="${MPI_RANKS:-16}"
readonly LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
readonly TOTAL_TASKS=25

if ! [[ "${MPI_RANKS}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'ERROR: MPI_RANKS must be a positive integer (got %q).\n' "${MPI_RANKS}" >&2
    exit 2
fi

if [[ ! -x "${EXECUTABLE}" ]]; then
    printf 'ERROR: executable is missing or not executable: %s\n' "${EXECUTABLE}" >&2
    exit 2
fi

if [[ ! -r "${CONFIG}" ]]; then
    printf 'ERROR: config file is not readable: %s\n' "${CONFIG}" >&2
    exit 2
fi

if ! command -v mpirun >/dev/null 2>&1; then
    printf 'ERROR: mpirun not found in PATH.\n' >&2
    exit 2
fi

mkdir -p -- "${LOG_DIR}"
cd -- "${RUN_DIR}"

declare -i submitted=0
declare -i failures=0

run_task() {
    local task_id="$1"
    local mass="$2"
    local sigma="$3"
    local cross_section_type="$4"
    local v_min="$5"
    local log_file
    local status

    printf -v log_file \
        '%s/task_%04d_mDM=%s_sigma_%s=%s_vMin=%s.log' \
        "${LOG_DIR}" "${task_id}" "${mass}" \
        "${cross_section_type}" "${sigma}" "${v_min}"

    printf '[%d/%d] START mDM=%s sigma_%s=%s vMin=%s\n' \
        "${task_id}" "${TOTAL_TASKS}" \
        "${mass}" "${cross_section_type}" "${sigma}" "${v_min}"

    if mpirun -n "${MPI_RANKS}" \
        "${EXECUTABLE}" "${CONFIG}" \
        "${mass}" "${sigma}" "${cross_section_type}" "${v_min}" \
        >"${log_file}" 2>&1; then

        printf '[%d/%d] DONE  mDM=%s sigma_%s=%s vMin=%s\n' \
            "${task_id}" "${TOTAL_TASKS}" \
            "${mass}" "${cross_section_type}" "${sigma}" "${v_min}"
        return 0
    else
        status=$?
        printf '[%d/%d] FAIL  mDM=%s sigma_%s=%s vMin=%s (exit=%d, log=%s)\n' \
            "${task_id}" "${TOTAL_TASKS}" \
            "${mass}" "${cross_section_type}" "${sigma}" "${v_min}" \
            "${status}" "${log_file}" >&2
        return "${status}"
    fi
}

queue_task() {
    submitted+=1

    if ! run_task "${submitted}" "$1" "$2" "$3" "$4"; then
        failures+=1
    fi
}
queue_task 1 2.15e-29 n 0.0
queue_task 1 2.15e-28 n 0.0
queue_task 1 2.15e-27 n 0.0
queue_task 1 2.15e-26 n 0.0
queue_task 1 2.15e-25 n 0.0
queue_task 10 3.51e-31 n 0.0
queue_task 10 3.51e-30 n 0.0
queue_task 10 3.51e-29 n 0.0
queue_task 10 3.51e-28 n 0.0
queue_task 10 3.51e-27 n 0.0
queue_task 100 8.49e-32 n 0.0
queue_task 100 8.49e-31 n 0.0
queue_task 100 8.49e-30 n 0.0
queue_task 100 8.49e-29 n 0.0
queue_task 100 8.49e-28 n 0.0
queue_task 1000 4.62e-31 n 0.0
queue_task 1000 4.62e-30 n 0.0
queue_task 1000 4.62e-29 n 0.0
queue_task 1000 4.62e-28 n 0.0
queue_task 1000 4.62e-27 n 0.0
queue_task 10000 4.96e-30 n 0.0
queue_task 10000 4.96e-29 n 0.0
queue_task 10000 4.96e-28 n 0.0
queue_task 10000 4.96e-27 n 0.0
queue_task 10000 4.96e-26 n 0.0

if ((failures > 0)); then
    printf 'Finished %d tasks with %d failure(s). See logs in %s\n' \
        "${submitted}" "${failures}" "${LOG_DIR}" >&2
    exit 1
fi

printf 'Finished all %d tasks successfully. Logs: %s\n' \
    "${submitted}" "${LOG_DIR}"
