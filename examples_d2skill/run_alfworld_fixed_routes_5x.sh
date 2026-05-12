#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ENGINE="${1:-vllm}"
if [[ $# -gt 0 ]]; then
    shift
fi

RUNS="${RUNS:-5}"
BASE_SEED="${BASE_SEED:-0}"
EVAL_DATASET="${EVAL_DATASET:-eval_in_distribution}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/fixed_route_5x_logs}"

mkdir -p "$LOG_ROOT"

run_one_model() {
    local model_label="$1"
    local script_path="$2"
    shift 2
    local model_log_dir="${LOG_ROOT}/${model_label}"
    mkdir -p "$model_log_dir"

    for ((run_idx = 0; run_idx < RUNS; run_idx++)); do
        local seed=$((BASE_SEED + run_idx))
        local log_path="${model_log_dir}/seed_${seed}.log"
        local experiment_name="fixed_${model_label}_seed${seed}"

        echo "[FixedRoute5x] model=${model_label} seed=${seed} log=${log_path}"
        bash "$script_path" "$ENGINE" \
            env.seed="$seed" \
            env.alfworld.eval_dataset="$EVAL_DATASET" \
            trainer.experiment_name="$experiment_name" \
            "$@" \
            2>&1 | tee "$log_path"
    done
}

run_one_model "deepseek" "${SCRIPT_DIR}/run_alfworld_deepseek_fixed.sh" "$@"
run_one_model "qwen3-8B" "${SCRIPT_DIR}/run_alfworld_qwen3_8b_fixed.sh" "$@"

python3 "${SCRIPT_DIR}/aggregate_fixed_route_metrics.py" "$LOG_ROOT"
