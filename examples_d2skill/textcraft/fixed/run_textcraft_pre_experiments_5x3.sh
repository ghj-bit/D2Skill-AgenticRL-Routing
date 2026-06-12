#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

ENGINE_ARGS=()
if [[ $# -gt 0 && ( "$1" == "vllm" || "$1" == "hf" || "$1" == "sglang" || "$1" == "ray" ) ]]; then
    ENGINE_ARGS=("$1")
    shift
fi

RUNS="${RUNS:-3}"
BASE_SEED="${BASE_SEED:-0}"
PREEXP_ROOT="${PREEXP_ROOT:-${PROJECT_DIR}/checkpoints/verl_agent_textcraft_fixed_route}"
SKILLS_JSON="${TEXTCRAFT_FIXED_SKILLS_JSON_PATH:-${SCRIPT_DIR}/textcraft_cost_planning_skills.json}"

run_experiment() {
    local name="$1"
    local script_path="$2"
    local skill_ids="$3"
    shift 3

    local model_log_dir="${PREEXP_ROOT}/${name}"
    local summary_json="${model_log_dir}/fixed_route_metric_summary.json"

    echo "============================================================"
    echo "[TextCraftPreExp] start ${name}"
    echo "[TextCraftPreExp] runs=${RUNS} base_seed=${BASE_SEED} skills='${skill_ids}' model_log_dir=${model_log_dir}"
    echo "============================================================"

    RUNS="${RUNS}" \
    BASE_SEED="${BASE_SEED}" \
    LOG_ROOT="${PREEXP_ROOT}" \
    MODEL_LOG_DIR="${model_log_dir}" \
    SUMMARY_JSON="${summary_json}" \
    TEXTCRAFT_FIXED_SKILLS_JSON_PATH="${SKILLS_JSON}" \
    TEXTCRAFT_FIXED_SKILL_IDS="${skill_ids}" \
    bash "${script_path}" "${ENGINE_ARGS[@]}" "$@"
}

run_experiment "qwen3-8B" \
    "${SCRIPT_DIR}/run_textcraft_qwen3_8b_fixed.sh" \
    "" \
    "$@"

run_experiment "qwen3-8B_skills" \
    "${SCRIPT_DIR}/run_textcraft_qwen3_8b_fixed.sh" \
    "0" \
    "$@"

run_experiment "qwen3-8B_2skills" \
    "${SCRIPT_DIR}/run_textcraft_qwen3_8b_fixed.sh" \
    "0,1" \
    "$@"

run_experiment "qwen3-8B_3skills" \
    "${SCRIPT_DIR}/run_textcraft_qwen3_8b_fixed.sh" \
    "0,1,2" \
    "$@"

run_experiment "deepseek-v3.2" \
    "${SCRIPT_DIR}/run_textcraft_deepseek_fixed.sh" \
    "" \
    "$@"

echo "============================================================"
echo "[TextCraftPreExp] all experiments finished"
echo "[TextCraftPreExp] root=${PREEXP_ROOT}"
echo "============================================================"
