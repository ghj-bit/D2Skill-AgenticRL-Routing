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
PREEXP_ROOT="${PREEXP_ROOT:-${PROJECT_DIR}/checkpoints/verl_agent_textcraft_fixed_route_manual_skills_5x3}"
SKILLS_JSON="${TEXTCRAFT_FIXED_SKILLS_JSON_PATH:-${SCRIPT_DIR}/textcraft_cost_planning_skills.json}"

run_experiment() {
    local name="$1"
    local model_name="$2"
    local use_skills="$3"
    local skill_ids="$4"
    shift 4

    local model_log_dir="${PREEXP_ROOT}/${name}"
    local summary_json="${model_log_dir}/fixed_route_metric_summary.json"

    echo "============================================================"
    echo "[TextCraftManualSkills5x3] start ${name}"
    echo "[TextCraftManualSkills5x3] model=${model_name} runs=${RUNS} base_seed=${BASE_SEED} use_skills=${use_skills} skill_ids='${skill_ids}'"
    echo "[TextCraftManualSkills5x3] skills_json=${SKILLS_JSON}"
    echo "[TextCraftManualSkills5x3] model_log_dir=${model_log_dir}"
    echo "============================================================"

    RUNS="${RUNS}" \
    BASE_SEED="${BASE_SEED}" \
    FIXED_ROUTE_MODEL="${model_name}" \
    LOG_ROOT="${PREEXP_ROOT}" \
    MODEL_LOG_DIR="${model_log_dir}" \
    SUMMARY_JSON="${summary_json}" \
    TEXTCRAFT_USE_FIXED_SKILLS="${use_skills}" \
    TEXTCRAFT_FIXED_SKILLS_BY_TASK_ID="0" \
    TEXTCRAFT_FIXED_SKILLS_JSON_PATH="${SKILLS_JSON}" \
    TEXTCRAFT_FIXED_SKILL_IDS="${skill_ids}" \
    bash "${SCRIPT_DIR}/run_textcraft_fixed_route.sh" "${ENGINE_ARGS[@]}" "$@"
}

run_experiment "qwen3-8B" \
    "qwen3-8B" \
    "0" \
    "" \
    "$@"

# run_experiment "qwen3-8B_skills" \
#     "qwen3-8B" \
#     "1" \
#     "0" \
#     "$@"

# run_experiment "qwen3-8B_2skills" \
#     "qwen3-8B" \
#     "1" \
#     "0,1" \
#     "$@"

# run_experiment "qwen3-8B_3skills" \
#     "qwen3-8B" \
#     "1" \
#     "0,1,2" \
#     "$@"

# run_experiment "deepseek-v3.2" \
#     "deepseek-v3.2" \
#     "0" \
#     "" \
#     "$@"

echo "============================================================"
echo "[TextCraftManualSkills5x3] all experiments finished"
echo "[TextCraftManualSkills5x3] root=${PREEXP_ROOT}"
echo "============================================================"
