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
QWEN3_8B_ALL_SEEDS_FAILED_TASK_IDS="${TEXTCRAFT_FIXED_SKILLS_TASK_IDS:-149,152,168,420,421,422,423,425,429,431,433,434,439,442,443,533,535}"

run_experiment() {
    local name="$1"
    local model_name="$2"
    local use_skills="$3"
    local skill_ids="$4"
    shift 4

    local model_log_dir="${PREEXP_ROOT}/${name}"
    local summary_json="${model_log_dir}/fixed_route_metric_summary.json"
    local skills_by_task_id="0"
    local skills_task_ids=""
    if [[ "${use_skills}" == "1" || "${use_skills}" == "true" ]]; then
        skills_by_task_id="1"
        skills_task_ids="${QWEN3_8B_ALL_SEEDS_FAILED_TASK_IDS}"
    fi

    echo "============================================================"
    echo "[TextCraftManualSkills5x3] start ${name}"
    echo "[TextCraftManualSkills5x3] model=${model_name} runs=${RUNS} base_seed=${BASE_SEED} use_skills=${use_skills} skill_ids='${skill_ids}'"
    echo "[TextCraftManualSkills5x3] skills_json=${SKILLS_JSON}"
    echo "[TextCraftManualSkills5x3] skills_by_task_id=${skills_by_task_id} skills_task_ids=${skills_task_ids:-none}"
    echo "[TextCraftManualSkills5x3] model_log_dir=${model_log_dir}"
    echo "============================================================"

    RUNS="${RUNS}" \
    BASE_SEED="${BASE_SEED}" \
    FIXED_ROUTE_MODEL="${model_name}" \
    LOG_ROOT="${PREEXP_ROOT}" \
    MODEL_LOG_DIR="${model_log_dir}" \
    SUMMARY_JSON="${summary_json}" \
    TEXTCRAFT_USE_FIXED_SKILLS="${use_skills}" \
    TEXTCRAFT_FIXED_SKILLS_BY_TASK_ID="${skills_by_task_id}" \
    TEXTCRAFT_FIXED_SKILLS_JSON_PATH="${SKILLS_JSON}" \
    TEXTCRAFT_FIXED_SKILL_IDS="${skill_ids}" \
    TEXTCRAFT_FIXED_SKILLS_TASK_IDS="${skills_task_ids}" \
    TEXTCRAFT_AGENT_PROMPT_STYLE="${TEXTCRAFT_AGENT_PROMPT_STYLE:-agentgym}" \
    TEXTCRAFT_HISTORY_LENGTH="${TEXTCRAFT_HISTORY_LENGTH:-0}" \
    TEXTCRAFT_KEEP_FULL_ASSISTANT_HISTORY_INCLUDE_THINK="${TEXTCRAFT_KEEP_FULL_ASSISTANT_HISTORY_INCLUDE_THINK:-1}" \
    ROUTING_LLM_MAX_TOKENS="${ROUTING_LLM_MAX_TOKENS:-4096}" \
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



