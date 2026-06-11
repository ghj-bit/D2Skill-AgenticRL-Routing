#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="$(cd -- "${EXAMPLES_DIR}/.." && pwd)"

ENGINE_ARGS=()
if [[ $# -gt 0 && ( "$1" == "vllm" || "$1" == "hf" || "$1" == "sglang" || "$1" == "ray" ) ]]; then
    ENGINE_ARGS=("$1")
    shift
fi

COMMON_LOG_ROOT="${TEXTCRAFT_COMPARE_LOG_ROOT:-${PROJECT_DIR}/checkpoints}"
NO_SKILLS_LOG_ROOT="${TEXTCRAFT_NO_SKILLS_LOG_ROOT:-${COMMON_LOG_ROOT}/verl_agent_textcraft_fixed_route}"
SKILLS_LOG_ROOT="${TEXTCRAFT_SKILLS_LOG_ROOT:-${COMMON_LOG_ROOT}/verl_agent_textcraft_fixed_route_skills}"
SKILL_IDS="${TEXTCRAFT_COMPARE_SKILL_IDS:-0,1}"

echo "[TextCraftCompare] Run 1/2: qwen3-8B without skills -> ${NO_SKILLS_LOG_ROOT}"
LOG_ROOT="$NO_SKILLS_LOG_ROOT" \
SUMMARY_JSON="${NO_SKILLS_LOG_ROOT}/qwen3-8B/fixed_route_metric_summary.json" \
TEXTCRAFT_FIXED_SKILL_IDS= \
bash "${SCRIPT_DIR}/run_textcraft_qwen3_8b_fixed.sh" "${ENGINE_ARGS[@]}" "$@"

echo "[TextCraftCompare] Run 2/2: qwen3-8B with skills (${SKILL_IDS}) -> ${SKILLS_LOG_ROOT}"
LOG_ROOT="$SKILLS_LOG_ROOT" \
SUMMARY_JSON="${SKILLS_LOG_ROOT}/qwen3-8B/fixed_route_metric_summary.json" \
TEXTCRAFT_FIXED_SKILL_IDS="$SKILL_IDS" \
bash "${SCRIPT_DIR}/run_textcraft_qwen3_8b_fixed.sh" "${ENGINE_ARGS[@]}" "$@"
