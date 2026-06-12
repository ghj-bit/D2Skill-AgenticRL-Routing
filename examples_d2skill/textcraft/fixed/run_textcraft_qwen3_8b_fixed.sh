#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export FIXED_ROUTE_MODEL="${FIXED_ROUTE_MODEL:-qwen3-8B}"
export TEXTCRAFT_FIXED_SKILLS_JSON_PATH="${TEXTCRAFT_FIXED_SKILLS_JSON_PATH:-${SCRIPT_DIR}/textcraft_cost_planning_skills.json}"
export TEXTCRAFT_FIXED_SKILL_IDS="${TEXTCRAFT_FIXED_SKILL_IDS-0,1,2}"

ENGINE_ARGS=()
if [[ $# -gt 0 && ( "$1" == "vllm" || "$1" == "hf" || "$1" == "sglang" || "$1" == "ray" ) ]]; then
    ENGINE_ARGS=("$1")
    shift
fi

bash "${SCRIPT_DIR}/run_textcraft_fixed_route.sh" "${ENGINE_ARGS[@]}" \
    +env.textcraft_fixed_skills_json_path="$TEXTCRAFT_FIXED_SKILLS_JSON_PATH" \
    "+env.textcraft_fixed_skill_ids='${TEXTCRAFT_FIXED_SKILL_IDS}'" \
    "$@"
