#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ENGINE="${1:-vllm}"
if [[ $# -gt 0 ]]; then
    shift
fi

export FIXED_ROUTE_MODEL="${FIXED_ROUTE_MODEL:-deepseek-v3.2}"
export COST_COE="${COST_COE:-0.0}"

bash "${SCRIPT_DIR}/run_alfworld_d2skill.sh" "$ENGINE" \
    routing.force_model_enable=True \
    routing.force_model_name="$FIXED_ROUTE_MODEL" \
    env.skills_only_memory.enable_dynamic_update=True \
    env.skills_only_memory.update_source=validation \
    env.skills_only_memory.update_save_traj=True \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.project_name='verl_agent_alfworld_fixed_route' \
    trainer.experiment_name="fixed_${FIXED_ROUTE_MODEL}" \
    "$@"
