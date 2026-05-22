#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export FIXED_ROUTE_MODEL="${FIXED_ROUTE_MODEL:-qwen3-8B}"

if [[ "${DIRECT_FIXED_MODEL_EVAL:-1}" == "1" ]]; then
    if [[ $# -gt 0 && ( "$1" == "vllm" || "$1" == "hf" || "$1" == "sglang" || "$1" == "ray" ) ]]; then
        shift
    fi

    export FIXED_EVAL_MODEL="${FIXED_EVAL_MODEL:-$FIXED_ROUTE_MODEL}"
    export FIXED_EVAL_API_BASE="${FIXED_EVAL_API_BASE:-https://notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-543feed4-0be2-4972-8987-a324af06c93f/vscode/4a7c22e1-2ea5-4c8a-8f1e-7c47a4734b85/84c7c462-172c-4370-af88-4c504b4dac10/proxy/8042/v1}"
    export FIXED_EVAL_API_KEY="${FIXED_EVAL_API_KEY:-empty}"

    echo "Launching direct AlfWorld fixed-model eval with model: ${FIXED_EVAL_MODEL}"
    python3 -m examples_d2skill.fixed_model_alfworld_eval "$@"
    exit 0
fi

ENGINE="${1:-vllm}"
if [[ $# -gt 0 ]]; then
    shift
fi

bash "${SCRIPT_DIR}/run_alfworld_d2skill.sh" "$ENGINE" \
    routing.force_model_enable=True \
    routing.force_model_name="$FIXED_ROUTE_MODEL" \
    env.skills_only_memory.enable_dynamic_update=True \
    env.skills_only_memory.update_source=validation \
    env.skills_only_memory.update_save_traj=True \
    trainer.val_only=True \
    trainer.val_before_train=True \
    +trainer.write_validation_alfworld_task_success=True \
    trainer.project_name='verl_agent_alfworld_fixed_route' \
    trainer.experiment_name="fixed_${FIXED_ROUTE_MODEL}" \
    "$@"
