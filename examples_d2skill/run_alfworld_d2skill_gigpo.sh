#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ENGINE="${1:-vllm}"
if [[ $# -gt 0 ]]; then
    shift
fi

mode="${GIGPO_MODE:-mean_norm}" # "mean_norm" or "mean_std_norm"

bash "${SCRIPT_DIR}/run_alfworld_d2skill.sh" "$ENGINE" \
    algorithm.adv_estimator=gigpo \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode="$mode" \
    trainer.project_name='verl_agent_alfworld_gigpo' \
    trainer.experiment_name='gigpo_qwen3-4b_skills_d2skill' \
    "$@"
