#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ENGINE="${1:-vllm}"
if [[ $# -gt 0 ]]; then
    shift
fi

bash "${SCRIPT_DIR}/run_alfworld_d2skill.sh" "$ENGINE" \
    algorithm.adv_estimator=gigpo \
    trainer.experiment_name='gigpo_qwen3-4b_skills_d2skill' \
    "$@"
