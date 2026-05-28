#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ENGINE="vllm"
if [[ $# -gt 0 && ( "$1" == "vllm" || "$1" == "hf" || "$1" == "sglang" || "$1" == "ray" ) ]]; then
  ENGINE="$1"
  shift
fi

# Defaults can be overridden by API_MODEL/FIXED_EVAL_MODEL.
FIXED_ROUTE_MODEL="${API_MODEL:-${FIXED_EVAL_MODEL:-deepseek}}"
export FIXED_ROUTE_MODEL

echo "Launching validation-only fixed-route AlfWorld eval with model: ${FIXED_ROUTE_MODEL}"
bash "${SCRIPT_DIR}/run_alfworld_d2skill.sh" "$ENGINE" \
  routing.force_model_enable=True \
  routing.force_model_name="$FIXED_ROUTE_MODEL" \
  routing.skip_router_generation=True \
  env.skills_only_memory.enable_dynamic_update=True \
  env.skills_only_memory.update_source=validation \
  env.skills_only_memory.update_save_traj=True \
  trainer.val_only=True \
  +trainer.write_validation_alfworld_task_success=True \
  trainer.n_gpus_per_node=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.max_num_seqs=256 \
  ray_init.num_cpus=40 \
  trainer.project_name='verl_agent_alfworld_fixed_route' \
  trainer.experiment_name="fixed_${FIXED_ROUTE_MODEL}" \
  "$@"
