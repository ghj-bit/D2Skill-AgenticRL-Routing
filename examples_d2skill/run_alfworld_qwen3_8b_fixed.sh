#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export FIXED_ROUTE_MODEL="${FIXED_ROUTE_MODEL:-qwen3-8B}"
export FIXED_EVAL_STYLE_LOGGING="${FIXED_EVAL_STYLE_LOGGING:-1}"
export FIXED_EVAL_DUMP_TRACE="${FIXED_EVAL_DUMP_TRACE:-1}"
export VAL_DATA_SIZE="${VAL_DATA_SIZE:-16}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
export FIXED_ALFWORLD_SKILLS_JSON_PATH="${FIXED_ALFWORLD_SKILLS_JSON_PATH:-/inspire/hdd/project/ai4education/qianhong-p-qianhong/ghj_workspace/D2Skill-AgenticRL-Routing/checkpoints/verl_agent_alfworld_gigpo/gigpo_qwen3-4b_skills_d2skill_0527/updated_skills_train_step140.json}"

ENGINE="vllm"
if [[ $# -gt 0 && ( "$1" == "vllm" || "$1" == "hf" || "$1" == "sglang" || "$1" == "ray" ) ]]; then
    ENGINE="$1"
    shift
fi

SKILL_ARGS=()
if [[ -n "${FIXED_ALFWORLD_SKILLS_JSON_PATH}" ]]; then
    SKILL_ARGS+=(
        env.skills_only_memory.skills_json_path="$FIXED_ALFWORLD_SKILLS_JSON_PATH"
    )
    echo "Loading AlfWorld skills from: ${FIXED_ALFWORLD_SKILLS_JSON_PATH}"
else
    SKILL_ARGS+=(
        env.skills_only_memory.skills_json_path=None
    )
    echo "No initial AlfWorld skills path configured; starting with empty skills."
fi

bash "${SCRIPT_DIR}/run_alfworld_d2skill.sh" "$ENGINE" \
    routing.force_model_enable=True \
    routing.force_model_name="$FIXED_ROUTE_MODEL" \
    data.val_batch_size="$VAL_DATA_SIZE" \
    env.seed=0 \
    +env.val_seed=1 \
    env.use_skills_only_memory=True \
    env.skills_only_memory.enable_dynamic_update=True \
    env.skills_only_memory.update_source=validation \
    env.skills_only_memory.update_save_traj=True \
    "${SKILL_ARGS[@]}" \
    trainer.val_only=True \
    +trainer.fixed_eval_style_logging="$FIXED_EVAL_STYLE_LOGGING" \
    +trainer.dump_random_trace_json="$FIXED_EVAL_DUMP_TRACE" \
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
