#!/usr/bin/env bash
set -Eeuo pipefail

export WANDB_MODE="${TEXTCRAFT_WANDB_MODE:-offline}"
export SWANLAB_MODE="${TEXTCRAFT_SWANLAB_MODE:-offline}"

usage() {
  cat <<'EOF'
Usage:
  ./run_textcraft_d2skill_gigpo.sh [ENGINE=vllm] [Hydra overrides...]

Prerequisite:
  Ensure the AgentGym TextCraft server is reachable.

Useful env vars:
  TEXTCRAFT_ENV_SERVER_URL=https://.../proxy/36001
  TEXTCRAFT_TRAIN_JSON=/path/to/textcraft_train.json
  TEXTCRAFT_VAL_JSON=/path/to/textcraft_test.json
  TEXTCRAFT_MINECRAFT_DIR=agentenv_textcraft/
  TRAIN_DATA_SIZE=32 VAL_DATA_SIZE=32 GROUP_SIZE=8 TEXTCRAFT_MAX_STEPS=30
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/env.sh"

export WANDB_MODE="${TEXTCRAFT_WANDB_MODE:-offline}"
export SWANLAB_MODE="${TEXTCRAFT_SWANLAB_MODE:-offline}"

ENGINE="${1:-vllm}"
shift 2>/dev/null || true

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export RAY_worker_register_timeout_seconds="${RAY_worker_register_timeout_seconds:-600}"

TEXTCRAFT_ENV_SERVER_URL="${TEXTCRAFT_ENV_SERVER_URL:-https://nat-notebook-inspire.sii.edu.cn/ws-1177d2a5-aef0-40d3-8777-fed9af13affc/project-b795c114-135a-40db-b3d0-19b60f25237b/user-543feed4-0be2-4972-8987-a324af06c93f/vscode/6372f346-4387-4ba6-86a2-bf8d931df3b4/00e01d55-95b3-40d5-825c-05697643fb2d/proxy/36001}"
TEXTCRAFT_TIMEOUT="${TEXTCRAFT_TIMEOUT:-600}"
TEXTCRAFT_DATA_LEN="${TEXTCRAFT_DATA_LEN:-374}"
TEXTCRAFT_VAL_OFFSET="${TEXTCRAFT_VAL_OFFSET:-10000}"
TEXTCRAFT_MINECRAFT_DIR="${TEXTCRAFT_MINECRAFT_DIR:-agentenv_textcraft/}"
TEXTCRAFT_COMMANDS="${TEXTCRAFT_COMMANDS:-null}"
TEXTCRAFT_GOAL="${TEXTCRAFT_GOAL:-null}"
TEXTCRAFT_TRAIN_JSON="${TEXTCRAFT_TRAIN_JSON:-/inspire/hdd/project/ai4education/qianhong-p-qianhong/ghj_workspace/AgentGym-RL/AgentGym-RL-Data-ID/train/textcraft_train.json}"
TEXTCRAFT_VAL_JSON="${TEXTCRAFT_VAL_JSON:-/inspire/hdd/project/ai4education/qianhong-p-qianhong/ghj_workspace/AgentGym-RL/AgentGym-RL-Data-ID/eval/textcraft_test.json}"
TEXTCRAFT_PARQUET_DIR="${TEXTCRAFT_PARQUET_DIR:-$HOME/data/verl-agent/textcraft}"

train_data_size="${TRAIN_DATA_SIZE:-32}"
val_data_size="${VAL_DATA_SIZE:-64}"
group_size="${GROUP_SIZE:-8}"
max_steps="${TEXTCRAFT_MAX_STEPS:-30}"

success_reward_weight="${SUCCESS_REWARD_WEIGHT:-1.0}"
cost_reward_weight="${COST_REWARD_WEIGHT:-1.0}"
step_cost_reward_weight="${STEP_COST_REWARD_WEIGHT:-1.0}"

ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-/inspire/hdd/project/ai4education/qianhong-p-qianhong/ghj_workspace/Qwen/Qwen2.5-3B-Instruct}"
mode="${GIGPO_MODE:-mean_norm}"

if [[ -f "$TEXTCRAFT_PARQUET_DIR/train.parquet" && -f "$TEXTCRAFT_PARQUET_DIR/test.parquet" ]]; then
  echo "TextCraft parquet already exists, skip prepare: $TEXTCRAFT_PARQUET_DIR"
else
  python3 "${SCRIPT_DIR}/prepare_agentgym_textcraft_data.py" \
    --train-json "$TEXTCRAFT_TRAIN_JSON" \
    --val-json "$TEXTCRAFT_VAL_JSON" \
    --output-dir "$TEXTCRAFT_PARQUET_DIR"
fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gigpo \
  algorithm.gamma=0.95 \
  algorithm.gigpo.step_advantage_w=1.0 \
  algorithm.gigpo.mode="$mode" \
  data.train_files="$TEXTCRAFT_PARQUET_DIR/train.parquet" \
  data.val_files="$TEXTCRAFT_PARQUET_DIR/test.parquet" \
  data.train_batch_size="$train_data_size" \
  data.val_batch_size="$val_data_size" \
  data.max_prompt_length=4096 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name="$ENGINE" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.max_num_seqs=512 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=1.0 \
  reward_model.success_reward_weight="$success_reward_weight" \
  reward_model.cost_reward_weight="$cost_reward_weight" \
  reward_model.step_cost_reward_weight="$step_cost_reward_weight" \
  algorithm.use_kl_in_reward=False \
  env.env_name=textcraft \
  env.seed=0 \
  env.max_steps="$max_steps" \
  env.rollout.n="$group_size" \
  env.resources_per_worker.num_cpus=0.1 \
  env.history_length=3 \
  +env.textcraft="{env_addr: '${TEXTCRAFT_ENV_SERVER_URL}', timeout: ${TEXTCRAFT_TIMEOUT}, minecraft_dir: '${TEXTCRAFT_MINECRAFT_DIR}', commands: ${TEXTCRAFT_COMMANDS}, goal: ${TEXTCRAFT_GOAL}, data_len: ${TEXTCRAFT_DATA_LEN}, val_offset: ${TEXTCRAFT_VAL_OFFSET}}" \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.project_name='verl_routing_textcraft_gigpo' \
  trainer.experiment_name='gigpo_qwen2.5-3b_textcraft_d2skill' \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.log_val_generations=10 \
  trainer.save_freq=20 \
  trainer.test_freq=5 \
  trainer.total_epochs=80 \
  trainer.val_before_train=True \
  trainer.ray_wait_register_center_timeout=3600 \
  +trainer.dump_random_trace_json=train_once \
  ray_init.num_cpus=80 \
  "$@" \
  2>&1 | tee run_textcraft_gigpo.log
