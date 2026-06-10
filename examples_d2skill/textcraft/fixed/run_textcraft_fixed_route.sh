#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="$(cd -- "${EXAMPLES_DIR}/.." && pwd)"

export FIXED_ROUTE_MODEL="${FIXED_ROUTE_MODEL:-qwen3-8B}"
export FIXED_EVAL_STYLE_LOGGING="${FIXED_EVAL_STYLE_LOGGING:-1}"
export FIXED_EVAL_DUMP_TRACE="${FIXED_EVAL_DUMP_TRACE:-1}"
export TEXTCRAFT_TIMEOUT="${TEXTCRAFT_TIMEOUT:-2400}"
export TEXTCRAFT_DATA_LEN="${TEXTCRAFT_DATA_LEN:-200}"
export VAL_DATA_SIZE="${VAL_DATA_SIZE:-1}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-15}"
export ROUTING_LLM_MAX_TOKENS="${ROUTING_LLM_MAX_TOKENS:-4096}"
export ROUTING_LLM_TEMPERATURE="${ROUTING_LLM_TEMPERATURE:-1.0}"
export ROUTING_LLM_TOP_P="${ROUTING_LLM_TOP_P:-1}"
export TEXTCRAFT_FIXED_GPU_MEMORY_UTILIZATION="${TEXTCRAFT_FIXED_GPU_MEMORY_UTILIZATION:-0.3}"
export TEXTCRAFT_MAX_STEPS="${TEXTCRAFT_MAX_STEPS:-20}"
export TEXTCRAFT_HISTORY_LENGTH="${TEXTCRAFT_HISTORY_LENGTH:-20}"
RUNS="${RUNS:-1}"
BASE_SEED="${BASE_SEED:-0}"
FIXED_TEXTCRAFT_OUTPUT_DIR="verl_agent_textcraft_fixed_route"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/checkpoints/${FIXED_TEXTCRAFT_OUTPUT_DIR}}"
MODEL_LOG_DIR="${LOG_ROOT}/${FIXED_ROUTE_MODEL}"
SUMMARY_JSON="${SUMMARY_JSON:-${MODEL_LOG_DIR}/fixed_route_metric_summary.json}"

ENGINE="vllm"
if [[ $# -gt 0 && ( "$1" == "vllm" || "$1" == "hf" || "$1" == "sglang" || "$1" == "ray" ) ]]; then
    ENGINE="$1"
    shift
fi

mkdir -p "$MODEL_LOG_DIR"

for ((run_idx = 0; run_idx < RUNS; run_idx++)); do
    seed=$((BASE_SEED + run_idx))
    log_path="${MODEL_LOG_DIR}/seed_${seed}.log"
    experiment_name="fixed_${FIXED_ROUTE_MODEL}_seed${seed}"
    trainer_output_dir="${MODEL_LOG_DIR}/${experiment_name}"

    echo "[TextCraftFixedRoute3x] model=${FIXED_ROUTE_MODEL} seed=${seed} max_steps=${TEXTCRAFT_MAX_STEPS} history_length=${TEXTCRAFT_HISTORY_LENGTH} val_data_size=${VAL_DATA_SIZE} max_concurrency=${MAX_CONCURRENCY} max_tokens=${ROUTING_LLM_MAX_TOKENS} temperature=${ROUTING_LLM_TEMPERATURE} top_p=${ROUTING_LLM_TOP_P} data_len=${TEXTCRAFT_DATA_LEN} timeout=${TEXTCRAFT_TIMEOUT} gpu_memory_utilization=${TEXTCRAFT_FIXED_GPU_MEMORY_UTILIZATION} log=${log_path}"
    bash "${EXAMPLES_DIR}/run_textcraft_d2skill_gigpo.sh" "$ENGINE" \
        routing.force_model_enable=True \
        routing.force_model_name="$FIXED_ROUTE_MODEL" \
        data.val_batch_size="$VAL_DATA_SIZE" \
        env.history_length="$TEXTCRAFT_HISTORY_LENGTH" \
        +env.textcraft_agentgym_prompt=True \
        env.seed="$seed" \
        +env.val_seed="$((seed + 1))" \
        trainer.val_only=True \
        +trainer.fixed_eval_style_logging="$FIXED_EVAL_STYLE_LOGGING" \
        trainer.dump_random_trace_json="$FIXED_EVAL_DUMP_TRACE" \
        trainer.n_gpus_per_node=1 \
        actor_rollout_ref.actor.ppo_mini_batch_size=16 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization="$TEXTCRAFT_FIXED_GPU_MEMORY_UTILIZATION" \
        actor_rollout_ref.rollout.max_num_seqs=256 \
        ray_init.num_cpus=5 \
        trainer.project_name="$FIXED_TEXTCRAFT_OUTPUT_DIR" \
        trainer.experiment_name="$experiment_name" \
        trainer.default_local_dir="$trainer_output_dir" \
        "$@" \
        2>&1 | tee "$log_path"
done

python3 "${EXAMPLES_DIR}/alfworld/fixed/aggregate_fixed_route_metrics.py" "$LOG_ROOT" \
    --json-out "$SUMMARY_JSON" \
    --wandb-name "fixed_${FIXED_ROUTE_MODEL}_3x_summary" \
    --no-wandb
