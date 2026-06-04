#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export FIXED_ROUTE_MODEL="${FIXED_ROUTE_MODEL:-qwen3-8B}"
export FIXED_EVAL_STYLE_LOGGING="${FIXED_EVAL_STYLE_LOGGING:-1}"
export FIXED_EVAL_DUMP_TRACE="${FIXED_EVAL_DUMP_TRACE:-1}"
export VAL_DATA_SIZE="${VAL_DATA_SIZE:-16}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
export FIXED_ALFWORLD_SKILLS_JSON_PATH="${FIXED_ALFWORLD_SKILLS_JSON_PATH:-/inspire/hdd/project/ai4education/qianhong-p-qianhong/ghj_workspace/D2Skill-AgenticRL-Routing/checkpoints/verl_agent_alfworld_gigpo/gigpo_qwen3-4b_skills_d2skill_0527/updated_skills_train_step140.json}"
export FIXED_EVAL_EMBEDDING_MODEL_PATH="${FIXED_EVAL_EMBEDDING_MODEL_PATH:-/inspire/hdd/project/ai4education/qianhong-p-qianhong/ghj_workspace/Qwen/Qwen3-Embedding-0.6B}"
RUNS="${RUNS:-3}"
BASE_SEED="${BASE_SEED:-0}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/fixed_route_3x_logs}"
MODEL_LOG_DIR="${LOG_ROOT}/${FIXED_ROUTE_MODEL}"
SUMMARY_JSON="${SUMMARY_JSON:-${MODEL_LOG_DIR}/fixed_route_metric_summary.json}"

ENGINE="vllm"
if [[ $# -gt 0 && ( "$1" == "vllm" || "$1" == "hf" || "$1" == "sglang" || "$1" == "ray" ) ]]; then
    ENGINE="$1"
    shift
fi

SKILL_ARGS=()
if [[ -n "${FIXED_ALFWORLD_SKILLS_JSON_PATH}" ]]; then
    SKILL_ARGS+=(
        --skills-json-path "$FIXED_ALFWORLD_SKILLS_JSON_PATH"
    )
    echo "Loading AlfWorld skills from: ${FIXED_ALFWORLD_SKILLS_JSON_PATH}"
else
    SKILL_ARGS+=(
        --skills-json-path None
    )
    echo "No initial AlfWorld skills path configured; starting with empty skills."
fi

mkdir -p "$MODEL_LOG_DIR"

for ((run_idx = 0; run_idx < RUNS; run_idx++)); do
    seed=$((BASE_SEED + run_idx))
    eval_seed=$((seed + 1000))
    log_path="${MODEL_LOG_DIR}/seed_${seed}.log"
    output_dir="${MODEL_LOG_DIR}/seed_${seed}"
    TRACE_ARGS=()
    if [[ "$FIXED_EVAL_DUMP_TRACE" == "1" || "$FIXED_EVAL_DUMP_TRACE" == "true" || "$FIXED_EVAL_DUMP_TRACE" == "True" ]]; then
        TRACE_ARGS+=(--record-trajectories)
    fi

    echo "[FixedRoute3x] model=${FIXED_ROUTE_MODEL} seed=${seed} log=${log_path}"
    python3 "${SCRIPT_DIR}/fixed_model_alfworld_eval.py" \
        --model "$FIXED_ROUTE_MODEL" \
        --env-num "$VAL_DATA_SIZE" \
        --seed "$eval_seed" \
        --test-times 1 \
        --max-steps 50 \
        --max-concurrency "$MAX_CONCURRENCY" \
        --history-length 2 \
        --eval-dataset eval_in_distribution \
        --embedding-model-path "$FIXED_EVAL_EMBEDDING_MODEL_PATH" \
        --output-dir "$output_dir" \
        "${SKILL_ARGS[@]}" \
        "${TRACE_ARGS[@]}" \
        "$@" \
        2>&1 | tee "$log_path"
done

python3 "${SCRIPT_DIR}/fixed_model_alfworld_eval.py" \
    --aggregate-root "$MODEL_LOG_DIR" \
    --json-out "$SUMMARY_JSON"
