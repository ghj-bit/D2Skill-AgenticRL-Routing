#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ENGINE="vllm"
if [[ $# -gt 0 && ( "$1" == "vllm" || "$1" == "hf" || "$1" == "sglang" || "$1" == "ray" ) ]]; then
    ENGINE="$1"
    shift
fi

RUNS="${RUNS:-3}"
BASE_SEED="${BASE_SEED:-0}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/fixed_route_5x_logs}"
FIXED_MODEL_EVAL_SCRIPT="${SCRIPT_DIR}/run_alfworld_fixed_model_eval.sh"

mkdir -p "$LOG_ROOT"

mapfile -t FIXED_MODELS < <(
    PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" python3 - <<'PY'
from routing.models_config.models_config import MODEL_CONF

for model_name in MODEL_CONF:
    print(model_name)
PY
)

if [[ "${#FIXED_MODELS[@]}" -eq 0 ]]; then
    echo "[FixedRoute5x] error: no models found in routing.models_config.models_config.MODEL_CONF" >&2
    exit 1
fi

echo "[FixedRoute5x] models from MODEL_CONF: ${FIXED_MODELS[*]}"

is_run_complete() {
    local log_path="$1"
    local done_path="$2"
    RUN_COMPLETE_REASON=""

    if [[ -f "$done_path" ]]; then
        RUN_COMPLETE_REASON="done_marker"
        return 0
    fi
    if [[ ! -s "$log_path" ]]; then
        return 1
    fi

    if python3 - "$log_path" "$SCRIPT_DIR" <<'PY'
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
script_dir = Path(sys.argv[2])
sys.path.insert(0, str(script_dir))

try:
    from aggregate_fixed_route_metrics import _extract_metric_dict
    metrics = _extract_metric_dict(log_path.read_text(encoding="utf-8", errors="ignore"))
except Exception:
    raise SystemExit(1)

raise SystemExit(0 if metrics else 1)
PY
    then
        RUN_COMPLETE_REASON="metrics_in_log"
        return 0
    fi

    return 1
}

run_one_model() {
    local model_label="$1"
    shift
    local model_log_dir="${LOG_ROOT}/${model_label}"
    mkdir -p "$model_log_dir"

    for ((run_idx = 0; run_idx < RUNS; run_idx++)); do
        local seed=$((BASE_SEED + run_idx))
        local log_path="${model_log_dir}/seed_${seed}.log"
        local done_path="${model_log_dir}/seed_${seed}.done"
        local experiment_name="fixed_${model_label}_seed${seed}"

        if is_run_complete "$log_path" "$done_path"; then
            touch "$done_path"
            echo "[FixedRoute5x] skip completed model=${model_label} seed=${seed} reason=${RUN_COMPLETE_REASON} log=${log_path}"
            continue
        fi

        echo "[FixedRoute5x] model=${model_label} seed=${seed} log=${log_path}"
        API_MODEL="$model_label" \
            FIXED_EVAL_MODEL="$model_label" \
            FIXED_EVAL_SEED="$seed" \
            FIXED_EVAL_OUTPUT_DIR="${model_log_dir}/seed_${seed}_output" \
            bash "$FIXED_MODEL_EVAL_SCRIPT" \
            --model "$model_label" \
            --seed "$seed" \
            --output-dir "${model_log_dir}/seed_${seed}_output" \
            "$@" \
            2>&1 | tee "$log_path"
        touch "$done_path"
        python3 "${SCRIPT_DIR}/aggregate_fixed_route_metrics.py" "$LOG_ROOT" --no-wandb \
            || echo "[FixedRoute5x] warning: aggregate_fixed_route_metrics.py found no legacy fixed-route metrics yet"
    done
}

for model_name in "${FIXED_MODELS[@]}"; do
    run_one_model "$model_name" "$@"
done

python3 "${SCRIPT_DIR}/aggregate_fixed_route_metrics.py" "$LOG_ROOT" \
    || echo "[FixedRoute5x] warning: aggregate_fixed_route_metrics.py found no legacy fixed-route metrics"
python3 "${SCRIPT_DIR}/aggregate_checkpoint_task_success.py" "${PROJECT_DIR}/checkpoints/verl_agent_alfworld_fixed_route" \
    || echo "[FixedRoute5x] warning: aggregate_checkpoint_task_success.py found no checkpoint task-success files"
