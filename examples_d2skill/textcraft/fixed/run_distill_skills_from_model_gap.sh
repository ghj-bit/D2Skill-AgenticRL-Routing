#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_CONF_KEY="${MODEL_CONF_KEY:-deepseek}"
DISTILL_MODEL="${DISTILL_MODEL:-${MODEL_CONF_KEY}}"
MODELS_CONFIG="${MODELS_CONFIG:-${REPO_ROOT}/routing/models_config/models_config.py}"
DISTILL_SCRIPT="${DISTILL_SCRIPT:-${SCRIPT_DIR}/distill_skills_from_model_gap.py}"

WEAK_DIR="${WEAK_DIR:-/inspire/hdd/project/ai4education/qianhong-p-qianhong/ghj_workspace/D2Skill-AgenticRL-Routing/checkpoints/verl_agent_textcraft_fixed_route/qwen3-8B/fixed_qwen3-8B_seed0/trajectories}"
STRONG_DIR="${STRONG_DIR:-/inspire/hdd/project/ai4education/qianhong-p-qianhong/ghj_workspace/D2Skill-AgenticRL-Routing/checkpoints/verl_agent_textcraft_fixed_route/deepseek-v3.2/fixed_deepseek-v3.2_seed0/trajectories}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/distilled_model_gap_skills}"
OUTPUT="${OUTPUT:-${OUT_DIR}/distilled_textcraft_model_gap_skills.json}"
PAIRS_OUTPUT="${PAIRS_OUTPUT:-${OUT_DIR}/model_gap_pairs.json}"
PROMPT_DIR="${PROMPT_DIR:-${OUT_DIR}/prompts}"
MAX_PAIRS="${MAX_PAIRS:-50}"
MAX_PAIRS_PER_DEPTH="${MAX_PAIRS_PER_DEPTH:-10}"
MAX_TURNS="${MAX_TURNS:-18}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TIMEOUT="${TIMEOUT:-240}"
RETRIES="${RETRIES:-2}"
INFER_SUCCESS_FROM_TEXT="${INFER_SUCCESS_FROM_TEXT:-0}"
DRY_RUN="${DRY_RUN:-0}"

read_model_conf() {
  "${PYTHON_BIN}" - "$MODELS_CONFIG" "$MODEL_CONF_KEY" <<'PY'
import importlib.util
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
spec = importlib.util.spec_from_file_location("routing_models_config_runtime", str(path))
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load models config: {path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
conf = getattr(module, "MODEL_CONF", {})
item = conf.get(key)
if not item:
    raise SystemExit(f"MODEL_CONF[{key!r}] not found in {path}")
print(item.get("api_base", ""))
print(item.get("api_key", ""))
PY
}

mapfile -t MODEL_CONF_VALUES < <(read_model_conf)
API_BASE="${API_BASE:-${MODEL_CONF_VALUES[0]:-}}"
API_KEY="${API_KEY:-${MODEL_CONF_VALUES[1]:-}}"

if [[ -z "$API_BASE" ]]; then
  echo "[error] api_base is empty for MODEL_CONF['${MODEL_CONF_KEY}'] in ${MODELS_CONFIG}" >&2
  exit 1
fi
if [[ -z "$API_KEY" ]]; then
  echo "[error] api_key is empty for MODEL_CONF['${MODEL_CONF_KEY}'] in ${MODELS_CONFIG}" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

ARGS=(
  --weak-dir "$WEAK_DIR"
  --strong-dir "$STRONG_DIR"
  --output "$OUTPUT"
  --pairs-output "$PAIRS_OUTPUT"
  --prompt-dir "$PROMPT_DIR"
  --max-pairs "$MAX_PAIRS"
  --max-pairs-per-depth "$MAX_PAIRS_PER_DEPTH"
  --max-turns "$MAX_TURNS"
  --api-key "$API_KEY"
  --base-url "$API_BASE"
  --model "$DISTILL_MODEL"
  --temperature "$TEMPERATURE"
  --max-tokens "$MAX_TOKENS"
  --timeout "$TIMEOUT"
  --retries "$RETRIES"
)

if [[ "$INFER_SUCCESS_FROM_TEXT" == "1" ]]; then
  ARGS+=(--infer-success-from-text)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  ARGS+=(--dry-run)
fi

echo "[distill] repo_root=$REPO_ROOT"
echo "[distill] model_conf=$MODELS_CONFIG key=$MODEL_CONF_KEY model=$DISTILL_MODEL"
echo "[distill] weak_dir=$WEAK_DIR"
echo "[distill] strong_dir=$STRONG_DIR"
echo "[distill] output=$OUTPUT"

exec "$PYTHON_BIN" "$DISTILL_SCRIPT" "${ARGS[@]}" "$@"