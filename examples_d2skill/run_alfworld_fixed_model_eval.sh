#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="alfoworld"

# ---------------- Fixed model API config ----------------
# Defaults can be overridden by API_MODEL/API_BASE/API_KEY or FIXED_EVAL_*.
# If API_BASE is empty, fixed_model_alfworld_eval.py resolves it from MODEL_CONF.
API_MODEL="${API_MODEL:-${FIXED_EVAL_MODEL:-qwen2.5-7B}}"
API_BASE="${API_BASE:-${FIXED_EVAL_API_BASE:-}}"
API_KEY="${API_KEY:-${FIXED_EVAL_API_KEY:-empty}}"

export FIXED_EVAL_MODEL="$API_MODEL"
export FIXED_EVAL_API_BASE="$API_BASE"
export FIXED_EVAL_API_KEY="$API_KEY"

if [[ "$ENV_NAME" == "alfoworld" ]]; then
  echo "Launching AlfWorld fixed-model eval with model: ${FIXED_EVAL_MODEL}"
  python3 -m examples_d2skill.fixed_model_alfworld_eval "$@"
else
  echo "Error: Unsupported environment '$ENV_NAME'. Use 'alfoworld'." >&2
  exit 1
fi
