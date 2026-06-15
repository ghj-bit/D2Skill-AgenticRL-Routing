#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

TOKEN_METHOD="${TOKEN_METHOD:-regex}"
WRITE_CALL_DETAILS="${WRITE_CALL_DETAILS:-0}"

ROOTS=(
  "${PROJECT_DIR}/verl_agent_textcraft_fixed_route_preexp_3x3"
  "${PROJECT_DIR}/verl_agent_textcraft_fixed_route_skills_5"
)

if [[ $# -gt 0 ]]; then
  ROOTS=("$@")
fi

ARGS=("${SCRIPT_DIR}/estimate_textcraft_fixed_tokens.py" --method "${TOKEN_METHOD}")
if [[ "${WRITE_CALL_DETAILS}" == "1" || "${WRITE_CALL_DETAILS}" == "true" ]]; then
  ARGS+=(--write-call-details)
fi
ARGS+=("${ROOTS[@]}")

python3 "${ARGS[@]}"
