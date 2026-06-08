#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export FIXED_ROUTE_MODEL="${FIXED_ROUTE_MODEL:-deepseek-v3.2}"

bash "${SCRIPT_DIR}/run_textcraft_fixed_route.sh" "$@"
