#!/usr/bin/env bash

ENV_NAME="alfoworld"

# ---------------- Fixed model API config ----------------
# Only one fixed model is evaluated in this script. Edit these three values
# when switching to another OpenAI-compatible model endpoint.
API_MODEL="qwen2.5-7B"
API_BASE="https://notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-543feed4-0be2-4972-8987-a324af06c93f/vscode/3a8e9a70-c91e-459d-ad61-e9b54493df6c/d4674774-18d0-401c-974c-178c2126a92e/proxy/8042/v1"
API_KEY="empty"

export FIXED_EVAL_MODEL="$API_MODEL"
export FIXED_EVAL_API_BASE="$API_BASE"
export FIXED_EVAL_API_KEY="$API_KEY"

if [[ "$ENV_NAME" == "alfoworld" ]]; then
  echo "Launching AlfWorld agent..."
  python3 -m examples_d2skill.fixed_model_alfworld_eval
else
  echo "Error: Unsupported environment '$ENV_NAME'. Use 'alfoworld'." >&2
  exit 1
fi
