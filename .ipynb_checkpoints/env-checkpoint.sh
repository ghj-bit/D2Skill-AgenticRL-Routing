#!/bin/bash
# Environment configuration template for SkillRL.
# Copy to env.sh and fill in real values:
#   cp env.example env.sh

# Ray Dashboard configuration (optional)
# RAY_DASHBOARD_CONFIG="trainer.ray_dashboard_host=0.0.0.0 trainer.ray_dashboard_port=8265"
# RAY_DASHBOARD_CONFIG=""

# SwanLab configuration
export SWANLAB_MODE="cloud"
export SWANLAB_API_KEY="YOUR_SWANLAB_API_KEY"
export SWANLAB_LOG_DIR="swanlog"
export SWANLAB_DISABLE_PROMPT=1
export SWANLAB_NON_INTERACTIVE=1

# CUDA / paths
# export CUDA_HOME="/path/to/your/conda/env"
export ALFWORLD_DATA="/inspire/hdd/project/ai4education/qianhong-p-qianhong/ghj_workspace/D2Skill-AgenticRL/agent_system/environments/env_package/alfworld/alfworld"

# Runtime tuning
export RAY_worker_register_timeout_seconds=600
# export RAY_START_METHOD=spawn
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export TORCH_NUM_THREADS=1
# export TORCH_INTRAOP_THREADS=1
# export TORCH_INTEROP_THREADS=1

# OpenAI-compatible API settings
export OPENAI_BASE_URL="https://notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-543feed4-0be2-4972-8987-a324af06c93f/vscode/4ff709dd-915e-4392-8a69-12c61dc95edb/46309a08-f7f1-4f88-b9f1-aabe05bace7a/proxy/8055/v1"
export OPENAI_API_KEY="empty"
export OPENAI_MODEL="deepseek"
# export SUMMARIZER_OPENAI_MODEL="gemini-3-flash-preview"
