#!/usr/bin/env bash
# Doctor: local Qwen3-8B; Expert: the configured DeepSeek API.
# Resume with: bash train_qwen.sh --resume
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

exec "${TINY_HOSPITAL_PYTHON:-python}" main.py \
  --stage train \
  --runtime sync \
  --doctor-backend local-qwen \
  --doctor-max-new-tokens 1024 \
  --experiment-dir experiment/new_training \
  --dataset data/traincut.txt \
  --count 686 \
  --chat-timeout 180 \
  --api-retries 5 \
  "$@"
