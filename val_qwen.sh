#!/usr/bin/env bash
# First run: bash val_qwen.sh; resume: bash val_qwen.sh --resume
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

exec "${TINY_HOSPITAL_PYTHON:-python}" main.py \
  --stage test \
  --runtime sync \
  --doctor-backend local-qwen \
  --memory-dir memories/final \
  --experiment-dir experiment/qwen_validation \
  --doctor-max-new-tokens 1024 \
  --dataset data/devcut.txt \
  --count 81 \
  --chat-timeout 120 \
  --api-retries 5 \
  "$@"
