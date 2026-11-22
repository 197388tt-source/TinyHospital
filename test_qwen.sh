#!/usr/bin/env bash
# First run: bash test_qwen.sh; resume: bash test_qwen.sh --resume
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

exec "${TINY_HOSPITAL_PYTHON:-python}" main.py \
  --stage test \
  --runtime sync \
  --doctor-backend local-qwen \
  --memory-dir memories/final \
  --experiment-dir experiment/qwen3_betterex_N \
  --doctor-max-new-tokens 1024 \
  --dataset data/testcut.txt \
  --count 93 \
  --chat-timeout 120 \
  --api-retries 5 \
  "$@"
