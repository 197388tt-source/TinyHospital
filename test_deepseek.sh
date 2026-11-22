#!/usr/bin/env bash
# First run: bash test_deepseek.sh; resume: bash test_deepseek.sh --resume
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

exec "${TINY_HOSPITAL_PYTHON:-python}" main.py \
  --stage test \
  --runtime sync \
  --doctor-backend api \
  --memory-dir memories/final \
  --experiment-dir experiment/deepseek_history \
  --dataset data/testcut.txt \
  --count 93 \
  --chat-timeout 120 \
  --api-retries 5 \
  "$@"
