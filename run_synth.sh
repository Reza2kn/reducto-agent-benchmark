#!/bin/bash
export PYTHONUNBUFFERED=1
source "$(dirname "$0")/.env"

cd /Users/reducto-reza/AI/reducto-agent-benchmark

echo "=== Synthetic data gen started: $(date) ===" >> /tmp/synth_gen.log

python3 -u benchmark/scripts/gen_synthetic_data.py \
  --target 36669 \
  --resume \
  >> /tmp/synth_gen.log 2>&1
