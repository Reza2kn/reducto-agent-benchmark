#!/bin/bash
export PYTHONUNBUFFERED=1
source "$(dirname "$0")/.env"

cd /Users/reducto-reza/AI/reducto-agent-benchmark

echo "=== Gemini probe run: $(date) ===" >> /tmp/hard_probe_run.log

python3 -u benchmark/scripts/bench_param_probe.py \
  --model "gemini" \
  --probe-set hard \
  --skip-done \
  >> /tmp/hard_probe_run.log 2>&1
