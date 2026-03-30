#!/bin/bash
# Periodically syncs the synthetic dataset to HuggingFace as it's generated.
# Pushes the live checkpoint + any finished train/val splits.

REPO="Reza2kn/reducto-api-tool-calls"
DATA_DIR="/Users/reducto-reza/AI/reducto-agent-benchmark/benchmark/data/synthetic_training"
SYNC_INTERVAL=300   # push every 5 minutes
LOG="/tmp/hf_sync.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "HF sync started → private:$REPO  (every ${SYNC_INTERVAL}s)"

while true; do
    sleep "$SYNC_INTERVAL"

    LINES=$(wc -l < "$DATA_DIR/.checkpoint.jsonl" 2>/dev/null || echo 0)

    python3 - <<PYEOF
from huggingface_hub import HfApi
import os, sys

api  = HfApi()
repo = "$REPO"
data = "$DATA_DIR"
lines = $LINES

files_to_push = []

# Always push the live checkpoint
ckpt = os.path.join(data, ".checkpoint.jsonl")
if os.path.exists(ckpt):
    files_to_push.append((ckpt, "data/checkpoint.jsonl"))

# Push finished splits if they exist
for fname in ("train.jsonl", "val.jsonl"):
    path = os.path.join(data, fname)
    if os.path.exists(path):
        files_to_push.append((path, f"data/{fname}"))

# README with current progress
readme = f"""---
license: mit
tags:
- tool-use
- function-calling
- reducto
- synthetic
size_categories:
- 10K<n<100K
---

# Reducto API Tool-Call Dataset

Synthetic training data for fine-tuning **Qwen3.5-35B-A3B** on Reducto document-processing API tool use.

Generated from hard probe benchmark results using top-5 teacher models:
Kimi K2.5, MiniMax M2.7, Gemini 3.1 Flash Lite, Qwen3.5-122B-A10B, Inception Mercury 2.

## Coverage
- All 7 Reducto endpoints (parse, extract, split, classify, edit, upload, get_job)
- All 13 accepted file formats (PDF, DOCX, XLSX, PPTX, PNG, JPG, TIFF, HTML, CSV, MSG, EML, TXT, BMP)
- Multi-hop chains (upload→parse→extract, classify→route→edit, async polling)
- Conflict resolution (mutually exclusive params)
- Negative examples and edge cases
- Targeted remediation for hardest probes

## Progress
**{lines:,} / 84,500 checkpoint entries** generated so far (~{lines//5:,} unique prompts × 5 teachers).
After verification + consensus filtering → target **16,900 final training examples**.

## Format
Each line is a JSON object with `messages` (system/user/assistant with tool_calls) and `tools` (full OpenAI-format schemas).
Compatible with HF TRL, Axolotl, and LLaMA Factory.
"""

try:
    for local_path, repo_path in files_to_push:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"sync: {lines:,} examples",
        )

    # Update README
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"readme: {lines:,} examples",
    )
    print(f"Pushed {len(files_to_push)} file(s) — {lines:,} lines")
except Exception as e:
    print(f"Push failed: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

    if [ $? -eq 0 ]; then
        log "✓ Pushed $LINES lines to $REPO"
    else
        log "✗ Push failed (will retry in ${SYNC_INTERVAL}s)"
    fi

    # Stop once train.jsonl exists and checkpoint is at/above target
    if [ -f "$DATA_DIR/verified_train.jsonl" ] && [ "$LINES" -ge 84500 ]; then
        log "Dataset complete. Final push done. Exiting."
        exit 0
    fi
done
