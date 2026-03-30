#!/bin/bash
# setup_vastai.sh — One-shot setup for Vast.ai training instances.
#
# Two training runs IN PARALLEL on separate instances:
#   Instance A: H100 SXM 80GB  → train_35b.py  (Qwen3.5-35B-A3B MoE)
#   Instance B: RTX 4090 24GB  → train_0.8b.py (Qwen3.5-0.6B AgentJSON)
#
# Usage (run on each instance immediately after SSH):
#   bash setup_vastai.sh --model 35b  [--push-to-hub Reza2kn/qwen35-reducto-lora]
#   bash setup_vastai.sh --model 0.8b [--push-to-hub Reza2kn/qwen-0.8b-reducto-lora]
#
# Vast.ai instance selection:
#   35b  → Search: "H100 SXM" 80GB VRAM, ~$2.50-3.50/hr. Pick SXM (not PCIe) — MoE benefits from HBM bandwidth.
#   0.8b → Search: "RTX 4090" 24GB VRAM, ~$0.40-0.60/hr. Or any 16GB+ GPU.
#   Disk: request 80GB+ for 35B (model weights ~70GB + adapter), 20GB for 0.8B.
#   Image: Use "pytorch:2.3.0-py3.11-cuda12.1-devel" or any recent CUDA 12.x image.
#
# Required env vars (set in Vast.ai instance env or export before running):
#   HF_TOKEN       — HuggingFace token with read+write access to Reza2kn/*
#   REDUCTO_API_KEY — Only needed if you want L4 verification on the instance

set -e

MODEL="35b"
PUSH_TO_HUB=""
HF_TOKEN_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)       MODEL="$2";       shift 2 ;;
        --push-to-hub) PUSH_TO_HUB="$2"; shift 2 ;;
        --hf-token)    HF_TOKEN="$2";    shift 2 ;;
        *) echo "Unknown arg: $1"; shift ;;
    esac
done

if [[ "$MODEL" != "35b" && "$MODEL" != "0.8b" ]]; then
    echo "Error: --model must be '35b' or '0.8b'"
    exit 1
fi

echo "=== Reducto fine-tune setup — model: $MODEL ==="
echo "CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'check manually')"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'check manually')"
echo ""

# ── 1. Install Unsloth (latest from git for MoE + Qwen3.5 support) ──────────
echo "[1/5] Installing Unsloth..."
pip install --quiet "unsloth @ git+https://github.com/unslothai/unsloth.git"

# ── 2. Install deps ──────────────────────────────────────────────────────────
echo "[2/5] Installing trl, datasets, huggingface_hub, bitsandbytes..."
pip install --quiet trl datasets huggingface_hub bitsandbytes accelerate

# ── 3. Clone repo + get training scripts ────────────────────────────────────
echo "[3/5] Cloning repo..."
# If GITHUB_TOKEN or repo is public:
# git clone https://github.com/Reza2kn/reducto-agent-benchmark && cd reducto-agent-benchmark
# For now, assume scripts are already in the working directory (scp'd or mounted).
# Minimum files needed:
#   train_35b.py       (or train_0.8b.py)
#   benchmark/data/synthetic_training/verified_train.jsonl
#   benchmark/data/synthetic_training/verified_val.jsonl   (optional but recommended)
# The training scripts will auto-download verified data from HF if not present locally.

# ── 4. HF login ──────────────────────────────────────────────────────────────
echo "[4/5] Logging into HuggingFace..."
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    echo "  Logged in via HF_TOKEN env var"
else
    echo "  HF_TOKEN not set — running interactive login"
    echo "  Paste your HF token (needs read+write access to Reza2kn/*)"
    huggingface-cli login
fi

# ── 5. Upload AgentJSON base model to HF (0.8b run only) ────────────────────
# The local AgentJSON model needs to be accessible on Vast.ai.
# On the SOURCE machine (your Mac), run this ONCE before the Vast.ai training:
#   python3 -c "
#   from huggingface_hub import HfApi
#   api = HfApi()
#   api.create_repo('Reza2kn/qwen35-agentjson-base', private=True, exist_ok=True)
#   api.upload_folder(
#       folder_path='local-models/Qwen-0.8B-AgentJSON',
#       repo_id='Reza2kn/qwen35-agentjson-base',
#       repo_type='model'
#   )
#   "
# Then on Vast.ai set --model Reza2kn/qwen35-agentjson-base

echo "[5/5] Setup complete. Starting training..."
echo ""

# ── Launch ───────────────────────────────────────────────────────────────────
PUSH_ARG=""
if [ -n "$PUSH_TO_HUB" ]; then
    PUSH_ARG="--push-to-hub $PUSH_TO_HUB"
fi

if [ "$MODEL" = "35b" ]; then
    echo "=== Running 35B training on H100 SXM 80GB ==="
    echo "    Expected: ~2–3 hrs  |  LoRA adapter ~1GB  |  bf16 MoE"
    echo ""
    # Key config decisions — see train_35b.py comments for rationale:
    #   r=32       : more capacity than v1's r=16, still safe for 80GB MoE
    #   accum=16   : effective batch=16 (smaller than v1's 32 → better gradient signal)
    #   epochs=3   : standard for this dataset size; stop early at epoch 2 if val loss rises
    python train_35b.py \
        --model       unsloth/Qwen3.5-35B-A3B \
        --max-seq-len 2048 \
        --lora-r      32 \
        --epochs      3 \
        --batch-size  1 \
        --grad-accum  16 \
        --lr          2e-4 \
        $PUSH_ARG \
        2>&1 | tee /tmp/train_35b.log

    echo ""
    echo "=== 35B training done. Log: /tmp/train_35b.log ==="

elif [ "$MODEL" = "0.8b" ]; then
    echo "=== Running 0.8B training on RTX 4090 (or any 16GB+ GPU) ==="
    echo "    Expected: ~20–40 min  |  LoRA adapter ~50MB  |  bf16"
    echo ""
    # Key config decisions — see train_0.8b.py comments for rationale:
    #   r=64, rslora   : high rank + stable scaling for tiny model on structured task
    #   epochs=5       : small model needs more passes
    #   lr=1e-4        : conservative; high-rank small models are LR-sensitive
    python train_0.8b.py \
        --model       Reza2kn/qwen35-agentjson-base \
        --max-seq-len 4096 \
        --lora-r      64 \
        --epochs      5 \
        --batch-size  4 \
        --grad-accum  4 \
        --lr          1e-4 \
        $PUSH_ARG \
        2>&1 | tee /tmp/train_0.8b.log

    echo ""
    echo "=== 0.8B training done. Log: /tmp/train_0.8b.log ==="
fi
