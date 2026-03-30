# H100 Training Instructions — AgentReducto A3B

This file is the single source of truth for training the Qwen3.5-35B-A3B model on the
Reducto API tool-call dataset. Read it fully before running anything.

---

## What You're Training

**Student model:** `unsloth/Qwen3.5-35B-A3B` (3.5B active parameters, MoE architecture)
**Goal:** A model that natively understands the Reducto document-processing API —
correct tool selection, parameter formatting, multi-hop chaining, and knowing when to stop.

**Two-stage training:**
1. **SFT** — supervised fine-tuning on **51,069** tool-call traces (45,969 train · 5,100 val)
2. **DPO** — preference training on **1,669 pairs** to fix the repetition-loop failure mode

Run SFT first. DPO continues from the SFT adapter. Do not run them in parallel or in reverse.

---

## Hardware

- 1× H100 SXM 80GB
- CUDA 12.1+
- ~70 GB VRAM for SFT (bf16 LoRA, MoE weights loaded in 16-bit)
- ~60 GB VRAM for DPO (same model, smaller optimizer state)

---

## Repos

| What | HF Repo | Notes |
|------|---------|-------|
| SFT dataset | `Reza2kn/reducto-api-tool-calls` | `data/verified_train.jsonl` + `data/verified_val.jsonl` |
| DPO dataset | `Reza2kn/reducto-dpo-termination` | `data/dpo_pairs.jsonl` — upload this first (Step 0) |
| SFT output | `Reza2kn/qwen35-reducto-lora` | LoRA adapter (private) |
| DPO output | `Reza2kn/qwen35-reducto-lora-dpo` | DPO-tuned adapter (private) |

---

## Environment Setup

```bash
# 1. Clone the repo
git clone https://github.com/Reza2kn/reducto-agent-benchmark
cd reducto-agent-benchmark

# 2. Install dependencies
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install trl transformers peft accelerate bitsandbytes datasets huggingface_hub

# 3. HF login (needed for pushing models + pulling private datasets)
huggingface-cli login   # paste your HF token when prompted

# 4. Verify GPU
python3 -c "import torch; print(torch.cuda.get_device_name(0)); print(f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')"
# Expected: NVIDIA H100 SXM5 80GB  /  80.x GB
```

---

## Step 0 — Verify Datasets Are on HF ✅

Both datasets are already uploaded. Confirm before starting:

```bash
python3 - << 'EOF'
from huggingface_hub import HfApi
api = HfApi()

sft = api.dataset_info("Reza2kn/reducto-api-tool-calls")
dpo = api.dataset_info("Reza2kn/reducto-dpo-termination")
print(f"SFT dataset : {sft.id}  (last modified: {sft.last_modified})")
print(f"DPO dataset : {dpo.id}  (last modified: {dpo.last_modified})")
print("Both present — proceed to Step 1.")
EOF
```

Expected:
- `Reza2kn/reducto-api-tool-calls` — 51,069 examples (45,969 train · 5,100 val)
- `Reza2kn/reducto-dpo-termination` — 1,669 preference pairs

---

## Step 1 — SFT Training

`train_35b.py` is already in the repo root. It handles:
- Pulling data from HF if not present locally
- `normalise_tool_calls()` to convert checkpoint format → Qwen3.5 template format
- `train_on_responses_only` masking (critical — do not remove)
- Sequence packing for throughput
- Saving LoRA adapter + merged 16-bit model

```bash
python3 train_35b.py \
  --hf-dataset Reza2kn/reducto-api-tool-calls \
  --output-dir runs/sft \
  --epochs 3 \
  --push-to-hub Reza2kn/qwen35-reducto-lora \
  2>&1 | tee /tmp/train_sft.log
```

**Expected runtime:** ~3–4 hours on H100 SXM 80GB (51,069 examples × 3 epochs, packing enabled)

**Expected VRAM:** ~68–72 GB (bf16 MoE weights + LoRA + optimizer states)

**Monitor:**
```bash
watch -n 5 nvidia-smi   # in a second terminal
tail -f /tmp/train_sft.log
```

**Sanity checks to verify it's working:**
1. First log line: `✓ Label masking OK — NNN/2048 tokens unmasked on sample[0]`
   - If you see `⚠ WARNING: All labels are -100`, the response masking is broken.
     Fix: verify the Qwen3.5 tokenizer uses `<|im_start|>user\n` / `<|im_start|>assistant\n` markers.
2. Training loss should drop from ~1.5–2.0 (step 0) to <0.4 by end of epoch 1.
   - If loss is flat or rising: data format issue, check `normalise_tool_calls()` output.
3. No OOM in the first 50 steps → you're fine for the full run.

**If OOM:**
```bash
# Drop effective batch (slower but safer):
python3 train_35b.py --grad-accum 8 --push-to-hub Reza2kn/qwen35-reducto-lora

# Or use QLoRA as last resort (slightly lower quality):
# Edit train_35b.py: set load_in_4bit=True, load_in_16bit=False
```

---

## Step 2 — DPO Training

Run `train_dpo.py` (also in repo root) **after SFT completes**. It loads the SFT adapter
and fine-tunes further with DPO preference pairs targeting the repetition-loop failure.

```bash
python3 train_dpo.py \
  --sft-model Reza2kn/qwen35-reducto-lora \
  --dpo-dataset Reza2kn/reducto-dpo-termination \
  --output-dir runs/dpo \
  --push-to-hub Reza2kn/qwen35-reducto-lora-dpo \
  2>&1 | tee /tmp/train_dpo.log
```

**Expected runtime:** ~30–45 minutes (1,727 pairs × 2 epochs — DPO converges fast)

**Expected VRAM:** ~60–65 GB (same model, DPO has smaller optimizer state than SFT)

**DPO data format** (what the dataset looks like — do not reformat):
```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Classify this doc and route it to extract..."},
    {"role": "assistant", "content": null, "tool_calls": [{"type": "function", "function": {"name": "reducto_classify", ...}}]},
    {"role": "tool", "content": "{\"category\": \"Financial Statement\", \"confidence\": 0.96}"}
  ],
  "chosen": [
    {"role": "assistant", "content": "I have analyzed the document and identified it as a Financial Statement...", "tool_calls": []}
  ],
  "rejected": [
    {"role": "assistant", "content": null, "tool_calls": [{"type": "function", "function": {"name": "reducto_classify", "arguments": "..."}}]}
  ],
  "tools": [...],
  "metadata": {"source": "generated", "teacher": "Claude Haiku 4.5 + thinking", "round": "dpo_termination"}
}
```

**Chosen = final text response (stop)**
**Rejected = repeating the last tool call (loop)**

**Sanity checks:**
1. DPO loss should start ~0.6–0.7 and drop toward ~0.4–0.5 within 100 steps.
2. Reward margin (chosen_reward - rejected_reward) should be positive and growing.
3. If rewards collapse (both go very negative), reduce `--beta` from 0.1 to 0.05.

---

## Data Format Reference

Both SFT and DPO data use the same base message format:

```json
{
  "messages": [
    {"role": "system", "content": "You are an AI assistant..."},
    {"role": "user", "content": "Parse this PDF..."},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_xyz", "type": "function",
       "function": {"name": "reducto_parse", "arguments": "{\"input\": \"https://...\", \"persist_results\": true}"}}
    ]},
    {"role": "tool", "content": "{\"job_id\": \"parse-abc123\", \"status\": \"pending\"}", "tool_call_id": "call_xyz"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_yz2", "type": "function",
       "function": {"name": "reducto_get_job", "arguments": "{\"job_id\": \"parse-abc123\"}"}}
    ]},
    ...
  ],
  "tools": [{"type": "function", "function": {"name": "reducto_parse", "description": "...", "parameters": {...}}}]
}
```

`normalise_tool_calls()` in `train_35b.py` handles format normalization automatically.

---

## What to Push and Where

| Artifact | Command | Notes |
|----------|---------|-------|
| SFT LoRA adapter | `--push-to-hub Reza2kn/qwen35-reducto-lora` | Auto-pushed by train_35b.py |
| DPO LoRA adapter | `--push-to-hub Reza2kn/qwen35-reducto-lora-dpo` | Auto-pushed by train_dpo.py |
| Merged 16-bit SFT | Saved locally to `runs/sft/merged_16bit/` | Push manually if needed for vLLM |
| GGUF (for llama.cpp) | `python3 convert_to_gguf.py` | Optional, post-training |

Push merged model manually if needed:
```bash
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="runs/dpo/merged_16bit",
    repo_id="Reza2kn/qwen35-reducto-merged",
    repo_type="model",
    private=True,
)
```

---

## Common Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `All labels are -100` | Response masking markers don't match template | Verify `<\|im_start\|>assistant\n` in tokenizer output |
| OOM at step 1 | bf16 too large | Add `--grad-accum 8` or set `load_in_4bit=True` |
| Loss NaN after epoch 1 | LR too high | Restart with `--lr 1e-4` |
| DPO rewards both collapse | Beta too high | Use `--beta 0.05` |
| `safetensors` not found | Missing adapter files | Check `runs/sft/lora_adapter/` exists before DPO |
| HF 401 on push | Not logged in | Run `huggingface-cli login` |
| Slow throughput (<500 tok/s) | Packing disabled or Flash Attention off | Ensure `packing=True` and `UNSLOTH_MOE_BACKEND=unsloth_triton` |

---

## Post-Training Benchmark

After both runs complete, benchmark the DPO model against the MCP probes:

```bash
# From the Mac (benchmark scripts aren't on H100)
# Update bench_mcp_probe.py to add the new model config, then:
python3 benchmark/scripts/bench_mcp_probe.py --model qwen35-reducto-lora-dpo
```

Target scores (based on training data coverage):
- MCP probes: ≥16/21 (up from 9/21 on ReductoLoRA-Q6K)
- API probes: ≥25/30 (up from 20/30)
- Zero repetition loops (that's the whole point of DPO)
