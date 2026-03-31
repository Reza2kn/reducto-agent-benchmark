# H100 Training Instructions — 0.8B AgentJSON Targeted Fine-tune

Single source of truth for training the targeted 0.8B fix. Read it fully before running anything.

---

## What You're Training

**Starting checkpoint:** `Qwen-0.8B-AgentJSON` — the function-calling model, NOT base Qwen and NOT ReductoLoRA.
**Why AgentJSON and not base or ReductoLoRA:**
- AgentJSON already knows tool-call format, when-to-call vs when-to-respond, JSON structure. Throwing that away means relearning mechanics from scratch.
- ReductoLoRA (R1 trained) actively learned the wrong thing: 51K examples with zero chain-termination signal taught it to loop. Don't continue from a model that learned to be wrong.

**Goal:** Fix 15 specific measured failures without touching the function-calling mechanics AgentJSON already has.

**Two-stage training:**
1. **SFT** — 2,669 targeted traces across 15 failure clusters
2. **DPO** — 969 preference pairs targeting 3 failure modes

Run SFT first. DPO starts from the SFT adapter. Do not reverse.

---

## Hardware

- 1× H100 SXM 80GB (or H100 PCIe — fits easily, model is 0.8B)
- CUDA 12.1+
- ~8 GB VRAM for SFT (0.8B in bf16 + LoRA r=64)
- ~6 GB VRAM for DPO

This is fast. SFT will finish in under 30 minutes. DPO under 15.

---

## Repos

| What | HF Repo / Local Path |
|------|---------------------|
| SFT dataset (generate first) | `benchmark/data/synthetic_training_0_8b/.checkpoint_0_8b.jsonl` |
| DPO dataset (generate first) | `benchmark/data/dpo_0_8b/.checkpoint_dpo_0_8b.jsonl` |
| Starting model | `Qwen-0.8B-AgentJSON` (download from wherever you have it, or HF) |
| SFT output | `Reza2kn/qwen-0_8b-agentjson-reducto-lora` |
| DPO output | `Reza2kn/qwen-0_8b-agentjson-reducto-lora-dpo` |

---

## Step 0 — Generate datasets on the Mac first

Both generators run on CPU/Mac (Haiku API calls). Start these before you touch the H100.

```bash
cd benchmark/scripts

# SFT: ~2,669 examples, ~1-2h depending on Haiku rate limits
python gen_0_8b_targeted.py --workers 8

# DPO: ~969 pairs, ~30-45min
python gen_dpo_0_8b.py --workers 8
```

Monitor with `wc -l benchmark/data/synthetic_training_0_8b/.checkpoint_0_8b.jsonl`.

If interrupted: `--resume` flag on both scripts.

---

## Step 1 — Upload datasets to HF

```bash
python - << 'EOF'
from datasets import Dataset
import json, pathlib

# SFT
rows = [json.loads(l) for l in
        pathlib.Path("benchmark/data/synthetic_training_0_8b/.checkpoint_0_8b.jsonl")
        .read_text().splitlines() if l.strip()]
print(f"SFT rows: {len(rows)}")
ds = Dataset.from_list(rows)
ds.push_to_hub("Reza2kn/reducto-0_8b-sft-targeted", private=True)

# DPO
rows = [json.loads(l) for l in
        pathlib.Path("benchmark/data/dpo_0_8b/.checkpoint_dpo_0_8b.jsonl")
        .read_text().splitlines() if l.strip()]
print(f"DPO rows: {len(rows)}")
ds = Dataset.from_list(rows)
ds.push_to_hub("Reza2kn/reducto-0_8b-dpo-targeted", private=True)
EOF
```

---

## Step 2 — SFT on H100

```bash
python train_0_8b.py \
  --base-model Qwen-0.8B-AgentJSON \
  --sft-dataset Reza2kn/reducto-0_8b-sft-targeted \
  --output-dir runs/0_8b_sft \
  --epochs 3 \
  --push-to-hub Reza2kn/qwen-0_8b-agentjson-reducto-lora
```

Key training params (hardcoded in `train_0_8b.py`, do not change without reason):

| Param | Value | Why |
|-------|-------|-----|
| `lora_r` | 64 | 15 new behaviors need more adapter capacity than the r=16 ReductoLoRA used |
| `lora_alpha` | 128 | 2× r — standard |
| `epochs` | 3 | Literature confirms overfitting at 5 epochs on 0.8B with small datasets |
| `lr` | 5e-5 | Conservative — lower than R1's 1e-4 to avoid overwriting AgentJSON priors |
| `packing` | False | Tiny dataset (2,669 examples) — packing hurts quality at this scale |
| `train_on_responses_only` | True | Only supervise assistant turns |

---

## Step 3 — DPO on H100

```bash
python train_dpo_0_8b.py \
  --sft-model Reza2kn/qwen-0_8b-agentjson-reducto-lora \
  --dpo-dataset Reza2kn/reducto-0_8b-dpo-targeted \
  --output-dir runs/0_8b_dpo \
  --push-to-hub Reza2kn/qwen-0_8b-agentjson-reducto-lora-dpo
```

DPO config:

| Param | Value | Why |
|-------|-------|-----|
| `beta` | 0.1 | Standard — same as 35B DPO |
| `epochs` | 2 | Sub-1B DPO needs fewer epochs; reward collapse risk at 3+ |
| `lr` | 5e-5 | Same as SFT — cautious |
| `max_length` | 2048 | DPO pairs are short (termination = one tool call + result + stop) |
| `ref_model` | None | TRL auto-creates frozen copy from SFT adapter |

---

## Step 4 — Benchmark the result

Merge LoRA → GGUF, serve locally, run against the same probes that measured the original failures.

```bash
# On Mac — merge and quantize
python benchmark/scripts/merge_lora_to_gguf.py \
  --base Qwen-0.8B-AgentJSON \
  --adapter Reza2kn/qwen-0_8b-agentjson-reducto-lora-dpo \
  --output local-models/Qwen-0.8B-ReductoLoRA-V2-Q6K.gguf \
  --quant Q6_K

# Serve (port 18083 to not conflict with existing models)
llama-server \
  --model local-models/Qwen-0.8B-ReductoLoRA-V2-Q6K.gguf \
  --port 18083 --parallel 4 --ctx-size 8192 &

# Add to models.py, then run probes
python benchmark/scripts/bench_param_probe.py --model qwen-0.8b-v2 --all-probes
python benchmark/scripts/bench_mcp_probe.py   --model qwen-0.8b-v2 --all-probes
```

---

## Expected outcomes

| Benchmark | Before (AgentJSON) | Before (ReductoLoRA) | Target after |
|-----------|-------------------|---------------------|--------------|
| Hard probes (22×3=66) | 38/66 | ~38/66 | ≥52/66 |
| Standard probes (10×3=30) | 21/30 | 20/30 | ≥27/30 |
| MCP probes (7×3=21) | 9/21 | 9/21 | ≥15/21 |
| `classify_route_extract` | 0/3 | 0/3 | 2-3/3 |
| `upload_persist` time | ~458s | ~885s | <60s |
| `split_preserve` time | ~603s | ~916s | <60s |

The probe timing is the most direct loop-fix signal. If `upload_persist` still takes >120s after DPO, the termination training didn't take — check DPO loss curve for reward collapse.

---

## Failure modes and fixes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| SFT val loss plateaus at epoch 1 | 2,669 is too many epochs for this model | Reduce to 2 epochs |
| DPO reward collapses to 0 | beta too low or data has near-identical chosen/rejected | Increase beta to 0.15, re-check pairs with identical prompts |
| Probe timing still >120s after training | Termination DPO didn't dominate — loop priors too strong | Add 669 more termination pairs and re-run DPO only |
| `citations_bbox` still 0/3 | A1 cluster underfit — model ignores citations param | Run 200 more A1 examples with `--target +200` and re-SFT |
| `classify_route_extract` still 0/3 | C1 cluster didn't converge — need more weight | C1 is already 1.8× weighted; if still failing, increase to 2.5× in gen script |
| ReductoLoRA-style regression on simple probes | lr too high — AgentJSON priors overwritten | Drop lr to 2e-5 and retrain from SFT checkpoint |

---

## Quick validation (before full benchmark)

Run just the 5 worst-performing probes to check improvement before committing to a full run:

```bash
python benchmark/scripts/bench_mcp_probe.py \
  --model qwen-0.8b-v2 \
  --probes classify_route_extract,upload_persist_array_extract,agentic_parse_citations,dual_doc_fan_out,extract_then_edit_form
```

If `classify_route_extract` is still 0/3 and timing on `upload_persist` is still >300s, training didn't take. Check loss curves before running the full benchmark suite.
