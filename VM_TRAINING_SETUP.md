# H100 Training Setup — Reducto API Fine-tune

## What We're Training

| Run | Model | Script | Expected time |
|-----|-------|--------|---------------|
| **35B** | `unsloth/Qwen3.5-35B-A3B` | `train_35b.py` | ~2–3 hrs |

**Dataset:** 9,764 verified train / 1,084 val — Reducto API tool-call traces, L1–L3 verified.
These land on the VM at `~/benchmark/data/synthetic_training/`.

---

## Step 1 — Install deps

```bash
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
pip install trl datasets huggingface_hub bitsandbytes accelerate
```

---

## Step 2 — HF login (needed to pull Qwen3.5-35B-A3B)

```bash
huggingface-cli login
# paste your HF token (Reza2kn account, read access is enough to pull the model)
```

---

## Step 3 — Create the training script

Create `train_35b.py` in the working directory with this content:

```python
#!/usr/bin/env python3
"""
train_35b.py — Optimized fine-tune of Qwen3.5-35B-A3B on Reducto API tool-call data.

Target hardware : 1x H100 SXM 80GB
Precision       : bf16 LoRA  (QLoRA disabled — unstable for Qwen3.5 MoE)
Expected runtime: ~2–3 hrs for ~9,700 examples x 3 epochs

Key decisions (don't change without reading the comments):
  train_on_responses_only  — only assistant/tool-call turns contribute to loss.
                             Without this the model learns to parrot user inputs.
  r=32                     — sweet spot for 35B MoE: enough capacity for the
                             structured tool-call format, safe for 80GB VRAM.
  enable_thinking=False    — Qwen3.5 has a CoT <think> mode. Our data has none.
                             Leaving it on creates garbage loss signal.
  packing=True             — tool traces are 300-800 tokens; packing gives 3-5x
                             throughput. Unsloth uses Flash Attention varlen to
                             prevent cross-example attention leakage.
  effective_batch=16       — batch=1 x grad_accum=16. Larger batches stabilise
                             gradient direction on structured-format tasks.
"""

import os, sys, json, argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="unsloth/Qwen3.5-35B-A3B")
    p.add_argument("--data-dir",     default="benchmark/data/synthetic_training")
    p.add_argument("--hf-dataset",   default="Reza2kn/reducto-api-tool-calls")
    p.add_argument("--output-dir",   default="runs/qwen35-reducto-lora")
    p.add_argument("--max-seq-len",  type=int,   default=2048)
    p.add_argument("--lora-r",       type=int,   default=32)
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--batch-size",   type=int,   default=1)
    p.add_argument("--grad-accum",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--push-to-hub",  default=None)
    return p.parse_args()


def normalise_tool_calls(messages):
    out = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tcs = []
            for tc in msg["tool_calls"]:
                args = tc.get("arguments", {})
                if isinstance(args, dict):
                    args = json.dumps(args)
                tcs.append({
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": args},
                })
            out.append({**msg, "tool_calls": tcs, "content": ""})
        else:
            out.append(msg)
    return out


def load_jsonl(path):
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def build_dataset(rows, tokenizer, max_seq_len):
    from datasets import Dataset
    texts, skipped = [], 0
    for row in rows:
        msgs  = normalise_tool_calls(row.get("messages", []))
        tools = row.get("tools")
        try:
            text = tokenizer.apply_chat_template(
                msgs,
                tools                = tools,
                tokenize             = False,
                add_generation_prompt= False,
                enable_thinking      = False,
            )
            if len(text) // 3 <= max_seq_len:
                texts.append({"text": text})
            else:
                skipped += 1
        except Exception:
            skipped += 1
    if skipped:
        print(f"  Skipped {skipped} examples (malformed / over seq-len)")
    return Dataset.from_list(texts)


def ensure_data(data_dir, hf_dataset):
    train_path = data_dir / "verified_train.jsonl"
    val_path   = data_dir / "verified_val.jsonl"
    if train_path.exists():
        return train_path, val_path
    print(f"Local data not found — pulling from HF: {hf_dataset}")
    try:
        from huggingface_hub import hf_hub_download
        data_dir.mkdir(parents=True, exist_ok=True)
        for fname, dest in [("data/verified_train.jsonl", train_path),
                             ("data/verified_val.jsonl",   val_path)]:
            try:
                local = hf_hub_download(repo_id=hf_dataset, filename=fname,
                                        repo_type="dataset", local_dir=str(data_dir))
                Path(local).rename(dest)
            except Exception as e:
                print(f"  Could not download {fname}: {e}")
    except ImportError:
        pass
    if not train_path.exists():
        sys.exit("ERROR: verified_train.jsonl not found locally or on HF.")
    return train_path, val_path


def main():
    args = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enable Unsloth's Triton MoE backend — ~2.5x faster than native on H100
    os.environ.setdefault("UNSLOTH_MOE_BACKEND", "unsloth_triton")

    from unsloth import FastLanguageModel

    print(f"\nLoading {args.model}  (bf16 LoRA, no QLoRA)")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = args.model,
        max_seq_length = args.max_seq_len,
        load_in_4bit   = False,
        load_in_16bit  = True,
        full_finetuning= False,
    )

    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"\nApplying LoRA  r={args.lora_r}  alpha={args.lora_r}  dropout=0")
    model = FastLanguageModel.get_peft_model(
        model,
        r              = args.lora_r,
        lora_alpha     = args.lora_r,
        lora_dropout   = 0,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",
        random_state               = 42,
        use_rslora                 = False,
    )
    model.print_trainable_parameters()

    train_path, val_path = ensure_data(data_dir, args.hf_dataset)
    train_rows = load_jsonl(train_path)
    val_rows   = load_jsonl(val_path) if val_path.exists() else []
    print(f"\nData  — Train: {len(train_rows):,}   Val: {len(val_rows):,}")

    train_ds = build_dataset(train_rows, tokenizer, args.max_seq_len)
    val_ds   = build_dataset(val_rows,   tokenizer, args.max_seq_len) if val_rows else None
    print(f"After filtering — Train: {len(train_ds):,}  Val: {len(val_ds) if val_ds else 0:,}")

    from trl import SFTTrainer, SFTConfig

    effective_batch = args.batch_size * args.grad_accum
    total_steps     = (len(train_ds) // effective_batch) * args.epochs
    print(f"\nTraining plan")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Effective batch : {args.batch_size} x {args.grad_accum} = {effective_batch}")
    print(f"  LR              : {args.lr}  (cosine, {args.warmup_ratio:.0%} warmup)")
    print(f"  Steps           : ~{total_steps:,}")

    cfg = SFTConfig(
        output_dir                  = str(output_dir),
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = args.warmup_ratio,
        bf16                        = True,
        fp16                        = False,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        max_grad_norm               = 1.0,
        logging_steps               = 10,
        eval_strategy               = "epoch" if val_ds else "no",
        save_strategy               = "epoch",
        save_total_limit            = 2,
        load_best_model_at_end      = bool(val_ds),
        metric_for_best_model       = "eval_loss" if val_ds else None,
        max_seq_length              = args.max_seq_len,
        dataset_text_field          = "text",
        packing                     = True,
        report_to                   = "none",
        seed                        = 42,
        dataloader_num_workers      = 4,
        dataset_num_proc            = 4,
    )

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        args          = cfg,
    )

    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part    = "<|im_start|>assistant\n",
    )

    # Sanity check — if this prints 0, the masking is broken
    sample_labels = trainer.train_dataset[0]["labels"]
    non_masked = sum(1 for l in sample_labels if l != -100)
    if non_masked == 0:
        print("\nWARNING: all labels are -100 on sample[0] — check template markers")
    else:
        print(f"\n✓ Label masking OK — {non_masked}/{len(sample_labels)} tokens unmasked on sample[0]")

    print("\nTraining...")
    trainer.train()

    adapter_path = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n✓ LoRA adapter saved -> {adapter_path}")

    merged_path = output_dir / "merged_16bit"
    print(f"Saving merged 16-bit model -> {merged_path}")
    model.save_pretrained_merged(str(merged_path), tokenizer, save_method="merged_16bit")
    print(f"✓ Merged model saved")

    if args.push_to_hub:
        print(f"\nPushing to HF -> {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub, private=True)
        tokenizer.push_to_hub(args.push_to_hub, private=True)
        print(f"  https://huggingface.co/{args.push_to_hub}")

    print("\nDone.")


if __name__ == "__main__":
    main()
```

---

## Step 4 — Create the data directory and drop in the verified files

```bash
mkdir -p benchmark/data/synthetic_training
# The two files to paste/upload here:
#   verified_train.jsonl   (9,764 examples, 64 MB)
#   verified_val.jsonl     (1,084 examples, 7 MB)
# They are on the Mac at:
#   ~/AI/reducto-agent-benchmark/benchmark/data/synthetic_training/
```

If you have SSH access from the Mac:
```bash
# Run this on the Mac to push data to the VM:
scp -P <PORT> \
  benchmark/data/synthetic_training/verified_train.jsonl \
  benchmark/data/synthetic_training/verified_val.jsonl \
  root@<VM_IP>:~/benchmark/data/synthetic_training/
```

Alternatively the script will auto-pull from `Reza2kn/reducto-api-tool-calls` on HF
if the local files aren't found — but the HF dataset may not have the latest splits yet.

---

## Step 5 — Launch training

```bash
# Recommended: run in tmux so it survives disconnects
tmux new -s train

python train_35b.py \
  --push-to-hub Reza2kn/qwen35-reducto-lora \
  2>&1 | tee /tmp/train_35b.log
```

Monitor in a second pane:
```bash
tmux new-window
watch -n 30 tail -20 /tmp/train_35b.log
```

---

## Expected output

```
Loading unsloth/Qwen3.5-35B-A3B  (bf16 LoRA, no QLoRA)
trainable params: 83,886,080 || all params: 36,742,914,048 || trainable%: 0.2284

Data  — Train: 9,764   Val: 1,084
After filtering — Train: 9,764  Val: 1,084

Training plan
  Epochs          : 3
  Effective batch : 1 x 16 = 16
  LR              : 0.0002  (cosine, 5% warmup)
  Steps           : ~1,830

✓ Label masking OK — 312/2048 tokens unmasked on sample[0]

Training...
{'loss': 1.42, 'grad_norm': 0.8, 'learning_rate': 1.2e-05, 'epoch': 0.09}
...
```

Loss should drop from ~1.4 → ~0.3–0.5 by epoch 3. If val loss starts rising after epoch 2, the adapter at epoch 2 is the keeper.

---

## After training — what you get

```
runs/qwen35-reducto-lora/
├── lora_adapter/          # LoRA weights only (~500MB) — pushed to HF
└── merged_16bit/          # Full merged model in bf16 (~70GB) — for GGUF conversion
```

Convert merged to GGUF for local benchmarking:
```bash
# On the VM or back on Mac with llama.cpp:
llama-quantize runs/qwen35-reducto-lora/merged_16bit output_q4.gguf Q4_K_M
```

---

## Data note

Current verified set is 9,764 train examples (L1+L2+L3 quality filters).
Full generation finishes in ~4–5 hrs → expected ~18,000 verified total.
We're training on the partial set now to get a benchmark signal tonight.
Re-train on the full set after generation completes for the final model.
