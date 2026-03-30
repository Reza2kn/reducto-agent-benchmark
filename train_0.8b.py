#!/usr/bin/env python3
"""
train_0.8b.py — Optimized fine-tune of Qwen3.5-0.6B (AgentJSON) on Reducto API tool-call data.

Target hardware : 1× RTX 4090 24GB or any H100 (trivially fits either)
Precision       : bf16 LoRA
Expected runtime: ~20–40 min on RTX 4090, ~10–20 min on H100

Model choice: We fine-tune from the local Qwen3.5-0.6B "AgentJSON" weights
(already specialized for tool calling) rather than a raw base model. This gives
a better starting point — the model already understands tool-call JSON format,
and we're adding Reducto-specific API knowledge on top.

Architecture note: This model is Qwen3.5-Text, a hybrid linear/full-attention
architecture (linear attention every 4 layers with full attention). The LoRA
target modules (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
exist in ALL layer types — both linear-attention and full-attention layers use
the same projection names, so targeting is straightforward.

Key differences from 35B config:
  • r=64, rslora=True    — Higher rank for small model. A 0.6B model has limited
                           capacity; we need to give the adapter maximum expressive
                           power to learn the Reducto API schema. rslora stabilizes
                           gradients at high rank (uses alpha/sqrt(r) scaling).
  • 5 epochs             — Small models need more passes to converge on structured tasks.
  • lr=1e-4              — More conservative than 35B: at r=64 with a 0.6B model,
                           large LR causes format collapse after epoch 2.
  • batch=4, accum=4     — Effective BS=16. Small model fits easily; no VRAM pressure.
  • adamw_torch_fused    — VRAM is not the constraint here; fused AdamW is fastest.
  • max_seq_len=4096     — Tool traces are short; 4096 gives headroom for complex
                           multi-turn examples without wasting memory.

Usage:
    python train_0.8b.py
    python train_0.8b.py --push-to-hub Reza2kn/qwen-0.8b-reducto-lora
    python train_0.8b.py --model Qwen/Qwen3-0.6B-Instruct   # use base model instead
"""

import os, sys, json, argparse
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    # Default: fine-tune from the local AgentJSON weights.
    # Fallback: use --model Qwen/Qwen3-0.6B-Instruct or unsloth/Qwen3-0.6B-Instruct
    # to start from a clean base.
    p.add_argument("--model", default="local-models/Qwen-0.8B-AgentJSON",
                   help="HF model ID or local path. Use 'unsloth/Qwen3-0.6B-Instruct' for clean base.")
    p.add_argument("--data-dir",     default="benchmark/data/synthetic_training")
    p.add_argument("--hf-dataset",   default="Reza2kn/reducto-api-tool-calls")
    p.add_argument("--output-dir",   default="runs/qwen-0.8b-reducto-lora")
    p.add_argument("--max-seq-len",  type=int,   default=4096)
    p.add_argument("--lora-r",       type=int,   default=64,
                   help="LoRA rank. 64 with rslora=True is the sweet spot for 0.6B tool-call SFT.")
    p.add_argument("--epochs",       type=int,   default=5,
                   help="Small models need more epochs. Watch val loss from epoch 3.")
    p.add_argument("--batch-size",   type=int,   default=4)
    p.add_argument("--grad-accum",   type=int,   default=4,   # Effective BS = 16
                   help="Effective batch = batch_size × grad_accum.")
    p.add_argument("--lr",           type=float, default=1e-4,
                   help="Lower LR than 35B: high-rank small models are LR-sensitive.")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--push-to-hub",  default=None)
    return p.parse_args()


# ── Data helpers (shared with train_35b.py) ───────────────────────────────────

def normalise_tool_calls(messages: list) -> list:
    """Convert checkpoint's tool_call format to Qwen3.5's expected format."""
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


def load_jsonl(path: Path) -> list[dict]:
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


def build_dataset(rows: list[dict], tokenizer, max_seq_len: int):
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
                enable_thinking      = False,   # No <think> blocks in our training data
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


def ensure_data(data_dir: Path, hf_dataset: str) -> tuple[Path, Path]:
    train_path = data_dir / "verified_train.jsonl"
    val_path   = data_dir / "verified_val.jsonl"
    if train_path.exists():
        return train_path, val_path

    print(f"Local verified splits not found. Pulling from HF: {hf_dataset}")
    try:
        from huggingface_hub import hf_hub_download
        data_dir.mkdir(parents=True, exist_ok=True)
        for fname, dest in [("data/verified_train.jsonl", train_path),
                             ("data/verified_val.jsonl",   val_path)]:
            try:
                local = hf_hub_download(
                    repo_id=hf_dataset, filename=fname,
                    repo_type="dataset", local_dir=str(data_dir),
                )
                Path(local).rename(dest)
            except Exception as e:
                print(f"  Could not download {fname}: {e}")
    except ImportError:
        pass

    if not train_path.exists():
        sys.exit(
            "ERROR: verified_train.jsonl not found locally or on HF.\n"
            "Run: python benchmark/scripts/verify_synthetic_data.py\n"
        )
    return train_path, val_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
    from unsloth import FastLanguageModel

    # Handle local path vs HF model ID
    model_path = args.model
    if Path(model_path).exists() and not model_path.startswith("unsloth/"):
        model_path = str(Path(model_path).resolve())

    print(f"\nLoading {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = model_path,
        max_seq_length = args.max_seq_len,
        load_in_4bit   = False,
        load_in_16bit  = True,    # bf16
        full_finetuning= False,
        # If loading local weights, Unsloth may need dtype specified explicitly
        # dtype          = None,  # auto-detected as bfloat16 from config.json
    )

    # ── Tokenizer padding ────────────────────────────────────────────────────
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── LoRA ─────────────────────────────────────────────────────────────────
    print(f"\nApplying LoRA  r={args.lora_r}  alpha={args.lora_r}  rslora=True  dropout=0")
    model = FastLanguageModel.get_peft_model(
        model,
        r              = args.lora_r,
        lora_alpha     = args.lora_r,
        lora_dropout   = 0,          # No dropout: small focused dataset, low overfitting risk
        target_modules = [
            # Standard projection layers — present in both linear-attention
            # and full-attention layers of this hybrid architecture
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",
        random_state               = 42,
        use_rslora                 = True,   # Stabilizes high-rank (r=64) training.
                                             # alpha/sqrt(r) scaling keeps gradient
                                             # magnitudes stable as r increases.
        loftq_config               = None,
    )
    model.print_trainable_parameters()

    # ── Data ─────────────────────────────────────────────────────────────────
    train_path, val_path = ensure_data(data_dir, args.hf_dataset)
    train_rows = load_jsonl(train_path)
    val_rows   = load_jsonl(val_path) if val_path.exists() else []
    print(f"\nData  — Train: {len(train_rows):,}   Val: {len(val_rows):,}")

    train_ds = build_dataset(train_rows, tokenizer, args.max_seq_len)
    val_ds   = build_dataset(val_rows, tokenizer, args.max_seq_len) if val_rows else None
    print(f"After filtering — Train: {len(train_ds):,}  Val: {len(val_ds) if val_ds else 0:,}")

    # ── Trainer ──────────────────────────────────────────────────────────────
    from trl import SFTTrainer, SFTConfig

    effective_batch = args.batch_size * args.grad_accum
    total_steps     = (len(train_ds) // effective_batch) * args.epochs
    print(f"\nTraining plan")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Effective batch : {args.batch_size} × {args.grad_accum} = {effective_batch}")
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
        # Fused AdamW: VRAM not the constraint for 0.6B; fused is fastest on GPU
        optim                       = "adamw_torch_fused",
        weight_decay                = 0.01,
        max_grad_norm               = 1.0,
        logging_steps               = 10,
        eval_strategy               = "epoch" if val_ds else "no",
        save_strategy               = "epoch",
        save_total_limit            = 3,          # Keep all 3 epoch checkpoints to pick best
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

    # ── Response-only masking ────────────────────────────────────────────────
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part    = "<|im_start|>assistant\n",
    )

    # Sanity check
    sample_labels = trainer.train_dataset[0]["labels"]
    non_masked = sum(1 for l in sample_labels if l != -100)
    if non_masked == 0:
        print("\n⚠  WARNING: All labels are -100 on sample[0]. Check template markers.")
    else:
        print(f"\n✓ Label masking OK — {non_masked}/{len(sample_labels)} tokens unmasked on sample[0]")

    # ── Train ────────────────────────────────────────────────────────────────
    print("\nTraining…  (tail -f /tmp/train_0.8b.log for progress)")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────────
    adapter_path = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n✓ LoRA adapter saved → {adapter_path}")

    # Merged model: useful for converting back to GGUF for local benchmarking
    merged_path = output_dir / "merged_16bit"
    print(f"Saving merged 16-bit model → {merged_path}")
    model.save_pretrained_merged(str(merged_path), tokenizer, save_method="merged_16bit")
    print(f"✓ Merged model saved — convert to GGUF with: llama-quantize {merged_path} output.gguf Q6_K")

    if args.push_to_hub:
        print(f"\nPushing adapter to HF → {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub, private=True)
        tokenizer.push_to_hub(args.push_to_hub, private=True)
        print(f"  https://huggingface.co/{args.push_to_hub}")

    print("\nDone.")


if __name__ == "__main__":
    main()
