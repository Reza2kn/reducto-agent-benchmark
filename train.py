#!/usr/bin/env python3
"""
Fine-tune Qwen3.5-35B-A3B on the Reducto API tool-call dataset using Unsloth.

Target hardware : 1× H100 SXM 80GB (Vast.ai)
Precision       : bf16 LoRA  (QLoRA/4-bit is NOT recommended for Qwen3.5 MoE)
Expected runtime: ~40–55 min for 16,900 examples × 3 epochs

Usage:
    python train.py
    python train.py --push-to-hub Reza2kn/qwen35-reducto-lora
    python train.py --data-dir /path/to/data --epochs 5 --lora-r 32
"""

import os, sys, json, argparse
from pathlib import Path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="unsloth/Qwen3.5-35B-A3B")
    p.add_argument("--data-dir",     default="benchmark/data/synthetic_training",
                   help="Dir with verified_train.jsonl + verified_val.jsonl")
    p.add_argument("--hf-dataset",   default="Reza2kn/reducto-api-tool-calls",
                   help="HF dataset repo to pull verified splits from if local files missing")
    p.add_argument("--output-dir",   default="runs/qwen35-reducto-lora")
    p.add_argument("--max-seq-len",  type=int,   default=2048)
    p.add_argument("--lora-r",       type=int,   default=16)
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--batch-size",   type=int,   default=1,
                   help="Per-device batch size. Keep at 1 for H100 80GB bf16.")
    p.add_argument("--grad-accum",   type=int,   default=32,
                   help="Gradient accumulation → effective batch = batch_size × grad_accum")
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--push-to-hub",  default=None,
                   help="HF repo to push LoRA adapter after training")
    return p.parse_args()


# ── Data helpers ──────────────────────────────────────────────────────────────

def normalise_tool_calls(messages: list) -> list:
    """
    Our checkpoint stores tool_calls as {"name": ..., "arguments": {...}}.
    Qwen3.5's chat template expects {"type": "function", "function": {"name": ..., "arguments": "<json string>"}}.
    """
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

    texts = []
    skipped = 0
    for row in rows:
        msgs  = normalise_tool_calls(row.get("messages", []))
        tools = row.get("tools")
        try:
            text = tokenizer.apply_chat_template(
                msgs,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
            )
            # Drop examples that would be truncated (rough token estimate: chars/3)
            if len(text) // 3 <= max_seq_len:
                texts.append({"text": text})
            else:
                skipped += 1
        except Exception:
            skipped += 1

    if skipped:
        print(f"  Skipped {skipped} examples (malformed or over max_seq_len)")
    return Dataset.from_list(texts)


def ensure_data(data_dir: Path, hf_dataset: str) -> tuple[Path, Path]:
    """
    Return (train_path, val_path). Downloads from HF if not present locally.
    """
    train_path = data_dir / "verified_train.jsonl"
    val_path   = data_dir / "verified_val.jsonl"

    if train_path.exists():
        return train_path, val_path

    # Try pulling from HF dataset repo
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
            "Either run verify_synthetic_data.py first, or wait for generation + verification to complete."
        )

    return train_path, val_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Model + tokenizer ────────────────────────────────────────────────────
    from unsloth import FastLanguageModel

    print(f"\nLoading {args.model}")
    print("  Precision: bf16 LoRA  (QLoRA disabled — unstable for Qwen3.5 MoE)")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = args.model,
        max_seq_length  = args.max_seq_len,
        load_in_4bit    = False,   # QLoRA not recommended for MoE
        load_in_16bit   = True,    # bf16
        full_finetuning = False,
    )

    # ── LoRA ─────────────────────────────────────────────────────────────────
    print(f"\nApplying LoRA  r={args.lora_r}  alpha={args.lora_r}")
    model = FastLanguageModel.get_peft_model(
        model,
        r           = args.lora_r,
        lora_alpha  = args.lora_r,   # Unsloth recommends alpha = r, not 2×r
        lora_dropout = 0,
        target_modules = [
            # Attention
            "q_proj", "k_proj", "v_proj", "o_proj",
            # Expert FFNs — don't skip these, tool-call patterns live here
            "gate_proj", "up_proj", "down_proj",
        ],
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",  # more VRAM-efficient than True
        random_state               = 42,
    )
    model.print_trainable_parameters()

    # ── Data ─────────────────────────────────────────────────────────────────
    train_path, val_path = ensure_data(data_dir, args.hf_dataset)

    print(f"\nLoading data")
    train_rows = load_jsonl(train_path)
    val_rows   = load_jsonl(val_path) if val_path.exists() else []
    print(f"  Train: {len(train_rows):,}   Val: {len(val_rows):,}")

    train_ds = build_dataset(train_rows, tokenizer, args.max_seq_len)
    val_ds   = build_dataset(val_rows,   tokenizer, args.max_seq_len) if val_rows else None
    print(f"  After template + length filter — Train: {len(train_ds):,}  Val: {len(val_ds) if val_ds else 0:,}")

    # ── Trainer ──────────────────────────────────────────────────────────────
    from trl import SFTTrainer, SFTConfig

    effective_batch = args.batch_size * args.grad_accum
    total_steps     = (len(train_ds) // effective_batch) * args.epochs
    print(f"\nTraining plan")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Effective batch : {args.batch_size} × {args.grad_accum} accum = {effective_batch}")
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
        packing                     = True,   # pack short examples → fewer padding tokens
        report_to                   = "none",
        seed                        = 42,
    )

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        args          = cfg,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print("\nTraining…")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────────
    adapter_path = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n✓ LoRA adapter saved → {adapter_path}")

    if args.push_to_hub:
        print(f"Pushing to HF → {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub, private=True)
        tokenizer.push_to_hub(args.push_to_hub, private=True)
        print(f"  https://huggingface.co/{args.push_to_hub}")

    print("\nDone.")


if __name__ == "__main__":
    main()
