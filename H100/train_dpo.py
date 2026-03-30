#!/usr/bin/env python3
"""
train_dpo.py — DPO fine-tune on top of the SFT adapter to fix the repetition-loop failure.

Must run AFTER train_35b.py (SFT). Loads the SFT LoRA adapter and continues training
with preference pairs: chosen = final text answer, rejected = repeat-tool-call loop.

Target hardware : 1× H100 SXM 80GB
Expected runtime: ~30–45 minutes (1,727 pairs × 2 epochs)

Key design decisions:
  • beta=0.1          — standard KL penalty; controls how far DPO diverges from SFT.
                        if chosen/rejected rewards both collapse toward -inf, drop to 0.05.
  • lr=5e-5           — 4× lower than SFT. DPO signal is dense; high LR causes instability.
  • 2 epochs          — DPO converges fast on small, high-quality paired datasets.
  • load_in_16bit     — same as SFT; avoids quantization error compounding in MoE layers.
  • train_on_responses_only=False — DPO loss is computed differently from SFT;
                        TRL's DPOTrainer handles its own masking internally.

Usage:
    python train_dpo.py
    python train_dpo.py --sft-model Reza2kn/qwen35-reducto-lora --push-to-hub Reza2kn/qwen35-reducto-lora-dpo
    python train_dpo.py --beta 0.05   # if rewards collapse
"""

import os, sys, json, argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft-model",    default="Reza2kn/qwen35-reducto-lora",
                   help="HF repo or local path to the SFT LoRA adapter")
    p.add_argument("--base-model",   default="unsloth/Qwen3.5-35B-A3B",
                   help="Base model the SFT adapter was trained on")
    p.add_argument("--dpo-dataset",  default="Reza2kn/reducto-dpo-termination",
                   help="HF dataset repo containing data/dpo_pairs.jsonl")
    p.add_argument("--output-dir",   default="runs/dpo")
    p.add_argument("--max-seq-len",  type=int,   default=2048)
    p.add_argument("--epochs",       type=int,   default=2)
    p.add_argument("--batch-size",   type=int,   default=1)
    p.add_argument("--grad-accum",   type=int,   default=8,
                   help="Effective batch = batch × grad_accum. DPO needs larger effective batch.")
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--beta",         type=float, default=0.1,
                   help="DPO KL penalty. Drop to 0.05 if rewards collapse.")
    p.add_argument("--push-to-hub",  default=None)
    return p.parse_args()


def load_dpo_pairs(dataset_repo: str) -> list[dict]:
    """Pull DPO pairs from HF dataset, return as list of dicts."""
    print(f"Loading DPO pairs from {dataset_repo}")
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=dataset_repo,
            filename="data/dpo_pairs.jsonl",
            repo_type="dataset",
        )
        pairs = [json.loads(l) for l in Path(local).read_text().splitlines() if l.strip()]
        print(f"  Loaded {len(pairs):,} preference pairs")
        return pairs
    except Exception as e:
        sys.exit(f"ERROR: Could not load DPO dataset from {dataset_repo}: {e}\n"
                 "Run Step 0 in H100_TRAINING.md to upload the dataset first.")


def format_dpo_row(row: dict, tokenizer) -> dict | None:
    """
    Convert one DPO pair from messages format to the string format TRL expects.

    DPO data structure:
      prompt   : list of messages ending with the last tool result
      chosen   : [{"role": "assistant", "content": "final answer text", "tool_calls": []}]
      rejected : [{"role": "assistant", "content": null, "tool_calls": [{repeat call}]}]
      tools    : list of tool definitions (needed for chat template)
    """
    try:
        prompt_msgs = row["prompt"]
        chosen_msg  = row["chosen"][0]
        rejected_msg = row["rejected"][0]
        tools = row.get("tools")

        # Apply chat template to prompt (stops before the final assistant turn)
        prompt_str = tokenizer.apply_chat_template(
            prompt_msgs,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,   # add the opening <|im_start|>assistant\n
            enable_thinking=False,
        )

        # Format chosen: final text response
        chosen_str = tokenizer.apply_chat_template(
            prompt_msgs + [chosen_msg],
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )[len(prompt_str):]   # strip the prompt prefix, keep only the response

        # Format rejected: repeated tool call
        # Rejected has tool_calls → need to render it as a tool-call turn
        rejected_str = tokenizer.apply_chat_template(
            prompt_msgs + [rejected_msg],
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )[len(prompt_str):]

        if not chosen_str.strip() or not rejected_str.strip():
            return None

        return {
            "prompt":   prompt_str,
            "chosen":   chosen_str,
            "rejected": rejected_str,
        }
    except Exception as e:
        return None


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("UNSLOTH_MOE_BACKEND", "unsloth_triton")

    # ── Load base model + SFT adapter ────────────────────────────────────────
    from unsloth import FastLanguageModel

    print(f"\nLoading base model: {args.base_model}")
    print(f"SFT adapter      : {args.sft_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = args.base_model,
        max_seq_length = args.max_seq_len,
        load_in_4bit   = False,
        load_in_16bit  = True,
        full_finetuning= False,
    )

    # Load SFT LoRA weights on top of base
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.sft_model, is_trainable=True)
    print("  ✓ SFT adapter loaded")

    tokenizer.padding_side = "left"   # DPO needs left-padding for consistent reward computation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Load + format DPO pairs ───────────────────────────────────────────────
    raw_pairs = load_dpo_pairs(args.dpo_dataset)

    from datasets import Dataset
    formatted, skipped = [], 0
    for row in raw_pairs:
        r = format_dpo_row(row, tokenizer)
        if r:
            formatted.append(r)
        else:
            skipped += 1

    if skipped:
        print(f"  Skipped {skipped} malformed pairs")
    print(f"  Using {len(formatted):,} pairs for DPO training")

    # 90/10 split for monitoring
    split_idx   = int(len(formatted) * 0.9)
    train_pairs = Dataset.from_list(formatted[:split_idx])
    val_pairs   = Dataset.from_list(formatted[split_idx:])
    print(f"  Train: {len(train_pairs):,}  Val: {len(val_pairs):,}")

    # ── DPO Trainer ──────────────────────────────────────────────────────────
    from trl import DPOTrainer, DPOConfig

    effective_batch = args.batch_size * args.grad_accum
    total_steps     = (len(train_pairs) // effective_batch) * args.epochs

    print(f"\nDPO training plan")
    print(f"  Beta            : {args.beta}  (KL penalty vs SFT reference)")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Effective batch : {args.batch_size} × {args.grad_accum} = {effective_batch}")
    print(f"  LR              : {args.lr}  (cosine decay)")
    print(f"  Steps           : ~{total_steps:,}")

    cfg = DPOConfig(
        output_dir                  = str(output_dir),
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.1,
        bf16                        = True,
        fp16                        = False,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        max_grad_norm               = 1.0,
        beta                        = args.beta,
        max_length                  = args.max_seq_len,
        max_prompt_length           = args.max_seq_len // 2,
        logging_steps               = 5,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        save_total_limit            = 2,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        report_to                   = "none",
        seed                        = 42,
        # DPO-specific: use the SFT model as reference policy.
        # TRL will create a frozen copy of the model automatically.
        is_encoder_decoder          = False,
    )

    trainer = DPOTrainer(
        model         = model,
        ref_model     = None,   # None = use model copy as reference (default, correct for LoRA DPO)
        args          = cfg,
        train_dataset = train_pairs,
        eval_dataset  = val_pairs,
        tokenizer     = tokenizer,
    )

    print("\nTraining…  (tail -f /tmp/train_dpo.log for progress)")
    print("Watch for: rewards/chosen and rewards/rejected in logs.")
    print("  Good: rewards/chosen rises, rewards/rejected falls, margin grows positive.")
    print("  Bad : both collapse toward -inf → run with --beta 0.05\n")

    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────────
    adapter_path = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n✓ DPO adapter saved → {adapter_path}")

    merged_path = output_dir / "merged_16bit"
    print(f"Saving merged 16-bit model → {merged_path}")
    # Unsloth merge works on PeftModel too
    model.save_pretrained_merged(str(merged_path), tokenizer, save_method="merged_16bit")
    print(f"✓ Merged model saved → {merged_path}")

    if args.push_to_hub:
        print(f"\nPushing DPO adapter → {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub, private=True)
        tokenizer.push_to_hub(args.push_to_hub, private=True)
        print(f"  https://huggingface.co/{args.push_to_hub}")

    print("\nDone. Next: benchmark with bench_mcp_probe.py (target ≥16/21 MCP, zero loops).")


if __name__ == "__main__":
    main()
