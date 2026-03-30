#!/usr/bin/env python3
"""
upload_combined.py — Push the final merged SFT dataset to HF.

Uploads benchmark/data/combined_training/verified_{train,val}.jsonl
to Reza2kn/reducto-api-tool-calls.

Run from repo root after verify + merge is complete.
"""

import argparse, json
from pathlib import Path
from huggingface_hub import HfApi


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",  default="benchmark/data/combined_training")
    p.add_argument("--repo",      default="Reza2kn/reducto-api-tool-calls")
    p.add_argument("--message",   default=None, help="Custom commit message")
    return p.parse_args()


def main():
    args   = parse_args()
    src    = Path(args.data_dir)
    api    = HfApi()

    train_path = src / "verified_train.jsonl"
    val_path   = src / "verified_val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found. Run verify + merge first.")

    train_n = sum(1 for l in train_path.read_text().splitlines() if l.strip())
    val_n   = sum(1 for l in val_path.read_text().splitlines()   if l.strip()) if val_path.exists() else 0
    total   = train_n + val_n

    message = args.message or f"Update dataset: {train_n:,} train + {val_n:,} val = {total:,} total"
    print(f"Uploading to {args.repo}")
    print(f"  train: {train_n:,}  val: {val_n:,}  total: {total:,}")
    print(f"  commit: {message}\n")

    for local, remote in [(train_path, "data/verified_train.jsonl"),
                          (val_path,   "data/verified_val.jsonl")]:
        if not local.exists():
            print(f"  Skipping {remote} (not found)")
            continue
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=args.repo,
            repo_type="dataset",
            commit_message=message,
        )
        print(f"  ✅ {remote}")

    print(f"\nDone → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
