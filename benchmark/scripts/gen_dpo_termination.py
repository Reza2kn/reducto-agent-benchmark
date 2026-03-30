#!/usr/bin/env python3
"""
gen_dpo_termination.py — DPO preference pairs for chain termination.

Fixes the #1 ReductoLoRA failure: model loops 50-75× after getting a valid result.

Each pair:
  prompt   : [system, user, assistant(tool_calls), tool_result(s)]
  chosen   : assistant writes final text response — task complete, NO more tools
  rejected : assistant blindly repeats the last tool call — the exact loop failure

The DPO gradient directly penalises "call same tool again after getting a result"
and rewards "output final answer and stop."

Target  : 669 pairs
Output  : benchmark/data/dpo_termination/.checkpoint_dpo.jsonl
"""

import argparse
import json
import os
import random
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))
from models import PREMIUM_MODELS

OUTPUT_DIR = Path("benchmark/data/dpo_termination")
CHECKPOINT = OUTPUT_DIR / ".checkpoint_dpo.jsonl"

TEACHER_NAME   = "Claude Haiku 4.5 + thinking"
REPHRASE_MODEL = "google/gemini-3.1-flash-lite-preview"

# ---------------------------------------------------------------------------
# Import shared helpers from gap 9
# ---------------------------------------------------------------------------

try:
    from gen_mcp_r3_gaps import (
        GAP9_TERMINATION,
        MCP_TOOL_SCHEMAS,
        MCP_SYSTEM_PROMPT,
        _synthetic_result,
        _get_final_text,
        run_through_teacher,
        generate_variations,
    )
except ImportError as e:
    print(f"ERROR: could not import from gen_mcp_r3_gaps.py: {e}")
    print("Run from benchmark/scripts/ directory.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# DPO pair builder
# ---------------------------------------------------------------------------

def _make_rejected_tool_call(last_tool_call: dict) -> dict:
    """
    Construct the rejected assistant turn: blindly repeats the last tool call.
    This is the exact loop failure observed in ReductoLoRA Q6K:
      - 75 calls on upload_persist_array_extract
      - 77 calls on split_preserve_extract_range
    Same tool name, same args, fresh call_id (model doesn't recognise it already ran).
    """
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": last_tool_call.get("function", {}).get("name", ""),
                "arguments": last_tool_call.get("function", {}).get("arguments", "{}"),
            },
        }],
    }


def generate_dpo_pair(model, prompt: str, dry_run: bool = False) -> Optional[dict]:
    """
    Build one DPO preference pair:

      prompt   = [system, user, assistant(tool_calls), tool_result(s)]
      chosen   = [assistant(final_text)]          ← correct: stop after result
      rejected = [assistant(repeat_tool_call)]    ← wrong: loop failure
    """
    # Step 1: teacher generates tool calls
    initial = run_through_teacher(model, prompt, "gap9_dpo", dry_run)
    if initial is None:
        return None

    tool_calls = initial["messages"][-1].get("tool_calls", [])
    if not tool_calls:
        return None

    # Step 2: inject synthetic tool results (same helpers as gap9 SFT)
    tool_result_msgs = []
    results = []
    for tc in tool_calls:
        tool_name = tc.get("function", {}).get("name", "")
        args = {}
        try:
            args = json.loads(tc.get("function", {}).get("arguments", "{}"))
        except Exception:
            pass
        result = _synthetic_result(tool_name, args)
        results.append(result)
        tool_result_msgs.append({
            "role": "tool",
            "tool_call_id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
            "content": json.dumps(result),
        })

    # The prompt ends right after the last tool result —
    # this is the decision point: "do I call again or do I stop?"
    prompt_msgs = [
        initial["messages"][0],   # system
        initial["messages"][1],   # user
        initial["messages"][2],   # assistant (tool_calls)
        *tool_result_msgs,        # tool results
    ]

    # Step 3: chosen — write final answer, no more tools
    final_text = _get_final_text(prompt, tool_calls, results, dry_run)
    chosen = [{"role": "assistant", "content": final_text, "tool_calls": []}]

    # Step 4: rejected — repeat last tool call (synthetic, no LLM needed)
    rejected = [_make_rejected_tool_call(tool_calls[-1])]

    return {
        "prompt":   prompt_msgs,
        "chosen":   chosen,
        "rejected": rejected,
        "tools":    MCP_TOOL_SCHEMAS,
        "metadata": {
            "source":  "dpo_synthetic",
            "teacher": model.display,
            "round":   "dpo_termination",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint: Path) -> tuple[int, set]:
    seen = set()
    if not checkpoint.exists():
        return 0, seen
    count = 0
    for line in checkpoint.read_text().splitlines():
        try:
            d = json.loads(line)
            user_msg = next(
                (m["content"] for m in d.get("prompt", []) if m["role"] == "user"), ""
            )
            seen.add(user_msg[:120])
            count += 1
        except Exception:
            pass
    return count, seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",  type=int, default=669,
                        help="DPO pairs to generate (default: 669)")
    parser.add_argument("--resume",  action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_models = {m.display: m for m in PREMIUM_MODELS}
    teacher = all_models.get(TEACHER_NAME)
    if not teacher:
        print(f"ERROR: teacher '{TEACHER_NAME}' not found in PREMIUM_MODELS")
        sys.exit(1)

    current, seen_prompts = load_checkpoint(CHECKPOINT) if args.resume else (0, set())

    print("=== DPO Termination Pair Generator ===")
    print(f"  Teacher  : {TEACHER_NAME}")
    print(f"  Workers  : {args.workers}")
    print(f"  Target   : {args.target} pairs")
    print(f"  Resume   : {current} existing pairs")
    print(f"  Scenarios: {len(GAP9_TERMINATION)} base (termination + classify-route)")
    print(f"  Rejected : synthetic repeat of last tool call (no extra LLM call)")
    print()

    n_scenarios    = len(GAP9_TERMINATION)
    variations_per = max(3, (args.target - current) // n_scenarios)
    print(f"  ~{variations_per} variations per base scenario")

    generated = current
    _lock     = threading.Lock()
    scenarios = list(GAP9_TERMINATION)
    random.shuffle(scenarios)

    def process_one(probe_id: str, base_prompt: str) -> int:
        nonlocal generated

        with _lock:
            if generated >= args.target:
                return 0
            n_to_gen = min(variations_per, args.target - generated)

        variations = generate_variations(base_prompt, n_to_gen, dry_run=args.dry_run)
        if not variations:
            variations = [base_prompt]

        added = 0
        for variation in variations:
            with _lock:
                if generated >= args.target:
                    break
                if variation[:120] in seen_prompts:
                    continue
                seen_prompts.add(variation[:120])

            pair = generate_dpo_pair(teacher, variation, dry_run=args.dry_run)

            if pair is None:
                with _lock:
                    seen_prompts.discard(variation[:120])
                continue

            with _lock:
                with CHECKPOINT.open("a") as f:
                    f.write(json.dumps(pair) + "\n")
                generated += 1
                added     += 1
                if generated % 25 == 0:
                    print(f"  [{generated}/{args.target}] pairs written")

        return added

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, pid, prompt): pid
                   for pid, prompt in scenarios}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"  [worker error] {futures[fut]}: {e}")

    print(f"\n=== Done: {generated} DPO pairs in {CHECKPOINT} ===")


if __name__ == "__main__":
    main()
