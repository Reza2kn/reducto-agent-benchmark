#!/usr/bin/env python3
"""
gen_dpo_0_8b.py — DPO preference pairs targeted at the 0.8B AgentJSON failures.

Three distinct DPO failure types (all observed in probe traces):

  TYPE 1 — Termination (669 pairs)
    Prompt  : [system, user, assistant(tool_calls), tool_result(s)]
    Chosen  : assistant writes final text, no more tools
    Rejected: assistant repeats last tool call (the loop failure, 600-900s runs)

  TYPE 2 — Param format (201 pairs)
    Prompt  : [system, user]
    Chosen  : assistant calls reducto_parse with filter_blocks=["Header","Footer"]
    Rejected: assistant calls reducto_parse with filter_blocks='["Header","Footer"]'
    Also covers agentic_scopes — same pattern, native array vs JSON string.

  TYPE 3 — Tool routing (99 pairs)
    Prompt  : [system, user asking to process unknown doc by type]
    Chosen  : assistant calls reducto_classify first
    Rejected: assistant calls reducto_extract directly (skips classify)

Rejected turns for TYPE 2 and TYPE 3 are synthetic (no LLM call needed) —
we know exactly what the wrong behaviour looks like from the probe traces.

Target  : 969 pairs
Output  : benchmark/data/dpo_0_8b/.checkpoint_dpo_0_8b.jsonl
Teacher : Claude Haiku 4.5 + thinking (for chosen turns)

Usage:
    python gen_dpo_0_8b.py               # full run
    python gen_dpo_0_8b.py --target 30   # quick test
    python gen_dpo_0_8b.py --dry-run
    python gen_dpo_0_8b.py --resume
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

OUTPUT_DIR = Path("benchmark/data/dpo_0_8b")
CHECKPOINT = OUTPUT_DIR / ".checkpoint_dpo_0_8b.jsonl"

TEACHER_NAME   = "Claude Haiku 4.5 + thinking"
REPHRASE_MODEL = "claude-haiku-4-5-20251001"

try:
    from gen_mcp_r3_gaps import (
        MCP_TOOL_SCHEMAS,
        MCP_SYSTEM_PROMPT,
        _synthetic_result,
        _get_final_text,
        run_through_teacher,
        generate_variations,
    )
    from gen_dpo_termination import _make_rejected_tool_call
    from gen_0_8b_targeted import CLUSTER_D_TERMINATION, CLUSTER_C1_CLASSIFY_ROUTE
except ImportError as e:
    print(f"ERROR: {e}")
    print("Run from benchmark/scripts/ directory.")
    sys.exit(1)

_random = random.Random(42)

# ---------------------------------------------------------------------------
# Type 1: Termination — reuses gap9 mechanism from R3 DPO
# Base scenarios: CLUSTER_D_TERMINATION + CLUSTER_C1_CLASSIFY_ROUTE
# (both are completion points where the 0.8B loops)
# ---------------------------------------------------------------------------

TERMINATION_SCENARIOS = CLUSTER_D_TERMINATION + CLUSTER_C1_CLASSIFY_ROUTE

# ---------------------------------------------------------------------------
# Type 2: Param format — chosen=array, rejected=JSON string
# Base prompts that naturally elicit filter_blocks or agentic_scopes calls
# ---------------------------------------------------------------------------

FORMAT_SCENARIOS = [
    # filter_blocks: native array vs serialised string
    ("fmt_filter_two_blocks",
     "Parse https://example.com/report.pdf and remove headers and footers."),
    ("fmt_filter_three_blocks",
     "Parse https://example.com/annual.pdf. Strip headers, footers, "
     "and page numbers."),
    ("fmt_filter_four_blocks",
     "Parse https://cdn.reducto.ai/samples/fidelity-example.pdf. "
     "Remove decorative elements: headers, footers, page numbers, watermarks."),
    ("fmt_filter_natural",
     "Parse https://example.com/kb.pdf for embedding. Clean text only — "
     "remove all noise blocks."),
    ("fmt_filter_specific",
     "Parse https://example.com/legal.pdf. Exclude headers, footers, "
     "section headers, and signature blocks."),
    ("fmt_filter_persist",
     "Parse https://example.com/doc.pdf, filter out headers and footers, "
     "persist results."),
    ("fmt_filter_with_agentic",
     "Parse https://example.com/tables.pdf. Remove headers and footers. "
     "Apply agentic correction for tables."),
    ("fmt_filter_minimal",
     "Parse https://example.com/brief.pdf — strip page headers and footers."),
    # agentic_scopes: native array vs comma string
    ("fmt_agentic_table",
     "Parse https://example.com/financials.pdf with agentic table correction."),
    ("fmt_agentic_table_figure",
     "Parse https://example.com/report.pdf with agentic correction for "
     "both tables and figures."),
    ("fmt_agentic_text_table",
     "Parse https://example.com/scan.pdf with agentic correction for "
     "text and tables."),
    ("fmt_agentic_figure_only",
     "Parse https://example.com/charts.pdf with agentic figure extraction."),
    ("fmt_agentic_persist",
     "Parse https://cdn.reducto.ai/samples/fidelity-example.pdf with "
     "agentic table correction, persist results."),
    ("fmt_agentic_filter_combined",
     "Parse https://example.com/messy.pdf. Filter headers/footers AND "
     "apply agentic table correction."),
]

# ---------------------------------------------------------------------------
# Type 3: Tool routing — chosen=classify-first, rejected=extract-direct
# Same base prompts as c1_classify_route — different DPO construction
# ---------------------------------------------------------------------------

ROUTING_SCENARIOS = [
    ("rt_classify_first_basic",
     "Process https://example.com/unknown.pdf: classify it, then extract "
     "appropriate fields."),
    ("rt_classify_first_triage",
     "Triage https://example.com/intake.pdf — it could be an invoice, "
     "contract, or statement. Extract the right fields based on type."),
    ("rt_classify_then_extract",
     "I don't know what's in https://example.com/doc.pdf. Figure out "
     "the document type first, then pull out the relevant data."),
    ("rt_classify_route_financial",
     "Unknown financial document at https://example.com/financial.pdf. "
     "Identify what it is, then extract appropriately."),
    ("rt_classify_route_explicit",
     "Two-step: classify https://example.com/filing.pdf, then extract "
     "based on the document category."),
    ("rt_classify_natural",
     "Process https://example.com/client_doc.pdf — "
     "what is it, and what are the key fields?"),
    ("rt_classify_then_schema",
     "https://example.com/intake_001.pdf could be many things. "
     "Classify it first before deciding what to extract."),
]


# ---------------------------------------------------------------------------
# Synthetic rejected turn constructors
# ---------------------------------------------------------------------------

def _make_rejected_format_call(chosen_msg: dict) -> dict:
    """
    Convert chosen tool call (native array) → rejected (JSON string).
    Targets: filter_blocks and agentic_scopes array→string regression.
    """
    tool_calls = chosen_msg.get("tool_calls", [])
    if not tool_calls:
        return chosen_msg

    new_calls = []
    for tc in tool_calls:
        args_str = tc.get("function", {}).get("arguments", "{}")
        try:
            args = json.loads(args_str)
        except Exception:
            new_calls.append(tc)
            continue

        modified = False
        # Convert filter_blocks array → JSON string
        if isinstance(args.get("filter_blocks"), list):
            args["filter_blocks"] = json.dumps(args["filter_blocks"])
            modified = True
        # Convert agentic_scopes array → comma-separated string
        if isinstance(args.get("agentic_scopes"), list):
            args["agentic_scopes"] = ",".join(args["agentic_scopes"])
            modified = True

        if not modified:
            # If neither param was an array, introduce the string form anyway
            # to ensure there's always a meaningful chosen/rejected contrast
            if "filter_blocks" not in args and "agentic_scopes" not in args:
                new_calls.append(tc)
                continue

        new_tc = {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": tc.get("function", {}).get("name", ""),
                "arguments": json.dumps(args),
            },
        }
        new_calls.append(new_tc)

    if not new_calls:
        return chosen_msg
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": new_calls,
    }


def _make_rejected_routing_call(prompt_msgs: list) -> dict:
    """
    Rejected routing turn: call reducto_extract directly without classifying.
    Extracted from the user message — use the document URL as input.
    """
    user_content = next(
        (m["content"] for m in prompt_msgs if m["role"] == "user"), ""
    )
    # Extract first URL from user message
    import re
    urls = re.findall(r'https?://\S+\.pdf[^\s"]*', user_content)
    doc_url = urls[0] if urls else "https://cdn.reducto.ai/samples/fidelity-example.pdf"

    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": "reducto_extract",
                "arguments": json.dumps({
                    "input": doc_url,
                    "schema": json.dumps({
                        "document_type": {"type": "string"},
                        "key_fields": {"type": "object"},
                    }),
                }),
            },
        }],
    }


# ---------------------------------------------------------------------------
# DPO pair generators per type
# ---------------------------------------------------------------------------

def generate_termination_pair(
    model, prompt: str, dry_run: bool = False
) -> Optional[dict]:
    """TYPE 1: tool_result → stop (chosen) vs loop (rejected)."""
    initial = run_through_teacher(model, prompt, "dpo_termination", dry_run)
    if initial is None:
        return None
    tool_calls = initial["messages"][-1].get("tool_calls", [])
    if not tool_calls:
        return None

    tool_result_msgs, results = [], []
    for tc in tool_calls:
        tool_name = tc.get("function", {}).get("name", "")
        try:
            args = json.loads(tc.get("function", {}).get("arguments", "{}"))
        except Exception:
            args = {}
        result = _synthetic_result(tool_name, args)
        results.append(result)
        tool_result_msgs.append({
            "role": "tool",
            "tool_call_id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
            "content": json.dumps(result),
        })

    prompt_msgs = [
        initial["messages"][0],  # system
        initial["messages"][1],  # user
        initial["messages"][2],  # assistant tool_calls
        *tool_result_msgs,
    ]
    final_text = _get_final_text(prompt, tool_calls, results, dry_run)
    chosen   = [{"role": "assistant", "content": final_text, "tool_calls": []}]
    rejected = [_make_rejected_tool_call(tool_calls[-1])]

    return {
        "prompt":   prompt_msgs,
        "chosen":   chosen,
        "rejected": rejected,
        "tools":    MCP_TOOL_SCHEMAS,
        "metadata": {
            "source":   "dpo_0_8b",
            "teacher":  model.display,
            "dpo_type": "termination",
        },
    }


def generate_format_pair(
    model, prompt: str, dry_run: bool = False
) -> Optional[dict]:
    """TYPE 2: native array (chosen) vs JSON string (rejected)."""
    initial = run_through_teacher(model, prompt, "dpo_format", dry_run)
    if initial is None:
        return None
    chosen_msg = initial["messages"][-1]
    tool_calls  = chosen_msg.get("tool_calls", [])
    if not tool_calls:
        return None

    # Check chosen actually has filter_blocks or agentic_scopes as array
    has_array = False
    for tc in tool_calls:
        try:
            args = json.loads(tc.get("function", {}).get("arguments", "{}"))
            if isinstance(args.get("filter_blocks"), list) or \
               isinstance(args.get("agentic_scopes"), list):
                has_array = True
                break
        except Exception:
            pass
    if not has_array:
        return None   # teacher didn't use array params — skip

    prompt_msgs = [
        initial["messages"][0],  # system
        initial["messages"][1],  # user
    ]
    chosen   = [chosen_msg]
    rejected = [_make_rejected_format_call(chosen_msg)]

    # If rejected is identical to chosen (no array params to corrupt), skip
    if (json.dumps(chosen, sort_keys=True) ==
            json.dumps(rejected, sort_keys=True)):
        return None

    return {
        "prompt":   prompt_msgs,
        "chosen":   chosen,
        "rejected": rejected,
        "tools":    MCP_TOOL_SCHEMAS,
        "metadata": {
            "source":   "dpo_0_8b",
            "teacher":  model.display,
            "dpo_type": "param_format",
        },
    }


def generate_routing_pair(
    model, prompt: str, dry_run: bool = False
) -> Optional[dict]:
    """TYPE 3: classify-first (chosen) vs extract-direct (rejected)."""
    initial = run_through_teacher(model, prompt, "dpo_routing", dry_run)
    if initial is None:
        return None
    chosen_msg = initial["messages"][-1]
    tool_calls  = chosen_msg.get("tool_calls", [])
    if not tool_calls:
        return None

    # Chosen must start with classify (otherwise teacher didn't route)
    first_tool = tool_calls[0].get("function", {}).get("name", "")
    if first_tool != "reducto_classify":
        return None   # teacher didn't classify first — skip this variation

    prompt_msgs = [
        initial["messages"][0],  # system
        initial["messages"][1],  # user
    ]
    chosen   = [chosen_msg]
    rejected = [_make_rejected_routing_call(prompt_msgs)]

    return {
        "prompt":   prompt_msgs,
        "chosen":   chosen,
        "rejected": rejected,
        "tools":    MCP_TOOL_SCHEMAS,
        "metadata": {
            "source":   "dpo_0_8b",
            "teacher":  model.display,
            "dpo_type": "tool_routing",
        },
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint: Path) -> tuple[int, set]:
    seen: set = set()
    if not checkpoint.exists():
        return 0, seen
    count = 0
    for line in checkpoint.read_text().splitlines():
        try:
            d = json.loads(line)
            user_msg = next(
                (m["content"] for m in d.get("prompt", []) if m["role"] == "user"),
                ""
            )
            seen.add(user_msg[:120])
            count += 1
        except Exception:
            pass
    return count, seen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Target split: 669 termination + 201 format + 99 routing = 969
DPO_TYPE_TARGETS = {
    "termination": 669,
    "format":      201,
    "routing":      99,
}

DPO_GENERATORS = {
    "termination": (TERMINATION_SCENARIOS, generate_termination_pair),
    "format":      (FORMAT_SCENARIOS,      generate_format_pair),
    "routing":     (ROUTING_SCENARIOS,     generate_routing_pair),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",  type=int, default=969,
                        help="Total DPO pairs to generate (default: 969)")
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

    # Scale targets proportionally if --target overrides default
    scale = args.target / sum(DPO_TYPE_TARGETS.values())
    type_targets = {
        t: max(1, int(n * scale)) for t, n in DPO_TYPE_TARGETS.items()
    }

    print("=== 0.8B Targeted DPO Generator ===")
    print(f"  Teacher   : {TEACHER_NAME}")
    print(f"  Target    : {args.target} pairs")
    print(f"  Workers   : {args.workers}")
    print(f"  Resume    : {current} existing pairs")
    for t, n in type_targets.items():
        print(f"  {t:15s}: {n} pairs")
    print()

    generated = current
    _lock = threading.Lock()

    # Track per-type counts
    type_counts: dict[str, int] = {t: 0 for t in DPO_TYPE_TARGETS}
    if args.resume and CHECKPOINT.exists():
        for line in CHECKPOINT.read_text().splitlines():
            try:
                d = json.loads(line)
                t = d.get("metadata", {}).get("dpo_type", "")
                if t in type_counts:
                    type_counts[t] += 1
            except Exception:
                pass

    def process_one(dpo_type: str, probe_id: str, base_prompt: str) -> int:
        nonlocal generated
        scenarios, gen_fn = DPO_GENERATORS[dpo_type]

        with _lock:
            if generated >= args.target:
                return 0
            if type_counts[dpo_type] >= type_targets[dpo_type]:
                return 0

        variations = generate_variations(
            base_prompt,
            n=max(3, type_targets[dpo_type] // len(scenarios)),
            dry_run=args.dry_run,
        )
        if not variations:
            variations = [base_prompt]

        added = 0
        for variation in variations:
            with _lock:
                if generated >= args.target:
                    break
                if type_counts[dpo_type] >= type_targets[dpo_type]:
                    break
                if variation[:120] in seen_prompts:
                    continue
                seen_prompts.add(variation[:120])

            pair = gen_fn(teacher, variation, dry_run=args.dry_run)
            if pair is None:
                with _lock:
                    seen_prompts.discard(variation[:120])
                continue

            with _lock:
                with CHECKPOINT.open("a") as f:
                    f.write(json.dumps(pair) + "\n")
                generated += 1
                type_counts[dpo_type] += 1
                added += 1
                if generated % 50 == 0:
                    print(f"  [{generated}/{args.target}]  "
                          f"term={type_counts['termination']} "
                          f"fmt={type_counts['format']} "
                          f"rt={type_counts['routing']}")
        return added

    # Build work list — ordered termination first (most pairs), then format, routing
    work: list[tuple[str, str, str]] = []
    for dpo_type, (scenarios, _) in DPO_GENERATORS.items():
        for probe_id, prompt in scenarios:
            work.append((dpo_type, probe_id, prompt))
    random.shuffle(work)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_one, dt, pid, prompt): (dt, pid)
            for dt, pid, prompt in work
        }
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"  [worker error] {futures[fut]}: {e}")

    print(f"\n=== Done: {generated}/{args.target} DPO pairs → {CHECKPOINT} ===")
    for t, cnt in type_counts.items():
        print(f"  {t:15s}: {cnt:>5,} / {type_targets[t]}")


if __name__ == "__main__":
    main()
