"""
Reducto Agent Matrix Benchmark
================================
Tests every model in the registry against the Reducto API via a real
multi-turn agentic tool-calling loop — no framework wrapper, just raw model
APIs driving actual Reducto calls.

This is the stress test: which models can use Reducto tools correctly?
- Some will hallucinate results instead of calling tools
- Some don't support function calling at all → error captured
- Some will call wrong endpoints or pass bad params
- The best ones will parse → extract → return structured ground truth

Run:
    cd benchmark/scripts
    python bench_agent_matrix.py                      # all models, all caps
    python bench_agent_matrix.py --model qwen3-32b    # single model (substring match)
    python bench_agent_matrix.py --capability extract  # single capability
    python bench_agent_matrix.py --premium-only        # skip OpenRouter models
"""

import asyncio
import json
import os
import sys
import time
import argparse
import traceback
from typing import Optional

import httpx

# Lazy-import; handled gracefully if missing
try:
    import openai
except ImportError:
    openai = None
try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None

sys.path.insert(0, os.path.dirname(__file__))
from bench_utils import (
    BenchmarkScore, save_result,
    evaluate_extract_output, evaluate_parse_output, evaluate_error_handling,
    TEST_DOC_URL, EXTRACT_SCHEMA, PROMPTS,
)
from models import ModelConfig, ALL_MODELS, PREMIUM_MODELS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REDUCTO_BASE  = os.getenv("REDUCTO_BASE_URL", "https://platform.reducto.ai")
REDUCTO_KEY   = os.environ["REDUCTO_API_KEY"]
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_KEY    = os.getenv("OPENAI_API_KEY", "")
OR_KEY        = os.getenv("OPENROUTER_API_KEY", "")

MAX_TURNS  = 6    # per task
MAX_PARALLEL = 12  # concurrent model calls (OpenRouter has no limit; be reasonable)

CAPABILITIES = ["extract", "parse", "error_handling"]

SYSTEM_PROMPT = f"""You are an AI assistant with access to Reducto's document intelligence API.
Reducto can parse PDFs, extract structured data, split documents, and classify document types.

IMPORTANT:
- Always use the provided tools to process documents — never guess or hallucinate content.
- For extraction tasks, use reducto_extract (not reducto_parse) — it returns typed JSON.
- If a tool call fails, read the error message and fix the parameters before retrying.
- After getting tool results, return a clean final answer.
"""

# ---------------------------------------------------------------------------
# Reducto tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "reducto_parse",
            "description": (
                "Parse a document into structured markdown with tables and figures. "
                "Returns content chunks and a job_id. Use persist_results=true to "
                "enable jobid:// reuse in subsequent calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Document URL, reducto:// file ID, or jobid:// from a previous parse",
                    },
                    "merge_tables": {
                        "type": "boolean",
                        "description": "Merge tables split across page boundaries (recommended for financial docs)",
                    },
                    "filter_blocks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Block types to exclude: Header, Footer, Page Number, etc.",
                    },
                    "persist_results": {
                        "type": "boolean",
                        "description": "Keep result for jobid:// reuse in extract/split calls",
                    },
                },
                "required": ["input"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reducto_extract",
            "description": (
                "Extract specific fields from a document into typed JSON. "
                "Provide a JSON Schema to define what to extract. "
                "Input can be a URL OR a jobid:// from a previous parse call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Document URL or jobid:// from reducto_parse",
                    },
                    "schema": {
                        "type": "string",
                        "description": "JSON Schema string defining the fields to extract",
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Custom instructions: e.g. 'Return null for missing fields. Values are in USD.'",
                    },
                    "deep_extract": {
                        "type": "boolean",
                        "description": "Run agentic re-pass for higher accuracy on complex documents",
                    },
                },
                "required": ["input", "schema"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reducto_classify",
            "description": "Classify a document into one of the provided categories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Document URL or reducto:// file ID"},
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Category names to classify into",
                    },
                },
                "required": ["input", "categories"],
            },
        },
    },
]

# Anthropic tool format (different schema)
TOOLS_ANTHROPIC = [
    {
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "input_schema": t["function"]["parameters"],
    }
    for t in TOOLS
]

# ---------------------------------------------------------------------------
# Reducto tool executor
# ---------------------------------------------------------------------------

async def _reducto_post(client: httpx.AsyncClient, path: str, body: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {REDUCTO_KEY}",
        "Content-Type": "application/json",
    }
    r = await client.post(f"{REDUCTO_BASE}{path}", json=body, headers=headers, timeout=90)
    if not r.is_success:
        # Parse structured errors
        try:
            err = r.json()
            if isinstance(err.get("detail"), list):
                msg = "; ".join(d.get("msg", "") for d in err["detail"])
            else:
                msg = str(err.get("detail", r.text[:300]))
        except Exception:
            msg = r.text[:300]
        raise ValueError(f"HTTP {r.status_code}: {msg}")
    return r.json()


async def execute_tool(name: str, args_raw: str | dict) -> str:
    """Execute a Reducto tool call and return a string result for the model."""
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
    except json.JSONDecodeError as e:
        return f"Error: could not parse tool arguments as JSON — {e}"

    async with httpx.AsyncClient() as client:
        try:
            if name == "reducto_parse":
                body: dict = {"input": args["input"]}
                if args.get("merge_tables"):
                    body["formatting"] = {"merge_tables": True}
                if args.get("filter_blocks"):
                    body.setdefault("retrieval", {})["filter_blocks"] = args["filter_blocks"]
                if args.get("persist_results"):
                    body.setdefault("settings", {})["persist_results"] = True

                data = await _reducto_post(client, "/parse", body)
                chunks = data.get("result", {}).get("chunks", [])
                # Return compact summary + first chunk preview
                table_count = sum(
                    1 for c in chunks for b in c.get("blocks", []) if b.get("type") == "Table"
                )
                preview = chunks[0].get("content", "")[:1500] if chunks else "(no content)"
                return (
                    f"Parse OK. Job ID: {data.get('job_id')}\n"
                    f"Pages: {data.get('usage', {}).get('num_pages', '?')} | Tables: {table_count}\n\n"
                    f"Content preview:\n{preview}"
                )

            elif name == "reducto_extract":
                raw_schema = args.get("schema", "{}")
                schema = json.loads(raw_schema) if isinstance(raw_schema, str) else raw_schema

                instructions: dict = {"schema": schema}
                if args.get("system_prompt"):
                    instructions["system_prompt"] = args["system_prompt"]

                body = {"input": args["input"], "instructions": instructions}
                settings: dict = {}
                if args.get("deep_extract"):
                    settings["deep_extract"] = True
                if settings:
                    body["settings"] = settings

                data = await _reducto_post(client, "/extract", body)
                result = data.get("result", [])
                return (
                    f"Extract OK. Job ID: {data.get('job_id')}\n\n"
                    f"Result:\n{json.dumps(result, indent=2)[:3000]}"
                )

            elif name == "reducto_classify":
                cats = args.get("categories", [])
                schema = [{"category": c, "criteria": [c]} for c in cats]
                data = await _reducto_post(client, "/classify", {
                    "input": args["input"],
                    "classification_schema": schema,
                })
                return f"Classify OK. Category: {data.get('result', {}).get('category')}"

            else:
                return f"Unknown tool: {name}"

        except Exception as e:
            return f"Tool error ({name}): {e}"


# ---------------------------------------------------------------------------
# Model callers — unified interface → returns (text, tool_calls_made)
# ---------------------------------------------------------------------------

async def _call_openai_compat(
    client,           # AsyncOpenAI client (OpenAI or OpenRouter)
    model_id: str,
    messages: list,
    reasoning_effort: Optional[str],
    extra_body: Optional[dict],
) -> tuple[str, list]:
    """Single-turn call. Returns (content_text, list_of_tool_calls)."""
    kwargs: dict = {
        "model": model_id,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if extra_body:
        kwargs["extra_body"] = extra_body

    try:
        resp = await client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        return msg.content or "", msg.tool_calls or []
    except Exception as e:
        return f"[model error] {e}", []


async def _call_anthropic(
    model_id: str,
    messages: list,
    thinking: bool,
    budget: int,
) -> tuple[str, list]:
    """Anthropic call with optional extended thinking."""
    if _anthropic is None:
        return "[anthropic not installed]", []

    client = _anthropic.AsyncAnthropic(api_key=ANTHROPIC_KEY)

    # Separate system from conversation
    system = next((m["content"] for m in messages if m["role"] == "system"), SYSTEM_PROMPT)
    conv = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "tool":
            # Convert tool result messages to Anthropic format
            for tr in (m if isinstance(m, list) else [m]):
                conv.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tr.get("tool_call_id", ""),
                        "content": tr.get("content", ""),
                    }],
                })
        elif m["role"] == "assistant" and m.get("tool_calls"):
            # Re-encode tool calls in Anthropic format
            content = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                content.append({
                    "type": "tool_use",
                    "id": tc.id if hasattr(tc, "id") else tc.get("id", ""),
                    "name": tc.function.name if hasattr(tc, "function") else tc["function"]["name"],
                    "input": json.loads(
                        tc.function.arguments if hasattr(tc, "function") else tc["function"]["arguments"]
                    ),
                })
            conv.append({"role": "assistant", "content": content})
        else:
            conv.append({"role": m["role"], "content": m["content"]})

    kwargs: dict = {
        "model": model_id,
        "max_tokens": budget + 4096 if thinking else 4096,
        "system": system,
        "messages": conv,
        "tools": TOOLS_ANTHROPIC,
    }

    try:
        if thinking:
            # Beta endpoint required for interleaved thinking; use adaptive for better perf
            kwargs["thinking"] = {"type": "adaptive", "budget_tokens": budget}
            resp = await client.beta.messages.create(
                betas=["interleaved-thinking-2025-05-14"],
                **kwargs,
            )
        else:
            resp = await client.messages.create(**kwargs)
    except Exception as e:
        return f"[anthropic error] {e}", []

    text_parts = [b.text for b in resp.content if hasattr(b, "text")]
    tool_uses  = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]

    # Normalize tool uses to OpenAI-like dicts for our shared loop
    norm_tools = []
    for tu in tool_uses:
        norm_tools.append({
            "id": tu.id,
            "function": {
                "name": tu.name,
                "arguments": json.dumps(tu.input),
            },
        })

    return " ".join(text_parts), norm_tools


# ---------------------------------------------------------------------------
# Multi-turn agent loop
# ---------------------------------------------------------------------------

async def run_agent(model: ModelConfig, task: str) -> tuple[str, list, int]:
    """
    Run the agent loop for one (model, task).
    Returns (final_text, all_messages, tool_call_count).
    """
    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": task},
    ]
    tool_call_count = 0

    # Build the right client
    if model.provider == "openai":
        if openai is None:
            return "[openai not installed]", messages, 0
        oai_client = openai.AsyncOpenAI(api_key=OPENAI_KEY)
    elif model.provider == "openrouter":
        if openai is None:
            return "[openai not installed]", messages, 0
        extra_body: dict = {}
        if model.or_provider_order:
            provider_cfg: dict = {
                "order": model.or_provider_order,
                "allow_fallbacks": True,
            }
            if model.or_quantization:
                provider_cfg["quantizations"] = [model.or_quantization]
            extra_body["provider"] = provider_cfg
        oai_client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OR_KEY,
        )
    else:
        oai_client = None

    for turn in range(MAX_TURNS):
        # --- Call the model ---
        if model.provider == "anthropic":
            text, tool_calls = await _call_anthropic(
                model.id, messages, model.reasoning, model.budget_tokens
            )
        else:
            text, tool_calls = await _call_openai_compat(
                oai_client,
                model.id,
                messages,
                model.reasoning_effort if model.reasoning and model.provider == "openai" else None,
                extra_body if model.provider == "openrouter" else None,
            )

        if not tool_calls:
            # Model gave a final text answer
            return text, messages, tool_call_count

        # --- Execute each tool call ---
        tool_call_count += len(tool_calls)
        tool_results = []
        for tc in tool_calls:
            fn   = tc["function"] if isinstance(tc, dict) else tc.function
            name = fn["name"]    if isinstance(fn, dict) else fn.name
            args = fn["arguments"] if isinstance(fn, dict) else fn.arguments
            result = await execute_tool(name, args)
            tc_id = tc["id"] if isinstance(tc, dict) else tc.id
            tool_results.append({"tool_call_id": tc_id, "role": "tool", "content": result})

        # Append to conversation history
        messages.append({"role": "assistant", "content": text, "tool_calls": tool_calls})
        messages.extend(tool_results)

    return "(max turns reached)", messages, tool_call_count


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

async def run_one(model: ModelConfig, capability: str, sem: asyncio.Semaphore) -> BenchmarkScore:
    task_map = {
        "extract":        PROMPTS["extract"],
        "parse":          PROMPTS["parse"],
        "error_handling": PROMPTS["error_handling"],
    }
    task = task_map.get(capability, PROMPTS["extract"])

    score = BenchmarkScore(
        platform=model.display,
        model=model.id,
        integration_path="tool_calling",
        capability=capability,
        # fixed dimensions for this test type
        setup_friction=5 if model.provider in ("openai", "anthropic") else 4,
        feature_coverage=3,   # REST via tool calling: parse/extract/classify reachable
        mcp_compatibility=0,  # not using MCP here
        token_efficiency=0 if model.is_oss else 3,
    )

    t0 = time.time()
    try:
        async with sem:
            final_text, messages, tc_count = await run_agent(model, task)

        score.wall_time_seconds  = round(time.time() - t0, 2)
        score.tool_calls_count   = tc_count
        score.notes              = model.notes

        # Score output quality
        if capability == "extract":
            quality, details = evaluate_extract_output(final_text)
        elif capability == "parse":
            quality, details = evaluate_parse_output(final_text)
        else:  # error_handling
            quality, details = evaluate_error_handling(final_text)

        score.output_quality = quality

        # Did the model actually call tools?
        used_tools = tc_count > 0
        score.discovery = 5 if used_tools else 2

        # Integration complexity: fewer turns = better
        turns = sum(1 for m in messages if m["role"] == "assistant")
        score.integration_complexity = max(1, 6 - turns)

        # Error recovery: check if error messages appear and were handled
        all_tool_content = " ".join(
            m.get("content", "") for m in messages if m.get("role") == "tool"
        )
        score.error_recovery = 4 if "error" not in all_tool_content.lower() else 3

        score.notes += (
            f" | turns={turns} tool_calls={tc_count}"
            f" | used_tools={'yes' if used_tools else 'NO — hallucinated'}"
        )

        subdir = "oss_models" if model.is_oss else "premium_models"
        save_result(score, subdir=subdir, raw_output=final_text[:4000])

    except Exception as e:
        score.notes = f"EXCEPTION: {traceback.format_exc()[-300:]}"
        score.output_quality = 1
        subdir = "oss_models" if model.is_oss else "premium_models"
        save_result(score, subdir=subdir)

    return score


# ---------------------------------------------------------------------------
# Main — parallel runner
# ---------------------------------------------------------------------------

async def main(models: list[ModelConfig], capabilities: list[str]):
    sem = asyncio.Semaphore(MAX_PARALLEL)

    tasks = [
        run_one(model, cap, sem)
        for model in models
        for cap in capabilities
    ]

    total = len(tasks)
    print(f"\n🚀  Running {total} tasks  ({len(models)} models × {len(capabilities)} capabilities)")
    print(f"    Parallelism: {MAX_PARALLEL} concurrent calls\n")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Model':<40} {'Cap':<16} {'Quality':>7} {'Calls':>6} {'Time':>6}")
    print("-" * 80)
    for r in results:
        if isinstance(r, BenchmarkScore):
            print(
                f"{r.platform:<40} {r.capability:<16}"
                f" {r.output_quality}/5    {r.tool_calls_count:>4}   {r.wall_time_seconds:>5.1f}s"
            )
        else:
            print(f"(error) {r}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reducto agent matrix benchmark")
    parser.add_argument("--model",        type=str, help="Filter models by substring")
    parser.add_argument("--capability",   type=str, help="Single capability to test")
    parser.add_argument("--premium-only", action="store_true", help="Skip OpenRouter models")
    parser.add_argument("--oss-only",     action="store_true", help="Only OpenRouter models")
    args = parser.parse_args()

    models_to_run = ALL_MODELS
    if args.premium_only:
        models_to_run = PREMIUM_MODELS
    elif args.oss_only:
        models_to_run = [m for m in ALL_MODELS if m.is_oss]
    if args.model:
        models_to_run = [m for m in models_to_run if args.model.lower() in m.id.lower() or args.model.lower() in m.display.lower()]

    caps = [args.capability] if args.capability else CAPABILITIES

    if not models_to_run:
        print("No models matched the filter.")
        sys.exit(1)

    asyncio.run(main(models_to_run, caps))
