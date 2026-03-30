"""
Benchmark: LlamaIndex ReActAgent + Reducto

Tests LlamaIndex's ReActAgent across ALL models in the registry (premium + 12 OSS
via OpenRouter). FunctionTool objects wrap Reducto's REST API directly.

Run:
    pip install llama-index llama-index-llms-openai llama-index-llms-anthropic requests
    python bench_llamaindex.py [--model <id>] [--capability <cap>] [--premium-only] [--oss-only]
"""

import argparse
import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(__file__))
from bench_utils import (
    BenchmarkScore,
    PROMPTS,
    EXTRACT_SCHEMA,
    evaluate_output,
    save_result,
)
from models import FRAMEWORK_MODELS, LOCAL_MODELS, ModelConfig

CAPABILITIES = ["parse", "extract", "split", "classify", "error_handling"]
REDUCTO_BASE = "https://platform.reducto.ai"


# ---------------------------------------------------------------------------
# Reducto tool functions — type-annotated for LlamaIndex FunctionTool introspection
# ---------------------------------------------------------------------------

def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ.get('REDUCTO_API_KEY', '')}",
        "Content-Type": "application/json",
    }


def reducto_parse(input_url: str) -> str:
    """
    Parse a document URL with Reducto's /parse endpoint.

    Args:
        input_url: URL of the document to parse (PDF, DOCX, XLSX, etc.)

    Returns:
        Extracted markdown text with tables and figures.
    """
    resp = requests.post(
        f"{REDUCTO_BASE}/parse",
        headers=_headers(),
        json={"input": input_url},
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    chunks = data.get("result", {}).get("chunks", [])
    return "\n\n".join(c.get("content", "") for c in chunks) or json.dumps(data)


def reducto_extract(input_url: str, schema_json: str) -> str:
    """
    Extract structured JSON from a document using Reducto's /extract endpoint.

    Args:
        input_url:   URL of the document to process.
        schema_json: JSON Schema string describing the fields to extract.

    Returns:
        JSON string with the extracted structured data.
    """
    try:
        schema = json.loads(schema_json)
    except json.JSONDecodeError as exc:
        return f"Error: invalid schema_json — {exc}"
    resp = requests.post(
        f"{REDUCTO_BASE}/extract",
        headers=_headers(),
        json={"input": input_url, "schema": schema},
        timeout=300,
    )
    resp.raise_for_status()
    return json.dumps(resp.json().get("result", resp.json()), indent=2)


def reducto_split(input_url: str, sections_json: str) -> str:
    """
    Split a document into named sections using Reducto's /split endpoint.

    Args:
        input_url:    URL of the document to split.
        sections_json: JSON array of section name strings, e.g. '["Cover Page","Holdings"]'.

    Returns:
        JSON with page ranges for each section.
    """
    try:
        sections = json.loads(sections_json)
    except json.JSONDecodeError:
        sections = [s.strip() for s in sections_json.split(",")]
    split_desc = [{"name": s, "description": s} for s in sections] if sections and isinstance(sections[0], str) else sections
    resp = requests.post(
        f"{REDUCTO_BASE}/split",
        headers=_headers(),
        json={"input": input_url, "split_description": split_desc},
        timeout=300,
    )
    resp.raise_for_status()
    return json.dumps(resp.json(), indent=2)


def reducto_classify(input_url: str, categories_json: str) -> str:
    """
    Classify a document into one of the provided categories using Reducto.

    Args:
        input_url:       URL of the document to classify.
        categories_json: JSON array of category name strings to classify into.

    Returns:
        JSON with the classification result.
    """
    try:
        categories = json.loads(categories_json)
    except json.JSONDecodeError:
        categories = [c.strip() for c in categories_json.split(",")]
    schema = [{"category": c, "criteria": [c.replace("_", " ")]} for c in categories] if categories and isinstance(categories[0], str) else categories
    resp = requests.post(
        f"{REDUCTO_BASE}/classify",
        headers=_headers(),
        json={"input": input_url, "classification_schema": schema},
        timeout=300,
    )
    resp.raise_for_status()
    return json.dumps(resp.json(), indent=2)


# ---------------------------------------------------------------------------
# LlamaIndex LLM builder
# ---------------------------------------------------------------------------

def _build_llm(model: ModelConfig):
    if model.provider == "anthropic":
        from llama_index.llms.anthropic import Anthropic
        # LlamaIndex's Anthropic backend rejects the thinking parameter via
        # additional_kwargs — omit it and test baseline tool-calling performance.
        return Anthropic(model=model.id, max_tokens=16000, temperature=0.0)

    # OpenAI, OpenRouter, or local llama-server — all via OpenAI-compatible endpoint
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
    from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS as _oai_models
    import tiktoken

    if model.provider == "local":
        _oai_models.setdefault(model.id, 32_768)
        tiktoken.model.MODEL_TO_ENCODING.setdefault(model.id, "cl100k_base")
        return LlamaOpenAI(
            model=model.id,
            api_base=model.local_base_url,
            api_key="local",
            temperature=0,
        )

    if model.provider == "openai":
        kwargs = dict(model=model.id, temperature=0)
        if model.reasoning:
            kwargs["additional_kwargs"] = {"reasoning_effort": model.reasoning_effort}
        return LlamaOpenAI(**kwargs)

    # OpenRouter — provider routing via additional_kwargs (passed to completions.create)
    _oai_models.setdefault(model.id, 128_000)
    tiktoken.model.MODEL_TO_ENCODING.setdefault(model.id, "cl100k_base")

    extra: dict = {}
    if model.or_provider_order:
        provider: dict = {"order": model.or_provider_order, "allow_fallbacks": True}
        if model.or_quantization:
            provider["quantizations"] = [model.or_quantization]
        extra["extra_body"] = {"provider": provider}

    return LlamaOpenAI(
        model=model.id,
        api_base="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        temperature=0,
        additional_kwargs=extra if extra else {},
    )


def _build_agent(model: ModelConfig):
    # LlamaIndex 0.14+: ReActAgent dropped from_tools(); use direct constructor.
    # .run() returns an awaitable WorkflowHandler.
    from llama_index.core.agent import ReActAgent
    from llama_index.core.tools import FunctionTool

    tools = [
        FunctionTool.from_defaults(fn=reducto_parse,     name="reducto_parse"),
        FunctionTool.from_defaults(fn=reducto_extract,   name="reducto_extract"),
        FunctionTool.from_defaults(fn=reducto_split,     name="reducto_split"),
        FunctionTool.from_defaults(fn=reducto_classify,  name="reducto_classify"),
    ]
    llm = _build_llm(model)
    # timeout caps the whole agent run; prevents infinite loops in the workflow.
    return ReActAgent(tools=tools, llm=llm, verbose=False, timeout=360.0)


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_llamaindex_model(model: ModelConfig, capability: str) -> BenchmarkScore:
    print(f"  [{model.display}] {capability} …")
    prompt = PROMPTS.get(capability, PROMPTS["extract"])
    errors: list[str] = []
    output = ""
    start = time.time()

    try:
        import asyncio as _asyncio
        agent = _build_agent(model)
        # LlamaIndex 0.14+: agent.run() calls asyncio.create_task() internally
        # (sync!) so it MUST be called from inside a running event loop.
        # Wrap both the agent.run() call and the await in one coroutine.
        async def _run():
            return await agent.run(prompt)
        _loop = _asyncio.new_event_loop()
        _asyncio.set_event_loop(_loop)
        try:
            response = _loop.run_until_complete(_run())
        finally:
            _loop.close()
        # AgentOutput.response is a ChatMessage; .content gives the text.
        # Fall back to str() if the attribute isn't there.
        msg = getattr(response, "response", response)
        output = getattr(msg, "content", None) or str(response)
    except Exception as exc:
        errors.append(str(exc))
        output = f"[error] {exc}"
        print(f"    !! {model.display}/{capability}: {exc}")

    wall_time = time.time() - start
    quality_score, details = evaluate_output(output, capability)

    score = BenchmarkScore(
        platform=f"LlamaIndex/{model.display}",
        model=model.display,
        integration_path="framework",
        capability=capability,
        discovery=3,
        setup_friction=3,
        integration_complexity=4,  # FunctionTool.from_defaults is concise
        feature_coverage=4,
        error_recovery=3 if not errors else 2,
        output_quality=quality_score,
        token_efficiency=2 if model.reasoning else 3,
        mcp_compatibility=0,
        wall_time_seconds=round(wall_time, 2),
        errors_encountered=errors,
        notes=(
            f"LlamaIndex ReActAgent — {model.display}. "
            f"FunctionTool.from_defaults introspects type annotations automatically. "
            f"Eval details: {details}"
        ),
    )

    subdir = ("local_models/llamaindex" if model.provider == "local"
              else "oss_models/llamaindex" if model.is_oss
              else "premium_models/llamaindex")
    save_result(score, subdir=subdir, raw_output=output)
    return score


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------

async def run_all(models: list[ModelConfig], capabilities: list[str]) -> None:
    sem = asyncio.Semaphore(4)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=4)

    async def bounded(model: ModelConfig, cap: str):
        async with sem:
            return await loop.run_in_executor(executor, run_llamaindex_model, model, cap)

    tasks = [bounded(m, c) for m in models for c in capabilities]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    passed = sum(1 for r in results if isinstance(r, BenchmarkScore))
    failed = len(results) - passed
    print(f"\nLlamaIndex done: {passed} passed, {failed} failed out of {len(results)}")
    executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LlamaIndex × Reducto benchmark")
    parser.add_argument("--model", help="Filter to a single model ID")
    parser.add_argument("--capability", help="Run a single capability")
    parser.add_argument("--premium-only", action="store_true")
    parser.add_argument("--oss-only", action="store_true")
    parser.add_argument("--local-only", action="store_true", help="Run local GGUF models only")
    args = parser.parse_args()

    missing = []
    for pkg in ("llama_index.core", "llama_index.llms.openai", "requests"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {missing}")
        print("Install: pip install llama-index llama-index-llms-openai llama-index-llms-anthropic requests")
        sys.exit(1)

    if not os.environ.get("REDUCTO_API_KEY"):
        print("Error: REDUCTO_API_KEY not set")
        sys.exit(1)

    models = LOCAL_MODELS if args.local_only else FRAMEWORK_MODELS
    if args.premium_only:
        models = [m for m in models if not m.is_oss]
    if args.oss_only:
        models = [m for m in models if m.is_oss]
    if args.model:
        all_models = FRAMEWORK_MODELS + LOCAL_MODELS
        models = [m for m in all_models if args.model in m.id]

    capabilities = [args.capability] if args.capability else CAPABILITIES

    print(f"LlamaIndex benchmark: {len(models)} models × {len(capabilities)} capabilities "
          f"= {len(models) * len(capabilities)} runs")
    asyncio.run(run_all(models, capabilities))


if __name__ == "__main__":
    main()
