"""
Benchmark: LangChain agent + Reducto

Tests a LangChain tool-calling agent (LangGraph-backed, v1.x API) across ALL models
in the registry. Each model × capability combo runs in parallel via asyncio + ThreadPoolExecutor.

LangChain 1.x dropped AgentExecutor in favour of create_agent() which returns a
LangGraph CompiledStateGraph. Tool functions are decorated with @tool so schema
inference is automatic.

Run:
    pip install langchain langchain-openai langchain-anthropic requests
    python bench_langchain.py [--model <id>] [--capability <cap>] [--premium-only] [--oss-only]
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
# Reducto tool implementations (raw requests, no SDK — measures real integration cost)
# ---------------------------------------------------------------------------

def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ.get('REDUCTO_API_KEY', '')}",
        "Content-Type": "application/json",
    }


def _reducto_parse(input_url: str) -> str:
    """Parse a document URL with Reducto. Returns extracted markdown + tables."""
    resp = requests.post(
        f"{REDUCTO_BASE}/parse",
        headers=_headers(),
        json={"input": input_url},
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    chunks = data.get("result", {}).get("chunks", [])
    parts = [c.get("content", "") for c in chunks]
    return "\n\n".join(parts) if parts else json.dumps(data)


def _reducto_extract(input_url: str, schema_json: str) -> str:
    """Extract structured JSON from a document. Args: url, JSON Schema string."""
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


def _reducto_split(input_url: str, sections_json: str) -> str:
    """Split a document into sections. Args: url, JSON array of section name strings."""
    try:
        sections = json.loads(sections_json)
    except json.JSONDecodeError:
        sections = [s.strip() for s in sections_json.split(",")]
    split_desc = [{"name": s, "description": s} for s in sections] if isinstance(sections, list) and sections and isinstance(sections[0], str) else sections
    resp = requests.post(
        f"{REDUCTO_BASE}/split",
        headers=_headers(),
        json={"input": input_url, "split_description": split_desc},
        timeout=300,
    )
    resp.raise_for_status()
    return json.dumps(resp.json(), indent=2)


def _reducto_classify(input_url: str, categories_json: str) -> str:
    """Classify a document into a category. Args: url, JSON array of category strings."""
    try:
        categories = json.loads(categories_json)
    except json.JSONDecodeError:
        categories = [c.strip() for c in categories_json.split(",")]
    schema = [{"category": c, "criteria": [c.replace("_", " ")]} for c in categories] if isinstance(categories, list) and categories and isinstance(categories[0], str) else categories
    resp = requests.post(
        f"{REDUCTO_BASE}/classify",
        headers=_headers(),
        json={"input": input_url, "classification_schema": schema},
        timeout=300,
    )
    resp.raise_for_status()
    return json.dumps(resp.json(), indent=2)


# ---------------------------------------------------------------------------
# LangChain @tool definitions — schema auto-inferred from type hints + docstrings
# ---------------------------------------------------------------------------

def _make_tools():
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def reducto_parse(input_url: str) -> str:
        """Parse a document URL with Reducto. Returns extracted markdown text and tables."""
        return _reducto_parse(input_url.strip())

    @lc_tool
    def reducto_extract(input_url: str, schema_json: str) -> str:
        """Extract structured JSON from a document. Pass the document URL and a JSON Schema string."""
        return _reducto_extract(input_url.strip(), schema_json.strip())

    @lc_tool
    def reducto_split(input_url: str, sections_json: str) -> str:
        """Split a document into named sections. Pass the URL and a JSON array of section name strings."""
        return _reducto_split(input_url.strip(), sections_json.strip())

    @lc_tool
    def reducto_classify(input_url: str, categories_json: str) -> str:
        """Classify a document into one of the provided categories. Pass URL and JSON array of category names."""
        return _reducto_classify(input_url.strip(), categories_json.strip())

    return [reducto_parse, reducto_extract, reducto_split, reducto_classify]


def _build_llm(model: ModelConfig):
    """Build the right LangChain LLM backend for a given model config."""
    if model.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # Note: LangChain's ChatAnthropic doesn't support the thinking parameter directly
        # (it rejects "adaptive" type as an extra field in its Pydantic schema).
        # We run without extended thinking here — measures baseline LangChain integration.
        return ChatAnthropic(model=model.id, max_tokens=8096, temperature=0.0)

    if model.provider == "local":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model.id,
            base_url=model.local_base_url,
            api_key="local",
            temperature=0,
        )

    if model.provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict = dict(model=model.id)
        if model.reasoning:
            # o-series: no temperature support; reasoning_effort is a first-class param
            kwargs["reasoning_effort"] = model.reasoning_effort
        else:
            kwargs["temperature"] = 0
        return ChatOpenAI(**kwargs)

    # OpenRouter
    from langchain_openai import ChatOpenAI

    # Provider routing goes via extra_body (added to raw JSON body by openai SDK),
    # NOT model_kwargs (which adds top-level API params and causes 422 errors).
    extra_body: dict = {}
    if model.or_provider_order:
        extra_body["provider"] = {"order": model.or_provider_order, "allow_fallbacks": True}
    if model.or_quantization:
        # OpenRouter expects plural "quantizations" as an array
        extra_body.setdefault("provider", {})["quantizations"] = [model.or_quantization]

    return ChatOpenAI(
        model=model.id,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        temperature=0,
        model_kwargs={"extra_body": extra_body} if extra_body else {},
    )


def _build_agent(model: ModelConfig):
    # LangChain 1.x: create_agent returns a LangGraph CompiledStateGraph
    from langchain.agents import create_agent

    tools = _make_tools()
    llm = _build_llm(model)
    return create_agent(model=llm, tools=tools)


# ---------------------------------------------------------------------------
# Per-model runner (sync — called from executor)
# ---------------------------------------------------------------------------

def run_langchain_model(model: ModelConfig, capability: str) -> BenchmarkScore:
    print(f"  [{model.display}] {capability} …")
    prompt = PROMPTS.get(capability, PROMPTS["extract"])
    errors: list[str] = []
    output = ""
    start = time.time()

    try:
        agent = _build_agent(model)
        # LangChain 1.x create_agent → LangGraph; messages-based invoke
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        messages = result.get("messages", [])
        # Last AI message is the final answer
        output = next(
            (m.content for m in reversed(messages) if hasattr(m, "role") and getattr(m, "role", "") != "tool"),
            ""
        )
        if not output and messages:
            last = messages[-1]
            output = last.content if hasattr(last, "content") else str(last)
    except Exception as exc:
        errors.append(str(exc))
        output = f"[error] {exc}"
        print(f"    !! {model.display}/{capability}: {exc}")

    wall_time = time.time() - start
    quality_score, details = evaluate_output(output, capability)

    # Reasoning models tend to use more tokens — penalise accordingly
    token_eff = 2 if model.reasoning else 3

    score = BenchmarkScore(
        platform=f"LangChain/{model.display}",
        model=model.display,
        integration_path="framework",
        capability=capability,
        discovery=3,
        setup_friction=3,
        integration_complexity=3,
        feature_coverage=4,
        error_recovery=3 if not errors else 2,
        output_quality=quality_score,
        token_efficiency=token_eff,
        mcp_compatibility=0,
        wall_time_seconds=round(wall_time, 2),
        errors_encountered=errors,
        notes=(
            f"LangChain ReAct agent — {model.display}. "
            f"Provider: {model.provider}. "
            f"Eval details: {details}"
        ),
    )

    subdir = ("local_models/langchain" if model.provider == "local"
              else "oss_models/langchain" if model.is_oss
              else "premium_models/langchain")
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
            return await loop.run_in_executor(executor, run_langchain_model, model, cap)

    tasks = [bounded(m, c) for m in models for c in capabilities]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    passed = sum(1 for r in results if isinstance(r, BenchmarkScore))
    failed = len(results) - passed
    print(f"\nLangChain done: {passed} passed, {failed} failed out of {len(results)}")
    executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain × Reducto benchmark")
    parser.add_argument("--model", help="Filter to a single model ID")
    parser.add_argument("--capability", help="Run a single capability")
    parser.add_argument("--premium-only", action="store_true")
    parser.add_argument("--oss-only", action="store_true")
    parser.add_argument("--local-only", action="store_true", help="Run local GGUF models only")
    args = parser.parse_args()

    # Dependency check
    missing = []
    for pkg in ("langchain", "langchain_openai", "langchain_anthropic", "requests"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing: {', '.join(missing)}")
        print("Install: pip install langchain langchain-openai langchain-anthropic requests")
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

    print(f"LangChain benchmark: {len(models)} models × {len(capabilities)} capabilities "
          f"= {len(models) * len(capabilities)} runs")
    asyncio.run(run_all(models, capabilities))


if __name__ == "__main__":
    main()
