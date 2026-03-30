"""
Benchmark: HuggingFace smolagents + Reducto

Tests smolagents ToolCallingAgent across ALL models in the registry (premium + 12 OSS
via OpenRouter). @tool-decorated functions wrap Reducto's REST API.

Run:
    pip install smolagents requests
    python bench_smolagents.py [--model <id>] [--capability <cap>] [--premium-only] [--oss-only]

Note: smolagents requires Python 3.10+.
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
# Reducto tools — @tool decorator introspects Google-style docstrings + type hints
# ---------------------------------------------------------------------------

def _build_tools():
    """Build smolagents Tool objects. Deferred so ImportError is handled gracefully."""
    from smolagents import tool

    @tool
    def reducto_parse(input_url: str) -> str:
        """
        Parse a document with Reducto and return extracted text as markdown.

        Args:
            input_url: URL of the document to parse (PDF, DOCX, XLSX, etc.)
        """
        api_key = os.environ.get("REDUCTO_API_KEY", "")
        resp = requests.post(
            f"{REDUCTO_BASE}/parse",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"input": input_url},
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        chunks = data.get("result", {}).get("chunks", [])
        return "\n\n".join(c.get("content", "") for c in chunks) or json.dumps(data)

    @tool
    def reducto_extract(input_url: str, schema_json: str) -> str:
        """
        Extract structured JSON from a document using Reducto's /extract endpoint.

        Args:
            input_url:   URL of the document to process.
            schema_json: JSON Schema string describing the fields to extract.
        """
        api_key = os.environ.get("REDUCTO_API_KEY", "")
        try:
            schema = json.loads(schema_json)
        except json.JSONDecodeError as exc:
            return f"Error: invalid schema_json — {exc}"
        resp = requests.post(
            f"{REDUCTO_BASE}/extract",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"input": input_url, "schema": schema},
            timeout=300,
        )
        resp.raise_for_status()
        return json.dumps(resp.json().get("result", resp.json()), indent=2)

    @tool
    def reducto_split(input_url: str, sections_json: str) -> str:
        """
        Split a document into named sections using Reducto's /split endpoint.

        Args:
            input_url:     URL of the document to split.
            sections_json: JSON array of section name strings, e.g. '["Cover Page","Holdings"]'.
        """
        api_key = os.environ.get("REDUCTO_API_KEY", "")
        try:
            sections = json.loads(sections_json)
        except json.JSONDecodeError:
            sections = [s.strip() for s in sections_json.split(",")]
        split_desc = [{"name": s, "description": s} for s in sections] if sections and isinstance(sections[0], str) else sections
        resp = requests.post(
            f"{REDUCTO_BASE}/split",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"input": input_url, "split_description": split_desc},
            timeout=300,
        )
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2)

    @tool
    def reducto_classify(input_url: str, categories_json: str) -> str:
        """
        Classify a document into one of the provided categories using Reducto.

        Args:
            input_url:       URL of the document to classify.
            categories_json: JSON array of category name strings.
        """
        api_key = os.environ.get("REDUCTO_API_KEY", "")
        try:
            categories = json.loads(categories_json)
        except json.JSONDecodeError:
            categories = [c.strip() for c in categories_json.split(",")]
        schema = [{"category": c, "criteria": [c.replace("_", " ")]} for c in categories] if categories and isinstance(categories[0], str) else categories
        resp = requests.post(
            f"{REDUCTO_BASE}/classify",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"input": input_url, "classification_schema": schema},
            timeout=300,
        )
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2)

    return [reducto_parse, reducto_extract, reducto_split, reducto_classify]


# ---------------------------------------------------------------------------
# smolagents model builder
# ---------------------------------------------------------------------------

def _build_model(model: ModelConfig):
    if model.provider == "anthropic":
        # LiteLLMModel: use bare model ID (no "anthropic/" prefix) so litellm
        # routes via its native Anthropic handler rather than falling through to
        # OpenRouter when OPENROUTER_API_KEY is also present in the environment.
        # litellm must be installed via: pip install 'smolagents[litellm]'
        from smolagents import LiteLLMModel
        # Note: litellm doesn't forward the "thinking" param to Anthropic's
        # extended-thinking API correctly — omit it and benchmark baseline performance.
        return LiteLLMModel(
            model_id=model.id,
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )

    if model.provider == "local":
        from smolagents import OpenAIServerModel
        return OpenAIServerModel(
            model_id=model.id,
            api_base=model.local_base_url,
            api_key="local",
        )

    if model.provider == "openai":
        from smolagents import OpenAIServerModel

        kwargs = dict(
            model_id=model.id,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
        if model.reasoning:
            # reasoning_effort goes into self.kwargs → passed to completions.create()
            kwargs["reasoning_effort"] = model.reasoning_effort
        return OpenAIServerModel(**kwargs)

    # OpenRouter — extra_body carries provider routing hints
    from smolagents import OpenAIServerModel

    kwargs = dict(
        model_id=model.id,
        api_base="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    )
    if model.or_provider_order:
        provider: dict = {"order": model.or_provider_order, "allow_fallbacks": True}
        if model.or_quantization:
            provider["quantizations"] = [model.or_quantization]
        kwargs["extra_body"] = {"provider": provider}

    return OpenAIServerModel(**kwargs)


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_smolagents_model(model: ModelConfig, capability: str) -> BenchmarkScore:
    print(f"  [{model.display}] {capability} …")
    prompt = PROMPTS.get(capability, PROMPTS["extract"])
    errors: list[str] = []
    output = ""
    start = time.time()

    try:
        from smolagents import ToolCallingAgent

        tools = _build_tools()
        smol_model = _build_model(model)
        agent = ToolCallingAgent(
            tools=tools,
            model=smol_model,
            max_steps=8,
            verbosity_level=0,
        )
        raw = agent.run(prompt)
        output = raw if isinstance(raw, str) else json.dumps(raw, indent=2)
    except Exception as exc:
        errors.append(str(exc))
        output = f"[error] {exc}"
        print(f"    !! {model.display}/{capability}: {exc}")

    wall_time = time.time() - start
    quality_score, details = evaluate_output(output, capability)

    score = BenchmarkScore(
        platform=f"smolagents/{model.display}",
        model=model.display,
        integration_path="framework",
        capability=capability,
        discovery=3,
        setup_friction=3,
        integration_complexity=4,  # @tool decorator is very concise
        feature_coverage=4,
        error_recovery=3 if not errors else 2,
        output_quality=quality_score,
        token_efficiency=2 if model.reasoning else 3,
        mcp_compatibility=0,
        wall_time_seconds=round(wall_time, 2),
        errors_encountered=errors,
        notes=(
            f"smolagents ToolCallingAgent — {model.display}. "
            f"@tool decorator introspects docstrings + type annotations. "
            f"Eval details: {details}"
        ),
    )

    subdir = ("local_models/smolagents" if model.provider == "local"
              else "oss_models/smolagents" if model.is_oss
              else "premium_models/smolagents")
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
            return await loop.run_in_executor(executor, run_smolagents_model, model, cap)

    tasks = [bounded(m, c) for m in models for c in capabilities]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    passed = sum(1 for r in results if isinstance(r, BenchmarkScore))
    failed = len(results) - passed
    print(f"\nsmolagents done: {passed} passed, {failed} failed out of {len(results)}")
    executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="smolagents × Reducto benchmark")
    parser.add_argument("--model", help="Filter to a single model ID")
    parser.add_argument("--capability", help="Run a single capability")
    parser.add_argument("--premium-only", action="store_true")
    parser.add_argument("--oss-only", action="store_true")
    parser.add_argument("--local-only", action="store_true", help="Run local GGUF models only")
    args = parser.parse_args()

    try:
        import smolagents  # noqa: F401
    except ImportError:
        print("Missing: smolagents")
        print("Install: pip install smolagents requests")
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

    print(f"smolagents benchmark: {len(models)} models × {len(capabilities)} capabilities "
          f"= {len(models) * len(capabilities)} runs")
    asyncio.run(run_all(models, capabilities))


if __name__ == "__main__":
    main()
