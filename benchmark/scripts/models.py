"""
Model registry for the Reducto agent benchmark.

All (framework × model) combinations draw from this single source of truth.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    id: str                                   # API model identifier
    display: str                              # human-readable label for reports
    provider: str                             # "anthropic" | "openai" | "openrouter" | "local"
    is_oss: bool = False                      # community / open-weight model

    # High-reasoning settings
    reasoning: bool = False
    budget_tokens: int = 8000                 # Claude extended-thinking token budget
    reasoning_effort: str = "high"            # OpenAI o-series: "low"|"medium"|"high"

    # OpenRouter provider routing
    or_provider_order: list = field(default_factory=list)
    or_quantization: Optional[str] = None    # "bf16" | "fp8" | "fp4" etc.

    # Local llama-server (provider="local")
    local_model_path: str = ""               # path to .gguf file
    local_base_url: str = "http://127.0.0.1:18080/v1"

    notes: str = ""


# ---------------------------------------------------------------------------
# Premium models — native APIs, high reasoning
# ---------------------------------------------------------------------------

PREMIUM_MODELS: list[ModelConfig] = [
    ModelConfig(
        id="claude-opus-4-6",
        display="Claude Opus 4.6 + thinking",
        provider="anthropic",
        reasoning=True,
        budget_tokens=10_000,
        notes="Extended thinking enabled; interleaved-thinking-2025-05-14 beta",
    ),
    ModelConfig(
        id="claude-haiku-4-5-20251001",
        display="Claude Haiku 4.5 + thinking",
        provider="anthropic",
        reasoning=True,
        budget_tokens=5_000,
        notes="Smallest Claude with thinking — good stress test for small models",
    ),
    ModelConfig(
        id="o3",
        display="OpenAI o3 (reasoning=high)",
        provider="openai",
        reasoning=True,
        reasoning_effort="high",
    ),
    ModelConfig(
        id="o4-mini",
        display="OpenAI o4-mini (reasoning=high)",
        provider="openai",
        reasoning=True,
        reasoning_effort="high",
    ),
]

# ---------------------------------------------------------------------------
# OpenRouter community models — parallel stress test
# Provider hints from user's spec; allow_fallbacks=True for resilience.
# ---------------------------------------------------------------------------

OPENROUTER_MODELS: list[ModelConfig] = [
    ModelConfig(
        id="stepfun/step-3.5-flash",
        display="StepFun Step-3.5 Flash",
        provider="openrouter",
        is_oss=True,
    ),
    ModelConfig(
        id="arcee-ai/trinity-large-preview:free",
        display="Arcee Trinity Large (prime)",
        provider="openrouter",
        is_oss=True,
        or_provider_order=["ArceeAI"],
        notes="Free tier; arcee-ai/prime preferred provider",
    ),
    ModelConfig(
        id="openai/gpt-oss-20b:nitro",
        display="GPT-OSS 20B via Groq",
        provider="openrouter",
        is_oss=True,
        or_provider_order=["Groq"],
    ),
    ModelConfig(
        id="nvidia/nemotron-3-super-120b-a12b",
        display="Nemotron Super 120B (Nebius bf16)",
        provider="openrouter",
        is_oss=True,
        or_provider_order=["Nebius AI"],
        or_quantization="bf16",
    ),
    ModelConfig(
        id="nvidia/nemotron-3-nano-30b-a3b:nitro",
        display="Nemotron Nano 30B (DeepInfra fp4)",
        provider="openrouter",
        is_oss=True,
        or_provider_order=["DeepInfra"],
        or_quantization="fp4",
    ),
    ModelConfig(
        id="qwen/qwen3-coder-next",
        display="Qwen3 Coder Next (ionstream fp8)",
        provider="openrouter",
        is_oss=True,
        or_provider_order=["IonStream"],
        or_quantization="fp8",
    ),
    ModelConfig(
        id="qwen/qwen3.5-122b-a10b",
        display="Qwen3.5-122B-A10B",
        provider="openrouter",
        is_oss=True,
        or_provider_order=["Alibaba Cloud"],
    ),
    ModelConfig(
        id="qwen/qwen3-32b:nitro",
        display="Qwen3 32B via Groq",
        provider="openrouter",
        is_oss=True,
        or_provider_order=["Groq"],
    ),
    ModelConfig(
        id="minimax/minimax-m2.7",
        display="MiniMax M2.7 (highspeed)",
        provider="openrouter",
        is_oss=True,
        or_provider_order=["MiniMax"],
    ),
    ModelConfig(
        id="xiaomi/mimo-v2-pro",
        display="Xiaomi MiMo V2 Pro (fp8)",
        provider="openrouter",
        is_oss=True,
        or_provider_order=["Xiaomi"],
        or_quantization="fp8",
    ),
    ModelConfig(
        id="moonshotai/kimi-k2.5:nitro",
        display="Kimi K2.5 via Fireworks",
        provider="openrouter",
        is_oss=True,
        or_provider_order=["Fireworks"],
    ),
    ModelConfig(
        id="z-ai/glm-5-turbo",
        display="GLM-5 Turbo",
        provider="openrouter",
        is_oss=True,
    ),
    ModelConfig(
        id="qwen/qwen3.5-35b-a3b",
        display="Qwen3.5-35B-A3B (Atlas Cloud fp8)",
        provider="openrouter",
        is_oss=True,
        or_provider_order=["Atlas Cloud"],
        or_quantization="fp8",
        notes="MoE model with 3B active params; 115 tok/s on Atlas Cloud fp8",
    ),
    # --- New probe candidates ---
    ModelConfig(
        id="openai/gpt-5.4-nano",
        display="GPT-5.4 Nano",
        provider="openrouter",
        is_oss=False,
    ),
    ModelConfig(
        id="openai/gpt-5.4-mini",
        display="GPT-5.4 Mini",
        provider="openrouter",
        is_oss=False,
    ),
    ModelConfig(
        id="inception/mercury-2",
        display="Inception Mercury 2",
        provider="openrouter",
        is_oss=True,
    ),
    # LFM dropped — too small to be useful as teacher, costs not worth it
    # ModelConfig(id="liquid/lfm-2.5-1.2b-thinking:free", display="Liquid LFM-2.5 1.2B Thinking (free)", ...),
    ModelConfig(
        id="qwen/qwen3.5-flash-02-23",
        display="Qwen3.5 Flash (02-23)",
        provider="openrouter",
        is_oss=True,
    ),
    ModelConfig(
        id="mistralai/devstral-small",
        display="Mistral Devstral Small",
        provider="openrouter",
        is_oss=True,
        notes="Code-focused Mistral; strong tool use",
    ),
    # --- Google Gemini ---
    ModelConfig(
        id="google/gemini-3.1-pro-preview-customtools",
        display="Gemini 3.1 Pro (custom-tools preview)",
        provider="openrouter",
        is_oss=False,
        notes="Gemini 3.1 Pro variant tuned for reliable function selection in multi-tool workflows; 1M ctx",
    ),
    ModelConfig(
        id="google/gemini-3.1-flash-lite-preview",
        display="Gemini 3.1 Flash Lite (preview)",
        provider="openrouter",
        is_oss=False,
        notes="Lightweight Gemini 3.1 Flash variant; Google AI Studio preview endpoint",
    ),
]

# ---------------------------------------------------------------------------
# Local GGUF models — served via llama-server (OpenAI-compatible endpoint)
# These require llama-server running at local_base_url before benchmark starts.
# ---------------------------------------------------------------------------

LOCAL_MODELS: list[ModelConfig] = [
    ModelConfig(
        id="qwen-0.8b-agentjson",
        display="Qwen-0.8B-AgentJSON-Q6K (local)",
        provider="local",
        is_oss=True,
        local_model_path="/Users/reducto-reza/AI/reducto-agent-benchmark/local-models/Qwen-0.8B-AgentJSON-Q6K.gguf",
        local_base_url="http://127.0.0.1:18080/v1",
        notes="0.8B Qwen3.5 fine-tuned on tool/JSON calling. Q6_K quantized locally.",
    ),
    ModelConfig(
        id="lfm2.5-1.2b-thinking",
        display="LFM-2.5 1.2B Thinking Q6K (local)",
        provider="local",
        is_oss=True,
        local_model_path="/Users/reducto-reza/AI/reducto-agent-benchmark/local-models/LFM2.5-1.2B-Thinking/LFM2.5-1.2B-Thinking-Q6_K.gguf",
        local_base_url="http://127.0.0.1:18081/v1",
        notes="LiquidAI 1.2B reasoning model. Q6_K GGUF ~918MB. Served on port 18081.",
    ),
]

# ---------------------------------------------------------------------------
# Combined lists
# ---------------------------------------------------------------------------

ALL_MODELS: list[ModelConfig] = PREMIUM_MODELS + OPENROUTER_MODELS

# Frameworks that support arbitrary backends use all models
FRAMEWORK_MODELS: list[ModelConfig] = ALL_MODELS
