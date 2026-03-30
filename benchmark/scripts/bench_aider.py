"""
Benchmark: Aider + Reducto
==========================
Aider (https://aider.chat) is an open-source CLI coding assistant that pairs with
your editor and rewrites files in response to natural-language instructions. It works
by sending the conversation — including repo context — to a backing LLM.

Install:
    pip install aider-chat

Typical usage:
    aider --model gpt-4.1 --no-auto-commits

MCP support: None. Aider has no MCP integration; all Reducto interaction must go
through the Python SDK or direct REST calls written by the model.

Integration paths tested:
    python_sdk  — agent writes Python using `pip install reductoai`
    rest_api    — agent writes requests.post() calls against platform.reducto.ai

Capabilities tested:
    parse          — document → structured markdown
    extract        — document + schema → typed JSON
    error_handling — auth failure, bad URL, malformed schema

Run:
    python bench_aider.py
    Then paste each printed prompt into an `aider` session and score the results
    by editing the stub JSON files created in benchmark/results/.
"""

import json
from pathlib import Path

from bench_utils import (
    BenchmarkScore,
    PROMPTS,
    TEST_DOC_URL,
    save_result,
)

# ---------------------------------------------------------------------------
# Platform-level constants
# ---------------------------------------------------------------------------

PLATFORM = "Aider"
MODEL = "gpt-4.1"

# Dimensions that are fixed for this platform regardless of capability tested.
# mcp_compatibility=0 because Aider has no MCP support (0 = N/A, excluded from total).
PLATFORM_SCORES = dict(
    setup_friction=4,    # pip install aider-chat + set OPENAI_API_KEY — 2 steps
    feature_coverage=3,  # SDK + REST paths work; no MCP, no native tool calling
    mcp_compatibility=0, # N/A
    discovery=3,         # Agent can find Reducto if given the package name; zero-shot unlikely
)

# ---------------------------------------------------------------------------
# Test matrix: (integration_path, capability, prompt_key)
# ---------------------------------------------------------------------------

TESTS = [
    ("python_sdk", "parse",          "parse"),
    ("python_sdk", "extract",        "extract"),
    ("rest_api",   "error_handling", "error_handling"),
]

# ---------------------------------------------------------------------------
# MCP config
# ---------------------------------------------------------------------------

MCP_CONFIG = "N/A — Aider does not support MCP."

# ---------------------------------------------------------------------------
# Setup instructions
# ---------------------------------------------------------------------------

SETUP = """
Setup — Aider + Reducto
-----------------------
1. Install Aider:
       pip install aider-chat

2. Set your model's API key (e.g. OpenAI):
       export OPENAI_API_KEY=sk-...
   Or for Anthropic:
       export ANTHROPIC_API_KEY=...

3. Set Reducto key:
       export REDUCTO_API_KEY=your_reducto_key

4. Launch Aider in a scratch directory:
       mkdir /tmp/reducto-aider-test && cd /tmp/reducto-aider-test
       aider --model gpt-4.1 --no-auto-commits

5. Paste each prompt below into the Aider chat.
   Aider will write and optionally run the code.
   Score results by editing the stub JSON files in benchmark/results/.

MCP: {mcp_config}
""".format(mcp_config=MCP_CONFIG)


def main():
    print("Reducto Agent Benchmark — Aider")
    print("=" * 60)
    print(SETUP)

    stubs = []

    for integration_path, capability, prompt_key in TESTS:
        prompt = PROMPTS[prompt_key]

        print(f"\n--- Test: {integration_path} / {capability} ---")
        print(f"Prompt to paste into Aider:\n")
        print(prompt)
        print()

        score = BenchmarkScore(
            platform=PLATFORM,
            model=MODEL,
            integration_path=integration_path,
            capability=capability,
            # Platform-level scores
            **PLATFORM_SCORES,
            # Capability-level scores — fill after running
            integration_complexity=0,
            error_recovery=0,
            output_quality=0,
            token_efficiency=0,
            notes=(
                f"STUB — paste the prompt above into `aider --model {MODEL} "
                f"--no-auto-commits` and fill scores after review."
            ),
        )

        out_path = save_result(score)
        stubs.append(out_path)

    print("\n" + "=" * 60)
    print(f"Created {len(stubs)} stub result file(s):")
    for p in stubs:
        print(f"  {p}")
    print("\nAfter running each prompt in Aider, fill in:")
    print("  integration_complexity, error_recovery, output_quality, token_efficiency")
    print("Then re-run benchmark/scripts/generate_report.py to update the comparison table.")


if __name__ == "__main__":
    main()
