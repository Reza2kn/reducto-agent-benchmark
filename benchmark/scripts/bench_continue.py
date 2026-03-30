"""
Benchmark: Continue.dev + Reducto
==================================
Continue (https://continue.dev) is an open-source VS Code / JetBrains AI extension
that provides inline completions, chat, and agent-style task execution. It supports
MCP natively via ~/.continue/config.json.

Install:
    VS Code Extensions → search "Continue" → Install
    (or JetBrains Plugin Marketplace → "Continue")

Model:
    Configurable per user; default used here is claude-sonnet-4-6.

MCP config — add to ~/.continue/config.json:
    {
      "mcpServers": [
        {
          "name": "reducto",
          "command": "npx",
          "args": ["tsx", "<absolute-path-to>/mcp-server/src/index.ts"],
          "env": { "REDUCTO_API_KEY": "YOUR_KEY" }
        }
      ]
    }

Integration paths tested:
    mcp         — uses MCP tools (reducto_parse, reducto_extract, etc.)
    python_sdk  — agent writes Python using `pip install reductoai`

Capabilities tested:
    mcp_parse          — parse via MCP tool, check table/figure counts
    mcp_extract        — parse with persist, then extract via jobid://
    jobid_chaining     — parse once, reuse job ID for extract + split
    python_sdk/extract — Python SDK extract with JSON schema

Run:
    python bench_continue.py
    Follow the printed setup instructions, paste each prompt into Continue's
    chat panel, then fill in scores in the generated stub JSON files.
"""

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

PLATFORM = "Continue.dev"
MODEL = "claude-sonnet-4-6"

PLATFORM_SCORES = dict(
    setup_friction=3,    # Install extension + edit config.json — 2-3 steps
    feature_coverage=4,  # MCP + SDK paths; missing CLI/workflow integrations
    mcp_compatibility=4, # Native MCP support; tool discovery solid but no auto-approval
    discovery=4,         # With MCP registered, tools appear in context automatically
)

# ---------------------------------------------------------------------------
# Test matrix: (integration_path, capability, prompt_key)
# ---------------------------------------------------------------------------

TESTS = [
    ("mcp",        "parse",        "mcp_parse"),
    ("mcp",        "extract",      "mcp_extract"),
    ("mcp",        "jobid_chaining", "jobid_chaining"),
    ("python_sdk", "extract",      "extract"),
]

# ---------------------------------------------------------------------------
# MCP config (printed verbatim for copy-paste)
# ---------------------------------------------------------------------------

MCP_CONFIG_JSON = """{
  "mcpServers": [
    {
      "name": "reducto",
      "command": "npx",
      "args": ["tsx", "<absolute-path-to>/mcp-server/src/index.ts"],
      "env": { "REDUCTO_API_KEY": "YOUR_KEY" }
    }
  ]
}"""

# ---------------------------------------------------------------------------
# Setup instructions
# ---------------------------------------------------------------------------

SETUP = f"""
Setup — Continue.dev + Reducto
-------------------------------
1. Install the Continue extension:
       VS Code: Extensions panel → search "Continue" → Install
       JetBrains: Plugins → Marketplace → "Continue"

2. Set your Reducto API key:
       export REDUCTO_API_KEY=your_reducto_key

3. Add the Reducto MCP server to ~/.continue/config.json:

{MCP_CONFIG_JSON}

   Replace <absolute-path-to> with the real path to this repo's mcp-server/.

4. Restart VS Code (or reload the Continue extension) so it picks up the new
   MCP server.

5. Open Continue's chat panel (Cmd+L / Ctrl+L) and paste each prompt below.
   Score results by editing the stub JSON files in benchmark/results/.
"""


def main():
    print("Reducto Agent Benchmark — Continue.dev")
    print("=" * 60)
    print(SETUP)

    stubs = []

    for integration_path, capability, prompt_key in TESTS:
        prompt = PROMPTS[prompt_key]

        print(f"\n--- Test: {integration_path} / {capability} ---")
        print("Prompt to paste into Continue chat:\n")
        print(prompt)
        print()

        score = BenchmarkScore(
            platform=PLATFORM,
            model=MODEL,
            integration_path=integration_path,
            capability=capability,
            **PLATFORM_SCORES,
            # Capability-level scores — fill after running
            integration_complexity=0,
            error_recovery=0,
            output_quality=0,
            token_efficiency=0,
            notes=(
                "STUB — paste the prompt above into Continue's chat panel and "
                "fill scores after review."
            ),
        )

        out_path = save_result(score)
        stubs.append(out_path)

    print("\n" + "=" * 60)
    print(f"Created {len(stubs)} stub result file(s):")
    for p in stubs:
        print(f"  {p}")
    print("\nAfter running each prompt in Continue, fill in:")
    print("  integration_complexity, error_recovery, output_quality, token_efficiency")
    print("Then re-run benchmark/scripts/generate_report.py to update the comparison table.")


if __name__ == "__main__":
    main()
