"""
Benchmark: Cline + Reducto
===========================
Cline (https://github.com/cline/cline) is an open-source VS Code extension that
provides an autonomous coding agent with strong MCP support. Each tool call triggers
an explicit approval prompt — making it a good test of how readable our MCP tool
descriptions are in practice.

Install:
    VS Code Extensions → search "Cline" → Install

Model:
    Configurable per user; default used here is claude-sonnet-4-6.

MCP config — VS Code extension settings → Cline → MCP Servers:
    {
      "mcpServers": {
        "reducto": {
          "command": "npx",
          "args": ["tsx", "<absolute-path-to>/mcp-server/src/index.ts"],
          "env": { "REDUCTO_API_KEY": "YOUR_KEY" }
        }
      }
    }

Integration paths tested:
    mcp         — MCP tools (reducto_parse, reducto_extract, reducto_split, etc.)
    python_sdk  — agent writes Python using `pip install reductoai`

Capabilities tested:
    parse           — parse doc, report tables and figures
    extract         — extract typed JSON fields via schema
    full_pipeline   — parse → extract → split → classify in one session
    jobid_chaining  — parse once, reuse job ID for extract + split
    python_sdk/extract — Python SDK extract with JSON schema

Run:
    python bench_cline.py
    Follow the printed setup instructions, paste each prompt into Cline's chat
    in VS Code, then fill in scores in the generated stub JSON files.
"""

from bench_utils import (
    BenchmarkScore,
    PROMPTS,
    TEST_DOC_URL,
    save_result,
)

# ---------------------------------------------------------------------------
# Platform-level constants
# ---------------------------------------------------------------------------

PLATFORM = "Cline"
MODEL = "Configurable (default: claude-sonnet-4-6)"

# Cline is renowned for its best-in-class MCP support — mcp_compatibility=5.
PLATFORM_SCORES = dict(
    setup_friction=3,    # Install extension + add MCP config — 2-3 steps
    feature_coverage=4,  # MCP + SDK paths; full tool coverage possible
    mcp_compatibility=5, # Cline is famous for excellent MCP discovery and invocation
    discovery=4,         # With MCP registered, tools are listed in Cline's context
)

# ---------------------------------------------------------------------------
# Test matrix: (integration_path, capability, prompt_key)
# ---------------------------------------------------------------------------

TESTS = [
    ("mcp",        "parse",         "mcp_parse"),
    ("mcp",        "extract",       "mcp_extract"),
    ("mcp",        "full_pipeline", "mcp_full_pipeline"),
    ("mcp",        "jobid_chaining","jobid_chaining"),
    ("python_sdk", "extract",       "extract"),
]

# ---------------------------------------------------------------------------
# MCP config (printed verbatim for copy-paste)
# ---------------------------------------------------------------------------

MCP_CONFIG_JSON = """{
  "mcpServers": {
    "reducto": {
      "command": "npx",
      "args": ["tsx", "<absolute-path-to>/mcp-server/src/index.ts"],
      "env": { "REDUCTO_API_KEY": "YOUR_KEY" }
    }
  }
}"""

# ---------------------------------------------------------------------------
# Setup instructions
# ---------------------------------------------------------------------------

SETUP = f"""
Setup — Cline + Reducto
-----------------------
1. Install the Cline extension:
       VS Code: Extensions panel → search "Cline" → Install

2. Configure your LLM provider in Cline's settings (API key for Claude, GPT, etc.).

3. Set your Reducto API key:
       export REDUCTO_API_KEY=your_reducto_key

4. Add the Reducto MCP server via VS Code → Settings → Cline → MCP Servers:

{MCP_CONFIG_JSON}

   Replace <absolute-path-to> with the real path to this repo's mcp-server/.

5. Each Reducto MCP tool call will show an approval prompt — accept to proceed.
   This tests how well our tool descriptions communicate intent.

6. Paste each prompt below into Cline's chat panel and score the results.
"""


def main():
    print("Reducto Agent Benchmark — Cline")
    print("=" * 60)
    print(SETUP)

    stubs = []

    for integration_path, capability, prompt_key in TESTS:
        prompt = PROMPTS[prompt_key]

        print(f"\n--- Test: {integration_path} / {capability} ---")
        print("Prompt to paste into Cline:\n")
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
                "STUB — paste the prompt above into Cline and fill scores after "
                "review. Note whether approval prompts were clearly worded."
            ),
        )

        out_path = save_result(score)
        stubs.append(out_path)

    print("\n" + "=" * 60)
    print(f"Created {len(stubs)} stub result file(s):")
    for p in stubs:
        print(f"  {p}")
    print("\nAfter running each prompt in Cline, fill in:")
    print("  integration_complexity, error_recovery, output_quality, token_efficiency")
    print("Then re-run benchmark/scripts/generate_report.py to update the comparison table.")


if __name__ == "__main__":
    main()
