"""
Benchmark: Gemini CLI + Reducto
================================
Gemini CLI (https://github.com/google-gemini/gemini-cli) is Google's open-source
terminal agent powered by Gemini models. It supports MCP servers via a config file
at ~/.gemini/settings.json.

Install:
    npm install -g @google/gemini-cli
    gemini auth   # log in with a Google account or set GEMINI_API_KEY

Model: gemini-2.5-pro

MCP config — create or edit ~/.gemini/settings.json:
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
    mcp         — MCP tools via ~/.gemini/settings.json config
    python_sdk  — agent writes Python using `pip install reductoai`
    rest_api    — agent writes direct HTTP calls against platform.reducto.ai

Capabilities tested:
    mcp_parse          — parse doc via MCP tool
    mcp_extract        — parse with persist + extract via jobid://
    python_sdk/extract — Python SDK extract with JSON schema
    error_handling     — auth failure, bad URL, malformed schema (REST path)

Run:
    python bench_gemini_cli.py
    Follow the setup instructions, run each prompt with `gemini`, then fill in
    scores in the generated stub JSON files.
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

PLATFORM = "Gemini CLI"
MODEL = "gemini-2.5-pro"

PLATFORM_SCORES = dict(
    setup_friction=4,    # npm install + auth + settings.json — about 3 steps
    feature_coverage=4,  # MCP + SDK + REST paths; missing workflow integrations
    mcp_compatibility=4, # MCP supported via settings.json; discovery is solid
    discovery=4,         # Gemini 2.5 Pro has strong tool use; finds Reducto tools reliably
)

# ---------------------------------------------------------------------------
# Test matrix: (integration_path, capability, prompt_key)
# ---------------------------------------------------------------------------

TESTS = [
    ("mcp",        "parse",          "mcp_parse"),
    ("mcp",        "extract",        "mcp_extract"),
    ("python_sdk", "extract",        "extract"),
    ("rest_api",   "error_handling", "error_handling"),
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
Setup — Gemini CLI + Reducto
-----------------------------
1. Install Gemini CLI:
       npm install -g @google/gemini-cli

2. Authenticate:
       gemini auth
   Or set an API key directly:
       export GEMINI_API_KEY=your_gemini_key

3. Set your Reducto API key:
       export REDUCTO_API_KEY=your_reducto_key

4. Add the Reducto MCP server to ~/.gemini/settings.json:

{MCP_CONFIG_JSON}

   Replace <absolute-path-to> with the real path to this repo's mcp-server/.

5. Verify MCP is loaded:
       gemini --list-tools   # should include reducto_parse, reducto_extract, etc.

6. Run each prompt below with:
       gemini -p "<prompt>"
   or interactively:
       gemini   # then paste the prompt
"""


def main():
    print("Reducto Agent Benchmark — Gemini CLI")
    print("=" * 60)
    print(SETUP)

    stubs = []

    for integration_path, capability, prompt_key in TESTS:
        prompt = PROMPTS[prompt_key]

        print(f"\n--- Test: {integration_path} / {capability} ---")
        print("Prompt to run with `gemini`:\n")
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
                "STUB — run the prompt above with `gemini` and fill scores after "
                "review."
            ),
        )

        out_path = save_result(score)
        stubs.append(out_path)

    print("\n" + "=" * 60)
    print(f"Created {len(stubs)} stub result file(s):")
    for p in stubs:
        print(f"  {p}")
    print("\nAfter running each prompt in Gemini CLI, fill in:")
    print("  integration_complexity, error_recovery, output_quality, token_efficiency")
    print("Then re-run benchmark/scripts/generate_report.py to update the comparison table.")


if __name__ == "__main__":
    main()
