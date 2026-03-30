"""
Benchmark: Codex CLI + Reducto
================================
Codex CLI (https://github.com/openai/codex) is OpenAI's open-source terminal agent.
It runs autonomously in a sandboxed shell, writes and executes code, and supports
MCP servers via ~/.codex/config.json.

Install:
    npm install -g @openai/codex

Typical usage:
    codex "your task here"
    codex --model o4-mini "your task here"

MCP config — create or edit ~/.codex/config.json:
    {
      "mcp": {
        "servers": {
          "reducto": {
            "command": "npx",
            "args": ["tsx", "<absolute-path-to>/mcp-server/src/index.ts"],
            "env": { "REDUCTO_API_KEY": "YOUR_KEY" }
          }
        }
      }
    }

Integration paths tested:
    mcp       — MCP tools via ~/.codex/config.json
    node_sdk  — agent writes TypeScript/JS using `npm install reductoai`
    rest_api  — agent writes direct fetch() calls against platform.reducto.ai

Capabilities tested:
    mcp_parse          — parse doc via MCP tool
    mcp_extract        — parse with persist + extract via jobid://
    node_sdk/extract   — Node SDK (reductoai npm package) extract with JSON schema
    error_handling     — auth failure, bad URL, malformed schema (REST path)

Run:
    python bench_codex_cli.py
    Follow the setup instructions, run each prompt with `codex`, then fill in
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

PLATFORM = "Codex CLI"
MODEL = "o4-mini"

PLATFORM_SCORES = dict(
    setup_friction=4,    # npm install -g + set OPENAI_API_KEY + config.json — ~3 steps
    feature_coverage=4,  # MCP + Node SDK + REST; missing Python SDK path
    mcp_compatibility=4, # MCP via config.json; solid tool discovery
    discovery=4,         # o4-mini strong at tool use; finds Reducto tools with MCP loaded
)

# ---------------------------------------------------------------------------
# Test matrix: (integration_path, capability, prompt_key)
# ---------------------------------------------------------------------------

TESTS = [
    ("mcp",      "parse",          "mcp_parse"),
    ("mcp",      "extract",        "mcp_extract"),
    ("node_sdk", "extract",        "extract"),
    ("rest_api", "error_handling", "error_handling"),
]

# ---------------------------------------------------------------------------
# MCP config (printed verbatim for copy-paste)
# ---------------------------------------------------------------------------

MCP_CONFIG_JSON = """{
  "mcp": {
    "servers": {
      "reducto": {
        "command": "npx",
        "args": ["tsx", "<absolute-path-to>/mcp-server/src/index.ts"],
        "env": { "REDUCTO_API_KEY": "YOUR_KEY" }
      }
    }
  }
}"""

# ---------------------------------------------------------------------------
# Setup instructions
# ---------------------------------------------------------------------------

SETUP = f"""
Setup — Codex CLI + Reducto
----------------------------
1. Install Codex CLI:
       npm install -g @openai/codex

2. Set your OpenAI API key:
       export OPENAI_API_KEY=sk-...

3. Set your Reducto API key:
       export REDUCTO_API_KEY=your_reducto_key

4. Add the Reducto MCP server to ~/.codex/config.json:

{MCP_CONFIG_JSON}

   Replace <absolute-path-to> with the real path to this repo's mcp-server/.

5. Run each prompt below with:
       codex --model o4-mini "<prompt>"
   or interactively:
       codex   # then paste the prompt

Note: The node_sdk/extract test asks Codex to install `reductoai` (npm) and write
a TypeScript snippet. It will execute in a sandboxed shell — check that
REDUCTO_API_KEY is visible in the shell environment.
"""


def main():
    print("Reducto Agent Benchmark — Codex CLI")
    print("=" * 60)
    print(SETUP)

    stubs = []

    for integration_path, capability, prompt_key in TESTS:
        prompt = PROMPTS[prompt_key]

        # The node_sdk/extract test needs a slight framing tweak — the shared
        # PROMPTS["extract"] is written for Python. Print a wrapper note.
        node_sdk_note = ""
        if integration_path == "node_sdk":
            node_sdk_note = (
                "\n[Note for Codex: use the `reductoai` npm package, not the Python SDK. "
                "Write a TypeScript or JavaScript script.]\n"
            )

        print(f"\n--- Test: {integration_path} / {capability} ---")
        print("Prompt to run with `codex`:\n")
        print(node_sdk_note + prompt)
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
                "STUB — run the prompt above with `codex --model o4-mini` and "
                "fill scores after review."
            ),
        )

        out_path = save_result(score)
        stubs.append(out_path)

    print("\n" + "=" * 60)
    print(f"Created {len(stubs)} stub result file(s):")
    for p in stubs:
        print(f"  {p}")
    print("\nAfter running each prompt in Codex CLI, fill in:")
    print("  integration_complexity, error_recovery, output_quality, token_efficiency")
    print("Then re-run benchmark/scripts/generate_report.py to update the comparison table.")


if __name__ == "__main__":
    main()
