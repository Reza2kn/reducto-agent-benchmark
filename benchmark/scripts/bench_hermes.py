"""
Benchmark: Hermes Agent (NousResearch) + Reducto via MCP

Hermes Agent is a CLI tool (https://github.com/nousresearch/hermes-agent), not a
Python library — so this script generates the MCP config and prompts to paste into
the CLI, then provides a manual scoring scaffold once you run them.

How to use:
    1. Install Hermes Agent:
           npm install -g @nousresearch/hermes-agent
       (or clone & build from https://github.com/nousresearch/hermes-agent)

    2. Run this script to get the config + prompts:
           python bench_hermes.py

    3. Copy the .mcp.json snippet to your project root, then run each prompt via:
           hermes "<prompt>"

    4. Manually record results in benchmark/results/hermes/ and update scores below.

Run:
    python bench_hermes.py [--score]   # --score: print current stub scores
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from bench_utils import BenchmarkScore, PROMPTS, save_result

MCP_CONFIG = {
    "mcpServers": {
        "reducto": {
            "command": "npx",
            "args": ["tsx", "mcp-server/src/index.ts"],
            "env": {
                "REDUCTO_API_KEY": "${REDUCTO_API_KEY}"
            }
        }
    }
}

HERMES_PROMPTS = {
    "mcp_parse":          PROMPTS["mcp_parse"],
    "mcp_extract":        PROMPTS["mcp_extract"],
    "mcp_full_pipeline":  PROMPTS["mcp_full_pipeline"],
    "error_handling":     PROMPTS["error_handling"],
}


def print_setup() -> None:
    print("=" * 70)
    print("Hermes Agent × Reducto — Manual Benchmark Setup")
    print("=" * 70)

    print("\n[1] Install Hermes Agent")
    print("    npm install -g @nousresearch/hermes-agent")

    print("\n[2] Project .mcp.json — add to your repo root:")
    print(json.dumps(MCP_CONFIG, indent=2))

    print("\n[3] Set env vars:")
    print(f"    export REDUCTO_API_KEY=<your key>")
    print(f"    export ANTHROPIC_API_KEY=<your key>  # Hermes uses Claude by default")

    print("\n[4] Run each prompt with hermes CLI:")
    for cap, prompt in HERMES_PROMPTS.items():
        print(f"\n--- {cap} ---")
        print(f'hermes "{prompt[:120]}..."')
        print(f"Full prompt ({len(prompt)} chars):")
        print(prompt)
        print()

    print("\n[5] After each run, record results in benchmark/results/hermes/")
    print("    Then re-run:  python bench_hermes.py --score")


def record_stub_scores() -> None:
    """
    Stub scores for Hermes Agent. Update these after manual testing.
    Hermes uses MCP natively via its tool-calling loop, so mcp_compatibility is scored.
    """
    stubs = [
        dict(
            capability="mcp_parse",
            discovery=4,         # Hermes discovers MCP tools from server manifest
            setup_friction=3,    # CLI install + .mcp.json config
            integration_complexity=5,  # zero user code — pure CLI + MCP
            feature_coverage=3,
            error_recovery=3,
            output_quality=0,    # fill in after running
            token_efficiency=3,
            mcp_compatibility=4, # MCP-native; tool invocation is direct
            notes="Hermes Agent MCP parse test — stub, update after manual run.",
        ),
        dict(
            capability="mcp_extract",
            discovery=4,
            setup_friction=3,
            integration_complexity=5,
            feature_coverage=4,
            error_recovery=3,
            output_quality=0,
            token_efficiency=3,
            mcp_compatibility=4,
            notes="Hermes Agent MCP extract with jobid:// — stub.",
        ),
        dict(
            capability="mcp_full_pipeline",
            discovery=4,
            setup_friction=3,
            integration_complexity=5,
            feature_coverage=5,
            error_recovery=3,
            output_quality=0,
            token_efficiency=3,
            mcp_compatibility=5,
            notes="Hermes Agent full pipeline test — stub.",
        ),
        dict(
            capability="error_handling",
            discovery=3,
            setup_friction=3,
            integration_complexity=5,
            feature_coverage=2,
            error_recovery=0,    # fill in after running
            output_quality=0,
            token_efficiency=3,
            mcp_compatibility=3,
            notes="Hermes Agent error handling test — stub.",
        ),
    ]

    for s in stubs:
        score = BenchmarkScore(
            platform="Hermes",
            model="hermes-3-llama-3.1-70b",  # default Hermes model
            integration_path="mcp",
            **s,
        )
        saved = save_result(score, subdir="hermes")
        print(f"Saved stub: {saved}")

    print("\nStub scores saved. Re-run after manual testing to fill in output_quality/error_recovery.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes Agent × Reducto benchmark")
    parser.add_argument("--score", action="store_true", help="Save stub score files")
    args = parser.parse_args()

    if args.score:
        record_stub_scores()
    else:
        print_setup()


if __name__ == "__main__":
    main()
