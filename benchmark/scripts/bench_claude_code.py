"""
Benchmark: Claude Code + Reducto

Tests three integration paths:
  1. SDK-only (Python SDK, no MCP)
  2. MCP server (the one we built)
  3. CLI plugin (reducto-cli)

Run: python bench_claude_code.py
Requires: REDUCTO_API_KEY env var set
"""

import json
import subprocess
import time
from bench_utils import (
    BenchmarkScore,
    BenchmarkRun,
    STANDARD_PROMPT,
    STANDARD_PROMPT_WITH_MCP,
    TEST_DOC_URL,
    evaluate_output,
    save_result,
)


def test_sdk_integration() -> dict:
    """Test 1: Can Claude Code use the Reducto Python SDK correctly?"""
    
    print("\n📋 Test 1: SDK Integration")
    print("=" * 50)
    
    # We use `claude` CLI with a prompt that asks it to use the SDK
    prompt = f"""Write and execute a Python script that:
1. Installs the reductoai package
2. Uses it to parse this PDF: {TEST_DOC_URL}
3. Prints the first 3 chunks of extracted content
4. Counts the number of tables found

Do not ask for confirmation, just write and run the code.
"""
    
    start = time.time()
    
    # In practice, you'd run:
    # result = subprocess.run(
    #     ["claude", "--print", "--dangerously-skip-permissions", "-p", prompt],
    #     capture_output=True, text=True, timeout=120
    # )
    
    elapsed = time.time() - start
    
    return {
        "test": "sdk_integration",
        "prompt": prompt,
        "wall_time": elapsed,
        "notes": "Run manually with: claude -p '<prompt>'",
    }


def test_mcp_integration() -> dict:
    """Test 2: Can Claude Code discover and use Reducto via MCP?"""
    
    print("\n📋 Test 2: MCP Integration")
    print("=" * 50)
    
    prompt = STANDARD_PROMPT_WITH_MCP
    
    # Requires MCP server configured in ~/.claude/settings.json or project .mcp.json:
    # {
    #   "mcpServers": {
    #     "reducto": {
    #       "command": "npx",
    #       "args": ["tsx", "../mcp-server/src/index.ts"],
    #       "env": { "REDUCTO_API_KEY": "<key>" }
    #     }
    #   }
    # }
    
    start = time.time()
    
    # result = subprocess.run(
    #     ["claude", "--print", "--dangerously-skip-permissions", "-p", prompt],
    #     capture_output=True, text=True, timeout=120
    # )
    
    elapsed = time.time() - start
    
    return {
        "test": "mcp_integration",
        "prompt": prompt,
        "wall_time": elapsed,
        "notes": "Requires MCP server in .mcp.json",
    }


def test_cli_plugin() -> dict:
    """Test 3: Can Claude Code use the official Reducto CLI plugin?"""
    
    print("\n📋 Test 3: CLI Plugin")
    print("=" * 50)
    
    prompt = f"""Use the reducto CLI to parse this document: {TEST_DOC_URL}

Run: uvx --from reducto-cli reducto parse "{TEST_DOC_URL}"

Then summarize the key financial data from the output.
"""
    
    start = time.time()
    elapsed = time.time() - start
    
    return {
        "test": "cli_plugin",
        "prompt": prompt,
        "wall_time": elapsed,
        "notes": "Requires reducto-cli installed",
    }


def score_claude_code(test_results: list) -> BenchmarkScore:
    """Score Claude Code based on test results.
    
    Fill these in manually after running the tests — the scoring
    is qualitative on a 1-5 scale per dimension.
    """
    score = BenchmarkScore(
        platform="Claude Code",
        model="Opus 4.6",
        
        # FILL AFTER RUNNING:
        discovery=0,          # 1-5: Did it find reducto without help?
        integration_time=0,   # 1-5: How fast from prompt to result?
        error_recovery=0,     # 1-5: Did it handle errors gracefully?
        output_quality=0,     # 1-5: Was extracted data correct?
        token_efficiency=0,   # 1-5: Token usage reasonable?
        mcp_compatibility=0,  # 1-5: MCP tool discovery + invocation
        
        notes="Three integration paths tested: SDK, MCP, CLI plugin",
    )
    return score


def main():
    print("🚀 Reducto Agent Benchmark — Claude Code")
    print("=" * 60)
    
    run = BenchmarkRun(platform="Claude Code", model="Opus 4.6")
    
    results = []
    
    with run.start_phase("sdk_integration"):
        results.append(test_sdk_integration())
    
    with run.start_phase("mcp_integration"):
        results.append(test_mcp_integration())
    
    with run.start_phase("cli_plugin"):
        results.append(test_cli_plugin())
    
    # Print timing
    print("\n⏱️  Phase Timing:")
    for phase in run.phases:
        print(f"  {phase['phase']}: {phase['duration_seconds']}s")
    
    # Score (fill manually after running)
    score = score_claude_code(results)
    
    print("\n📊 Score (fill in after manual review):")
    print(f"  Total: {score.total_score}/{score.max_score}")
    
    # Save
    save_result(score)
    
    print("\n💡 Next: Run the actual prompts through Claude Code and fill in scores.")
    print("   Edit the score in benchmark/results/claude_code.json")


if __name__ == "__main__":
    main()
