"""
Benchmark: Cursor Agent Mode + Reducto

Test in Cursor by opening a new Composer/Agent chat and pasting the prompt.
Requires Reducto MCP server in Cursor's mcp.json.

Run: python bench_cursor.py (prints prompts, scoring is manual)
"""

from bench_utils import BenchmarkScore, STANDARD_PROMPT_WITH_MCP, save_result

def main():
    print("🚀 Reducto Agent Benchmark — Cursor")
    print("=" * 60)
    print("\n1. Add to ~/.cursor/mcp.json:")
    print("""   {
     "mcpServers": {
       "reducto": {
         "command": "npx",
         "args": ["tsx", "<path>/mcp-server/src/index.ts"],
         "env": { "REDUCTO_API_KEY": "<key>" }
       }
     }
   }""")
    print(f"\n2. Open Cursor Agent mode and paste:\n{STANDARD_PROMPT_WITH_MCP}")
    
    score = BenchmarkScore(platform="Cursor", model="Sonnet 4.6")
    save_result(score)

if __name__ == "__main__":
    main()
