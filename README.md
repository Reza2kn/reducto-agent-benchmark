# Reducto Agent Experience Benchmark

**How well do today's major AI agent platforms integrate with Reducto?**

This project benchmarks 13 AI agent platforms and model frameworks on a
standardized document processing task using Reducto's API, identifies friction
points in the current integration surface, and includes a **standalone MCP
server** to close the biggest gap found.

## Why This Exists

AI agents are becoming the primary way developers discover and integrate
document processing tools. If Reducto isn't frictionless inside the agent
platforms where developers already live, we lose deals we never even hear about.

This benchmark answers:
1. Which agent platforms integrate with Reducto most smoothly today?
2. Where do agents get stuck, hallucinate, or drop off?
3. What's the highest-leverage improvement to Reducto's agent-facing surface?

## Platforms Tested

### Coding Agents

| Platform | Model | MCP Support | Integration Methods |
|----------|-------|:-----------:|-------------------|
| Claude Code | Sonnet 4.6 | Native | MCP, Python SDK, CLI plugin |
| OpenAI Codex CLI | GPT-5.3 | Native | MCP, TypeScript SDK, REST |
| Cursor | Sonnet 4.6 | Native | MCP, Python/TS SDK |
| Gemini CLI | Gemini 3.1 Pro | Native | Python SDK, REST |
| Cline (OSS) | Configurable | Native | MCP, Python/TS SDK |
| Continue.dev (OSS) | Configurable | Native | MCP, SDK |
| Aider (OSS) | Configurable | Partial | Python SDK, REST |

### Open-Source Model Frameworks

| Platform | Notes |
|----------|-------|
| LangChain (OSS) | Python tool integration via REST / SDK |
| LlamaIndex (OSS) | Document pipeline integration |
| smolagents (OSS) | HuggingFace agent framework |

### OSS Models via Ollama

Llama 3.3, Qwen 2.5, DeepSeek Coder, Phi-4, Mistral — tested against the same
task using local inference + REST integration.

### Automation Platforms

| Platform | Notes |
|----------|-------|
| n8n | Node-based workflow automation |
| Dify | LLM app builder with REST tool support |

## Test Task

Each agent receives the same prompt:

> "Parse this financial statement PDF, extract the portfolio value table,
> account info, income summary, and top holdings. Return structured JSON."

**Test document:** [Fidelity sample statement](https://cdn.reducto.ai/samples/fidelity-example.pdf)
— multi-table layout, headers, account summaries, formatted text.

## Scoring Dimensions (0-5 each, 0 = N/A)

| Dimension | What We Measure |
|-----------|----------------|
| **Discovery** | Finds Reducto and picks the right endpoint without hand-holding |
| **Setup Friction** | Effort to get credentials, install deps, configure the integration |
| **Integration Complexity** | Tool calls / lines of code from prompt to working result |
| **Feature Coverage** | How many of the 7 capabilities the agent successfully exercises |
| **Error Recovery** | Auth failures, rate limits, malformed input — does it recover? |
| **Output Quality** | Accuracy of extracted data vs ground truth |
| **Token Efficiency** | Total tokens consumed to complete the task |
| **MCP Compatibility** | Correct discovery and invocation of MCP tools |

## Key Finding: MCP Compatibility Gates Feature Access

Reducto has excellent SDKs. The original gap — no official MCP server — is now
closed by `@reductoai/mcp-server` v0.2.0. The deeper finding: an agent using
REST typically discovers `/parse` and `/extract`. An agent using the MCP server
gets all 7 tools, plus `jobid://` chaining — parse once, reuse across N
operations, saving ~60% of credits on multi-step workflows.

## The Fix: `@reductoai/mcp-server` v0.2.0

This repo includes a production-ready MCP server wrapping all core endpoints:

```bash
cd mcp-server
npm install
REDUCTO_API_KEY=your_key npx reducto-mcp
```

Exposes these tools to any MCP-compatible agent:

| Tool | Description |
|------|-------------|
| `reducto_parse` | Parse documents → structured markdown + tables + figures |
| `reducto_extract` | Extract specific fields → typed JSON via schema |
| `reducto_split` | Split documents into named sections by page range |
| `reducto_edit` | Fill forms / modify documents with natural language |
| `reducto_classify` | Classify document type before processing |
| `reducto_upload` | Upload a local/remote file, returns `reducto://` file ID |
| `reducto_get_job` | Poll a job by ID — enables `jobid://` chaining |

### Connect to Claude Desktop / Code

```json
{
  "mcpServers": {
    "reducto": {
      "command": "npx",
      "args": ["@reductoai/mcp-server"],
      "env": { "REDUCTO_API_KEY": "your_key" }
    }
  }
}
```

### Connect to Cursor / Cline / Codex

```json
{
  "mcpServers": {
    "reducto": {
      "command": "npx",
      "args": ["@reductoai/mcp-server"],
      "env": { "REDUCTO_API_KEY": "your_key" }
    }
  }
}
```

## Running the Benchmark

```bash
cd benchmark/scripts

export REDUCTO_API_KEY=your_key

# Per-platform runners (print prompts to use with each platform)
python bench_claude_code.py
python bench_openai_codex.py
python bench_cursor.py
python bench_gemini_cli.py
python bench_cline.py

# Generate comparison report from scored JSON files
python generate_report.py
```

Results are saved to `benchmark/results/` (flat files + `automation/` subdir).
The report generator writes `report/results_matrix.md`.

## Project Structure

```
reducto-agent-benchmark/
├── mcp-server/            # Standalone Reducto MCP server (TypeScript)
│   ├── src/
│   │   └── index.ts       # 7 MCP tools, retry/backoff, jobid:// support
│   ├── package.json       # v0.2.0
│   └── README.md
├── benchmark/
│   ├── scripts/           # Per-platform benchmark runners + report generator
│   ├── results/           # Raw benchmark output (JSON)
│   │   └── automation/    # n8n, Dify results
│   └── test-docs/         # Test documents
├── report/
│   ├── findings.md        # Benchmark results + recommendations
│   └── results_matrix.md  # Auto-generated comparison tables
└── README.md
```

## Author

Reza Sayar — Data Engineer @ Reducto
