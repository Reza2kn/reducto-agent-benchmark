# CLAUDE.md — Reducto Agent Experience Benchmark

## What This Project Is

A benchmark of how well major AI agent platforms integrate with Reducto's document processing API, plus a **standalone MCP server** that closes the biggest gap found (Reducto has no official MCP server).

This is a weekend project by Reza Sayar (Data Engineer @ Reducto) to demonstrate the value of improving Reducto's agent-facing surface — specifically for the "Agent Experience Engineer" role.

## Project Structure

```
reducto-agent-benchmark/
├── mcp-server/              # Standalone Reducto MCP server (TypeScript)
│   ├── src/index.ts         # MCP server implementation — 6 tools
│   ├── package.json
│   ├── tsconfig.json
│   └── README.md
├── benchmark/
│   ├── scripts/
│   │   ├── bench_utils.py   # Shared scoring framework + test prompts
│   │   ├── bench_claude_code.py
│   │   ├── bench_openai_codex.py
│   │   ├── bench_cursor.py
│   │   ├── bench_gemini_cli.py
│   │   ├── bench_cline.py
│   │   └── generate_report.py
│   ├── results/             # JSON score files (generated)
│   └── test-docs/           # Test documents
├── report/
│   └── findings.md          # Benchmark results + recommendations
├── README.md
└── CLAUDE.md                # This file
```

## Reducto API — Quick Reference

- **Base URL:** `https://platform.reducto.ai`
- **Auth:** `Authorization: Bearer $REDUCTO_API_KEY` (env var should be set)
- **Docs:** https://docs.reducto.ai/agent-guide (dense agent reference)
- **Python SDK:** `pip install reductoai` → `from reducto import Reducto`
- **Node SDK:** `npm install reductoai` → `import Reducto from 'reductoai'`

### Core Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /parse` | Document → structured markdown + tables + figures with bboxes |
| `POST /extract` | Document + JSON Schema → typed JSON fields |
| `POST /split` | Document + section defs → page ranges per section |
| `POST /edit` | Document + NL instructions → edited document URL |
| `POST /classify` | Document + categories → classification result |
| `POST /upload` | File → `reducto://` file ID |

### Test Document

Standard test doc for all benchmarks:
`https://cdn.reducto.ai/samples/fidelity-example.pdf`

This is a Fidelity financial statement with multi-table layout, headers, account summaries, and formatted text.

## MCP Server

Located in `mcp-server/`. Uses `@modelcontextprotocol/sdk` and `zod`.

### Setup & Run

```bash
cd mcp-server
npm install
REDUCTO_API_KEY=$REDUCTO_API_KEY npm run dev    # development (tsx)
REDUCTO_API_KEY=$REDUCTO_API_KEY npm start      # production (compiled)
```

### Test with MCP Inspector

```bash
cd mcp-server
npm run build
npx @modelcontextprotocol/inspector node dist/index.js
```

### Tools Exposed

| MCP Tool | Wraps Endpoint | Key Params |
|----------|---------------|------------|
| `reducto_parse` | `/parse` | input, agentic_scopes, chunk_mode, table_format, page_range |
| `reducto_extract` | `/extract` | input, schema (JSON string), array_extract |
| `reducto_split` | `/split` | input, split_description (JSON string) |
| `reducto_edit` | `/edit` | document_url, edit_instructions |
| `reducto_classify` | `/classify` | input, classification_schema (JSON string) |
| `reducto_upload` | `/upload` | file_url |

### How to Use This MCP Server From Claude Code

Add to project `.mcp.json`:

```json
{
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
```

## Benchmark Framework

### Scoring (6 dimensions × 1-5 scale = 30 max)

| Dimension | What It Measures |
|-----------|-----------------|
| Discovery | Can the agent find Reducto / pick the right endpoint without hand-holding? |
| Integration Time | Lines of code or tool calls from prompt to working result |
| Error Recovery | Auth failures, rate limits, malformed input — does it recover? |
| Output Quality | Accuracy of extracted data vs the known content of the test PDF |
| Token Efficiency | Total tokens consumed for the task |
| MCP Compatibility | Correct discovery + invocation of MCP tools |

### Running Benchmarks

Each `bench_*.py` script prints the prompts to use with that platform. The actual running is manual (you paste prompts into the agent), then score in the generated JSON files in `benchmark/results/`.

```bash
cd benchmark/scripts
python bench_claude_code.py   # prints prompts for 3 Claude Code tests
python bench_openai_codex.py  # prints prompts for Codex CLI
python bench_cursor.py        # prints MCP config + prompt for Cursor
python bench_gemini_cli.py    # prints prompt for Gemini CLI
python bench_cline.py         # prints MCP config + prompt for Cline
python generate_report.py     # compiles results into comparison table
```

### Ground Truth (what we expect agents to extract)

The Fidelity sample PDF contains:
- Portfolio value table with beginning/ending values
- Account number and account type
- Income summary by tax category (taxable, tax-exempt, etc.)
- Top holdings with names, values, and percentages
- ~5 pages, at least 4 tables

## Common Tasks You'll Help With

1. **Improving the MCP server** — adding error handling, better tool descriptions, new params, Streamable HTTP transport for remote hosting
2. **Running benchmark tests** — executing Reducto API calls, evaluating output quality against ground truth
3. **Writing the report** — filling in scores, adding per-platform notes in `report/findings.md`
4. **Testing edge cases** — malformed URLs, auth errors, large docs, spreadsheets, images
5. **Polishing for sharing** — the README, MCP server README, and findings report should be clean enough to share with Reducto's ML team lead

## Style Notes

- Keep code clean and minimal — this is a proof of concept, not a production system
- TypeScript for the MCP server (matches Reducto's Node SDK ecosystem)
- Python for benchmark scripts (matches Reducto's primary SDK)
- Comments should explain *why*, not *what*
- The report should read as professional but not corporate — Reza's voice, not a committee's

## Environment

- Node 18+ required for MCP server
- Python 3.9+ for benchmark scripts
- `REDUCTO_API_KEY` must be set as environment variable
- MCP server communicates over stdio (standard MCP transport)
