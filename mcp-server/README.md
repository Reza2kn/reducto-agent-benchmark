# @reductoai/mcp-server

A standalone MCP (Model Context Protocol) server for Reducto's document
intelligence API. Lets any MCP-compatible AI agent — Claude Code, Cursor,
Codex, Cline, Gemini CLI, Continue.dev — parse, extract, split, edit, and
classify documents without writing integration code.

**Version:** 0.2.0

## Quick Start

```bash
# Install and run
npm install @reductoai/mcp-server
REDUCTO_API_KEY=your_key npx reducto-mcp

# Or run from source
git clone https://github.com/reductoai/mcp-server
cd mcp-server
npm install
REDUCTO_API_KEY=your_key npm run dev
```

## Connect to Your Agent

### Claude Desktop / Claude Code

Add to `~/.claude/settings.json` or project `.mcp.json`:

```json
{
  "mcpServers": {
    "reducto": {
      "command": "npx",
      "args": ["@reductoai/mcp-server"],
      "env": {
        "REDUCTO_API_KEY": "your_key"
      }
    }
  }
}
```

### Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "reducto": {
      "command": "npx",
      "args": ["@reductoai/mcp-server"],
      "env": {
        "REDUCTO_API_KEY": "your_key"
      }
    }
  }
}
```

### Cline (VS Code) / Continue.dev

Add to Cline or Continue MCP settings:

```json
{
  "mcpServers": {
    "reducto": {
      "command": "npx",
      "args": ["@reductoai/mcp-server"],
      "env": {
        "REDUCTO_API_KEY": "your_key"
      }
    }
  }
}
```

### OpenAI Codex CLI

```json
{
  "mcpServers": {
    "reducto": {
      "command": "npx",
      "args": ["@reductoai/mcp-server"]
    }
  }
}
```

## Available Tools

| Tool | Wraps | Key Params |
|------|-------|-----------|
| `reducto_parse` | `POST /parse` | input, agentic_scopes, chunk_mode, table_format, page_range, filter_blocks, merge_tables, extraction_mode, persist_results |
| `reducto_extract` | `POST /extract` | input, schema, array_extract, deep_extract, citations, system_prompt |
| `reducto_split` | `POST /split` | input, split_description, split_rules |
| `reducto_edit` | `POST /edit` | document_url, edit_instructions |
| `reducto_classify` | `POST /classify` | input, classification_schema |
| `reducto_upload` | `POST /upload` | file_url |
| `reducto_get_job` | `GET /job/{id}` | job_id |

---

### `reducto_parse`

Parse any document into structured markdown with tables, figures, and bounding
boxes.

```
Input:        URL or reducto:// file ID
Options:
  agentic_scopes    - restrict parse to specific content types
  chunk_mode        - how to chunk the output (page, section, etc.)
  table_format      - markdown, html, or json
  page_range        - e.g. "1-3" to parse a subset
  filter_blocks     - exclude block types (e.g. headers, footers)
  merge_tables      - merge fragmented tables across pages
  extraction_mode   - standard or high_quality
  persist_results   - keep results available for jobid:// reuse
Output: Markdown chunks with block-type annotations
```

**Example:** "Parse this PDF and show me the tables"

---

### `reducto_extract`

Extract specific fields from a document into typed JSON using a JSON Schema.

```
Input:        URL or file ID + JSON Schema string
Options:
  array_extract     - enable array extraction for repeating items
  deep_extract      - multi-pass extraction for complex nested fields
  citations         - return source page/location for each extracted value
  system_prompt     - override the system prompt for extraction
Output: Structured JSON matching your schema
```

**Example:** "Extract the invoice number, date, and line items from this PDF"

---

### `reducto_split`

Divide a document into named logical sections by page range.

```
Input:        URL or file ID + section definitions (JSON string)
Options:
  split_rules       - additional constraints on how splits are determined
Output: Section names with page ranges
```

**Example:** "Split this contract into cover page, terms, and signature sections"

---

### `reducto_edit`

Fill forms or modify documents using natural language instructions.

```
Input:        URL or file ID + natural language instructions
Output: URL to the edited document
```

**Example:** "Fill in name as John Doe, check the US Citizen box"

---

### `reducto_classify`

Classify a document's type based on provided categories.

```
Input:        URL or file ID + classification schema (JSON string)
Output: Matched category with confidence score
```

**Example:** "Is this an invoice, contract, or financial statement?"

---

### `reducto_upload`

Upload a file to Reducto and get a file ID for use with other tools.

```
Input:        Public URL of the file
Output: reducto:// file ID
```

---

### `reducto_get_job`

Poll the status and result of a previously submitted job.

```
Input:        job_id (returned by parse, extract, split, etc.)
Output: Job status and result when complete
```

Primarily used for `jobid://` chaining — see below.

---

## `jobid://` Chaining

The most credit-efficient pattern for multi-step workflows: parse once, reuse
the cached result for all subsequent operations.

### How It Works

1. Call `reducto_parse` with `persist_results: true`. The response includes a
   `job_id`.
2. Use `jobid://<job_id>` as the `input` for any subsequent tool call
   (`reducto_extract`, `reducto_split`, `reducto_classify`).
3. Reducto serves the cached parse result — no re-processing, no re-billing
   for the initial parse.

### Example Agent Workflow

```
# Step 1: Parse and persist
reducto_parse(input="https://example.com/report.pdf", persist_results=true)
→ { job_id: "abc123", chunks: [...] }

# Step 2: Extract using the cached parse
reducto_extract(input="jobid://abc123", schema="{...}")
→ { portfolio_value: ..., account_number: ... }

# Step 3: Split using the same cached parse
reducto_split(input="jobid://abc123", split_description="[...]")
→ { sections: [...] }

# Step 4: Classify using the same cached parse
reducto_classify(input="jobid://abc123", classification_schema="[...]")
→ { category: "financial_statement" }
```

Steps 2-4 each consume minimal credits because the document was already parsed.
On a 5-step workflow this typically saves ~60% compared to passing the URL
to each tool separately.

---

## Environment Variables

| Variable | Required | Description |
|----------|:--------:|-------------|
| `REDUCTO_API_KEY` | Yes | Your Reducto API key ([get one](https://studio.reducto.ai/)) |
| `REDUCTO_BASE_URL` | No | Override API base URL (default: `https://platform.reducto.ai`) |

## Supported File Types

PDF, DOCX, DOC, XLSX, XLS, CSV, PPTX, PPT, PNG, JPG, TIFF, GIF, BMP, HEIC,
RTF, TXT, and 20+ more.

## Development

```bash
# Install dependencies
npm install

# Run in development mode (hot reload)
npm run dev

# Build for production
npm run build

# Test with MCP Inspector
npm run build
npx @modelcontextprotocol/inspector node dist/index.js
```

## What's New in v0.2.0

- **`reducto_get_job`** — new tool for polling job status and enabling
  `jobid://` chaining
- **`reducto_parse`** — added `filter_blocks`, `merge_tables`,
  `extraction_mode`, `persist_results`
- **`reducto_extract`** — added `deep_extract`, `citations`, `system_prompt`
- **`reducto_split`** — added `split_rules`
- **Retry / backoff** — automatic retry on 429 and 5xx responses
- **Bug fixes** — `array_extract` placement, `split` confidence key, structured
  error parsing

## License

Apache-2.0
