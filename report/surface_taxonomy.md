# Reducto Agent Surface Taxonomy

**Purpose:** Complete map of every surface an agent can touch when working with Reducto —
what exists, what works well, what's missing, and what we need to generate training data for.
**Author:** Reza Sayar
**Date:** 2026-03-28

---

## 1. Existing API Surfaces

### 1.1 Core Endpoints

| Endpoint | Method | Sync? | Async? | Primary Use |
|----------|--------|-------|--------|-------------|
| `/parse` | POST | ✅ | ✅ (`/parse_async`) | Document → structured markdown + blocks |
| `/extract` | POST | ✅ | ✅ (`/extract_async`) | Document + JSON schema → typed JSON |
| `/split` | POST | ✅ | ✅ (`/split_async`) | Document + section defs → page ranges |
| `/edit` | POST | ✅ | ✅ (`/edit_async`) | Document + NL instructions → edited doc URL |
| `/classify` | POST | ✅ | ✅ (`/classify_async`) | Document + categories → classification |
| `/upload` | POST | ✅ | — | File URL → `reducto://` stable file ID |
| `/job/{id}` | GET | ✅ | — | Poll async job status / retrieve result |

---

### 1.2 Input Types (cross-endpoint)

All processing endpoints accept the same input variants:

| Input Type | Format | Notes |
|-----------|--------|-------|
| Public URL | `https://...` | Direct download, must be publicly accessible |
| Presigned cloud URL | `https://s3.amazonaws.com/...`, GCS, Azure | Signed URLs work; short-lived ones should use `/upload` first |
| Reducto file ID | `reducto://uuid` | From `/upload`; stable for 24h, re-usable across calls |
| Job ID reference | `jobid://uuid` | Reuse a previous parse result without re-processing |
| URL array | `["url1","url2"]` | Multi-document batch — treated as one logical document |
| Edit instruction string | `"string"` | Only for `/edit` — the NL edit instruction |

**Training coverage needed:** All 6 input types × all 5 processing endpoints.

---

### 1.3 `/parse` — Full Parameter Map

**Enhance:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `enhance.agentic` | `[{scope: "text"|"table"|"figure"}]` | — | Vision LLM correction pass per block type |
| `enhance.summarize_figures` | bool | false | Generate text summary of figures |
| `enhance.intelligent_ordering` | bool | false | Reorder blocks by reading flow, not DOM order |

**Retrieval:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `retrieval.chunking.chunk_mode` | enum | `disabled` | `variable`, `section`, `page`, `block`, `page_sections` |
| `retrieval.chunking.chunk_size` | int | 500 | Target tokens per chunk (250–1500) |
| `retrieval.chunking.chunk_overlap` | int | 0 | Token overlap between adjacent chunks |
| `retrieval.filter_blocks` | enum[] | — | Block types to exclude from output |
| `retrieval.embedding_optimized` | bool | false | Format output for embedding (removes markup) |

**Block types** (for `filter_blocks`): `Header`, `Footer`, `Title`, `Section Header`, `Page Number`, `List Item`, `Figure`, `Table`, `Key Value`, `Text`, `Comment`, `Signature`

**Formatting:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `formatting.table_output_format` | enum | `dynamic` | `html`, `md`, `json`, `csv`, `jsonbbox` |
| `formatting.page_markers` | bool | false | Insert `## Page N` headers |
| `formatting.merge_tables` | bool | false | Stitch tables split across page breaks |
| `formatting.include` | object | — | `change_tracking`, `highlight`, `comments`, `hyperlinks`, `signatures` |

**Spreadsheet:**
| Param | Type | Notes |
|-------|------|-------|
| `spreadsheet.split_large_tables` | bool | Break oversized tables into smaller chunks |
| `spreadsheet.clustering` | enum | `accurate`/`fast`/`disabled` |
| `spreadsheet.cell_colors` | bool | Include cell background color |
| `spreadsheet.formulas` | bool | Return formula strings not computed values |
| `spreadsheet.dropdowns` | bool | Return dropdown option lists |
| `spreadsheet.hidden_rows/sheets/cols` | bool | Include hidden content |

**Settings:**
| Param | Type | Notes |
|-------|------|-------|
| `settings.ocr_system` | enum | `standard`/`legacy` |
| `settings.extraction_mode` | enum | `ocr`/`hybrid` (hybrid = native text + OCR fallback) |
| `settings.return_ocr_data` | bool | Return raw OCR confidence + coordinates |
| `settings.return_images` | enum[] | `figure`, `table`, `page` — return presigned image URLs |
| `settings.page_range` | `{start, end}` | 1-indexed page slice |
| `settings.document_password` | string | Password for encrypted PDFs |
| `settings.timeout` | int | Per-request timeout seconds |
| `settings.force_url_result` | bool | Always return URL result even for small docs |
| `settings.persist_results` | bool | Keep result accessible via `jobid://` for 1 hour |

**Response:**
```json
{
  "job_id": "uuid",
  "usage": { "num_pages": 5, "credits": 10 },
  "duration": 2.3,
  "result": {
    "type": "full" | "url",
    "chunks": [
      {
        "content": "markdown text",
        "embed": "embedding-optimized text",
        "blocks": [
          {
            "type": "Table",
            "bbox": {"left":0.1,"top":0.2,"width":0.8,"height":0.3,"page":1},
            "content": "...",
            "image_url": "https://...",
            "chart_data": {...},
            "confidence": 0.97
          }
        ]
      }
    ]
  }
}
```

---

### 1.4 `/extract` — Full Parameter Map

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `input` | string/array | ✅ | Any input type from §1.2 |
| `instructions.schema` | JSON Schema | ✅ | Target schema; field `description` used as extraction hint |
| `instructions.system_prompt` | string | — | Custom extraction instructions |
| `settings.array_extract` | bool | — | Extract repeated rows (line items, transactions) |
| `settings.deep_extract` | bool | — | Agentic refinement pass; higher cost/latency |
| `settings.citations` | `{enabled, numerical_confidence}` | — | Return bbox for each extracted field |
| `settings.include_images` | bool | — | Include chart/figure context in extraction |
| `settings.optimize_for_latency` | bool | — | Faster, slightly less accurate |
| Parsing options | object | — | Full `enhance`/`retrieval`/`formatting`/`settings` from `/parse` |

**Response:**
```json
{
  "job_id": "uuid",
  "usage": { "num_pages": 5, "fields": 12, "credits": 15 },
  "studio_link": "https://studio.reducto.ai/...",
  "result": { "invoice_number": "INV-001", "total": 1250.00 }
}
```

---

### 1.5 `/split` — Full Parameter Map

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `input` | string/array | ✅ | Any input type from §1.2 |
| `split_description` | `[{name, description, partition_key?}]` | ✅ | Section definitions |
| `split_rules` | string | — | NL override rules for splitting logic |
| `settings.table_cutoff` | enum | — | `truncate`/`preserve` at section boundaries |
| Parsing options | object | — | Forwarded to internal parse step |

**Response:**
```json
{
  "job_id": "uuid",
  "usage": { "num_pages": 20, "credits": 8 },
  "result": {
    "section_mapping": {...},
    "splits": [
      { "name": "Executive Summary", "pages": [[1,2]], "conf": 0.95 },
      { "name": "Financials", "pages": [[3,8]], "conf": 0.88 }
    ]
  }
}
```

---

### 1.6 `/classify` — Full Parameter Map

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `input` | string/array | ✅ | Any input type from §1.2 |
| `classification_schema` | `[{category, criteria[]}]` | ✅ | Category definitions with criteria list |
| `document_metadata` | string | — | Context hint about the document origin/type |
| `page_range` | `{start, end}` | — | Only classify first N pages; max 10 pages |

---

### 1.7 `/edit` — Full Parameter Map

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `document_url` | string | ✅ | URL or `reducto://` file ID |
| `edit_instructions` | string | ✅ | NL description of edits to make |
| `edit_options.color` | hex string | — | Highlight color for filled fields. Default: `#FF0000` |
| `edit_options.font_size` | int (1–72) | — | Font size for new text |
| `edit_options.llm_provider_preference` | enum | — | `openai`, `anthropic`, `google` |
| `edit_options.enable_overflow_pages` | bool | — | Allow content to add new pages |
| `edit_options.flatten` | bool | — | Convert form fields to static text |
| `edit_options.form_schema` | EditWidget[] | — | Structured form field definitions |

---

### 1.8 `/upload` — Full Parameter Map

| Param | Type | Notes |
|-------|------|-------|
| `url` | string | Via JSON body — upload from URL |
| `file` | binary | Via multipart/form-data — direct file upload |
| `extension` | string | Optional file extension hint |

**Response:** `{ "file_id": "reducto://uuid", "presigned_url": "https://..." }`

---

### 1.9 `/job/{id}` — Async Polling

```
GET /job/{uuid}
Authorization: Bearer $KEY

Response: { "status": "processing"|"completed"|"failed", "result": {...}, "error": "..." }
```

**Polling pattern agents must know:**
```python
import time
while True:
    r = client.get_job(job_id)
    if r.status in ("completed", "failed"):
        break
    time.sleep(2)
```

---

## 2. SDK Surfaces

### 2.1 Python SDK (`reductoai`)

```python
from reducto import Reducto
client = Reducto(api_key=os.environ["REDUCTO_API_KEY"])

client.parse(input="url", ...)         # → ParseResult
client.extract(input="url", ...)       # → ExtractResult
client.split(input="url", ...)         # → SplitResult
client.edit(document_url="url", ...)   # → EditResult
client.classify(input="url", ...)      # → ClassifyResult
client.upload(url="url")               # → UploadResult

# Async variants
job = client.parse_async(input="url", ...)    # → AsyncJob
result = client.get_job(job.job_id)           # → poll
```

**Key agent patterns to learn:**
- `from reducto import Reducto` (not `reductoai`)
- Environment variable: `REDUCTO_API_KEY`
- Chaining: `upload()` → `reducto://` → `parse()` → `jobid://` → `extract()`
- Structured schema: `json.dumps(schema_dict)` before passing to `extract()`

---

### 2.2 Node.js SDK (`reductoai`)

```typescript
import Reducto from "reductoai";
const client = new Reducto({ apiKey: process.env.REDUCTO_API_KEY });

await client.parse({ input: "url", ... });
await client.extract({ input: "url", instructions: { schema: {} }, ... });
await client.split({ input: "url", splitDescription: [...], ... });
await client.edit({ documentUrl: "url", editInstructions: "...", ... });
await client.classify({ input: "url", classificationSchema: [...], ... });
await client.upload({ url: "url" });
await client.getJob("job_id");
```

**Key differences from Python SDK:**
- camelCase params (`splitDescription` not `split_description`)
- Schema passed as object, not JSON string
- `new Reducto({apiKey})` constructor pattern

---

### 2.3 Go SDK

Homepage confirms Go SDK exists. Full surface currently unknown — no public docs found.
**Gap:** No confirmed method signatures. Training data for Go needs to be derived from
the REST API with idiomatic Go patterns.

Likely surface:
```go
import "github.com/reductoai/reducto-go"

client := reducto.NewClient(os.Getenv("REDUCTO_API_KEY"))
result, err := client.Parse(ctx, reducto.ParseParams{Input: "url"})
result, err := client.Extract(ctx, reducto.ExtractParams{...})
// etc.
```

---

### 2.4 REST API (direct HTTP)

Any language. Pattern:
```bash
curl -X POST https://platform.reducto.ai/parse \
  -H "Authorization: Bearer $REDUCTO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "https://..."}'
```

**Agents must know:**
- Base URL: `https://platform.reducto.ai`
- Auth: `Authorization: Bearer $KEY` (not `API-Key`, not `X-API-Key`)
- Content-Type always `application/json` (except `/upload` multipart)
- Error format: `{ "detail": [{ "loc": [...], "msg": "...", "type": "..." }] }` (FastAPI)

---

### 2.5 MCP Server (our implementation)

7 tools exposed:

| Tool | Wraps | Status |
|------|-------|--------|
| `reducto_parse` | `/parse` | ✅ Full param coverage |
| `reducto_extract` | `/extract` | ✅ Full param coverage |
| `reducto_split` | `/split` | ✅ Full param coverage |
| `reducto_edit` | `/edit` | ✅ Full param coverage |
| `reducto_classify` | `/classify` | ✅ Full param coverage |
| `reducto_upload` | `/upload` | ✅ URL-based upload |
| `reducto_get_job` | `/job/{id}` | ✅ Status polling |

**Missing from MCP server:**
- Binary file upload (only URL-based upload)
- Async job submission + webhook registration
- Batch processing

---

## 3. Integration Patterns

These are the higher-level workflows agents actually perform — and the primary unit of training data.

### 3.1 Single-step patterns

| Pattern | Description | Surfaces |
|---------|-------------|---------|
| `parse_url` | Parse a public PDF URL | REST / Python / Node / Go / MCP |
| `extract_fields` | Extract JSON schema from a doc | REST / Python / Node / Go / MCP |
| `classify_then_route` | Classify doc type, pick extraction schema | REST / Python / Node / Go / MCP |
| `split_sections` | Identify logical sections in a long doc | REST / Python / Node / Go / MCP |
| `fill_form` | Fill a PDF form with given values | REST / Python / Node / Go / MCP |
| `upload_private_file` | Upload via URL, get stable ID | REST / Python / Node / Go / MCP |

### 3.2 Multi-step pipelines

| Pattern | Steps | Value |
|---------|-------|-------|
| **Upload → Parse** | `upload()` → `reducto://` → `parse()` | Private/expiring docs |
| **Parse → Extract** | `parse(persist=true)` → `jobid://` → `extract()` | Avoid re-parsing |
| **Parse → Split** | `parse(persist=true)` → `jobid://` → `split()` | Section-aware extraction |
| **Classify → Extract** | `classify()` → pick schema → `extract()` | Multi-doc-type pipelines |
| **Split → Extract per section** | `split()` → page ranges → `parse()` each → `extract()` | Long structured docs |
| **Full pipeline** | `classify()` → `split()` → `parse(section)` → `extract()` | Enterprise doc processing |
| **Async batch** | `parse_async()` × N → poll all → aggregate | High-volume ingestion |

### 3.3 Error recovery patterns

| Error | Code | Agent should... |
|-------|------|----------------|
| Invalid API key | 401 | Show clear auth error, never retry |
| Malformed input | 422 | Parse `detail` array, fix the specific field |
| Rate limit | 429 | Respect `Retry-After` header, exponential backoff |
| Server error | 500 | Retry up to 3× with backoff |
| Invalid JSON schema | 422 | Validate schema locally before sending |
| Doc too large (>100MB) | 422 | Use `/upload` presigned URL path instead |
| Unsupported file type | 422 | Check supported formats, suggest conversion |
| Job not found | 404 | `jobid://` expired (>1 hour); re-process original |
| Empty result | 200 + empty | May indicate scanned doc; retry with `ocr` mode |

---

## 4. Missing Surfaces (Gaps for Agents)

These don't exist today but would meaningfully improve agent experience.

### 4.1 API Gaps — High Priority

| Gap | Description | Impact |
|-----|-------------|--------|
| **`GET /account/usage`** | Current credits remaining, rate limit status, quota per tier | Agents can't proactively manage cost; silently fail on credit exhaustion |
| **`POST /parse/estimate`** | "How many credits will this doc cost?" before processing | Prevents surprise credit burns on large docs; enables agent cost-awareness |
| **`GET /jobs`** | List recent jobs with status, timestamps, input doc | Agents spawning many async jobs have no way to reconcile completed work |
| **`DELETE /job/{id}`** | Cancel an in-flight async job | No escape hatch; agents can't abort runaway jobs |
| **Batch endpoint** | `POST /batch` — submit array of inputs, get one batch job ID | Multi-doc workflows require N separate API calls with N separate polling loops |
| **`POST /upload/presigned`** | Return a presigned URL for direct browser/agent upload (no Reducto proxy) | Enables client-side upload for large files; reduces latency |

### 4.2 API Gaps — Medium Priority

| Gap | Description |
|-----|-------------|
| **`POST /redact`** | PII removal from documents — critical for HIPAA/healthcare use case that Reducto markets to |
| **`POST /compare`** | Diff two document versions, return changed regions as bboxes — useful for contract review |
| **`GET /document/{id}/info`** | Page count, file type, file size, encrypted? without processing credits |
| **Webhook registration** | `POST /webhooks` to register persistent endpoint vs per-request `webhook` param |
| **Streaming parse** | Server-Sent Events for `/parse` — stream chunks as they're processed (critical for UX in long docs) |

### 4.3 SDK Gaps

| Gap | Surface | Description |
|-----|---------|-------------|
| **Built-in retry logic** | Python, Node, Go SDKs | Currently no retry on 429/500 in SDKs (our MCP server had to add this manually) |
| **Async iteration** | Python SDK | `async for chunk in client.parse_stream(...)` — doesn't exist |
| **Schema validation helper** | Python, Node | `client.validate_schema(schema)` before sending to `/extract` |
| **Typed response models** | Python SDK | Pydantic models for `ParseResult`, `ExtractResult` etc. for IDE autocomplete |
| **LangChain document loader** | Python | `ReductoLoader` for drop-in RAG pipeline integration |
| **LlamaIndex reader** | Python | `ReductoReader` for LlamaIndex |
| **OpenAI tool definition exports** | Python, Node | `reducto.as_openai_tools()` → ready-made tool definitions for direct use in LLM calls |

### 4.4 MCP Server Gaps

| Gap | Description |
|-----|-------------|
| **Binary file upload** | Current `reducto_upload` only accepts URLs; agents can't upload local files |
| **Async job submission** | No way to kick off async jobs and poll — blocks the MCP call for full duration |
| **Streamable HTTP transport** | Only stdio currently; can't host remotely or use with non-local clients |
| **`reducto_batch`** | No batch tool |
| **Cost/quota tool** | `reducto_account_status` showing credits remaining |

### 4.5 Agent-Experience Gaps

These are softer, DX-level gaps that affect how well agents can self-serve:

| Gap | Description |
|-----|-------------|
| **Structured error messages** | Current 422s say "value is not a valid enum" — they should say "table_output_format must be one of: html, md, json, csv, jsonbbox, dynamic" |
| **`/openapi.json` with agent-friendly descriptions** | Field descriptions in the OpenAPI spec are currently terse; agent-tuned descriptions (like our MCP server has) should be canonical |
| **Example request library** | Per-endpoint curl/Python/Node examples in docs; agents learn patterns from examples |
| **Agent guide completeness** | `docs.reducto.ai/agent-guide` exists but is sparse; job chaining, async polling, error handling patterns not documented |
| **SDK version pinning in docs** | Docs don't specify which SDK version introduced which feature; agents pulling from training data may use deprecated patterns |

---

## 5. Training Data Generation Plan

### 5.1 Coverage Matrix

For each cell: generate N synthetic (task prompt, correct execution, validated output) triples.

| Capability | Python SDK | Node SDK | Go (REST) | REST | MCP |
|-----------|:---------:|:--------:|:----------:|:----:|:---:|
| parse — basic | 30 | 20 | 15 | 15 | 20 |
| parse — advanced options | 20 | 15 | 10 | 10 | 15 |
| parse — page range | 15 | 10 | 8 | 8 | 10 |
| parse — chunking/RAG | 15 | 10 | 8 | 8 | 10 |
| extract — simple schema | 30 | 20 | 15 | 15 | 20 |
| extract — complex schema | 20 | 15 | 10 | 10 | 15 |
| extract — array_extract | 15 | 10 | 8 | 8 | 10 |
| extract — citations | 10 | 8 | 5 | 5 | 8 |
| split | 20 | 15 | 10 | 10 | 15 |
| classify | 20 | 15 | 10 | 10 | 15 |
| edit / form fill | 15 | 10 | 8 | 8 | 10 |
| upload → parse chain | 20 | 15 | 10 | 10 | 15 |
| parse → jobid → extract | 20 | 15 | 10 | 10 | 15 |
| classify → extract routing | 20 | 15 | 10 | 10 | 15 |
| split → extract per section | 15 | 10 | 8 | 8 | 10 |
| full pipeline | 15 | 10 | 8 | 8 | 10 |
| async + polling | 15 | 10 | 8 | 8 | 0* |
| error: 401 recovery | 10 | 8 | 5 | 5 | 8 |
| error: 422 recovery | 10 | 8 | 5 | 5 | 8 |
| error: 429 backoff | 10 | 8 | 5 | 5 | 8 |
| error: empty result → retry ocr | 10 | 8 | 5 | 5 | 8 |
| **TOTAL** | **375** | **260** | **180** | **180** | **245** |

**Grand total: ~1,240 examples** (before teacher-model resampling)
With 3× resampling per example (3 different frontier teachers): **~3,700 examples**
After quality filtering at ≥4/5 output_quality: target **~2,500 clean training pairs**

*Async MCP: our server is sync-only; skip or add as "call reducto_get_job in a loop" examples

### 5.2 Document Type Coverage

Each example should use a realistic document. Use mix of:

| Document Type | Example | Covers |
|--------------|---------|--------|
| Financial statement | Fidelity PDF (our test doc) | Tables, multi-page, headers |
| Invoice | Single-page, line items | array_extract, key-value |
| Contract | Multi-section, signatures | split, classify, signatures |
| Medical record | HIPAA-sensitive | redaction gap, extraction |
| Insurance form | Fillable PDF | edit/form fill |
| Spreadsheet | Excel/CSV | spreadsheet options |
| Presentation | PPTX | figure extraction |
| Scanned image | Low-res scan | OCR mode, agentic |
| Multilingual | Non-English doc | system_prompt guidance |
| Mixed-format array | Multiple PDFs as one input | URL array input type |

### 5.3 Teacher Model Selection

Ranked by param probe score (10 advanced params × 3 pts = 30 max):

| Model | Probe Score | Strengths | Weaknesses |
|-------|:-----------:|-----------|------------|
| **Claude Haiku 4.5** | 29/30 | Perfect on 9/10 params; best judgment calls | url_array: fires redundant calls alongside array input |
| **OpenAI o4-mini** | 29/30 | Perfect on 9/10 params; best schema construction | url_array: same redundant-call pattern as Haiku |
| **Xiaomi MiMo V2 Pro** | 27/30 | Highest output quality (4.9/5) in main benchmark | array_extract, return_figure_images usage imprecise |
| Qwen3 Coder Next | 25/30 | Strong code generation patterns | Completely missed split_rules; url_array API-rejected |

**Student baseline (Qwen3.5-35B-A3B):** 27/30 — already matches MiMo. Gap to best teachers is 2 pts on 3 specific probes.

**Primary teachers:** Haiku + o4-mini (29/30 tied). Use for all examples.
**Secondary teacher:** MiMo (27/30). Use for output quality diversity — its API call style is stylistically closest to what A3B has seen in pretraining.
**Qwen3 Coder:** Use only for parse/extract/classify code patterns — exclude from split_rules and url_array examples.

**Training targets for A3B (priority order):**
1. `url_array` — use array input *exclusively*, no individual follow-up calls
2. `array_extract` — correct usage when extracting repeating rows (value/context)
3. `return_figure_images` — correct param placement and expected response handling

Generate with each teacher, keep all versions, filter by quality ≥4/5, use highest-scored for SFT,
keep preference pairs (winner vs loser) for DPO/GRPO.

### 5.4 Data Format

Each training example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert at using Reducto's document processing API. [tool definitions here]"
    },
    {
      "role": "user",
      "content": "Extract all line items and totals from this invoice: https://..."
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "type": "function",
        "function": {
          "name": "reducto_extract",
          "arguments": "{\"input\":\"https://...\",\"schema\":\"{\\\"type\\\":\\\"object\\\",\\\"properties\\\":{\\\"line_items\\\":{\\\"type\\\":\\\"array\\\",...}}}\",\"array_extract\":true}"
        }
      }]
    },
    {
      "role": "tool",
      "content": "{\"result\":{\"line_items\":[...]}}"
    },
    {
      "role": "assistant",
      "content": "Here are the line items extracted: ..."
    }
  ],
  "metadata": {
    "surface": "mcp",
    "capability": "extract",
    "pattern": "array_extract",
    "doc_type": "invoice",
    "teacher_model": "Xiaomi MiMo V2 Pro",
    "quality_score": 5,
    "validated_against_api": true
  }
}
```

---

## 6. Summary: Priority Order for Data Generation

1. **Python SDK** — largest existing user base; biggest training impact per example
2. **MCP** — zero existing examples in public training data (new surface); high novelty signal
3. **Node SDK** — second-largest surface; camelCase conventions different enough to warrant own examples
4. **REST** — foundation; agents writing integrations from scratch will default to this
5. **Go SDK** — verify surface first, then generate; smaller but growing enterprise segment

**Phase 1 (SFT, 2–4 weeks):** Python + MCP, ~620 examples
**Phase 2 (expand, 2–4 weeks):** Node + REST, ~440 more
**Phase 3 (GRPO, ongoing):** All surfaces with live API validation as reward signal
**Phase 4 (Go + gaps):** Go SDK + any new endpoints Reducto ships

---

*Next: build the synthetic data generator script (`benchmark/scripts/gen_synthetic_data.py`)*
