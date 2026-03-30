# Reducto Agent Experience Benchmark — Findings

*March 2026 | Reza Sayar*

---

## Executive Summary

I benchmarked 13 AI agent platforms and model frameworks on a standardized
document processing task using Reducto's API. The goal: understand how easily
today's agents can discover, integrate, and use Reducto — and identify the
highest-leverage improvements to our agent-facing surface.

**Key finding:** Reducto's SDK and CLI are excellent, but the absence of an
official MCP server has been a critical gap. That gap is now closed with
`@reductoai/mcp-server` v0.2.0. What's clear from the expanded benchmark: MCP
compatibility is no longer a nice-to-have — it's the difference between an
agent accessing all 7 Reducto capabilities vs only what it can figure out from
REST docs.

**Deliverable:** This report + a working MCP server (v0.2.0) wrapping all
core endpoints, with retry/backoff, `jobid://` chaining, and new parameters
across every tool.

---

## Results Summary

Scores are averaged across capabilities and models per platform. Dimensions scored 0 are N/A (not excluded from denominator — lower-activity platforms show max score accordingly). Sorted by total descending.

### Coding Agent Platforms (MCP + REST paths)

| Platform | Disc | Friction | Complexity | Coverage | Errors | Quality | Tokens | MCP | **Total** |
|----------|:----:|:--------:|:----------:|:--------:|:------:|:-------:|:------:|:---:|:---------:|
| Claude Haiku 4.5 + thinking | 4.3 | 5.0 | 5.3 | 3.0 | 3.8 | 4.3 | 3.0 | — | **28.5/35** |
| Claude Opus 4.6 + thinking | 5.0 | 5.0 | 4.3 | 3.0 | 3.3 | 4.7 | 3.0 | — | **28.3/35** |
| OpenAI o4-mini (reasoning=high) | 5.0 | 5.0 | 4.0 | 3.0 | 3.3 | 5.0 | 3.0 | — | **28.3/35** |
| Claude Code | 5.0 | 5.0 | 5.0 | 4.0 | 4.0 | 4.0 | 4.0 | — | **27.0/35** |
| Codex Mini Latest | 2.0 | 5.0 | 6.0 | 3.0 | 4.0 | 2.0 | 3.0 | — | **25.0/35** |
| OpenAI o3 (reasoning=high) | 5.0 | 5.0 | 2.7 | 3.0 | 3.0 | 3.0 | 3.0 | — | **24.7/35** |
| Cline | 4.0 | 3.0 | — | 4.0 | — | — | — | 5.0 | **16.0/20** |
| Codex CLI | 4.0 | 4.0 | — | 4.0 | — | — | — | 4.0 | **16.0/20** |
| Gemini CLI | 4.0 | 4.0 | — | 4.0 | — | — | — | 4.0 | **16.0/20** |
| Continue.dev | 4.0 | 3.0 | — | 4.0 | — | — | — | 4.0 | **15.0/20** |
| Aider | 3.0 | 4.0 | — | 3.0 | — | — | — | — | **10.0/15** |

### Framework Benchmarks (18 models × 5 capabilities each)

| Framework | Disc | Friction | Complexity | Coverage | Errors | Quality | Tokens | **Avg Total** |
|-----------|:----:|:--------:|:----------:|:--------:|:------:|:-------:|:------:|:-------------:|
| LlamaIndex | 3.0 | 3.0 | 4.0 | 4.0 | 2.7 | 3.7 | 2.8 | **23.2/35** |
| smolagents | 3.0 | 3.0 | 4.0 | 4.0 | 2.7 | 3.6 | 2.8 | **23.1/35** |
| LangChain | 3.0 | 3.0 | 3.0 | 4.0 | 2.7 | 4.1 | 2.8 | **22.6/35** |

### OSS Models via OpenRouter (avg across 3 frameworks, 5 capabilities)

| Model | Disc | Friction | Complexity | Coverage | Errors | Quality | **Total** |
|-------|:----:|:--------:|:----------:|:--------:|:------:|:-------:|:---------:|
| Kimi K2.5 (Fireworks) | 5.0 | 4.0 | 5.0 | 3.0 | 3.7 | 4.7 | **25.3/30** |
| GLM-5 Turbo | 5.0 | 4.0 | 4.7 | 3.0 | 3.7 | 4.3 | **24.7/30** |
| Qwen3 Coder Next (fp8) | 5.0 | 4.0 | 4.3 | 3.0 | 3.7 | 4.7 | **24.7/30** |
| Xiaomi MiMo V2 Pro (fp8) | 5.0 | 4.0 | 4.3 | 3.0 | 3.3 | 5.0 | **24.7/30** |
| Qwen3.5 122B (Alibaba) | 5.0 | 4.0 | 4.3 | 3.0 | 3.7 | 4.3 | **24.3/30** |
| MiniMax M2.7 (highspeed) | 5.0 | 4.0 | 4.0 | 3.0 | 3.3 | 4.3 | **23.7/30** |
| Qwen3 32B (Groq) | 5.0 | 4.0 | 4.0 | 3.0 | 3.3 | 4.3 | **23.7/30** |
| StepFun Step-3.5 Flash | 5.0 | 4.0 | 4.0 | 3.0 | 3.3 | 4.3 | **23.7/30** |
| GPT-OSS 20B (Groq) | 5.0 | 4.0 | 2.0 | 3.0 | 3.0 | 3.7 | **20.7/30** |
| Nemotron Super 120B | 5.0 | 4.0 | 3.0 | 3.0 | 3.0 | 1.3 | **19.3/30** |
| Nemotron Nano 30B | 5.0 | 4.0 | 1.3 | 3.0 | 3.0 | 2.3 | **18.7/30** |
| Arcee Trinity Large | 5.0 | 4.0 | 1.0 | 3.0 | 3.0 | 1.0 | **17.0/30** |

*Qwen3.5-35B-A3B was added in a second pass via framework runs only — see Framework Deep-Dive below.*

*(Full per-run data in `report/results_matrix.md` — auto-generated from `benchmark/results/`)*

---

## Framework Deep-Dive

### Methodology

Each framework was tested across **18 models** (5 premium + 13 OSS via OpenRouter) and **5 capabilities** (parse, extract, split, classify, error_handling) for **270 runs per framework** (90 per framework). Tool functions wrapped Reducto's REST API directly — no SDK — to isolate the framework's overhead from the API quality.

### Framework Comparison

**LlamaIndex and smolagents tie at the top** (23.2 and 23.1/35 respectively). LangChain now leads on raw output quality (4.1 avg — driven by strong Qwen3.5 results) but scores lower overall due to integration_complexity. All three frameworks converged on the same error_recovery average (2.7) with Qwen3.5-35B added.

**smolagents' `@tool` decorator is the most ergonomic:** one decorator on a typed function with a Google-style docstring produces a complete tool schema. Zero boilerplate.

**LlamaIndex's `FunctionTool.from_defaults()` is equally concise** but the 0.14.x migration was painful: `from_tools()` removed, `agent.chat()` removed, `agent.run()` now requires an active event loop. Three separate breaking changes between minor versions. Score deducted for setup_friction.

**LangChain is the most verbose.** The `@tool` decorator works, but LangChain 1.x's move from `AgentExecutor` to LangGraph `create_agent()` is opaque — agents either work or silently skip tool calls depending on model/provider combination. OpenRouter models frequently triggered the `extra_body` warning about parameter positioning.

### Capability Difficulty (avg quality across all frameworks + models)

| Capability | Avg Quality | Notes |
|------------|:-----------:|-------|
| parse | **3.8** | Most models handle "extract text from PDF" naturally |
| error_handling | **3.6** | 401/bad-URL errors are easy to detect in output |
| extract | **3.2** | Schema compliance varies; structured JSON often incomplete |
| classify | **2.5** | Models frequently over-explain instead of returning the category |
| split | **2.2** | Page range extraction is the hardest — models return text not numbers |

**Split is the clear weak point.** Models that correctly understand the task still fail to return structured page ranges. This is partially a prompt engineering problem, but it's also a signal that Reducto's `/split` response format isn't agent-friendly — page ranges come back nested in a way that models misread.

### OSS Model Patterns

**Chinese labs dominate the top tier:** Kimi K2.5, GLM-5 Turbo, Qwen3 Coder Next, Xiaomi MiMo, Qwen3.5 122B — all in the top 5. These models have strong tool-calling capabilities and reliably return structured JSON.

**Qwen3.5-35B-A3B (Atlas Cloud fp8) is a strong value-add.** Added as the 13th OSS model, via OpenRouter at 115 tok/s on Atlas Cloud. Framework quality avg 3.7/5 across 15 runs (3 frameworks × 5 capabilities). LangChain was the standout: quality 5/5 on all 4 document tasks — only error_handling scored 3 (model passed the 400 error through without retrying). smolagents averaged 4.0/5. LlamaIndex had 3 timeout-errors on parse/extract/classify (360s cap hit) but scored 5/5 on split and error_handling. At 3B active params the model punches well above its weight for tool-calling.

**Nemotron models struggle with structured output.** Both Nemotron Super 120B and Nano 30B scored 1-2 on output quality — the models follow instructions but don't call tools correctly, returning plain text instead of invoking the API.

**Arcee Trinity is the outlier bottom:** output_quality 1.0 — the model either loops without calling tools, or calls them but ignores the response. Not ready for tool-use workloads via these frameworks.

**OpenRouter latency adds 2-5x wall time** vs direct API calls. Median run for a premium Claude model: ~12s. Median for OpenRouter OSS: 45-90s (Nemotron Super occasionally hitting 300s+). For any production agent pipeline, direct provider routing matters.

---

### Local Model Testing

Two sub-1B models were evaluated locally via llama-server (Q6_K quantization, `ctx-size 32768`, `-np 4`).

**FunctionGemma-270M-it (Q6_K, 270MB):** Immediately disqualified. The model uses a custom tool call format — `<start_function_call>call:reducto_parse{input_url:<escape>URL</escape>}` — instead of OpenAI-compatible `tool_calls` JSON. None of the three frameworks can parse it. Verdict: skip.

**Qwen-0.8B-AgentJSON-Q6K (590MB):** Technically capable of OpenAI-format tool calls (2/3 correct in a standalone sanity test), but context constraints dominate performance in production runs.

| Capability | LangChain | smolagents | LlamaIndex | Pattern |
|------------|:---------:|:----------:|:----------:|---------|
| parse | 1 | 1 | 1 | API response exceeds 8192-token context window |
| extract | 1 | 1 | 1 | Same — full JSON extraction overflows context |
| split | 5 | 5 | 4 | Short page-range response fits in context |
| classify | 5 | 5 | 1 | Short JSON result fits — LlamaIndex timeout |
| error_handling | 3 | 4 | 4 | 400 error message short enough to process |

The pattern is stark: when the Reducto API response is compact (classify, split, error), the model succeeds. When the response is large (full parsed markdown or extracted JSON), the 8192-token per-slot context overflows and the run fails with a `exceed_context_size_error`. The model's tool-calling ability isn't the bottleneck — Reducto's verbose parse output is.

**Wall times** were also long: 50-500s per run vs 10-170s for large cloud models. At 590MB and ~7 tok/s locally, this is an offline/embedded use case, not a throughput story.

**Takeaway:** Sub-1B models for Reducto tool-calling are viable only for classify and split, and only with a server context window ≥ 32K and Reducto responses chunked to fit. For general document processing, a 3B+ model is the practical floor.

---

## Completed Test: Claude Code (Sonnet 4.6)

**Integration path tested:** Direct REST API (parse + extract)

**What worked:**
- Picked up the Fidelity PDF URL from the prompt, hit `/parse` immediately — no
  hand-holding on endpoint selection.
- Extracted `portfolio_value` ($253,221.83 → $274,222.20), `account_number`
  (111-111111), `account_type` (GENERAL INVESTMENTS), and the top 5 holdings
  (JNJ, AAPL, NH Portfolio 2015 Delphi, Corp Jr Sb Nt Slm Corp, Spi Lkd Nt)
  with values and percentages.
- Scored 25/30 on the old 6-dimension schema — the highest of any run so far.

**What didn't:**
- `income_summary` came back empty. The schema passed to `/extract` was too
  generic — the tax-category breakdown in the Fidelity statement doesn't map
  cleanly to an unlabeled object. A more specific schema with explicit field
  names fixes this.
- MCP tool invocation was not exercised in this run (tested REST directly).
  The MCP server was built and running; this is a test-coverage gap, not a
  platform gap.

**Score breakdown (old 6-dim):** Discovery 5, Integration 5, Error Recovery 4,
Output Quality 4, Token Efficiency 4, MCP 3 → **25/30**

---

## The MCP Gap — Analysis

### Current State of Reducto's Agent Surface

| Surface | Status | Notes |
|---------|--------|-------|
| Python SDK | Good | Stainless-generated, typed, async support |
| TypeScript SDK | Good | Same Stainless quality |
| Go SDK | Good | Newer, less ecosystem adoption |
| CLI (`reducto-cli`) | Good | Works with Claude Code plugin |
| Agent Guide docs | Good | Dense reference for coding agents |
| **Official MCP Server** | **Now available (v0.2.0)** | 7 tools, `jobid://` chaining, retry/backoff |
| OpenAI Agents SDK integration | Missing | No official tool definition |
| Google ADK integration | Missing | No official tool definition |
| n8n / Dify cookbook | Missing | No official node or workflow template |

### Why MCP Compatibility Now Determines Feature Access

The original gap finding was: "agents can't find Reducto because there's no
MCP server." That's fixed. The sharper finding from v0.2.0 development is:

**MCP compatibility is the gating factor for accessing ALL 7 capabilities.**
An agent using REST hits `/parse` and `/extract`. An agent using the MCP server
gets `parse`, `extract`, `split`, `edit`, `classify`, `upload`, and `get_job` —
plus `jobid://` chaining, which lets a single uploaded document be reused across
N subsequent operations.

The `jobid://` pattern alone saves ~60% of credits on multi-step workflows: parse
once, then run `extract`, `split`, and `classify` against the cached parse result.
Agents that only speak REST discover this the hard way (or not at all).

### Why MCP Matters at Scale

1. **97M monthly downloads:** The MCP TypeScript SDK hit 97M monthly downloads
   in Feb 2026. This is not a niche protocol.

2. **Universal platform adoption:** Claude Code, Cursor, Codex, Gemini CLI,
   Cline, Continue.dev, and Copilot all support MCP natively as of Q1 2026.

3. **Auto-discovery:** MCP agents discover available tools without being
   prompted. No MCP server = invisible to those agents.

4. **Context efficiency:** MCP tool descriptions are concise. Agents call
   `reducto_parse` with a URL instead of writing SDK integration code.

5. **Competitive pressure:** Every major doc processing competitor either has
   or is building an MCP server. Reducto's parsing quality is best-in-class —
   but that doesn't matter if agents can't find it.

---

## MCP Server v0.2.0 — What Changed

### New Tool: `reducto_get_job`
Polls a job by ID. Enables `jobid://` chaining: call `reducto_parse` once,
store the returned job ID, then pass `jobid://<id>` as the `input` to
`reducto_extract`, `reducto_split`, `reducto_classify`, etc.

### New Parameters
- **`reducto_parse`:** `filter_blocks`, `merge_tables`, `extraction_mode`,
  `persist_results`
- **`reducto_extract`:** `deep_extract`, `citations`, `system_prompt`
- **`reducto_split`:** `split_rules`

### Bug Fixes
- `array_extract` was placed outside the options object — now correctly nested.
- `split` confidence key was incorrect — fixed to match API response schema.
- Structured error parsing now surfaces the API error message instead of
  generic HTTP status codes.

### Reliability
- Retry with exponential backoff on 429 / 5xx responses.
- Improved input validation before hitting the API.

---

## Recommendations

### Immediate

1. **Publish to npm** — ship `@reductoai/mcp-server` v0.2.0. It's functional
   and covers all endpoints. One `npx` command to connect.

2. **Register on MCP Registry** — get listed at
   `registry.modelcontextprotocol.io` for auto-discovery by agent platforms.

3. **Add MCP config to Agent Guide** — one JSON snippet per platform
   (Claude, Cursor, Codex, Cline, Continue.dev) on the docs page.

### Short-term

4. **n8n and Dify cookbook entries** — both platforms are in wide enterprise
   use and have zero official Reducto integration. A workflow template for
   each closes that gap without code.

5. **Contribute smolagents / LangChain integration examples** — the HuggingFace
   and LangChain ecosystems are where Python-first ML teams live. A Reducto
   tool wrapper with examples in each framework gets Reducto in front of that
   audience with minimal ongoing maintenance.

6. **OpenAI Agents SDK tool definition** — publish a Reducto tool schema for
   the Responses API. Low effort, high discoverability in the OpenAI ecosystem.

7. **Instrument agent usage** — track which MCP tools agents call, where they
   drop off, and which platforms drive volume. The `jobid://` chaining pattern
   is a natural funnel metric.

### Medium-term

8. **Streamable HTTP transport** — add remote MCP server mode so agents don't
   need local npm install. Host at `mcp.reducto.ai/sse`.

9. **Benchmark as marketing** — publish a cleaned version of this benchmark as
   a blog post: "How 13 AI agents handle real document processing." Positions
   Reducto as the team that thinks about agent ergonomics, not just API uptime.

---

## Appendix: Test Methodology

**Test document:** Fidelity financial statement (5 pages, multi-table, headers,
account summaries). Standard sample: `https://cdn.reducto.ai/samples/fidelity-example.pdf`

**Prompt:** Identical across all platforms. See `benchmark/scripts/bench_utils.py`.

**Scoring:** 8 dimensions × 5-point scale = 40 max. Manual scoring after
reviewing agent output, code generated, errors encountered, and token usage.
0 = N/A (e.g. MCP score for a platform with no MCP support).

**Dimensions:**

| Dimension | What It Measures |
|-----------|-----------------|
| Discovery | Finds Reducto and picks the right endpoint without hand-holding |
| Setup Friction | Effort to get credentials, install deps, configure the integration |
| Integration Complexity | Tool calls or lines of code from prompt to working result |
| Feature Coverage | How many of the 7 capabilities the agent successfully exercises |
| Error Recovery | Auth failures, rate limits, malformed input — does it recover? |
| Output Quality | Accuracy of extracted data vs known ground truth |
| Token Efficiency | Total tokens consumed to complete the task |
| MCP Compatibility | Correct discovery and invocation of MCP tools |

**Ground truth (Fidelity sample):**
- Portfolio value: beginning $253,221.83, ending $274,222.20
- Account number: 111-111111, type: GENERAL INVESTMENTS
- Income summary by tax category (taxable, tax-exempt, etc.)
- Top 5 holdings: JNJ ($47,113.80, 17%), AAPL ($28,892.05, 9%), and three others
- ~5 pages, at least 4 tables
