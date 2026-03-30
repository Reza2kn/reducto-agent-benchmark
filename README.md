# AgentReducto

**agentreducto.com** — a benchmark suite and a family of fine-tuned models specializing in the Reducto document-processing API.

---

## What's Here

| Component | Status |
|-----------|--------|
| Hard-probe benchmark — 20 models × 22 probes | ✅ 432 runs complete |
| Live tracker UI | ✅ benchmark.agentreducto.com |
| MCP server v0.2.0 — all 7 tools | ✅ fully tested |
| Synthetic training data pipeline | ✅ 27k / 36,669 generated |
| Verification pipeline (L1–L4) | ✅ 10,848 verified examples on disk |
| Qwen3.5-35B-A3B fine-tune | 🚧 training on H100 |
| Qwen-0.8B-AgentJSON fine-tune | 🚧 base model pushed, training pending |
| HF Space — in-browser demo | 🚧 pending trained weights |
| Agent framework benchmarks (Cline, Cursor, Codex CLI, Gemini CLI…) | 🚧 manual runs not yet done |
| MCP-specific benchmark | 🚧 not yet run |

---

## Hard Probe Benchmark

22 API-specific probes scored 0–3 across 20 frontier models. Not a vibe check — each probe tests whether the model picks the right Reducto endpoint, constructs correct parameters, and handles edge cases without hand-holding.

**Live results:** [benchmark.agentreducto.com](https://benchmark.agentreducto.com)

### Models Tested (20)

| Model | Probes |
|-------|--------|
| Claude Opus 4.6 + thinking | 22/22 ✅ |
| Claude Haiku 4.5 + thinking | 22/22 ✅ |
| OpenAI o3 (reasoning=high) | 22/22 ✅ |
| OpenAI o4-mini (reasoning=high) | 22/22 ✅ |
| GPT-5.4 Mini | 22/22 ✅ |
| GPT-5.4 Nano | 22/22 ✅ |
| GPT-OSS 20B via Groq | 22/22 ✅ |
| Gemini 3.1 Flash Lite (preview) | 22/22 ✅ |
| Gemini 3.1 Pro (custom-tools preview) | 20/22 ✅ |
| Kimi K2.5 via Fireworks | 22/22 ✅ |
| MiniMax M2.7 (highspeed) | 22/22 ✅ |
| Qwen3.5-122B-A10B | 22/22 ✅ |
| Qwen3.5-35B-A3B (Atlas Cloud fp8) | 22/22 ✅ |
| Qwen3 32B via Groq | 20/22 ✅ |
| Inception Mercury 2 | 22/22 ✅ |
| GLM-5 Turbo | 22/22 ✅ |
| Xiaomi MiMo V2 Pro (fp8) | 22/22 ✅ |
| StepFun Step-3.5 Flash | 22/22 ✅ |
| Nemotron Nano 30B (DeepInfra fp4) | 22/22 ✅ |
| Qwen-0.8B-AgentJSON-Q6K (local) | 18/22 ✅ |

### 22 Hard Probes

Grouped by surface area — param formatting, endpoint selection, multi-hop chaining, edge cases:

```
fmt_parse_chunk_mode       fmt_parse_table_format     fmt_parse_page_range
fmt_parse_filter_blocks    fmt_parse_agentic_scopes   fmt_extract_schema
fmt_extract_array          fmt_split_description      fmt_classify_schema
chain_upload_then_parse    chain_parse_persist_jobid  chain_jobid_extract
chain_jobid_split          sel_parse_vs_extract       sel_extract_vs_classify
sel_split_vs_parse         edge_bad_url               edge_expired_jobid
edge_malformed_schema      edge_large_doc_page_range  hard_multiop_pipeline
agentjson_tool_dispatch
```

---

## MCP Server

Drop-in MCP server wrapping all 7 Reducto endpoints. **All tools tested end-to-end** including `jobid://` chaining and all error paths.

```bash
cd mcp-server && npm install
REDUCTO_API_KEY=your_key npx tsx src/index.ts
```

**Add to `.mcp.json` (Claude Code / Cursor / Cline / Codex):**

```json
{
  "mcpServers": {
    "reducto": {
      "command": "npx",
      "args": ["tsx", "mcp-server/src/index.ts"],
      "env": { "REDUCTO_API_KEY": "your_key" }
    }
  }
}
```

| Tool | Wraps | Status |
|------|-------|--------|
| `reducto_parse` | `POST /parse` | ✅ tested |
| `reducto_extract` | `POST /extract` | ✅ tested |
| `reducto_split` | `POST /split` | ✅ tested |
| `reducto_classify` | `POST /classify` | ✅ tested |
| `reducto_edit` | `POST /edit` | ✅ tested (correct 422 on non-form PDF) |
| `reducto_upload` | `POST /upload` | ✅ tested |
| `reducto_get_job` | `GET /job/{id}` | ✅ tested |

`jobid://` chaining (parse once → extract + split without re-processing) ✅ tested.

---

## Fine-Tuning Pipeline 🚧

Training two models on ~18k verified Reducto tool-call traces:

| Model | Base | Hardware | Status |
|-------|------|----------|--------|
| Qwen3.5-35B-A3B | `unsloth/Qwen3.5-35B-A3B` | H100 SXM 80GB (Vast.ai) | 🚧 training |
| Qwen-0.8B-AgentJSON | `Reza2kn/qwen35-agentjson-base` | RTX 4090 | 🚧 pending data |

**Data pipeline:** 116 seeds × 38 variations × 5 teacher models → 36,669 raw examples → ~18k verified via L1–L4 quality gates → `verified_train.jsonl` / `verified_val.jsonl`.

Target splits (exact): **train 16,669 · val 3,669**

HF dataset: `Reza2kn/reducto-api-tool-calls` (push pending gen completion)

---

## Pending 🚧

- [ ] Agent framework benchmark runs — Cursor, Cline, Codex CLI, Gemini CLI (manual, ~2–3h session)
- [ ] MCP benchmark — same 22 probes routed through the MCP server instead of direct API
- [ ] HF Space — quantized 0.8B model in-browser via WebGPU with extreme multi-hop demo cases
- [ ] Re-train on full ~18k verified set once gen pipeline completes
- [ ] Push final model weights + GGUF to HF Hub

---

## Project Structure

```
reducto-agent-benchmark/
├── mcp-server/                # MCP server (TypeScript, v0.2.0)
│   └── src/index.ts           # 7 tools, retry/backoff, jobid:// chaining
├── benchmark/
│   ├── scripts/               # Probe runners, scoring framework
│   ├── results/               # Per-model JSON score files (432 runs)
│   └── data/synthetic_training/  # Training data pipeline
├── site/                      # Landing page (agentreducto.com)
├── report/                    # findings.md, results tables
├── tracker.py                 # Live benchmark tracker UI (port 7842)
├── gen_synthetic_data.py      # Training data generator
├── verify_synthetic_data.py   # L1–L4 verification pipeline
├── train_35b.py               # Qwen3.5-35B training script
└── train_0.8b.py              # Qwen-0.8B training script
```

---

## Author

Reza Sayar — Data Engineer @ Reducto
