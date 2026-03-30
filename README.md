# AgentReducto

**agentreducto.com** — a benchmark suite and a family of fine-tuned models specializing in the Reducto document-processing API.

---

## What's Here

| Component | Status |
|-----------|--------|
| Hard-probe benchmark — 20 models × 22 probes | ✅ 432 runs complete |
| MCP Race — 26 models × 7 adversarial MCP probes | ✅ complete |
| Live tracker UI | ✅ benchmark.agentreducto.com |
| MCP server v0.2.0 — all 7 tools | ✅ fully tested |
| R1 training data — direct API traces | ✅ 16,669 verified examples |
| R2 training data — MCP multi-hop chains | ✅ 20,000 verified examples |
| R3 training data — gap-focused + termination | ✅ 14,781 verified examples |
| DPO preference pairs — loop termination fix | ✅ 1,669 pairs |
| SFT dataset on HF (`Reza2kn/reducto-api-tool-calls`) | ✅ 51,069 (45,969 train · 5,100 val) |
| DPO dataset on HF (`Reza2kn/reducto-dpo-termination`) | ✅ 1,669 pairs |
| ReductoLoRA-Q6K (0.8B) — baseline fine-tune | ✅ benchmarked: 9/21 MCP · 20/30 API |
| Qwen3.5-35B-A3B fine-tune | 🚧 dataset ready, training pending on H100 |
| HF Space — in-browser demo | 🚧 pending trained weights |

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

## MCP Race

26 frontier models on 7 adversarial enterprise workflows — multi-hop chains, persisted job fan-outs, dual-doc parallel processing, agentic scope arrays. Tests whether a model reads the MCP tool definition or hallucinates REST API params.

**Live results:** [mcp.agentreducto.com](https://mcp.agentreducto.com)

### Top 3 (perfect or near-perfect)

| Model | Score |
|-------|-------|
| Claude Haiku 4.5 + thinking | 21/21 ✅ |
| Gemini 3.1 Flash Lite (preview) | 20/21 |
| MiniMax M2.7 (highspeed) | 20/21 |

### ReductoLoRA-Q6K baseline (0.8B fine-tune)

First fine-tuned checkpoint, trained on R1 data only (direct API traces, no MCP chains):

| Benchmark | Score | Notes |
|-----------|-------|-------|
| MCP probes | 9/21 | same as base — R1 data had no chain examples |
| API probes | 20/30 | solid improvement over base |

**Key failure mode:** repetition loop — the model calls the same tool 50–75× after getting a valid result (`upload_persist_array_extract`: 75 calls, `split_preserve_extract_range`: 77 calls). Root cause: 0.8B cannot track "I already got a result" across context; R1 training data lacked any multi-step chain examples.

This analysis directly shaped the R2 and R3 dataset design.

---

## MCP Server

Drop-in MCP server wrapping all 7 Reducto endpoints. All tools tested end-to-end including `jobid://` chaining and error paths.

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
| `reducto_edit` | `POST /edit` | ✅ tested |
| `reducto_upload` | `POST /upload` | ✅ tested |
| `reducto_get_job` | `GET /job/{id}` | ✅ tested |

`jobid://` chaining (parse once → extract + split without re-processing) ✅ tested.

---

## Fine-Tuning Pipeline

Training **Qwen3.5-35B-A3B** (3.5B active params) on a three-round dataset targeting Reducto-specific tool-call behavior.

### Dataset Rounds

| Round | Focus | Raw | Verified | Status |
|-------|-------|-----|----------|--------|
| R1 | Direct API traces — all 6 endpoints, param coverage | 36,669 | 16,669 | ✅ |
| R2 | MCP multi-hop chains — 8 teacher models | 25,909 | 20,000 | ✅ |
| R3 | 9 targeted gaps from MCP race + termination fix | 17,690 | 14,781 | ✅ |

**R3 gap breakdown:**
1. `schema` param (not `schema_json`) — adversarial prompts
2. `filter_blocks` / `agentic_scopes` as native arrays
3. `table_cutoff="preserve"` when prompt says "allow" — adversarial trap
4. 4-hop chain: `upload → parse(persist) → get_job → extract(jobid://)`
5. Dual-doc fan-out: 2× parse / get_job / extract with distinct job IDs
6. `array_extract=True` for repeating-row schemas
7. `reducto_upload` first for presigned / expiring URLs
8. `citations=True` + `deep_extract=True` for compliance docs
9. **Chain termination** (9,669 examples) — fixes the repetition loop: explicit "got result → output answer → stop" training signal

### DPO Preference Pairs

1,669 pairs targeting the loop failure directly:
- **Chosen**: final text response after last tool result, no further tool calls
- **Rejected**: repeating the last tool call after a completed result

DPO creates an explicit "don't do this" gradient that SFT positive-only examples can't.

### Datasets on HF

| Repo | Contents | Split |
|------|----------|-------|
| `Reza2kn/reducto-api-tool-calls` | 51,069 SFT examples (R1 + R2 + R3) | 45,969 train · 5,100 val |
| `Reza2kn/reducto-dpo-termination` | 1,669 DPO preference pairs | — |

### Top teachers (by MCP race score)

L3 consensus in verification only counts votes from the top-performing teachers:
- Claude Haiku 4.5 + thinking (21/21)
- Gemini 3.1 Flash Lite (20/21)
- MiniMax M2.7 (20/21)

---

## Pending 🚧

- [ ] Kick off Qwen3.5-35B-A3B SFT + DPO training on H100 (see `H100_TRAINING.md`)
- [ ] HF Space — quantized model in-browser via WebGPU
- [ ] Agent framework benchmark runs (Cursor, Cline, Codex CLI, Gemini CLI)

---

## Project Structure

```
reducto-agent-benchmark/
├── mcp-server/                      # MCP server (TypeScript, v0.2.0)
│   └── src/index.ts                 # 7 tools, retry/backoff, jobid:// chaining
├── benchmark/
│   ├── scripts/                     # Probe runners, scoring, data generators
│   │   ├── bench_mcp_probe.py       # MCP race runner
│   │   ├── bench_param_probe.py     # API hard-probe runner
│   │   ├── gen_mcp_data.py          # R2 MCP chain generator
│   │   ├── gen_mcp_r3_gaps.py       # R3 gap-focused generator
│   │   ├── gen_dpo_termination.py   # DPO preference pair generator
│   │   ├── verify_synthetic_data.py # L1–L4 verification pipeline
│   │   └── upload_combined.py       # Push merged splits to HF
│   ├── results/                     # Per-model JSON score files
│   └── data/
│       ├── synthetic_training/      # R1 verified examples
│       ├── synthetic_training_mcp/  # R2 MCP examples
│       ├── synthetic_training_r3/   # R3 gap-focused examples
│       ├── dpo_termination/         # DPO preference pairs
│       └── combined_training/       # Merged train/val splits
├── site/                            # Landing page (agentreducto.com)
├── gen_tracker.py                   # Live generation dashboard (port 8081)
└── train_35b.py                     # Qwen3.5-35B-A3B training script
```

---

## Author

Reza Sayar — Data Engineer @ Reducto
