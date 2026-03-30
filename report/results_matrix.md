# Reducto Agent Benchmark — Results Matrix

## Platform Summary

Average score per dimension across all runs for each platform. Sorted by total score descending.

| Platform | Discovery | Setup Friction | Integration Complexity | Feature Coverage | Error Recovery | Output Quality | Token Efficiency | Mcp Compatibility | Avg Total |
|----------|:-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------:|:---------:|
| Claude Haiku 4.5 + thinking | 4.25 | 5.00 | 5.25 | 3.00 | 3.75 | 4.25 | 3.00 | - | **28.50/35.00** |
| Claude Opus 4.6 + thinking | 5.00 | 5.00 | 4.33 | 3.00 | 3.33 | 4.67 | 3.00 | - | **28.33/35.00** |
| OpenAI o4-mini (reasoning=high) | 5.00 | 5.00 | 4.00 | 3.00 | 3.33 | 5.00 | 3.00 | - | **28.33/35.00** |
| Claude Code | 5.00 | 5.00 | 5.00 | 4.00 | 4.00 | 4.00 | 4.00 | - | **27.00/35.00** |
| Codex Mini Latest | 2.00 | 5.00 | 6.00 | 3.00 | 4.00 | 2.00 | 3.00 | - | **25.00/35.00** |
| OpenAI o3 (reasoning=high) | 5.00 | 5.00 | 2.67 | 3.00 | 3.00 | 3.00 | 3.00 | - | **24.67/35.00** |
| LlamaIndex | 3.00 | 3.00 | 4.00 | 4.00 | 2.68 | 3.61 | 2.79 | - | **23.08/35.00** |
| smolagents | 3.00 | 3.00 | 4.00 | 4.00 | 2.66 | 3.58 | 2.79 | - | **23.03/35.00** |
| LangChain | 3.00 | 3.00 | 3.00 | 4.00 | 2.71 | 4.04 | 2.79 | - | **22.54/35.00** |
| Cline | 4.00 | 3.00 | - | 4.00 | - | - | - | 5.00 | **16.00/20.00** |
| Codex CLI | 4.00 | 4.00 | - | 4.00 | - | - | - | 4.00 | **16.00/20.00** |
| Gemini CLI | 4.00 | 4.00 | - | 4.00 | - | - | - | 4.00 | **16.00/20.00** |
| Continue.dev | 4.00 | 3.00 | - | 4.00 | - | - | - | 4.00 | **15.00/20.00** |
| Aider | 3.00 | 4.00 | - | 3.00 | - | - | - | - | **10.00/15.00** |

## Integration Path vs Output Quality

Average `output_quality` score broken down by integration path and platform.

| Integration Path | Aider | Arcee Trinity Large (prime) | Claude Code | Claude Haiku 4.5 + thinking | Claude Opus 4.6 + thinking | Cline | Codex CLI | Codex Mini Latest | Continue.dev | GLM-5 Turbo | GPT-OSS 20B via Groq | Gemini CLI | Kimi K2.5 via Fireworks | LangChain | LlamaIndex | MiniMax M2.7 (highspeed) | Nemotron Nano 30B (DeepInfra fp4) | Nemotron Super 120B (Nebius bf16) | OpenAI o3 (reasoning=high) | OpenAI o4-mini (reasoning=high) | Qwen3 32B via Groq | Qwen3 Coder Next (ionstream fp8) | Qwen3.5 122B (Alibaba) | StepFun Step-3.5 Flash | Xiaomi MiMo V2 Pro (fp8) | smolagents |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| framework | - | - | - | - | - | - | - | - | - | - | - | - | - | 4.04 | 3.61 | - | - | - | - | - | - | - | - | - | - | 3.58 |
| mcp | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| node_sdk | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| python_sdk | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| rest_api | - | - | 4.00 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| tool_calling | - | 1.00 | - | 4.25 | 4.67 | - | - | 2.00 | - | 4.33 | 3.67 | - | 4.67 | - | - | 4.33 | 2.33 | 1.33 | 3.00 | 5.00 | 4.33 | 4.67 | 4.33 | 4.33 | 5.00 | - |

## Capability Coverage vs Integration Path

Average `output_quality` for each capability type per integration path.

| Capability | framework | mcp | node_sdk | python_sdk | rest_api | tool_calling |
|---|---|---|---|---|---|---|
| classify | 3.74 | - | - | - | - | 3.00 |
| error_handling | 3.84 | - | - | - | - | 4.18 |
| extract | 3.33 | - | - | - | 4.00 | 3.18 |
| full_pipeline | - | - | - | - | - | - |
| jobid_chaining | - | - | - | - | - | - |
| parse | 3.74 | - | - | - | - | 3.88 |
| split | 4.07 | - | - | - | - | - |

## OSS / Community Models (OpenRouter)

| Platform | Discovery | Setup Friction | Integration Complexity | Feature Coverage | Error Recovery | Output Quality | Token Efficiency | Mcp Compatibility | Avg Total |
|----------|:-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------:|:---------:|
| Kimi K2.5 via Fireworks | 5.00 | 4.00 | 5.00 | 3.00 | 3.67 | 4.67 | - | - | **25.33/30.00** |
| GLM-5 Turbo | 5.00 | 4.00 | 4.67 | 3.00 | 3.67 | 4.33 | - | - | **24.67/30.00** |
| Qwen3 Coder Next (ionstream fp8) | 5.00 | 4.00 | 4.33 | 3.00 | 3.67 | 4.67 | - | - | **24.67/30.00** |
| Xiaomi MiMo V2 Pro (fp8) | 5.00 | 4.00 | 4.33 | 3.00 | 3.33 | 5.00 | - | - | **24.67/30.00** |
| Qwen3.5 122B (Alibaba) | 5.00 | 4.00 | 4.33 | 3.00 | 3.67 | 4.33 | - | - | **24.33/30.00** |
| MiniMax M2.7 (highspeed) | 5.00 | 4.00 | 4.00 | 3.00 | 3.33 | 4.33 | - | - | **23.67/30.00** |
| Qwen3 32B via Groq | 5.00 | 4.00 | 4.00 | 3.00 | 3.33 | 4.33 | - | - | **23.67/30.00** |
| StepFun Step-3.5 Flash | 5.00 | 4.00 | 4.00 | 3.00 | 3.33 | 4.33 | - | - | **23.67/30.00** |
| GPT-OSS 20B via Groq | 5.00 | 4.00 | 2.00 | 3.00 | 3.00 | 3.67 | - | - | **20.67/30.00** |
| Nemotron Super 120B (Nebius bf16) | 5.00 | 4.00 | 3.00 | 3.00 | 3.00 | 1.33 | - | - | **19.33/30.00** |
| Nemotron Nano 30B (DeepInfra fp4) | 5.00 | 4.00 | 1.33 | 3.00 | 3.00 | 2.33 | - | - | **18.67/30.00** |
| Arcee Trinity Large (prime) | 5.00 | 4.00 | 1.00 | 3.00 | 3.00 | 1.00 | - | - | **17.00/30.00** |
