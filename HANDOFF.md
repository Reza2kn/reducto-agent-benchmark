# What We're Building & Where We Are

## The Goal

We're training a small, fast AI model to be an expert at using the Reducto API. The idea: instead of a developer (or a general-purpose AI) having to figure out how to call Reducto's document-processing endpoints, our fine-tuned model just *knows* — it picks the right endpoint, constructs the right parameters, and handles errors cleanly.

The model we're training is **Qwen3.5-35B-A3B** (a lean mixture-of-experts model, ~3B active parameters). It'll live on a rented H100 GPU on Vast.ai.

---

## How We're Generating Training Data

We can't hand-write 18,000 training examples. So we used 5 powerful "teacher" LLMs to generate them for us:

- Kimi K2.5 (via Fireworks)
- Gemini 3.1 Flash Lite
- MiniMax M2.7
- Inception Mercury 2
- Qwen3.5-122B

Each teacher was given 116 different Reducto API scenarios (things like "parse this PDF", "extract structured data", "split a document into sections") and asked to produce realistic tool-call traces — essentially showing what a perfect API interaction looks like. Each scenario was varied 38 ways to maximize diversity.

Verifier layers on the 30,000 checkpoint:

Layer	What it does	Expected dropout
L1	Schema validation — invalid enums, bad JSON	~5%
L2	Tool hint check — e.g. "split this doc" but called reducto_parse	~5–10%
L3	Cross-model consensus — keeps examples where ≥3 of 5 teachers produced the same tool fingerprint	~20–30%
L4	Live Reducto API re-check — 4xx = reject (already passed once during gen, so mostly passes again)	~5%
L3 is the biggest filter. For each (probe_id, prompt) group, all 5 teacher responses need ≥3 to agree on the same tool call shape. For simple probes (fmt_parse_*, upload) all 5 agree → all pass. For hard probes (chain_classify_route_extract) teachers diverge → whole group can get dropped.

Net estimate:

30,000 checkpoint × ~55–65% pass rate ≈ 16,500–19,500 verified examples
  → verified_train.jsonl  ~90% ≈ 14,850–17,550 examples
  → verified_val.jsonl    ~10% ≈  1,650–1,950 examples

Best single-point estimate: ~18,000 total (~16,200 train / ~1,800 val). That's enough for 3 epochs at effective batch 32 = ~1,500 optimizer steps on the H100.

After generation, a separate "judge" AI (L4) filters out bad examples. We expect ~60% to pass, leaving us with ~18,000 high-quality training examples.

---

## Current Status

**We're halfway there.** As of now:

- **14,500 / 30,000 raw examples generated** (48%)
- Running at ~40–60 examples/minute
- ETA to finish generation: **~6–7 hours** (tonight)
- Zero API errors, process healthy

Once generation hits 30,000, a watchdog script automatically kicks off the quality-filtering step (no manual action needed).

---

## What Happens Next (in order)

1. **Tonight** — generation finishes, auto-filter runs, produces ~18k clean examples
2. **Tonight** — spin up an H100 on Vast.ai, fine-tune Qwen3.5-35B-A3B on the dataset
3. **Tonight** — test the fine-tuned model against the same benchmark we used to evaluate GPT-4, Claude, Gemini etc. on Reducto API tasks

---

## Why This Matters

Reducto currently has no purpose-built model for its own API. General-purpose agents (Claude Code, Cursor, Codex) score poorly on Reducto-specific tasks because they have to infer how the API works from sparse docs. A fine-tuned model that *knows* the API natively would dramatically improve agent experience — faster integration, fewer errors, no hand-holding.

This is a weekend proof-of-concept to demonstrate that value to Reducto's team.
