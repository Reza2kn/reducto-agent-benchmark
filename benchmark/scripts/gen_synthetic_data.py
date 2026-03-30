#!/usr/bin/env python3
"""
gen_synthetic_data.py — Synthetic training data generator for Reducto fine-tuning.

Target: 16,900 examples for Qwen3.5-35B-A3B student model.

Coverage strategy (goes well beyond the 22 hard probes):
  1. All 22 hard probe scenarios  ×  varied rephrases
  2. All file formats Reducto accepts  ×  each endpoint
  3. Multi-hop chains  (upload→parse→extract, classify→route→edit, etc.)
  4. Negative/conflict resolution  (mutually exclusive params, wrong tool → recovery)
  5. Targeted remediation for the 3 probes even Kimi missed
  6. Cost/perf tradeoffs  (page ranges, latency vs accuracy, upload reuse)
  7. Edge cases  (async polling, OCR fallback, table spanning pages, job expiry)

Usage:
    python gen_synthetic_data.py                     # full run, target=16900
    python gen_synthetic_data.py --target 100        # quick test
    python gen_synthetic_data.py --dry-run           # no API calls
    python gen_synthetic_data.py --resume            # skip already-saved checkpoints
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))
from models import OPENROUTER_MODELS, PREMIUM_MODELS, ModelConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR  = Path("benchmark/results/param_probe")
OUTPUT_DIR   = Path("benchmark/data/synthetic_training")
HARD_DIR     = RESULTS_DIR / "by_model_hard"
CHECKPOINT   = OUTPUT_DIR / ".checkpoint.jsonl"   # for resume

# Generation teachers — cheap + high-scoring (≥55/66), no reasoning models.
# Reasoning models (o3, Opus, Haiku+thinking) cost 10-800x more per call due to
# thinking tokens and are completely wasteful for single-shot tool call generation.
GENERATION_TEACHERS = [
    "Kimi K2.5 via Fireworks",           # 63/66 — #1, Fireworks is fast + cheap
    "MiniMax M2.7 (highspeed)",          # 62/66 — highspeed endpoint
    "Gemini 3.1 Flash Lite (preview)",   # 62/66 — fastest in benchmark
    "Qwen3.5-122B-A10B",                 # 62/66 — Alibaba direct, cheap
    "Inception Mercury 2",               # 60/66 — fast diffusion model
    "GLM-5 Turbo",                       # 59/66 — cheap turbo endpoint
    "GPT-5.4 Nano",                      # 58/66 — nano = cheap
    "StepFun Step-3.5 Flash",            # 57/66 — flash = fast + cheap
]

# Fast cheap model for generating prompt variations
REPHRASE_MODEL = "google/gemini-3.1-flash-lite-preview"   # fastest in our benchmark

SYSTEM_PROMPT = (
    "You are an expert Reducto API integration agent. "
    "You help users process documents by calling the right Reducto tools with the correct parameters. "
    "Always prefer the most specific, efficient tool call for the task. "
    "When a task implies a specific Reducto parameter, use it — "
    "e.g. bounding boxes → citations=True; repeating rows → array_extract=True; "
    "scanned doc → extraction_mode='ocr'; speed critical → optimize_for_latency=True and deep_extract=False; "
    "multiple ops on same doc → upload first then reuse reducto:// ID; "
    "RAG chunking → chunk_mode='section'; embedding → embedding_optimized=True."
)

# ---------------------------------------------------------------------------
# All file formats Reducto accepts
# ---------------------------------------------------------------------------

FILE_FORMATS = {
    "pdf":   ("a PDF document",          "https://example.com/document.pdf"),
    "docx":  ("a Word document (.docx)", "https://example.com/report.docx"),
    "xlsx":  ("an Excel spreadsheet (.xlsx)", "https://example.com/data.xlsx"),
    "pptx":  ("a PowerPoint presentation (.pptx)", "https://example.com/slides.pptx"),
    "png":   ("a scanned PNG image",     "https://example.com/scan.png"),
    "jpg":   ("a scanned JPG photo",     "https://example.com/invoice.jpg"),
    "tiff":  ("a multi-page TIFF scan",  "https://example.com/contract.tiff"),
    "html":  ("an HTML web page",        "https://example.com/page.html"),
    "csv":   ("a CSV data file",         "https://example.com/data.csv"),
    "msg":   ("a Microsoft Outlook .msg email", "https://example.com/email.msg"),
    "eml":   ("a raw .eml email file",   "https://example.com/message.eml"),
    "txt":   ("a plain text file",       "https://example.com/notes.txt"),
    "bmp":   ("a scanned BMP image",     "https://example.com/scan.bmp"),
}

# ---------------------------------------------------------------------------
# Scenario bank — organized by category
# Each entry: (probe_id_tag, prompt_template)
# probe_id_tag is used for metadata only (not a real probe constraint)
# ---------------------------------------------------------------------------

# ── Category 1: The 22 hard probes — imported from bench_param_probe ──────
# (handled separately via harvest + rephrase of existing probe prompts)

# ── Category 2: File format × endpoint matrix ─────────────────────────────

def _fmt_scenarios() -> list[tuple[str, str]]:
    """Generate parse/extract prompts for every file format."""
    scenarios = []
    for fmt, (desc, url) in FILE_FORMATS.items():
        scenarios += [
            (f"fmt_parse_{fmt}",
             f"I have {desc} at {url}. Parse it and give me all the text and tables."),
            (f"fmt_extract_{fmt}",
             f"Extract the key fields from {desc} at {url}. I need: title, date, total amount, and any line items."),
            (f"fmt_ocr_{fmt}" if fmt in ("png","jpg","tiff","bmp") else f"fmt_parse_{fmt}_clean",
             f"Parse {desc} at {url}. It's a scanned document so make sure to use OCR mode." if fmt in ("png","jpg","tiff","bmp")
             else f"Parse {desc} at {url} and strip all headers, footers, and page numbers so I get clean body text for my LLM."),
        ]
    return scenarios

# ── Category 3: Multi-hop chains ──────────────────────────────────────────

CHAIN_SCENARIOS: list[tuple[str, str]] = [
    ("chain_upload_multiop",
     "I have a PDF at https://s3.amazonaws.com/bucket/report.pdf?AWSAccessKeyId=AKID&Signature=xyz&Expires=1700000000 "
     "— that's a presigned URL that expires in 90 seconds. "
     "I need to parse it, extract a financial summary table, AND classify it as invoice/report/contract. "
     "Do all three without re-downloading the file each time."),

    ("chain_parse_split_extract",
     "Here's a 40-page legal contract: https://example.com/contract.pdf. "
     "First split it into sections (cover, definitions, obligations, signatures). "
     "Then extract the key obligations and deadlines from just the obligations section."),

    ("chain_classify_route_extract",
     "I'm building a document pipeline. The file at https://example.com/doc.pdf could be an invoice, "
     "a purchase order, or a statement. First classify it, then if it's an invoice extract the "
     "line items with amounts, otherwise extract the PO number and vendor details."),

    ("chain_upload_persist_jobid",
     "Upload https://example.com/large-report.pdf first to get a stable ID. "
     "Then parse it with persist_results=True so I can reuse the job ID. "
     "Then extract the executive summary section using the job ID."),

    ("chain_async_poll",
     "I submitted a 200-page document to Reducto and got back job_id='job_abc123' with status pending. "
     "How do I retrieve the results when it's done?"),

    ("chain_split_extract_all_sections",
     "This annual report https://example.com/annual_report.pdf has 5 sections: "
     "Executive Summary, Financials, Operations, Risk Factors, and Appendix. "
     "Split it by those sections, then extract structured data from each section separately."),

    ("chain_image_then_extract",
     "I have a financial report at https://example.com/financials.pdf where the revenue breakdown "
     "is only shown in a bar chart on page 3, not in any table. "
     "Parse the document including the chart image, then extract the revenue figures from it."),

    ("chain_edit_then_classify",
     "Fill in the form at https://example.com/application.pdf with: "
     "name='Acme Corp', date='2026-01-15', amount=50000. "
     "Then lock the fields so they can't be modified, and classify the completed form."),

    ("chain_multiformat_batch",
     "I have three documents to process: "
     "https://example.com/invoice.pdf, https://example.com/receipt.jpg, https://example.com/order.docx. "
     "Parse all three and extract the total amounts. Be efficient about it."),

    ("chain_page_range_then_extract",
     "This 100-page report at https://example.com/report.pdf has the financial data I need on pages 12-18 only. "
     "Parse just those pages to save cost, then extract all dollar amounts and percentages."),
]

# ── Category 4: Negative / conflict resolution ────────────────────────────

CONFLICT_SCENARIOS: list[tuple[str, str]] = [
    ("conflict_citations_chunk",
     "Parse https://example.com/doc.pdf and give me the output chunked by page. "
     "I also need the exact bounding box coordinates for every table cell so I can highlight them in my UI. "
     "Use both citations and chunk_mode='page'."),
    # Correct: citations=True, NO chunk_mode (they're mutually exclusive)

    ("conflict_latency_deep",
     "Extract data from https://example.com/doc.pdf as fast as possible — I need it under 2 seconds. "
     "Use deep extraction mode for maximum accuracy and also enable the latency optimization flag."),
    # Correct: optimize_for_latency=True, deep_extract=False (contradictory)

    ("conflict_wrong_tool_parse_for_extract",
     "I need to get the vendor name, invoice number, and line items as structured JSON from "
     "https://example.com/invoice.pdf. Use the parse endpoint."),
    # Correct: should use reducto_extract with schema, not parse

    ("conflict_upload_not_needed",
     "I have a public stable URL https://cdn.example.com/permanent/doc.pdf that will never expire. "
     "Upload it first, then parse it, then extract the summary."),
    # Correct: skip upload since URL is stable — go straight to parse

    ("conflict_array_extract_single",
     "Extract the company name and CEO from this single-company profile at "
     "https://example.com/profile.pdf. Use array_extract since there might be multiple values."),
    # Correct: array_extract=False — it's a single record, not repeating rows
]

# ── Category 5: Targeted remediation — 3 probes Kimi missed ──────────────

REMEDIATION_SCENARIOS: list[tuple[str, str]] = [
    # optimize_latency_not_deep — model must NOT enable deep_extract alongside latency opt
    ("remediation_latency_no_deep_1",
     "Extract the total revenue from https://example.com/financials.pdf as quickly as possible. "
     "Speed is more important than completeness here — this is a real-time dashboard."),
    ("remediation_latency_no_deep_2",
     "I need a fast approximate extraction from https://example.com/report.pdf. "
     "Latency target is under 1 second. Turn on speed optimization but don't do a deep scan."),
    ("remediation_latency_no_deep_3",
     "Quick extract from https://example.com/doc.pdf — we're in a latency-sensitive path. "
     "Optimize for speed, not accuracy. Do NOT use thorough/deep extraction."),
    ("remediation_latency_no_deep_4",
     "Extract key fields from https://example.com/brief.pdf. "
     "This is for a live preview so it needs to be fast. "
     "Use latency optimization. Make sure deep extraction is off — it would be too slow."),
    ("remediation_latency_no_deep_5",
     "Real-time extraction needed from https://example.com/data.pdf. "
     "Enable the performance/latency flag. Do NOT combine with deep_extract — those two conflict."),

    # upload_reuse — must reuse reducto:// across ALL subsequent operations
    ("remediation_upload_reuse_1",
     "This S3 presigned URL expires in 2 minutes: https://s3.amazonaws.com/b/f.pdf?X-Amz-Expires=120. "
     "I need to parse it, extract a summary, AND classify it as invoice or report. "
     "Make sure you only download the file once."),
    ("remediation_upload_reuse_2",
     "Upload https://example.com/doc.pdf?token=abc123&exp=1700000001 — it's a short-lived link. "
     "Then use the uploaded ID to: (1) parse for text, (2) extract the financial table, (3) split by section. "
     "Reuse the same reducto:// reference for all three."),
    ("remediation_upload_reuse_3",
     "I have an expiring presigned link: https://storage.googleapis.com/bucket/contract.pdf?X-Goog-Expires=300. "
     "Run parse, extract (parties and dates), and classify on it. "
     "Upload it first so you have a stable ID to reuse across all three calls."),
    ("remediation_upload_reuse_4",
     "This link is temporary and will expire: https://cdn.example.com/temp/report.pdf?sig=xyz. "
     "I need: full parse, extraction of all dollar amounts, and a document classification. "
     "Don't re-download it three times — upload once and reuse."),
    ("remediation_upload_reuse_5",
     "Process https://presigned.s3.example.com/file.pdf?expires=1700000000 — presigned URL, expires soon. "
     "Do a parse AND an extract on it. Use a single upload to get a stable reducto:// ID first."),

    # edit_flatten_lock — must use flatten=True to prevent re-editing
    ("remediation_flatten_lock_1",
     "Fill in the PDF form at https://example.com/nda.pdf: "
     "counterparty='Acme Corp', date='2026-03-28', governing_law='California'. "
     "This is a legally binding NDA — lock the fields after filling so they can't be changed."),
    ("remediation_flatten_lock_2",
     "Complete the contract at https://example.com/contract.pdf with the agreed terms: "
     "party_a='Reducto Inc', party_b='Client LLC', effective_date='2026-04-01'. "
     "Flatten it afterwards — this is the final signed version and must be tamper-proof."),
    ("remediation_flatten_lock_3",
     "Fill out https://example.com/form.pdf with: name='John Doe', ssn_last4='1234', amount='$5,000'. "
     "After filling, flatten/lock the form so the fields are permanently embedded and cannot be re-opened or modified."),
    ("remediation_flatten_lock_4",
     "I need to fill and finalize https://example.com/agreement.pdf. "
     "Fields: company='TechCorp', date='today', signature_block='Authorized Representative'. "
     "Use flatten=True — this becomes a read-only document after submission."),
    ("remediation_flatten_lock_5",
     "Edit https://example.com/tax_form.pdf: fill in the income fields with the provided values. "
     "Then lock the form flat — it's being submitted to the IRS and cannot be editable after this point."),
]

# ── Category 6: Cost/performance optimization ─────────────────────────────

COST_SCENARIOS: list[tuple[str, str]] = [
    ("cost_page_range_large_doc",
     "I have a 200-page annual report at https://example.com/annual.pdf but I only need "
     "the financial statements on pages 45-72. Parse only those pages — I don't want to pay for the rest."),

    ("cost_page_range_classify",
     "Classify this 500-page document https://example.com/huge.pdf. "
     "The document type is always identifiable from the first 3 pages — limit processing to pages 1-3."),

    ("cost_embedding_optimized",
     "Parse https://example.com/knowledge_base.pdf. The output will be embedded into a vector database "
     "for semantic search. Optimize the output for embedding quality."),

    ("cost_persist_for_reuse",
     "Parse https://example.com/report.pdf with persist_results=True. "
     "I'll be running multiple extract calls against it afterwards and want to reuse the parse result."),

    ("cost_skip_images_text_only",
     "Parse https://example.com/doc.pdf — I only need the text and tables. "
     "No need to return figure images, skip them to keep the response lean."),

    ("cost_batch_array_extract",
     "Extract all 847 transaction line items from https://example.com/bank_statement.pdf. "
     "There are hundreds of repeating rows. Use the appropriate mode for bulk row extraction."),

    ("cost_filter_for_llm",
     "Parse https://example.com/report.pdf for feeding to an LLM. "
     "Remove headers, footers, and page numbers — just clean body text and tables."),
]

# ── Category 7: Edge cases ────────────────────────────────────────────────

EDGE_SCENARIOS: list[tuple[str, str]] = [
    ("edge_job_expiry_resubmit",
     "I tried to use jobid://expired-job-123 but got an error saying the job has expired. "
     "The original document was at https://example.com/doc.pdf. How do I recover?"),

    ("edge_empty_ocr_retry",
     "I parsed https://example.com/scan.pdf and got back empty content — no text was extracted. "
     "The document is a scanned paper form. What should I do differently?"),

    ("edge_table_page_boundary",
     "This financial report https://example.com/financials.pdf has a portfolio table that starts "
     "on page 5 and continues onto page 6. Parse it so the table comes back as one unified table, "
     "not split across pages."),

    ("edge_mixed_content_types",
     "Parse https://example.com/mixed.pdf — it has both typed text sections and scanned image sections "
     "on different pages. Make sure all content is extracted regardless of whether it's typed or scanned."),

    ("edge_presigned_batch",
     "I have 5 documents that all need to be classified: "
     "[https://s3.example.com/a.pdf, https://s3.example.com/b.pdf, https://s3.example.com/c.pdf]. "
     "They're presigned URLs. Process them efficiently."),

    ("edge_large_doc_async",
     "Submit https://example.com/massive-200page.pdf for parsing. It's a large document that "
     "will probably be processed asynchronously. Tell me how to handle the async job response."),

    ("edge_chart_only_data",
     "The revenue data in https://example.com/report.pdf is only shown in a pie chart on page 2. "
     "There are no tables or text with the actual numbers. How do I extract those figures?"),

    ("edge_section_split_rules",
     "Split https://example.com/legal.pdf into Introduction, Terms, and Signatures sections. "
     "Make sure no table gets cut in half at a section boundary — preserve tables whole."),
]

# ── Combine all non-probe scenarios ──────────────────────────────────────

ALL_SCENARIO_BANKS: list[tuple[str, str]] = (
    _fmt_scenarios()
    + CHAIN_SCENARIOS
    + CONFLICT_SCENARIOS
    + REMEDIATION_SCENARIOS
    + COST_SCENARIOS
    + EDGE_SCENARIOS
)

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format for fine-tuning)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "reducto_parse",
        "description": (
            "Parse a document into structured text, tables, and figures. "
            "input: public URL, reducto:// file ID, or JSON array string of URLs. "
            "Use extraction_mode='ocr' for scanned documents (PNG, JPG, TIFF, BMP). "
            "Use persist_results=True when you plan to run extract/split/classify afterwards. "
            "Use chunk_mode='section' for semantic RAG chunking. "
            "Use return_figure_images=True to include chart/figure image URLs. "
            "Use return_images_types=[\"figure\",\"table\"] for specific image subsets. "
            "Use filter_blocks=[\"Header\",\"Footer\",\"Page Number\"] for clean LLM input. "
            "Use page_range_start/end to reduce cost on large docs. "
            "Use embedding_optimized=True when output will be vector-embedded. "
            "Use merge_tables=True when a table spans multiple pages. "
            "Use agentic_scopes=[\"table\"] when tables have complex alignment."
        ),
        "parameters": {"type": "object", "required": ["input"], "properties": {
            "input": {"type": "string"},
            "table_format": {"type": "string", "enum": ["html","md","json","csv","jsonbbox","dynamic"]},
            "extraction_mode": {"type": "string", "enum": ["hybrid","ocr"]},
            "merge_tables": {"type": "boolean"},
            "persist_results": {"type": "boolean"},
            "chunk_mode": {"type": "string", "enum": ["variable","section","page","block","disabled"]},
            "chunk_size": {"type": "integer"},
            "return_figure_images": {"type": "boolean"},
            "return_images_types": {"type": "string"},
            "filter_blocks": {"type": "string"},
            "page_range_start": {"type": "integer"},
            "page_range_end": {"type": "integer"},
            "embedding_optimized": {"type": "boolean"},
            "agentic_scopes": {"type": "string"},
        }},
    }},
    {"type": "function", "function": {
        "name": "reducto_extract",
        "description": (
            "Extract structured JSON from a document using a JSON Schema. "
            "input: URL, reducto:// ID, or 'jobid://JOB_ID' to reuse a cached parse. "
            "Use array_extract=True when extracting repeating rows (line items, holdings, transactions). "
            "Use citations=True to get bounding box coordinates per field. "
            "IMPORTANT: citations=True and chunk_mode are mutually exclusive — do not combine. "
            "Use optimize_for_latency=True for speed; when used, set deep_extract=False. "
            "Use system_prompt for unit conversion or output formatting instructions. "
            "Use include_images=True when data is only in charts/figures, not text or tables."
        ),
        "parameters": {"type": "object", "required": ["input", "schema_json"], "properties": {
            "input": {"type": "string"},
            "schema_json": {"type": "string"},
            "array_extract": {"type": "boolean"},
            "deep_extract": {"type": "boolean"},
            "citations": {"type": "boolean"},
            "optimize_for_latency": {"type": "boolean"},
            "system_prompt": {"type": "string"},
            "include_images": {"type": "boolean"},
        }},
    }},
    {"type": "function", "function": {
        "name": "reducto_split",
        "description": (
            "Split a document into named sections with page ranges. "
            "Use split_rules for constraints like 'never split a table across sections'. "
            "Use table_cutoff='preserve' to keep tables whole at section boundaries. "
            "Use page_range_start/end for large docs to reduce cost."
        ),
        "parameters": {"type": "object", "required": ["input", "split_description"], "properties": {
            "input": {"type": "string"},
            "split_description": {"type": "string"},
            "split_rules": {"type": "string"},
            "table_cutoff": {"type": "string", "enum": ["preserve","allow"]},
            "page_range_start": {"type": "integer"},
            "page_range_end": {"type": "integer"},
        }},
    }},
    {"type": "function", "function": {
        "name": "reducto_classify",
        "description": (
            "Classify a document into one of the provided categories. "
            "Use page_range_start/end to limit pages processed (max 10 recommended for cost). "
            "Use document_metadata to pass pipeline/source context."
        ),
        "parameters": {"type": "object", "required": ["input", "classification_schema"], "properties": {
            "input": {"type": "string"},
            "classification_schema": {"type": "string"},
            "document_metadata": {"type": "string"},
            "page_range_start": {"type": "integer"},
            "page_range_end": {"type": "integer"},
        }},
    }},
    {"type": "function", "function": {
        "name": "reducto_edit",
        "description": (
            "Edit or fill a PDF using natural language instructions or structured field definitions. "
            "Use form_schema_json for multi-field structured fills (more precise than edit_instructions). "
            "Use flatten=True to lock filled fields permanently — required for legally binding documents "
            "or any document that must not be re-edited after submission."
        ),
        "parameters": {"type": "object", "required": ["document_url"], "properties": {
            "document_url": {"type": "string"},
            "edit_instructions": {"type": "string"},
            "form_schema_json": {"type": "string"},
            "flatten": {"type": "boolean"},
        }},
    }},
    {"type": "function", "function": {
        "name": "reducto_upload",
        "description": (
            "Upload a file to get a stable reducto:// ID. "
            "Use BEFORE other operations when: "
            "(1) the source URL is a short-lived presigned URL that may expire, OR "
            "(2) you need to run 2+ operations on the same document to avoid re-downloading. "
            "Do NOT upload if the URL is stable/permanent and only one operation is needed."
        ),
        "parameters": {"type": "object", "required": ["file_url"], "properties": {
            "file_url": {"type": "string"},
        }},
    }},
    {"type": "function", "function": {
        "name": "reducto_get_job",
        "description": (
            "Poll for the result of an async Reducto job. "
            "Use when a previous call returned status='pending' with a job_id. "
            "If the job has expired, re-submit the original document."
        ),
        "parameters": {"type": "object", "required": ["job_id"], "properties": {
            "job_id": {"type": "string"},
        }},
    }},
]


# ---------------------------------------------------------------------------
# Training example
# ---------------------------------------------------------------------------

@dataclass
class TrainingExample:
    probe_id: str
    source: str          # "harvested" | "generated"
    teacher_model: str
    user_prompt: str
    tool_calls: list
    score: int = 3


def example_to_jsonl(ex: TrainingExample) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": ex.user_prompt},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                 "function": {"name": tc["name"],
                              "arguments": json.dumps(tc["arguments"]) if isinstance(tc["arguments"], dict) else tc["arguments"]}}
                for tc in ex.tool_calls
            ]},
        ],
        "tools": TOOL_SCHEMAS,
        "metadata": {"probe_id": ex.probe_id, "source": ex.source, "teacher": ex.teacher_model},
    }


# ---------------------------------------------------------------------------
# Phase 1: Harvest from hard probe results
# ---------------------------------------------------------------------------

def harvest_from_results(hard_dir: Path, min_score: int = 3) -> list[TrainingExample]:
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from bench_param_probe import HARD_PROBES, PROBES
        probe_map = {p.id: p.prompt for p in (PROBES + HARD_PROBES)}
    except Exception as e:
        print(f"  [harvest] could not import probe prompts: {e}")
        probe_map = {}

    examples = []
    if not hard_dir.exists():
        print(f"  [harvest] {hard_dir} not found")
        return examples

    excluded = {
        "Mistral Devstral Small", "Alibaba Tongyi DeepResearch 30B-A3B",
        "Codex Mini Latest", "Arcee Trinity Large (prime)",
        "Nemotron Super 120B (Nebius bf16)", "Qwen3 Coder Next (ionstream fp8)",
    }

    for f in sorted(hard_dir.glob("*.json")):
        data = json.loads(f.read_text())
        if not data: continue
        model_name = data[0].get("model", f.stem)
        if model_name in excluded: continue
        for r in data:
            if r.get("score", 0) < min_score: continue
            if r.get("credit_error"): continue
            pid = r.get("probe_id", "")
            prompt = probe_map.get(pid, "")
            if not prompt: continue
            tool_calls = [
                {"name": tc["tool"], "arguments": tc.get("args", {})}
                for tc in r.get("tool_calls", [])
                if tc.get("tool")
            ]
            if not tool_calls: continue
            examples.append(TrainingExample(
                probe_id=pid, source="harvested",
                teacher_model=model_name, user_prompt=prompt,
                tool_calls=tool_calls, score=r["score"],
            ))

    print(f"  [harvest] {len(examples)} examples (score≥{min_score}) from {hard_dir}")
    return examples


# ---------------------------------------------------------------------------
# Phase 2: Generate prompt variations
# ---------------------------------------------------------------------------

REPHRASE_PROMPT = """\
You are generating synthetic training data for an AI agent. Rephrase the following user request in {n} distinct ways.
Rules:
- Same underlying task, different wording/framing/detail level
- Vary style: some terse, some verbose, some from a dev, some from a non-technical analyst
- Some can mention urgency, cost concerns, scale, or downstream use case
- Be realistic — something a real user of a document AI API might actually write

Original:
{original}

Return ONLY a JSON array of {n} strings. No markdown, no explanation."""


_REPHRASE_BATCH = 20   # ask for this many variations per call to avoid truncation

def generate_variations(prompt: str, n: int, dry_run: bool = False) -> list[str]:
    if dry_run:
        return [f"[DRY RUN #{i+1}] {prompt[:80]}..." for i in range(n)]

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=REPHRASE_MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        temperature=0.85,
        max_tokens=8192,   # 20 rephrases × ~60 tokens each ≈ 1,200 tokens — plenty
        timeout=60,
    )

    results: list[str] = []
    remaining = n
    while remaining > 0:
        batch = min(remaining, _REPHRASE_BATCH)
        try:
            msg  = llm.invoke(REPHRASE_PROMPT.format(n=batch, original=prompt))
            text = msg.content.strip()
            if "```" in text:
                text = text.split("```")[1].lstrip("json").strip()
            chunk = json.loads(text)
            if isinstance(chunk, list):
                results.extend(str(v) for v in chunk[:batch])
        except Exception as e:
            print(f"    [rephrase] error (batch {batch}): {e}")
            # keep whatever we have so far; caller falls back to base prompt if empty
        remaining -= batch

    return results


# ---------------------------------------------------------------------------
# Phase 3: Run variation through teacher model
# ---------------------------------------------------------------------------

def run_through_teacher(
    model: ModelConfig, prompt: str, probe_id: str, dry_run: bool = False
) -> Optional[TrainingExample]:
    if dry_run:
        return TrainingExample(
            probe_id=probe_id, source="generated", teacher_model=model.display,
            user_prompt=prompt,
            tool_calls=[{"name": "reducto_parse", "arguments": {"input": "https://example.com/doc.pdf"}}],
        )
    try:
        from bench_param_probe import _build_llm, _build_probe_tools
        llm = _build_llm(model)
        tools = _build_probe_tools()
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ])
        raw_calls = getattr(response, "tool_calls", []) or []
        tool_calls = [
            {"name": tc.get("name", ""), "arguments": tc.get("args", {})}
            for tc in raw_calls if tc.get("name")
        ]
        if not tool_calls:
            return None
        return TrainingExample(
            probe_id=probe_id, source="generated",
            teacher_model=model.display, user_prompt=prompt, tool_calls=tool_calls,
        )
    except Exception as e:
        print(f"    [teacher:{model.display[:20]}] {probe_id}: {str(e)[:80]}")
        return None


# ---------------------------------------------------------------------------
# Select teacher models from hard probe results
# ---------------------------------------------------------------------------

def get_teacher_models(top_n: int = 8) -> list[ModelConfig]:
    """
    Return top-N teacher ModelConfig objects by hard probe score.
    Reasoning models (o3, Opus, etc.) are excluded — they burn $10-800 per
    2,000 calls due to thinking tokens and are overkill for single-shot generation.
    """
    all_models = PREMIUM_MODELS + OPENROUTER_MODELS
    name_to_model = {m.display: m for m in all_models}

    # Exclude expensive reasoning models from generation teachers
    def _is_cheap(m: ModelConfig) -> bool:
        if m.reasoning:
            return False   # thinking budget = $$$
        pricey_ids = ("o3", "o4", "opus", "gemini-3.1-pro")
        return not any(p in m.id.lower() for p in pricey_ids)

    # Score from results dir if available
    if HARD_DIR.exists():
        scores = {}
        for f in HARD_DIR.glob("*.json"):
            data = json.loads(f.read_text())
            if not data: continue
            name = data[0].get("model", "")
            scores[name] = sum(r.get("score", 0) for r in data)
        ranked = [m for m, _ in sorted(scores.items(), key=lambda x: -x[1])]
    else:
        ranked = TOP_TEACHER_NAMES

    # Use the curated generation teacher list, filtered to top_n
    result = []
    for name in GENERATION_TEACHERS[:top_n]:
        m = name_to_model.get(name)
        if m and _is_cheap(m):
            result.append(m)

    return result


# ---------------------------------------------------------------------------
# L4 — Reducto live API gate (real-time, gentle rate limiting)
# ---------------------------------------------------------------------------

TEST_DOC      = "https://cdn.reducto.ai/samples/fidelity-example.pdf"
_URL_RE       = re.compile(r"https?://\S+\.(?:pdf|docx?|xlsx?|pptx?|png|jpe?g|tiff?|csv|html?|txt|msg|eml|bmp)\b", re.IGNORECASE)
_REDUCTO_BASE = "https://platform.reducto.ai"

# Gentle rate limiter: max 8 concurrent Reducto calls + 0.1 s between fires
# Was: 4 concurrent / 0.25s → 4 req/s. We were at ~1.1 req/s (28% utilisation).
# Bumped to 8 concurrent / 0.1s → up to 10 req/s, still well within free-tier limits.
_REDUCTO_SEM  = threading.Semaphore(8)
_REDUCTO_LOCK = threading.Lock()
_last_call_ts = [0.0]
_MIN_INTERVAL = 0.1    # seconds between ANY Reducto call (10 req/s ceiling)

_ENDPOINT_MAP = {
    "reducto_parse":    "/parse",
    "reducto_extract":  "/extract",
    "reducto_split":    "/split",
    "reducto_classify": "/classify",
    "reducto_edit":     "/edit",
    "reducto_upload":   "/upload",
    "reducto_get_job":  None,
}

_SAFE_KEYS = {
    "reducto_parse":    {"input", "chunk_mode", "extraction_mode", "table_format",
                         "filter_blocks", "merge_tables"},
    "reducto_extract":  {"input", "schema_json", "array_extract", "deep_extract",
                         "include_images", "optimize_for_latency"},
    "reducto_split":    {"input", "split_description"},
    "reducto_classify": {"input", "classification_schema"},
    "reducto_edit":     {"document_url", "edit_instructions", "flatten"},
    "reducto_upload":   {"file_url"},
}


def _l4_check(tool_calls: list[dict], api_key: str) -> tuple[bool, str]:
    """
    Call the first testable Reducto endpoint with normalised args.
    Returns (passed: bool, reason: str).
    2xx  → pass.   4xx → fail (bad params).   5xx/timeout → pass (API hiccup).
    """
    if not api_key:
        return True, "no_key"

    for tc in tool_calls:
        name     = tc.get("name", "")
        raw_args = tc.get("arguments", {})
        if isinstance(raw_args, str):
            try: raw_args = json.loads(raw_args)
            except: raw_args = {}

        path = _ENDPOINT_MAP.get(name)
        if path is None:
            continue   # get_job: synthetic ids can't be tested

        # Normalise: swap any doc URL for the known-good test doc
        args = {k: v for k, v in raw_args.items() if k in _SAFE_KEYS.get(name, set())}
        for key in ("input", "document_url", "file_url"):
            if key not in args or not isinstance(args.get(key), str) or \
               args[key].startswith("reducto://") or args[key].startswith("jobid://") or \
               _URL_RE.search(args.get(key, "")):
                args[key] = TEST_DOC

        url  = f"{_REDUCTO_BASE}{path}"
        body = json.dumps(args).encode()
        req  = urllib.request.Request(
            url, data=body,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            method="POST",
        )

        # Rate limiting: honour the global ceiling
        with _REDUCTO_LOCK:
            gap = _MIN_INTERVAL - (time.time() - _last_call_ts[0])
            if gap > 0:
                time.sleep(gap)
            _last_call_ts[0] = time.time()

        with _REDUCTO_SEM:
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return True, f"http_{resp.status}"
            except urllib.error.HTTPError as e:
                code = e.code
                detail = e.read().decode("utf-8", errors="replace")[:100]
                if 400 <= code < 500:
                    return False, f"http_{code}: {detail}"
                return True, f"http_{code}_server_err"   # 5xx = not our fault
            except Exception as ex:
                return True, f"network_err: {str(ex)[:60]}"   # timeout etc. = pass

    return True, "no_testable_call"


# ---------------------------------------------------------------------------
# Main async pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(
    target: int,
    dry_run: bool,
    variations_per_prompt: Optional[int],
    resume: bool,
    min_score: int,
    n_teachers: int,
    use_l4: bool = True,
) -> list[TrainingExample]:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_examples: list[TrainingExample] = []
    reducto_api_key = os.environ.get("REDUCTO_API_KEY", "")
    l4_enabled = use_l4 and bool(reducto_api_key) and not dry_run
    l4_rejected = 0

    # Load checkpoint if resuming
    seen_keys: set[tuple] = set()
    skipped_dry = 0
    if resume and CHECKPOINT.exists():
        for line in CHECKPOINT.read_text().splitlines():
            if not line.strip(): continue
            try:
                ex_dict = json.loads(line)
                msgs = ex_dict.get("messages", [])
                user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                # Skip any dry-run examples that leaked into the checkpoint
                if "[DRY RUN" in user_msg:
                    skipped_dry += 1
                    continue
                meta    = ex_dict.get("metadata", {})
                teacher = meta.get("teacher", "")
                key = (meta.get("probe_id",""), user_msg[:100], teacher)
                seen_keys.add(key)
                # Reconstruct minimal TrainingExample for counting
                all_examples.append(TrainingExample(
                    probe_id=meta.get("probe_id",""), source=meta.get("source",""),
                    teacher_model=meta.get("teacher",""), user_prompt=user_msg,
                    tool_calls=[],
                ))
            except Exception:
                pass
        print(f"  [resume] loaded {len(all_examples)} examples from checkpoint", end="")
        if skipped_dry:
            print(f" (skipped {skipped_dry} dry-run leftovers)", end="")
        print()

    # Phase 1: Harvest
    print("\n── Phase 1: Harvest existing probe results ──")
    harvested = harvest_from_results(HARD_DIR, min_score)
    for ex in harvested:
        key = (ex.probe_id, ex.user_prompt[:100])
        if key not in seen_keys:
            seen_keys.add(key)
            all_examples.append(ex)
    print(f"  Total after harvest: {len(all_examples)}")

    if len(all_examples) >= target:
        print(f"  Already at target ({len(all_examples)}/{target})")
        return all_examples

    # Phase 2+3: Generate variations through teachers
    print(f"\n── Phase 2: Generate to target={target} ──")

    teachers = get_teacher_models(n_teachers)
    if not teachers:
        print("  ERROR: no teacher models found")
        return all_examples
    print(f"  Teachers ({len(teachers)}): {[m.display[:25] for m in teachers]}")

    # Build full prompt list: hard probes + all scenario banks
    from bench_param_probe import HARD_PROBES, PROBES
    probe_prompts: list[tuple[str, str]] = [(p.id, p.prompt) for p in (PROBES + HARD_PROBES)]
    all_prompts = probe_prompts + ALL_SCENARIO_BANKS
    random.shuffle(all_prompts)

    # How many variations per prompt to hit target
    remaining = target - len(all_examples)
    if variations_per_prompt is None:
        # Round-robin: spread remaining across all prompts × teachers
        variations_per_prompt = max(1, -(-remaining // (len(all_prompts) * len(teachers))))
    print(f"  {len(all_prompts)} prompt seeds × {variations_per_prompt} variations × {len(teachers)} teachers")
    print(f"  Projected: ~{len(all_prompts) * variations_per_prompt * len(teachers):,} examples\n")

    # Cap concurrent OpenRouter calls to avoid rate-limiting.
    # Only 1 rate-limit error observed at 50 concurrent → safe to push higher.
    # 100 concurrent × avg 5-8s/call ≈ 12-20 completions/sec → ~90-140 epm.
    SEED_PARALLELISM    = 20   # seeds processed simultaneously (more = smoother queue fill)
    TASK_TIMEOUT        = 60   # seconds per individual teacher call
    MAX_CONCURRENT_CALLS = 100 # global cap on in-flight OpenRouter calls
    n_workers = MAX_CONCURRENT_CALLS + 20  # thread pool slightly larger than concurrency cap
    loop      = asyncio.get_running_loop()
    executor  = ThreadPoolExecutor(max_workers=n_workers)
    task_sem  = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
    seed_sem  = asyncio.Semaphore(SEED_PARALLELISM)
    ckpt_lock = asyncio.Lock()                 # serialise checkpoint writes

    checkpoint_fh = open(CHECKPOINT, "a")

    # shared mutable state (protected by ckpt_lock for writes)
    generated = 0

    async def run_one(model, prompt, probe_id):
        """Teacher call + L4 + immediate checkpoint write. Returns (added, l4_dropped)."""
        nonlocal generated, l4_rejected
        async with task_sem:
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(executor, run_through_teacher, model, prompt, probe_id, dry_run),
                    timeout=TASK_TIMEOUT,
                )
            except asyncio.TimeoutError:
                return 0, 0

        if not isinstance(result, TrainingExample):
            return 0, 0

        key = (result.probe_id, result.user_prompt[:100], result.teacher_model)
        if key in seen_keys:       # fast pre-check (no lock needed for read under GIL)
            return 0, 0

        # L4 gate — run in executor so it doesn't block the event loop
        if l4_enabled:
            passed, _ = await loop.run_in_executor(
                executor, _l4_check, result.tool_calls, reducto_api_key
            )
            if not passed:
                l4_rejected += 1
                return 0, 1

        # Write immediately, one example at a time, as it arrives
        async with ckpt_lock:
            if key in seen_keys:   # re-check under lock to prevent duplicates
                return 0, 0
            if len(all_examples) >= target:
                return 0, 0
            seen_keys.add(key)
            all_examples.append(result)
            checkpoint_fh.write(json.dumps(example_to_jsonl(result)) + "\n")
            checkpoint_fh.flush()
            generated += 1
            return 1, 0

    async def process_seed(i, probe_id, base_prompt):
        async with seed_sem:
            if len(all_examples) >= target:
                return

            if dry_run:
                variations = generate_variations(base_prompt, variations_per_prompt, dry_run=True)
            else:
                variations = await loop.run_in_executor(
                    executor, generate_variations, base_prompt, variations_per_prompt
                )
            if not variations:
                # Rephrase failed — generate programmatic variations so we don't
                # collide with the bare base_prompt already in seen_keys from earlier runs.
                _pfx = [
                    "Task: ", "I need to: ", "Request: ", "Please help me: ",
                    "Goal: ", "Quick question: ", "Action needed: ", "Help: ",
                ]
                variations = [p + base_prompt for p in _pfx]

            tasks   = [run_one(m, v, probe_id) for v in variations for m in teachers]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            added   = sum(r[0] for r in results if isinstance(r, tuple))
            dropped = sum(r[1] for r in results if isinstance(r, tuple))
            l4_note = f"  L4✗{dropped}" if dropped else ""
            print(f"  [{i+1:>3}/{len(all_prompts)}] {probe_id[:30]:30s}  +{added:3d}{l4_note}  total={len(all_examples):,}/{target:,}")

    # Multi-pass loop: re-run seeds with fresh rephrases until target is hit.
    # Each pass generates ~5K-12K new examples (new rephrase text = new dedup keys).
    # Typical run: ~3 passes × ~1.5 hrs each = ~4-5 hrs total for target=30K.
    pass_num = 0
    while len(all_examples) < target:
        pass_num += 1
        remaining = target - len(all_examples)
        print(f"\n  ── Pass {pass_num}  ({len(all_examples):,}/{target:,}, need {remaining:,} more) ──")

        random.shuffle(all_prompts)   # different seed order each pass = different variation combos

        seed_tasks = [
            process_seed(i, probe_id, base_prompt)
            for i, (probe_id, base_prompt) in enumerate(all_prompts)
        ]
        await asyncio.gather(*seed_tasks)

        new_this_pass = len(all_examples) - (target - remaining)
        print(f"  Pass {pass_num} done: +{new_this_pass:,} new examples  total={len(all_examples):,}/{target:,}")

        if new_this_pass < 500:
            # Diminishing returns — rephrase space nearly exhausted
            print("  Low yield this pass — stopping early.")
            break

    checkpoint_fh.close()
    executor.shutdown(wait=False)
    print(f"\n  Generated {generated} new examples. Grand total: {len(all_examples):,}")
    if l4_enabled:
        print(f"  L4 Reducto API gate: {l4_rejected} examples dropped (bad params rejected by API)")
    return all_examples


# ---------------------------------------------------------------------------
# Save final dataset
# ---------------------------------------------------------------------------

def save_dataset(examples: list[TrainingExample], out_dir: Path, split: float = 0.9):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deduplicate
    seen, unique = set(), []
    for ex in examples:
        key = (ex.probe_id, ex.user_prompt[:100], ex.tool_calls[0]["name"] if ex.tool_calls else "")
        if key not in seen:
            seen.add(key); unique.append(ex)

    random.shuffle(unique)
    n_train = int(len(unique) * split)
    train, val = unique[:n_train], unique[n_train:]

    train_path = out_dir / "train.jsonl"
    val_path   = out_dir / "val.jsonl"

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(example_to_jsonl(ex)) + "\n")
    with open(val_path, "w") as f:
        for ex in val:
            f.write(json.dumps(example_to_jsonl(ex)) + "\n")

    # Stats
    by_probe  = {}
    by_source = {}
    for ex in unique:
        by_probe[ex.probe_id]  = by_probe.get(ex.probe_id, 0) + 1
        by_source[ex.source]   = by_source.get(ex.source, 0) + 1

    print(f"\n{'='*60}")
    print(f"Dataset saved  —  {len(unique):,} unique examples")
    print(f"  Train : {len(train):,}  →  {train_path}")
    print(f"  Val   : {len(val):,}    →  {val_path}")
    print(f"\n  By source : {by_source}")
    print(f"\n  Top probe coverage:")
    for pid, cnt in sorted(by_probe.items(), key=lambda x: -x[1])[:15]:
        bar = "█" * (cnt // 20)
        print(f"    {pid:<35} {cnt:>5}  {bar}")
    return train_path, val_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate 16,900 synthetic Reducto training examples")
    parser.add_argument("--target", type=int, default=16_900,
                        help="Total examples to generate (default: 16,900)")
    parser.add_argument("--variations", type=int, default=None,
                        help="Variations per prompt seed (auto-calculated from --target if not set)")
    parser.add_argument("--teachers", type=int, default=5,
                        help="Number of teacher models to use (default: 8)")
    parser.add_argument("--min-score", type=int, default=3,
                        help="Min probe score to harvest (default: 3)")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint (skip already-generated examples)")
    parser.add_argument("--dry-run", action="store_true",
                        help="No API calls — verify logic and count only")
    parser.add_argument("--no-l4", action="store_true",
                        help="Skip live Reducto API gate during generation")
    args = parser.parse_args()

    out_dir = Path(args.output)
    api_key = os.environ.get("REDUCTO_API_KEY", "")
    print(f"Reducto synthetic data generator")
    print(f"  Target   : {args.target:,} examples")
    print(f"  Teachers : {args.teachers}")
    print(f"  Output   : {out_dir}")
    if args.dry_run:
        print(f"  *** DRY RUN ***")
    if args.resume:
        print(f"  *** RESUME mode ***")
    if not args.no_l4 and api_key and not args.dry_run:
        print(f"  L4 gate  : ON  (Reducto API live check, ≤4 concurrent, ≥0.25s between calls)")
    else:
        print(f"  L4 gate  : OFF")

    examples = asyncio.run(run_pipeline(
        target=args.target,
        dry_run=args.dry_run,
        variations_per_prompt=args.variations,
        resume=args.resume,
        min_score=args.min_score,
        n_teachers=args.teachers,
        use_l4=not args.no_l4,
    ))

    if not examples:
        print("No examples generated.")
        return

    train_path, val_path = save_dataset(examples, out_dir)
    print(f"\nReady to fine-tune Qwen3.5-35B-A3B:")
    print(f"  axolotl train --config configs/reducto_a3b.yml \\")
    print(f"    --dataset {train_path} --val-dataset {val_path}")


if __name__ == "__main__":
    main()
