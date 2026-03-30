#!/usr/bin/env python3
"""
gen_mcp_data.py — Round-2 synthetic data generator: MCP tool-call format.

Generates training examples where the model uses the Reducto MCP server's exact
tool schemas (parameter names match mcp-server/src/index.ts, NOT the raw API).

Key differences from round-1 (gen_synthetic_data.py):
  - `reducto_extract` uses `schema` (not `schema_json`)
  - `reducto_parse` has `filter_blocks` as array, `add_page_markers`, `persist_results`
  - `reducto_edit` has `highlight_color`, `enable_overflow_pages` (no flatten)
  - `reducto_split` has `table_cutoff` enum ["truncate","preserve"]
  - `reducto_get_job` — 7th tool, MCP-only, for jobid:// reuse and async polling
  - System prompt is MCP-context aware

Usage:
    python gen_mcp_data.py                     # full run, target=36669
    python gen_mcp_data.py --target 100        # quick test
    python gen_mcp_data.py --dry-run           # no API calls
    python gen_mcp_data.py --resume            # skip already-saved checkpoints
"""

import argparse
import json
import os
import random
import re
import sys
import time
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

OUTPUT_DIR  = Path("benchmark/data/synthetic_training_mcp")
CHECKPOINT  = OUTPUT_DIR / ".checkpoint_mcp.jsonl"

GENERATION_TEACHERS = [
    "Kimi K2.5 via Fireworks",
    "MiniMax M2.7 (highspeed)",
    "Gemini 3.1 Flash Lite (preview)",
    "Qwen3.5-122B-A10B",
    "Inception Mercury 2",
    "GLM-5 Turbo",
    "GPT-5.4 Nano",
    "StepFun Step-3.5 Flash",
]

REPHRASE_MODEL = "google/gemini-3.1-flash-lite-preview"

# MCP-aware system prompt — agent knows it's in an MCP context
MCP_SYSTEM_PROMPT = (
    "You are an expert Reducto MCP integration agent. "
    "You help users process documents by calling the right Reducto MCP tools with correct parameters. "
    "You have access to 7 MCP tools: reducto_parse, reducto_extract, reducto_split, "
    "reducto_classify, reducto_edit, reducto_upload, and reducto_get_job. "
    "Key rules:\n"
    "- reducto_extract: use 'schema' (JSON string) not 'schema_json'\n"
    "- reducto_parse: filter_blocks is an array like ['Header','Footer']; "
    "set persist_results=true when you plan to reuse results via jobid://\n"
    "- reducto_get_job: use to retrieve async results or verify a job before passing jobid:// to extract/split\n"
    "- reducto_upload: use before multi-operation workflows or expiring presigned URLs\n"
    "- reducto_edit: highlight_color for field visibility, enable_overflow_pages for long fills\n"
    "- reducto_split: table_cutoff='preserve' to keep tables whole at section boundaries\n"
    "Always prefer the most specific, efficient tool for the task."
)

# ---------------------------------------------------------------------------
# MCP Tool Schemas — exact parameter names from mcp-server/src/index.ts
# ---------------------------------------------------------------------------

MCP_TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "reducto_parse",
        "description": (
            "Parse a document (PDF, image, DOCX, XLSX, etc.) into structured content "
            "including text, tables, and figures. Accepts public URLs, presigned S3/GCS/Azure "
            "URLs, reducto:// file IDs from reducto_upload, or jobid://JOB_ID to re-use a "
            "previous parse result. "
            "Set persist_results=true to get a job ID you can pass as 'jobid://JOB_ID' "
            "to reducto_extract or reducto_split — avoids re-parsing the same document."
        ),
        "parameters": {"type": "object", "required": ["input"], "properties": {
            "input": {"type": "string", "description": "Document source: URL, reducto:// file ID, or jobid://JOB_ID"},
            "filter_blocks": {
                "type": "array",
                "items": {"type": "string", "enum": [
                    "Header", "Footer", "Title", "Section Header", "Page Number",
                    "List Item", "Figure", "Table", "Key Value", "Text", "Comment", "Signature"
                ]},
                "description": "Block types to exclude. E.g. ['Header','Footer','Page Number'] for clean LLM input."
            },
            "merge_tables": {"type": "boolean", "description": "Merge tables split across page boundaries."},
            "extraction_mode": {"type": "string", "enum": ["ocr", "hybrid"],
                "description": "'hybrid' (default) or 'ocr' for scanned documents."},
            "chunk_mode": {"type": "string",
                "enum": ["disabled", "variable", "section", "page", "block", "page_sections"],
                "description": "Chunking strategy. 'variable' best for RAG."},
            "table_format": {"type": "string",
                "enum": ["dynamic", "html", "md", "json", "csv", "jsonbbox"],
                "description": "Table output format. 'html' best for merged-cell tables."},
            "add_page_markers": {"type": "boolean",
                "description": "Insert '## Page N' markers between pages."},
            "agentic_scopes": {
                "type": "array",
                "items": {"type": "string", "enum": ["text", "table", "figure"]},
                "description": "AI-powered correction: 'table' fixes misaligned columns, 'figure' extracts chart data."
            },
            "return_figure_images": {"type": "boolean",
                "description": "Return presigned image URLs for every detected figure."},
            "persist_results": {"type": "boolean",
                "description": "Keep result accessible for 1 hour as jobid://JOB_ID for reuse."},
            "page_range_start": {"type": "integer", "description": "First page (1-indexed)."},
            "page_range_end": {"type": "integer", "description": "Last page (1-indexed)."},
        }},
    }},
    {"type": "function", "function": {
        "name": "reducto_extract",
        "description": (
            "Extract specific fields from a document into structured JSON using a JSON Schema. "
            "Use 'schema' parameter (JSON string). "
            "Pass jobid://JOB_ID as input to skip re-parsing a document you already parsed. "
            "Use array_extract=true for repeating rows (line items, transactions, holdings). "
            "Use citations=true to get bounding box coordinates per extracted field."
        ),
        "parameters": {"type": "object", "required": ["input", "schema"], "properties": {
            "input": {"type": "string", "description": "URL, reducto:// ID, or jobid://JOB_ID"},
            "schema": {"type": "string",
                "description": "JSON Schema string defining fields to extract. Add 'description' to each property."},
            "system_prompt": {"type": "string",
                "description": "Custom extraction instructions — units, null handling, format preferences."},
            "array_extract": {"type": "boolean",
                "description": "Extract repeating rows. Schema must have at least one top-level array property."},
            "deep_extract": {"type": "boolean",
                "description": "Agentic refinement pass for difficult documents. Higher cost."},
            "citations": {"type": "boolean",
                "description": "Return bounding box per extracted value. Cannot combine with chunking."},
            "include_images": {"type": "boolean",
                "description": "Include figure/chart images in extraction context for chart data."},
        }},
    }},
    {"type": "function", "function": {
        "name": "reducto_split",
        "description": (
            "Divide a document into named logical sections by page range. "
            "Returns page ranges per section — pass them back to reducto_parse with "
            "page_range_start/end to process each section individually."
        ),
        "parameters": {"type": "object", "required": ["input", "split_description"], "properties": {
            "input": {"type": "string", "description": "URL, reducto:// ID, or jobid://JOB_ID"},
            "split_description": {"type": "string",
                "description": "JSON array of section definitions: [{\"name\":\"...\",\"description\":\"...\"}]"},
            "split_rules": {"type": "string",
                "description": "Override rules for splitting. E.g. 'Never split a table across sections.'"},
            "table_cutoff": {"type": "string", "enum": ["truncate", "preserve"],
                "description": "'preserve' extends section to include the full table at a boundary."},
        }},
    }},
    {"type": "function", "function": {
        "name": "reducto_classify",
        "description": (
            "Classify a document's type against a set of categories. "
            "Use before processing to route to the right pipeline. "
            "Limit page_range_start/end to 1-5 pages for cost efficiency."
        ),
        "parameters": {"type": "object", "required": ["input", "classification_schema"], "properties": {
            "input": {"type": "string", "description": "URL or reducto:// file ID"},
            "classification_schema": {"type": "string",
                "description": "JSON array of categories: [{\"category\":\"...\",\"criteria\":[...]}]"},
            "document_metadata": {"type": "string",
                "description": "Context string to help the classifier."},
            "page_range_start": {"type": "integer"},
            "page_range_end": {"type": "integer"},
        }},
    }},
    {"type": "function", "function": {
        "name": "reducto_edit",
        "description": (
            "Fill forms or modify a document using natural language instructions. "
            "Returns a presigned URL to the edited document (valid 24 hours). "
            "Use highlight_color to make filled fields visible. "
            "Use enable_overflow_pages if filled content might overflow a page."
        ),
        "parameters": {"type": "object", "required": ["document_url", "edit_instructions"], "properties": {
            "document_url": {"type": "string", "description": "URL or reducto:// file ID"},
            "edit_instructions": {"type": "string",
                "description": "Natural language fill/edit instructions."},
            "highlight_color": {"type": "string",
                "description": "Hex color for filled fields. E.g. '#FFFF00' for yellow. Default: red."},
            "enable_overflow_pages": {"type": "boolean",
                "description": "Allow adding new pages if filled content overflows."},
        }},
    }},
    {"type": "function", "function": {
        "name": "reducto_upload",
        "description": (
            "Upload a document to Reducto by URL and get a reducto:// file ID for use with other tools. "
            "Use when the source URL has a short-lived signature or you need a stable reference "
            "for multiple operations. The file ID can be passed as 'input' to any other reducto_* tool."
        ),
        "parameters": {"type": "object", "required": ["file_url"], "properties": {
            "file_url": {"type": "string",
                "description": "Public or presigned URL of the file to upload."},
        }},
    }},
    {"type": "function", "function": {
        "name": "reducto_get_job",
        "description": (
            "Retrieve the result of a previous Reducto job by its ID. "
            "Two main uses: (1) Re-use a previous parse: verify the job succeeded before passing "
            "jobid://JOB_ID to reducto_extract or reducto_split. "
            "(2) Check the status of a long-running async job. "
            "Accepts bare UUID or 'jobid://UUID' prefix."
        ),
        "parameters": {"type": "object", "required": ["job_id"], "properties": {
            "job_id": {"type": "string",
                "description": "Job ID from a previous reducto_* call. Accepts 'jobid://...' or bare UUID."},
        }},
    }},
]

# ---------------------------------------------------------------------------
# Scenario bank — MCP-specific, all 7 tools
# ---------------------------------------------------------------------------

# ── 1. Single-tool scenarios covering each MCP-specific feature ───────────

PARSE_MCP_SCENARIOS = [
    ("mcp_parse_filter_array",
     "Parse https://example.com/report.pdf — I'm feeding the output to an LLM so remove headers, footers, and page numbers. Use an array for the filter."),

    ("mcp_parse_persist_for_reuse",
     "Parse https://example.com/financials.pdf and persist the result — I'll run extract and split on it right after without re-parsing."),

    ("mcp_parse_page_markers",
     "Parse https://example.com/manual.pdf and add page markers so I know which page each chunk came from."),

    ("mcp_parse_agentic_table",
     "Parse https://example.com/portfolio.pdf — the tables have complex merged cells and misaligned columns. Use agentic correction on tables."),

    ("mcp_parse_agentic_figure",
     "Parse https://example.com/earnings.pdf. The revenue breakdown is only in a bar chart. I need the actual numbers extracted from the figure."),

    ("mcp_parse_agentic_all",
     "Parse https://example.com/annual.pdf — apply all agentic scopes: OCR cleanup on text, column fix on tables, and data extraction from figures."),

    ("mcp_parse_persist_then_jobid",
     "Parse https://cdn.reducto.ai/samples/fidelity-example.pdf with persist_results so I can pass the job ID to extract afterwards."),

    ("mcp_parse_page_range_persist",
     "Parse only pages 10-25 of https://example.com/bigdoc.pdf. Persist the result — I'll run extract on those pages using the job ID."),

    ("mcp_parse_chunk_variable",
     "Parse https://example.com/knowledge_base.pdf for a RAG pipeline. Chunk it by variable size and remove headers and footers."),

    ("mcp_parse_html_table",
     "Parse https://example.com/complex_tables.pdf. The tables have merged cells — use HTML table format and agentic table correction."),

    ("mcp_parse_ocr_with_markers",
     "Parse this scanned document: https://example.com/scan.pdf. Force OCR mode and add page markers."),

    ("mcp_parse_figure_images",
     "Parse https://example.com/charts.pdf and return the actual image URLs for each figure — I need to display them in my UI."),
]

GET_JOB_SCENARIOS = [
    ("mcp_getjob_verify_before_reuse",
     "I ran reducto_parse earlier and got job_id='abc-123'. Before I pass it as jobid://abc-123 to reducto_extract, I want to verify the parse succeeded. Check the job status."),

    ("mcp_getjob_retrieve_async",
     "I submitted a large 200-page document to reducto_parse and got back job_id='job-xyz-456' with status pending. Check if it's done now."),

    ("mcp_getjob_bare_uuid",
     "Retrieve the result of job_id 'f47ac10b-58cc-4372-a567-0e02b2c3d479' — that's the parse I ran on the contract earlier."),

    ("mcp_getjob_jobid_prefix",
     "Get the results of jobid://550e8400-e29b-41d4-a716-446655440000 — I need to see what was extracted."),

    ("mcp_getjob_expired_recovery",
     "I tried using jobid://expired-job-123 in reducto_extract and got an error that the job has expired. The original document is at https://example.com/doc.pdf. How do I recover?"),

    ("mcp_getjob_chain_verify",
     "I parsed https://example.com/report.pdf with persist_results=true and got job_id='parse-job-001'. "
     "First verify the parse completed successfully using get_job, then extract the executive summary using that job ID."),

    ("mcp_getjob_poll_status",
     "I ran reducto_parse on a 500-page document 2 minutes ago, got job_id='long-job-999'. Poll for the result now."),

    ("mcp_getjob_before_split",
     "I have job_id='parse-result-555' from an earlier parse. Verify it's still valid before using it as input to reducto_split."),
]

EXTRACT_MCP_SCENARIOS = [
    ("mcp_extract_schema_param",
     "Extract the vendor name, invoice number, line items, and total from https://example.com/invoice.pdf. "
     "Use the 'schema' parameter (it's a JSON string) — not schema_json."),

    ("mcp_extract_jobid_reuse",
     "I already parsed https://example.com/financials.pdf and got job_id='parse-job-888'. "
     "Now extract: account_number, portfolio_value, and top_5_holdings using jobid://parse-job-888 as input."),

    ("mcp_extract_array_line_items",
     "Extract all 200+ transaction line items from https://example.com/bank_statement.pdf. "
     "Use array_extract since it's repeating rows. Schema should have a 'transactions' array property."),

    ("mcp_extract_citations_bbox",
     "Extract the key fields from https://example.com/contract.pdf and return bounding box coordinates "
     "for each extracted value so my UI can highlight them. Use citations=true."),

    ("mcp_extract_chart_data",
     "The revenue figures in https://example.com/slides.pdf are only shown in charts, not tables. "
     "Extract revenue_q1, revenue_q2, revenue_q3, revenue_q4 — set include_images=true so the model can see the charts."),

    ("mcp_extract_deep_extract",
     "Extract contract_parties, effective_date, governing_law, and termination_conditions from "
     "https://example.com/complex_contract.pdf. Initial extraction was missing fields — use deep_extract=true for accuracy."),

    ("mcp_extract_system_prompt",
     "Extract all dollar amounts from https://example.com/statement.pdf. "
     "Use system_prompt to tell the model: all values are in USD, use null if not found, return numbers not strings."),

    ("mcp_extract_persist_then_extract",
     "Parse https://example.com/report.pdf with persist_results=true, then extract: "
     "title, author, date, and key_findings using the returned job ID."),
]

SPLIT_MCP_SCENARIOS = [
    ("mcp_split_preserve_tables",
     "Split https://example.com/annual.pdf into: Cover, Executive Summary, Financials, Operations, Appendix. "
     "Make sure no table gets cut in half — use table_cutoff='preserve'."),

    ("mcp_split_with_rules",
     "Split https://example.com/legal.pdf into Introduction, Terms and Conditions, Signatures. "
     "Add a split_rule: if a section is missing, mark it empty rather than merging with adjacent."),

    ("mcp_split_then_parse_sections",
     "This 50-page report https://example.com/report.pdf has 4 sections: Summary, Methodology, Results, References. "
     "Split it first, then use the page ranges to parse just the Results section."),

    ("mcp_split_jobid_reuse",
     "I parsed https://example.com/doc.pdf earlier with persist_results=true, job_id='parse-222'. "
     "Now split jobid://parse-222 into Introduction, Body, and Conclusion sections."),

    ("mcp_split_complex",
     "Split https://example.com/fund_report.pdf into: Fund Overview, Portfolio Holdings, "
     "Performance Attribution, Risk Metrics, Disclosures. table_cutoff should preserve whole tables."),
]

CLASSIFY_MCP_SCENARIOS = [
    ("mcp_classify_route",
     "Classify https://example.com/doc.pdf as one of: invoice, purchase_order, statement, contract, other. "
     "Only look at pages 1-3 to save cost."),

    ("mcp_classify_with_metadata",
     "Classify https://example.com/upload.pdf — this came from our AP automation pipeline so it's "
     "probably an invoice or PO. Pass that context as document_metadata."),

    ("mcp_classify_then_extract",
     "I'm building a document pipeline. First classify https://example.com/doc.pdf as invoice, contract, or report. "
     "Then if it's an invoice, extract vendor, total, line_items. If contract, extract parties and effective_date."),

    ("mcp_classify_pages_1_to_5",
     "Classify this 300-page document https://example.com/huge.pdf — "
     "the doc type is always clear from the cover page. Limit to pages 1-5."),
]

EDIT_MCP_SCENARIOS = [
    ("mcp_edit_highlight_yellow",
     "Fill in the NDA at https://example.com/nda.pdf: counterparty='Acme Corp', date='2026-04-01', "
     "governing_law='California'. Highlight the filled fields in yellow."),

    ("mcp_edit_overflow",
     "Fill out the application form at https://example.com/application.pdf. The additional_notes field "
     "might need more space than the form allows. Enable overflow pages."),

    ("mcp_edit_basic",
     "Edit https://example.com/form.pdf: fill in name='John Doe', company='TechCorp', date='2026-01-15'."),

    ("mcp_edit_highlight_custom",
     "Complete the purchase order at https://example.com/po.pdf: vendor='Supplier Inc', amount='$5,000', "
     "delivery_date='2026-05-01'. Use a light blue highlight (#ADD8E6) on the filled fields."),

    ("mcp_edit_no_overflow",
     "Fill the tax form at https://example.com/tax.pdf with the provided values. "
     "The form is fixed-size — don't allow overflow pages."),
]

UPLOAD_MCP_SCENARIOS = [
    ("mcp_upload_presigned",
     "This presigned URL expires in 60 seconds: https://s3.amazonaws.com/bucket/contract.pdf?X-Amz-Expires=60. "
     "Upload it first to get a stable reducto:// ID, then parse it."),

    ("mcp_upload_multi_op",
     "Upload https://example.com/report.pdf to get a stable ID. "
     "Then use that ID to: (1) classify the document, (2) parse it for full text."),

    ("mcp_upload_then_extract_split",
     "Upload https://storage.example.com/temp/annual.pdf?token=xyz&expires=300 — the URL is short-lived. "
     "Then split it into sections and extract financial data from the Financials section."),
]

# ── 2. Multi-hop MCP chains (all 7 tools involved) ────────────────────────

MCP_CHAIN_SCENARIOS = [
    ("mcp_chain_upload_persist_getjob_extract",
     "I have a presigned URL that expires soon: https://s3.example.com/docs/report.pdf?X-Amz-Expires=300. "
     "Upload it to get a stable ID. Then parse with persist_results=true. "
     "Verify the parse job with get_job. Then extract: title, total_assets, net_income using the job ID."),

    ("mcp_chain_classify_route_extract",
     "https://example.com/unknown.pdf could be an invoice, contract, or financial report. "
     "Classify it using pages 1-2 only. Then based on the result: "
     "if invoice → extract vendor, total, invoice_number; "
     "if contract → extract parties, effective_date, termination; "
     "if report → extract title, period, key_metrics."),

    ("mcp_chain_split_parse_each",
     "This 80-page document https://example.com/fund_report.pdf has 5 sections. "
     "First split it into sections with table_cutoff='preserve'. "
     "Then parse just the Financials section using the returned page range."),

    ("mcp_chain_parse_getjob_reuse",
     "Parse https://cdn.reducto.ai/samples/fidelity-example.pdf with persist_results=true. "
     "After parsing, use get_job to verify the result is ready. "
     "Then extract the account_number, portfolio_value, and holdings_list using jobid://[returned_id]."),

    ("mcp_chain_upload_classify_extract",
     "Upload https://example.com/intake.pdf?token=abc&exp=1700000000 — short-lived URL. "
     "Then classify it as invoice, purchase_order, or other. "
     "Then extract the appropriate fields based on the classification result."),

    ("mcp_chain_edit_classify",
     "Fill in the application form at https://example.com/app.pdf: "
     "name='Reza Sayar', role='Engineer', start_date='2026-06-01'. "
     "After editing, classify the filled document to verify it's tagged as 'job_application'."),

    ("mcp_chain_split_extract_all_sections",
     "https://example.com/annual_report.pdf has sections: Summary, Financials, Risk Factors, Appendix. "
     "Split it with table_cutoff='preserve'. "
     "Then extract key_metric, year from the Financials section using the page range from the split."),

    ("mcp_chain_getjob_then_reuse",
     "I ran reducto_parse 30 minutes ago on a large doc and got job_id='stale-parse-abc'. "
     "First check if the job is still valid using get_job. "
     "If valid, run reducto_extract with jobid://stale-parse-abc to get vendor and total. "
     "If expired, re-parse https://example.com/invoice.pdf and then extract."),

    ("mcp_chain_upload_multiop_fan_out",
     "I need to run parse, extract (financial summary), AND classify on https://example.com/report.pdf?sig=temp123. "
     "Upload first since the URL is signed. Then fan out: parse for full text AND extract for summary, reusing the reducto:// ID."),

    ("mcp_chain_parse_split_getjob",
     "Parse https://example.com/legal.pdf with persist_results=true and add_page_markers=true. "
     "Check the job with get_job. Then split the document into Introduction, Terms, Signatures using the job ID."),
]

# ── 3. MCP error / edge cases ─────────────────────────────────────────────

MCP_EDGE_SCENARIOS = [
    ("mcp_edge_schema_vs_schema_json",
     "Extract vendor_name and invoice_total from https://example.com/invoice.pdf. "
     "Use the 'schema' parameter — that's the JSON Schema string in the MCP tool, not 'schema_json'."),

    ("mcp_edge_getjob_expired",
     "I called reducto_get_job on 'job-expired-xxx' and got a 404. "
     "The original document is at https://example.com/doc.pdf. "
     "How do I recover and get the extracted data?"),

    ("mcp_edge_filter_blocks_array",
     "Parse https://example.com/report.pdf and exclude Header, Footer, and Page Number blocks. "
     "Remember: filter_blocks takes an array, not a comma-separated string."),

    ("mcp_edge_agentic_scopes_array",
     "Parse https://example.com/complex.pdf with agentic table correction. "
     "agentic_scopes is an array — pass ['table'] not a string."),

    ("mcp_edge_overflow_pages",
     "Fill in the contract at https://example.com/contract.pdf with a long legal clause in the notes field. "
     "The clause is about 3 paragraphs — it will definitely overflow. Enable overflow_pages."),

    ("mcp_edge_table_cutoff_preserve",
     "Split https://example.com/doc.pdf into three sections. "
     "Use table_cutoff='preserve' not 'allow' — the MCP tool uses 'truncate'/'preserve' not the raw API enum."),

    ("mcp_edge_jobid_prefix_normalize",
     "Use the job ID from my last parse: it's either 'abc-123' or 'jobid://abc-123' — reducto_get_job handles both formats."),

    ("mcp_edge_classify_page_limit",
     "Classify this 1000-page document https://example.com/massive.pdf. "
     "Only process pages 1-3 — classification works on a small window at the start."),
]

# ── 4. Covering all 7 tools explicitly ────────────────────────────────────

ALL_7_TOOLS_SCENARIOS = [
    ("mcp_all7_full_workflow",
     "Full MCP workflow on https://s3.example.com/intake.pdf?X-Amz-Expires=120: "
     "1. Upload to get a stable ID (presigned URL). "
     "2. Parse with persist_results=true. "
     "3. Verify parse with get_job. "
     "4. Classify the doc type (invoice/contract/report). "
     "5. Split into sections. "
     "6. Extract key fields from the main section. "
     "7. Fill in any missing fields via edit."),

    ("mcp_all7_get_job_central",
     "I'm building a document processing pipeline. For each step, I want to: "
     "parse (persist), get_job to confirm, classify, split, extract from sections, "
     "edit to fill a summary field, and upload the result. "
     "Document: https://example.com/pipeline_test.pdf"),

    ("mcp_all7_showcase",
     "Demonstrate all 7 Reducto MCP tools on the Fidelity sample: "
     "https://cdn.reducto.ai/samples/fidelity-example.pdf — "
     "upload, parse, get_job, classify, split, extract, edit."),
]

# ── Combine all MCP scenarios ─────────────────────────────────────────────

ALL_MCP_SCENARIOS = (
    PARSE_MCP_SCENARIOS
    + GET_JOB_SCENARIOS
    + EXTRACT_MCP_SCENARIOS
    + SPLIT_MCP_SCENARIOS
    + CLASSIFY_MCP_SCENARIOS
    + EDIT_MCP_SCENARIOS
    + UPLOAD_MCP_SCENARIOS
    + MCP_CHAIN_SCENARIOS
    + MCP_EDGE_SCENARIOS
    + ALL_7_TOOLS_SCENARIOS
)

# ---------------------------------------------------------------------------
# Training example
# ---------------------------------------------------------------------------

@dataclass
class TrainingExample:
    probe_id: str
    source: str
    teacher_model: str
    user_prompt: str
    tool_calls: list
    score: int = 3


def example_to_jsonl(ex: TrainingExample) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": MCP_SYSTEM_PROMPT},
            {"role": "user",      "content": ex.user_prompt},
            {"role": "assistant", "content": None, "tool_calls": [
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]) if isinstance(tc["arguments"], dict) else tc["arguments"],
                    },
                }
                for tc in ex.tool_calls
            ]},
        ],
        "tools": MCP_TOOL_SCHEMAS,
        "metadata": {
            "probe_id": ex.probe_id,
            "source": ex.source,
            "teacher": ex.teacher_model,
            "round": "mcp_round2",
        },
    }


# ---------------------------------------------------------------------------
# Phase 1: Generate prompt variations
# ---------------------------------------------------------------------------

REPHRASE_PROMPT = """\
You are generating synthetic training data for an AI agent that uses MCP tools. Rephrase the following user request in {n} distinct ways.
Rules:
- Same underlying task, different wording/framing/detail level
- Vary style: some terse, some verbose, some from a dev, some from a non-technical analyst
- Some can mention the MCP context: Claude Code, Cursor, Cline, or just "MCP tools"
- Be realistic — something a real user interacting with an MCP server might actually write

Original:
{original}

Return ONLY a JSON array of {n} strings. No markdown, no explanation."""

_REPHRASE_BATCH = 20


def generate_variations(prompt: str, n: int, dry_run: bool = False) -> list[str]:
    if dry_run:
        return [f"[DRY RUN #{i+1}] {prompt[:80]}..." for i in range(n)]

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=REPHRASE_MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        temperature=0.85,
        max_tokens=8192,
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
            print(f"    [rephrase] error: {e}")
        remaining -= batch

    return results


# ---------------------------------------------------------------------------
# Phase 2: Run variation through teacher model
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
        import concurrent.futures as _cf
        from bench_param_probe import _build_llm, _build_probe_tools
        llm   = _build_llm(model)
        tools = _build_probe_tools()
        llm_with_tools = llm.bind_tools(tools)
        def _call():
            return llm_with_tools.invoke([
                {"role": "system", "content": MCP_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ])
        with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
            _fut = _ex.submit(_call)
            try:
                response = _fut.result(timeout=90)
            except _cf.TimeoutError:
                print(f"    [teacher] timeout on {probe_id} — skipping")
                return None
        raw_calls = getattr(response, "tool_calls", []) or []
        tool_calls = [
            {"name": tc.get("name", ""), "arguments": tc.get("args", {})}
            for tc in raw_calls if tc.get("name")
        ]
        if not tool_calls:
            return None
        return TrainingExample(
            probe_id=probe_id, source="generated", teacher_model=model.display,
            user_prompt=prompt, tool_calls=tool_calls,
        )
    except Exception as e:
        print(f"    [teacher] error ({model.display}): {e}")
        return None


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def load_teachers(teacher_names: list[str]) -> list[ModelConfig]:
    all_models = {m.display: m for m in (OPENROUTER_MODELS + PREMIUM_MODELS)}
    teachers = []
    for name in teacher_names:
        if name in all_models:
            teachers.append(all_models[name])
        else:
            print(f"  [warn] teacher '{name}' not found in models registry, skipping")
    return teachers


def load_checkpoint(checkpoint: Path) -> set[str]:
    """Return set of already-generated prompt hashes for dedup."""
    seen = set()
    if not checkpoint.exists():
        return seen
    for line in checkpoint.read_text().splitlines():
        try:
            d = json.loads(line)
            msgs = d.get("messages", [])
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            seen.add(user_msg[:120])
        except Exception:
            pass
    return seen


def save_example(ex: TrainingExample, checkpoint: Path) -> None:
    row = example_to_jsonl(ex)
    with checkpoint.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",  type=int, default=36_669)
    parser.add_argument("--resume",  action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seen_prompts = load_checkpoint(CHECKPOINT) if args.resume else set()
    current = sum(1 for _ in CHECKPOINT.open()) if (args.resume and CHECKPOINT.exists()) else 0
    print(f"=== MCP Round-2 gen started ===  target={args.target}  resume_from={current}")

    teachers = load_teachers(GENERATION_TEACHERS)
    if not teachers:
        print("ERROR: No teachers loaded. Check models.py.")
        sys.exit(1)

    # How many variations per base scenario to hit target
    n_scenarios = len(ALL_MCP_SCENARIOS)
    # Each scenario → variations → each through a random teacher
    # Estimate: ~target / n_scenarios variations per scenario
    variations_per_scenario = max(4, args.target // n_scenarios)

    print(f"  {n_scenarios} base MCP scenarios × ~{variations_per_scenario} variations each")
    print(f"  {len(teachers)} teachers available")

    generated = current

    def process_scenario(scenario_idx: int, probe_id: str, base_prompt: str) -> int:
        nonlocal generated
        added = 0

        if generated >= args.target:
            return added

        # Generate variations
        n_to_gen = min(variations_per_scenario, args.target - generated)
        variations = generate_variations(base_prompt, n_to_gen, dry_run=args.dry_run)
        if not variations:
            variations = [base_prompt]

        for variation in variations:
            if generated >= args.target:
                break
            if variation[:120] in seen_prompts:
                continue

            teacher = random.choice(teachers)
            ex = run_through_teacher(teacher, variation, probe_id, dry_run=args.dry_run)
            if ex is None:
                continue

            save_example(ex, CHECKPOINT)
            seen_prompts.add(variation[:120])
            generated += 1
            added += 1

            if generated % 100 == 0:
                print(f"  [{generated}/{args.target}] scenario={probe_id} teacher={teacher.display}")

        return added

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(process_scenario, i, pid, prompt)
            for i, (pid, prompt) in enumerate(ALL_MCP_SCENARIOS)
        ]
        # If we still need more, cycle through scenarios again
        cycle = 0
        while generated < args.target and cycle < 10:
            cycle += 1
            for i, (pid, prompt) in enumerate(ALL_MCP_SCENARIOS):
                if generated >= args.target:
                    break
                pool.submit(process_scenario, i, f"{pid}_cycle{cycle}", prompt)

        for f in futures:
            try:
                f.result(timeout=300)
            except Exception as e:
                print(f"  [worker] error: {e}")

    final = sum(1 for _ in open(CHECKPOINT)) if CHECKPOINT.exists() else 0
    print(f"\n=== Done. {final} examples in {CHECKPOINT} ===")


if __name__ == "__main__":
    main()
