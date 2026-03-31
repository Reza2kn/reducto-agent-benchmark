#!/usr/bin/env python3
"""
gen_0_8b_targeted.py — Targeted SFT dataset for Qwen-0.8B-AgentJSON fine-tune.

Methodology mirrors R3: every scenario bank maps to a specific, measured failure
in the 0.8B probe results.  Teacher is Haiku (21/21 MCP).  Start fine-tune from
AgentJSON checkpoint (NOT base Qwen), LoRA r=64.

Failure clusters addressed:
  A1  citations=True on extract (param_present=False, hard probes: 0/3 × 2)
  A2  system_prompt on parse    (param_present=False, hard probe: 1/3)
  A3  optimize_for_latency flag (confused with deep_extract, 0/3)
  A4  merge_tables=True         (param_present=False, 1/3)
  A5  return_images typed enum  (param_present=False, 1/3)
  A6  document_metadata flags   (param_present=False, std: 0/3)
  A7  input as URL array        (param_present=False, std: 1/3)
  B1  filter_blocks native array vs JSON string  (correct=False, 2/3)
  B2  agentic_scopes correct values + array form (correct=False, 2/3)
  B3  output_format enum: json_bbox              (correct=False, std: 2/3)
  B4  ocr_mode correct values                    (correct=False, api=False)
  B5  table_cutoff="preserve" (adversarial trap)  (correct=False, 2/3)
  C1  classify → route → extract two-step chain  (0/3 on BOTH base and LoRA)
  C2  dual-doc fan-out: 2× parse, 2× job IDs    (1/3 — loses track of IDs)
  D   chain termination (upload-persist and split loops 600-900s)

Target  : 2,669 examples
Output  : benchmark/data/synthetic_training_0_8b/.checkpoint_0_8b.jsonl
Teacher : Claude Haiku 4.5 + thinking

Usage:
    python gen_0_8b_targeted.py               # full run
    python gen_0_8b_targeted.py --target 50   # quick test
    python gen_0_8b_targeted.py --dry-run
    python gen_0_8b_targeted.py --resume
"""

import argparse
import json
import os
import random
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))
from models import PREMIUM_MODELS, ModelConfig

OUTPUT_DIR = Path("benchmark/data/synthetic_training_0_8b")
CHECKPOINT = OUTPUT_DIR / ".checkpoint_0_8b.jsonl"

TEACHER_NAME   = "Claude Haiku 4.5 + thinking"
REPHRASE_MODEL = "claude-haiku-4-5-20251001"

# Reuse shared MCP infrastructure from R3
try:
    from gen_mcp_r3_gaps import (
        MCP_TOOL_SCHEMAS,
        MCP_SYSTEM_PROMPT,
        _synthetic_result,
        _get_final_text,
        run_through_teacher,
        generate_variations,
        REPHRASE_PROMPT,
    )
except ImportError as e:
    print(f"ERROR: could not import from gen_mcp_r3_gaps.py: {e}")
    print("Run from benchmark/scripts/ directory.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Cluster A — params the model never uses (param_present=False)
# ---------------------------------------------------------------------------

# A1: citations=True on extract — tested via citations_bbox, citations_no_chunking
# The model uses extract but never sets citations=True.
CLUSTER_A1_CITATIONS = [
    ("a1_citations_explicit",
     "Extract vendor_name, invoice_number, total_amount from "
     "https://example.com/invoice.pdf. I need bounding-box citations for every "
     "extracted field so I can highlight them in the UI."),
    ("a1_citations_compliance",
     "Extract patient_name, diagnosis_code, and treatment_plan from "
     "https://example.com/medical.pdf for a compliance audit. Every value must "
     "include its page location for traceability — use citations."),
    ("a1_citations_legal",
     "Extract contract_party_a, contract_party_b, effective_date, and "
     "governing_law from https://example.com/contract.pdf. We need bbox citations "
     "so our legal team can verify each field against the original document."),
    ("a1_citations_fidelity",
     "Pull account_number, portfolio_value, and top_holding from "
     "https://cdn.reducto.ai/samples/fidelity-example.pdf. Include spatial "
     "citations for each extracted value."),
    ("a1_citations_no_explicit_flag",
     "Extract company_name, fiscal_quarter, revenue, net_income from "
     "https://example.com/earnings.pdf. The output must be traceable to specific "
     "locations in the PDF — I'll need coordinates for each field."),
    ("a1_citations_deep",
     "Deep-extract holdings data from https://example.com/fund.pdf: "
     "fund_name, nav, total_assets, top_5_holdings. Include citations and use "
     "deep extraction mode."),
    ("a1_citations_after_parse",
     "Parse https://example.com/annual.pdf first, then extract: ceo_name, "
     "revenue, headcount, key_risks. Include bbox citations in the extraction."),
    ("a1_citations_array",
     "Extract all line items from https://example.com/purchase_order.pdf as an "
     "array: each row should have item_name, qty, unit_price, total. Use "
     "array_extract and include citations for each row."),
    ("a1_citations_no_chunking",
     "Extract summary_title, author, and key_findings from "
     "https://example.com/report.pdf without chunking the document. "
     "I need exact bounding boxes for each extracted value."),
    ("a1_citations_minimal",
     "From https://example.com/receipt.pdf, extract merchant and total. "
     "Add citations so I know where on the page each value came from."),
]

# A2: system_prompt on parse — model uses system_prompt on extract but not parse
CLUSTER_A2_SYSTEM_PROMPT_PARSE = [
    ("a2_sys_prompt_parse_basic",
     "Parse https://example.com/financial.pdf. The document uses non-standard "
     "table formatting — instruct the parser to treat any indented numeric row "
     "as a table cell, not free text."),
    ("a2_sys_prompt_parse_language",
     "Parse https://example.com/french_contract.pdf. It's in French — add a "
     "system prompt telling the parser to preserve French legal terminology "
     "without translation."),
    ("a2_sys_prompt_parse_units",
     "Parse https://example.com/lab_report.pdf. The numbers use European decimal "
     "notation (comma as decimal separator). Pass a system prompt to handle this."),
    ("a2_sys_prompt_parse_redaction",
     "Parse https://example.com/hr_doc.pdf. Use a system prompt to skip any "
     "sections labelled CONFIDENTIAL and treat them as empty blocks."),
    ("a2_sys_prompt_parse_structure",
     "Parse https://example.com/slides.pdf where every slide has a header, "
     "bullet list, and chart. Pass a system prompt so the parser understands "
     "this repeating structure."),
    ("a2_sys_prompt_parse_currency",
     "Parse https://example.com/invoice.pdf. All monetary amounts are in JPY "
     "with ¥ symbol — include a system prompt noting the currency format so "
     "the parser preserves it correctly."),
    ("a2_sys_prompt_parse_custom_terms",
     "Parse https://example.com/technical_manual.pdf. It contains proprietary "
     "part numbers like XR-4829-B. Use a system prompt to flag these as "
     "structured entities, not random strings."),
    ("a2_sys_prompt_parse_then_extract",
     "Parse https://example.com/earnings.pdf with a system prompt: treat every "
     "bold row in a table as a subtotal. Then extract revenue, expenses, "
     "and net_income."),
]

# A3: optimize_for_latency (model confuses this with deep_extract=False)
CLUSTER_A3_OPTIMIZE_LATENCY = [
    ("a3_optimize_latency_basic",
     "Extract company_name and total_revenue from "
     "https://example.com/report.pdf. Speed is critical — optimise for "
     "low latency, not deep accuracy."),
    ("a3_optimize_latency_realtime",
     "I have a real-time pipeline. Extract invoice_number and total from "
     "https://example.com/invoice.pdf as fast as possible — use the latency "
     "optimisation flag."),
    ("a3_optimize_latency_vs_deep",
     "Extract account_number and balance from "
     "https://cdn.reducto.ai/samples/fidelity-example.pdf. "
     "I do NOT need deep extraction — prioritise response speed."),
    ("a3_optimize_latency_high_volume",
     "High-volume pipeline: extract vendor and amount from "
     "https://example.com/invoice.pdf. Throughput matters more than accuracy "
     "on edge cases — set the latency optimisation option."),
    ("a3_optimize_latency_simple_doc",
     "This is a simple one-page form at https://example.com/form.pdf. "
     "Just extract name, date, and signature_present. No need for deep mode — "
     "optimise for speed."),
    ("a3_optimize_latency_not_deep",
     "Extract title and summary from https://example.com/brief.pdf. "
     "optimize_for_latency=True please — I don't need the deep extraction pass."),
    ("a3_optimize_latency_pipeline",
     "Pre-screen https://example.com/intake.pdf for classification. "
     "Extract doc_type and date_received quickly — this is a triage step, "
     "latency-optimised mode."),
]

# A4: merge_tables=True for cross-page tables
CLUSTER_A4_MERGE_TABLES = [
    ("a4_merge_tables_basic",
     "Parse https://example.com/annual.pdf. The financial tables span multiple "
     "pages — merge them into single continuous tables."),
    ("a4_merge_tables_crosspage",
     "The holdings table in https://cdn.reducto.ai/samples/fidelity-example.pdf "
     "continues across pages. Parse it with table merging so I get one table, "
     "not fragments."),
    ("a4_merge_tables_natural",
     "Parse https://example.com/10k.pdf. There's a large data table that "
     "flows from page 12 to page 14. I need it as a single merged table."),
    ("a4_merge_tables_with_persist",
     "Parse https://example.com/report.pdf with persist_results. The income "
     "statement spans pages 8-10 — merge tables so cross-page rows are joined."),
    ("a4_merge_tables_agentic",
     "Parse https://example.com/fund_factsheet.pdf. Enable agentic table "
     "correction and merge any tables that continue across page breaks."),
    ("a4_merge_tables_for_extract",
     "Parse https://example.com/earnings.pdf with table merging. Then extract "
     "all revenue line items as an array."),
    ("a4_merge_tables_simple",
     "The table at https://example.com/statement.pdf runs across two pages. "
     "Parse with merge_tables so the output has one clean table."),
]

# A5: return_images typed array (model ignores return_images_types)
CLUSTER_A5_RETURN_IMAGES = [
    ("a5_return_images_charts",
     "Parse https://example.com/annual.pdf and return the chart images — "
     "I need the PNG renders of the bar charts for my dashboard."),
    ("a5_return_images_figures",
     "Parse https://example.com/research.pdf. Extract figure images so I can "
     "display them inline with the extracted text."),
    ("a5_return_images_all",
     "Parse https://cdn.reducto.ai/samples/fidelity-example.pdf and include "
     "all embedded images in the output — charts, logos, everything."),
    ("a5_return_images_diagrams",
     "Parse https://example.com/technical_manual.pdf. The schematics and "
     "diagrams are critical — return image renders alongside the text."),
    ("a5_return_images_with_agentic",
     "Parse https://example.com/slides.pdf with agentic_scopes=['figure'] and "
     "return the figure images so I can post-process them."),
    ("a5_return_images_minimal",
     "Parse https://example.com/report.pdf and return any chart or graph "
     "images embedded in it."),
    ("a5_return_images_then_extract",
     "Parse https://example.com/fund.pdf with return_figure_images. Then "
     "extract nav, total_assets, and top_holdings."),
]

# A6: document_metadata (model doesn't pass metadata flags)
CLUSTER_A6_DOCUMENT_METADATA = [
    ("a6_metadata_basic",
     "Parse https://example.com/contract.pdf and include the document "
     "metadata: title, author, creation_date, page_count."),
    ("a6_metadata_audit",
     "Extract contract_parties and effective_date from "
     "https://example.com/agreement.pdf. Also retrieve document metadata "
     "for our audit log."),
    ("a6_metadata_pdf_properties",
     "Parse https://example.com/report.pdf. I need both the text content "
     "and the PDF document properties (title, author, created, modified)."),
    ("a6_metadata_with_extract",
     "From https://example.com/invoice.pdf, extract vendor, amount, and date. "
     "Also return document metadata fields."),
    ("a6_metadata_for_indexing",
     "Parse https://cdn.reducto.ai/samples/fidelity-example.pdf for indexing. "
     "Include document metadata so I can populate my search index title field."),
    ("a6_metadata_simple",
     "Parse https://example.com/policy.pdf and return the document title and "
     "page count alongside the parsed content."),
    ("a6_metadata_legal",
     "Parse https://example.com/court_filing.pdf. Include document properties "
     "(author, creation date) — these are required for court record keeping."),
    ("a6_metadata_url_array",
     "Parse https://example.com/memo.pdf. I need the document metadata "
     "fields — title, subject, creator."),
]

# A7: input as URL array (model only passes single string)
CLUSTER_A7_URL_ARRAY = [
    ("a7_url_array_two_docs",
     "Parse both https://example.com/q1_report.pdf and "
     "https://example.com/q2_report.pdf in a single call — pass them as an "
     "array of URLs."),
    ("a7_url_array_extract",
     "Extract vendor and total from all three invoices at once: "
     "https://example.com/inv_001.pdf, https://example.com/inv_002.pdf, "
     "https://example.com/inv_003.pdf. Pass the URLs as an array."),
    ("a7_url_array_three_docs",
     "I have a portfolio of documents to parse: "
     "https://example.com/doc_a.pdf, https://example.com/doc_b.pdf, "
     "https://example.com/doc_c.pdf. Process them together in one API call."),
    ("a7_url_array_classify_batch",
     "Classify these four documents in one call: "
     "https://example.com/doc1.pdf, https://example.com/doc2.pdf, "
     "https://example.com/doc3.pdf, https://example.com/doc4.pdf."),
    ("a7_url_array_two_statements",
     "Parse the January and February statements together: "
     "https://example.com/jan.pdf and https://example.com/feb.pdf. "
     "Pass them as an array input."),
    ("a7_url_array_single_wrong",
     "I need to pass two PDFs to the parser at the same time: "
     "https://cdn.reducto.ai/samples/fidelity-example.pdf and "
     "https://example.com/supplement.pdf."),
    ("a7_url_array_natural",
     "Batch parse: https://example.com/plan_a.pdf, "
     "https://example.com/plan_b.pdf. Both use the same schema — "
     "process in one request."),
]

# ---------------------------------------------------------------------------
# Cluster B — params used but wrong value/format (param_correct=False)
# ---------------------------------------------------------------------------

# B1: filter_blocks as native array (model passes JSON string)
CLUSTER_B1_FILTER_BLOCKS_ARRAY = [
    ("b1_filter_string_trap",
     "Parse https://example.com/report.pdf. Pass filter_blocks as a "
     "comma-separated string: 'Header,Footer,PageNumber'."),
    ("b1_filter_json_string_trap",
     "Parse https://example.com/annual.pdf. Set filter_blocks to the JSON "
     "string '[\"Header\",\"Footer\"]' to remove noise."),
    ("b1_filter_natural_two",
     "Parse https://example.com/statement.pdf and remove headers and footers."),
    ("b1_filter_quoted_array_trap",
     "Parse https://example.com/contract.pdf. I'll pass filter_blocks as a "
     "serialised array: '[\"Signature\",\"Comment\",\"Footer\"]'."),
    ("b1_filter_natural_four",
     "Parse https://cdn.reducto.ai/samples/fidelity-example.pdf for RAG. "
     "Strip decorative elements: headers, footers, page numbers, watermarks."),
    ("b1_filter_comma_trap",
     "Parse https://example.com/10k.pdf. filter_blocks='Header,Footer,"
     "Section Header,Page Number' — use a comma string."),
    ("b1_filter_six_types",
     "Parse https://example.com/legal.pdf. Exclude: Header, Footer, "
     "Page Number, Title, Section Header, Signature. "
     "Pass as a comma-separated string."),
    ("b1_filter_natural_clean",
     "Parse https://example.com/kb.pdf for embedding. Remove noise blocks: "
     "headers, footers, and page numbers. I want clean text only."),
    ("b1_filter_combined_with_agentic",
     "Parse https://example.com/tables.pdf. Remove headers and footers "
     "and apply agentic correction for tables. Pass filter_blocks as "
     "'Header,Footer' string."),
    ("b1_filter_natural_minimal",
     "Parse https://example.com/brief.pdf. Strip page headers and footers "
     "before I send the text to an LLM."),
]

# B2: agentic_scopes correct values + native array
CLUSTER_B2_AGENTIC_SCOPES = [
    ("b2_agentic_string_trap",
     "Parse https://example.com/report.pdf. Set agentic_scopes to the string "
     "'table' for column alignment correction."),
    ("b2_agentic_comma_trap",
     "Parse https://example.com/slides.pdf. Use agentic_scopes='text,figure' "
     "(comma-separated string) for OCR cleanup and figure extraction."),
    ("b2_agentic_natural_tables",
     "Parse https://example.com/financials.pdf. The tables have misaligned "
     "columns — apply AI correction for tables."),
    ("b2_agentic_natural_all",
     "Parse https://example.com/complex.pdf. Enable agentic correction for "
     "both tables and figures — the layout is dense."),
    ("b2_agentic_figure_trap",
     "Parse https://example.com/charts.pdf. Pass agentic_scopes='figure' "
     "as a string for chart extraction."),
    ("b2_agentic_text_table_trap",
     "Parse https://example.com/annual.pdf. agentic_scopes='text,table' "
     "for OCR cleanup and column correction."),
    ("b2_agentic_natural_persist",
     "Parse https://cdn.reducto.ai/samples/fidelity-example.pdf. Enable "
     "agentic correction for tables, persist results. I'll extract later."),
    ("b2_agentic_figure_table_trap",
     "Parse https://example.com/research.pdf. Use agentic_scopes='figure,table' "
     "(as a string) — I need chart AI and column correction."),
    ("b2_agentic_clean_no_trap",
     "Parse https://example.com/fund.pdf with AI table correction enabled. "
     "Complex multi-column layout, needs column realignment."),
    ("b2_agentic_single_scope",
     "Parse https://example.com/data.pdf. I only need agentic correction "
     "for tables — pass just that one scope."),
]

# B3: output_format=json_bbox (model uses wrong enum value)
CLUSTER_B3_JSON_BBOX = [
    ("b3_bbox_format_explicit",
     "Parse https://example.com/form.pdf and return the output as "
     "json_bbox format — I need bounding boxes for every text block."),
    ("b3_bbox_format_ui",
     "Parse https://example.com/invoice.pdf. I'm building a PDF highlighter "
     "UI — I need the output_format that includes spatial coordinates for "
     "each text block."),
    ("b3_bbox_format_natural",
     "Parse https://example.com/contract.pdf and give me the bounding-box "
     "JSON output so I can overlay annotations."),
    ("b3_bbox_format_wrong_name_trap",
     "Parse https://example.com/report.pdf with output_format='bbox_json' — "
     "I need the coordinates for each paragraph block."),
    ("b3_bbox_format_spatial",
     "Parse https://example.com/statement.pdf and return spatial layout data "
     "with coordinates for each text element."),
    ("b3_bbox_format_highlight",
     "I want to highlight search results in a PDF viewer. Parse "
     "https://example.com/doc.pdf and return bounding-box output format."),
    ("b3_bbox_format_ocr_plus_coords",
     "Parse https://example.com/scan.pdf. It's a scanned document — I need "
     "OCR output with bounding boxes for each recognised text block."),
    ("b3_bbox_format_annotation",
     "Parse https://cdn.reducto.ai/samples/fidelity-example.pdf. Output as "
     "json_bbox so my annotation layer can position labels correctly."),
]

# B4: ocr_mode correct value (model uses wrong enum or wrong param)
CLUSTER_B4_OCR_MODE = [
    ("b4_ocr_mode_basic",
     "Parse https://example.com/scanned.pdf — it's a scanned document with "
     "no embedded text. Enable OCR mode."),
    ("b4_ocr_mode_force",
     "Parse https://example.com/photo_invoice.pdf. This is a photograph of "
     "a document, not a digital PDF — force OCR processing."),
    ("b4_ocr_mode_mixed",
     "Parse https://example.com/hybrid.pdf. Some pages are digital text, "
     "some are scanned images. Use OCR mode to handle the scanned pages."),
    ("b4_ocr_mode_low_quality",
     "Parse https://example.com/fax.pdf — it's a low-quality fax scan. "
     "Set OCR mode so the parser doesn't skip these pages."),
    ("b4_ocr_mode_disabled",
     "Parse https://example.com/digital.pdf. This is a native digital PDF "
     "with embedded text — explicitly disable OCR to save processing time."),
    ("b4_ocr_mode_quality",
     "Parse https://example.com/archive.pdf. High-res scans from our archive. "
     "Enable OCR — these pages have no embedded text layer."),
    ("b4_ocr_mode_with_agentic",
     "Parse https://example.com/legacy_scan.pdf with OCR mode and agentic "
     "correction for tables. The scanned tables need column realignment."),
]

# B5: table_cutoff="preserve" when prompt says "allow" — same adversarial trap as R3
# but calibrated for 0.8B's specific failure pattern (it passes "allow")
CLUSTER_B5_TABLE_CUTOFF = [
    ("b5_allow_trap_direct",
     "Split https://example.com/report.pdf into Summary, Financials, Appendix. "
     "Allow tables to cross section boundaries — set table_cutoff='allow'."),
    ("b5_allow_trap_natural",
     "Split https://cdn.reducto.ai/samples/fidelity-example.pdf into: "
     "Account Overview, Portfolio Holdings, Income Summary, Transactions. "
     "Don't cut the holdings table in half — allow it to span."),
    ("b5_allow_trap_verbose",
     "Split https://example.com/10k.pdf into Management Discussion, "
     "Financial Statements, Notes, Exhibits. "
     "Set table_cutoff to 'allow' so tables aren't truncated."),
    ("b5_allow_trap_user_knows",
     "I want table_cutoff='allow' so tables aren't cut at section breaks. "
     "Split https://example.com/fund.pdf into Overview, Holdings, Fees."),
    ("b5_no_trap_preserve",
     "Split https://example.com/prospectus.pdf into Introduction, "
     "Risk Factors, Financial Data. Keep tables whole at boundaries."),
    ("b5_allow_trap_misremember",
     "Split https://example.com/agreement.pdf into Terms, Definitions, "
     "Signatures. I believe 'allow' mode prevents table truncation."),
    ("b5_allow_trap_soft",
     "Split https://example.com/annual_report.pdf into 4 sections. "
     "If a table sits at a section break — allow it to flow through."),
    ("b5_allow_trap_technical",
     "Split https://example.com/regulatory.pdf. The table_cutoff option — "
     "use 'allow' to prevent table rows being split across sections."),
    ("b5_allow_trap_api_wording",
     "According to the API docs, table_cutoff='allow' preserves table "
     "integrity. Split https://example.com/statement.pdf into 3 sections."),
    ("b5_no_trap_clean",
     "Split https://example.com/legal.pdf into Recitals, Obligations, "
     "Termination, Signatures. Tables must not be divided across sections."),
]

# ---------------------------------------------------------------------------
# Cluster C1 — classify → route → extract (0/3 on both base and LoRA)
# Critical: model must call classify FIRST, read the result, then call extract
# ---------------------------------------------------------------------------

CLUSTER_C1_CLASSIFY_ROUTE = [
    ("c1_classify_then_extract_basic",
     "Process https://example.com/unknown.pdf: first classify it to determine "
     "the document type, then extract the appropriate fields based on the result."),
    ("c1_classify_route_invoice",
     "I'm receiving documents that could be invoices, contracts, or statements. "
     "Start with https://example.com/incoming_001.pdf: classify it first, then "
     "extract the relevant fields for that document type."),
    ("c1_classify_route_financial",
     "Unknown document at https://example.com/doc.pdf — could be an annual "
     "report, earnings call transcript, or fund factsheet. Classify it first, "
     "then extract the financial fields appropriate for that type."),
    ("c1_classify_explicit_two_step",
     "Two-step workflow: (1) classify https://example.com/intake.pdf as one of: "
     "invoice, contract, medical_record, or other. (2) based on the result, "
     "extract the fields standard for that document type."),
    ("c1_classify_route_medical",
     "Incoming document https://example.com/patient_doc.pdf — could be a "
     "prescription, lab report, or referral. Classify it first, then extract "
     "the fields relevant to its type."),
    ("c1_classify_route_legal",
     "Triage https://example.com/legal_intake.pdf: classify as one of "
     "nda, employment_agreement, service_contract, or other. Then extract "
     "the standard fields for the identified category."),
    ("c1_classify_route_fidelity",
     "Classify https://cdn.reducto.ai/samples/fidelity-example.pdf — "
     "determine whether it's a portfolio statement, tax document, or "
     "trade confirmation. Then extract appropriate fields."),
    ("c1_classify_route_schema",
     "I need adaptive extraction: classify https://example.com/doc.pdf first, "
     "then use a schema appropriate for the identified document type to extract."),
    ("c1_classify_route_batch_single",
     "Run the classify-then-extract pipeline on https://example.com/report.pdf. "
     "Step 1: classify. Step 2: extract fields matched to the document type."),
    ("c1_classify_route_unknown_format",
     "I have no idea what's in https://example.com/mystery.pdf. "
     "Classify it first, then extract what makes sense for that type."),
    ("c1_classify_route_natural",
     "Process https://example.com/filing.pdf by first figuring out what type "
     "of document it is, then pulling the relevant data out."),
    ("c1_classify_route_categories",
     "Classify https://example.com/intake.pdf as invoice, purchase_order, "
     "receipt, or statement. Then extract the key financial fields for "
     "whichever type it is."),
]

# ---------------------------------------------------------------------------
# Cluster C2 — dual-doc fan-out (1/3 — loses job ID tracking)
# ---------------------------------------------------------------------------

CLUSTER_C2_DUAL_DOC = [
    ("c2_dual_basic",
     "Process two client documents independently: "
     "https://example.com/client_a.pdf and https://example.com/client_b.pdf. "
     "Parse both with persist_results, verify both with get_job, then extract "
     "account_number and balance from each."),
    ("c2_dual_fidelity_pair",
     "Extract portfolio data from two statements: "
     "https://cdn.reducto.ai/samples/fidelity-example.pdf and "
     "https://example.com/statement_q2.pdf. "
     "Process each separately: parse(persist) → get_job → extract. "
     "Keep the job IDs distinct."),
    ("c2_dual_invoice_pair",
     "Two invoices to process: https://example.com/inv_jan.pdf and "
     "https://example.com/inv_feb.pdf. "
     "For each: parse with persist, verify with get_job, then extract "
     "vendor, invoice_number, and total. Use separate job IDs for each."),
    ("c2_dual_parallel",
     "Parallel processing of https://example.com/doc1.pdf and "
     "https://example.com/doc2.pdf. "
     "Parse both (persist_results=True), verify both are complete, "
     "then extract title and key_findings from each using its own jobid://."),
    ("c2_dual_natural",
     "I have two contracts: https://example.com/contract_acme.pdf and "
     "https://example.com/contract_beta.pdf. "
     "Run the full parse-verify-extract pipeline on each independently."),
    ("c2_dual_upload_first",
     "Two documents with expiring URLs: "
     "https://s3.example.com/doc_a.pdf?token=abc123 and "
     "https://s3.example.com/doc_b.pdf?token=xyz789. "
     "Upload both, parse both with persist, verify both, extract from each."),
    ("c2_dual_compare",
     "Compare two quarterly reports: https://example.com/q3.pdf and "
     "https://example.com/q4.pdf. Parse each separately, persist results, "
     "verify, extract revenue and net_income from each."),
    ("c2_dual_strict_isolation",
     "Process https://example.com/fund_a.pdf and https://example.com/fund_b.pdf "
     "independently — do not mix their job IDs. "
     "Full pipeline for each: parse(persist) → get_job → extract."),
]

# ---------------------------------------------------------------------------
# Cluster D — Chain termination (upload-persist and split loops)
# Reuses generate_gap9_example() from gen_mcp_r3_gaps.py
# Scenarios specifically target the two probes where 0.8B loops longest
# ---------------------------------------------------------------------------

CLUSTER_D_TERMINATION = [
    # upload → parse(persist) → get_job → extract → STOP
    ("d_upload_persist_stop",
     "Upload https://example.com/report.pdf, parse it with persist_results, "
     "verify with get_job, then extract title and summary. Report the results."),
    ("d_upload_persist_invoice",
     "Presigned URL (expires in 10 min): "
     "https://s3.example.com/invoice.pdf?sig=xyz. "
     "Upload it for a stable ID, parse with persist, verify, "
     "extract vendor and total. Done."),
    ("d_upload_persist_array_extract",
     "Upload https://example.com/holdings.pdf, parse with persist_results, "
     "verify job, then extract all holdings as an array. Return the results."),
    ("d_upload_persist_financial",
     "Process https://cdn.reducto.ai/samples/fidelity-example.pdf: "
     "upload, parse(persist), get_job to verify, extract account_number and "
     "portfolio_value. Output the extracted data."),
    ("d_upload_persist_stop_clean",
     "Full async pipeline on https://example.com/doc.pdf: "
     "upload → parse(persist) → get_job → extract key_findings. "
     "Present findings to the user."),
    # split → extract from range → STOP
    ("d_split_preserve_stop",
     "Split https://example.com/annual.pdf into Introduction, "
     "Financial Statements, Appendix with table_cutoff=preserve. "
     "Extract revenue and net_income from Financial Statements. Report back."),
    ("d_split_extract_section",
     "Split https://cdn.reducto.ai/samples/fidelity-example.pdf into "
     "Account Overview, Holdings, Income Summary. "
     "Extract all holdings from the Holdings section. Return the data."),
    ("d_split_then_done",
     "Split https://example.com/report.pdf into 3 sections, then extract "
     "key_findings from the first section only. Output the result."),
    ("d_split_page_range",
     "Split https://example.com/10k.pdf into Management Discussion, "
     "Financials, Notes. Use the Financials page range to extract "
     "revenue, expenses, net_income. Present findings."),
    # classify → route → STOP (the other 0/3 failure)
    ("d_classify_extract_stop",
     "Classify https://example.com/unknown.pdf, then extract the relevant "
     "fields for that document type. Give me the results."),
    ("d_classify_route_stop",
     "Classify https://example.com/intake.pdf as invoice, contract, or "
     "statement. Extract the appropriate fields. Report the outcome."),
    ("d_upload_split_stop",
     "Upload https://example.com/fund.pdf, parse with persist, verify, "
     "split into sections, then extract holdings from the main section. "
     "Summarise the holdings you found."),
]

# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

GAP_SCENARIOS_0_8B = {
    "a1_citations":       CLUSTER_A1_CITATIONS,
    "a2_sys_prompt":      CLUSTER_A2_SYSTEM_PROMPT_PARSE,
    "a3_optimize_lat":    CLUSTER_A3_OPTIMIZE_LATENCY,
    "a4_merge_tables":    CLUSTER_A4_MERGE_TABLES,
    "a5_return_images":   CLUSTER_A5_RETURN_IMAGES,
    "a6_doc_metadata":    CLUSTER_A6_DOCUMENT_METADATA,
    "a7_url_array":       CLUSTER_A7_URL_ARRAY,
    "b1_filter_array":    CLUSTER_B1_FILTER_BLOCKS_ARRAY,
    "b2_agentic_scopes":  CLUSTER_B2_AGENTIC_SCOPES,
    "b3_json_bbox":       CLUSTER_B3_JSON_BBOX,
    "b4_ocr_mode":        CLUSTER_B4_OCR_MODE,
    "b5_table_cutoff":    CLUSTER_B5_TABLE_CUTOFF,
    "c1_classify_route":  CLUSTER_C1_CLASSIFY_ROUTE,
    "c2_dual_doc":        CLUSTER_C2_DUAL_DOC,
    "d_termination":      CLUSTER_D_TERMINATION,
}

ALL_SCENARIOS = [
    (probe_id, prompt)
    for scenarios in GAP_SCENARIOS_0_8B.values()
    for probe_id, prompt in scenarios
]

# Gap weights — C1 and D get heavier allocation (worst failures)
GAP_WEIGHTS = {
    "a1_citations":      1.0,
    "a2_sys_prompt":     1.0,
    "a3_optimize_lat":   1.0,
    "a4_merge_tables":   1.0,
    "a5_return_images":  1.0,
    "a6_doc_metadata":   1.0,
    "a7_url_array":      1.0,
    "b1_filter_array":   1.0,
    "b2_agentic_scopes": 1.0,
    "b3_json_bbox":      1.0,
    "b4_ocr_mode":       1.0,
    "b5_table_cutoff":   1.0,
    "c1_classify_route": 1.8,   # 0/3 failure — needs extra signal
    "c2_dual_doc":       1.2,
    "d_termination":     1.5,   # loop failure — termination needs more signal
}

_random = random.Random(42)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint: Path) -> tuple[int, set]:
    seen: set = set()
    if not checkpoint.exists():
        return 0, seen
    count = 0
    for line in checkpoint.read_text().splitlines():
        try:
            d = json.loads(line)
            msgs = d.get("messages", [])
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            seen.add(user_msg[:120])
            count += 1
        except Exception:
            pass
    return count, seen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",  type=int, default=2_669,
                        help="SFT examples to generate (default: 2669)")
    parser.add_argument("--resume",  action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_models = {m.display: m for m in PREMIUM_MODELS}
    teacher = all_models.get(TEACHER_NAME)
    if not teacher:
        print(f"ERROR: teacher '{TEACHER_NAME}' not found in PREMIUM_MODELS")
        sys.exit(1)

    current, seen_prompts = load_checkpoint(CHECKPOINT) if args.resume else (0, set())

    print("=== 0.8B Targeted SFT Generator ===")
    print(f"  Teacher  : {TEACHER_NAME}")
    print(f"  Target   : {args.target}")
    print(f"  Workers  : {args.workers}")
    print(f"  Resume   : {current} existing examples")
    print(f"  Clusters : {list(GAP_SCENARIOS_0_8B.keys())}")
    print(f"  Scenarios: {len(ALL_SCENARIOS)} base prompts")
    print()

    # Weighted allocation per scenario
    total_weight = sum(
        GAP_WEIGHTS.get(gname, 1.0) * len(scenarios)
        for gname, scenarios in GAP_SCENARIOS_0_8B.items()
    )
    remaining = args.target - current

    # Per-scenario target counts (weighted)
    scenario_targets: dict[str, int] = {}
    for gname, scenarios in GAP_SCENARIOS_0_8B.items():
        w = GAP_WEIGHTS.get(gname, 1.0)
        for pid, _ in scenarios:
            scenario_targets[pid] = max(
                3,
                int(remaining * (w / len(scenarios)) / total_weight * len(scenarios))
            )

    generated = current
    _lock = threading.Lock()

    def process_one(probe_id: str, base_prompt: str) -> int:
        nonlocal generated

        n_to_gen = scenario_targets.get(probe_id, 5)
        with _lock:
            if generated >= args.target:
                return 0
            n_to_gen = min(n_to_gen, args.target - generated)

        variations = generate_variations(base_prompt, n_to_gen, dry_run=args.dry_run)
        if not variations:
            variations = [base_prompt]

        is_termination = probe_id.startswith("d_")
        added = 0

        for variation in variations:
            with _lock:
                if generated >= args.target:
                    break
                if variation[:120] in seen_prompts:
                    continue
                seen_prompts.add(variation[:120])

            # Slow API call — no lock held
            if is_termination:
                from gen_mcp_r3_gaps import generate_gap9_example
                row = generate_gap9_example(
                    teacher, variation, probe_id, dry_run=args.dry_run
                )
            else:
                row = run_through_teacher(
                    teacher, variation, probe_id, dry_run=args.dry_run
                )

            if row is None:
                with _lock:
                    seen_prompts.discard(variation[:120])
                continue

            if row:
                row["metadata"]["round"] = "0_8b_targeted"
                row["metadata"]["cluster"] = probe_id.split("_")[0]

            with _lock:
                with CHECKPOINT.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                generated += 1
                added += 1
                if generated % 50 == 0:
                    print(f"  [{generated}/{args.target}] probe={probe_id}")

        return added

    scenarios = list(ALL_SCENARIOS)
    random.shuffle(scenarios)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_one, pid, prompt): pid
            for pid, prompt in scenarios
        }
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"  [worker error] {futures[fut]}: {e}")

    print(f"\n=== Done: {generated}/{args.target} examples → {CHECKPOINT} ===")

    # Distribution summary
    counts: dict[str, int] = {}
    if CHECKPOINT.exists():
        for line in CHECKPOINT.read_text().splitlines():
            try:
                d = json.loads(line)
                pid = d.get("metadata", {}).get("probe_id", "unknown")
                # Map to cluster
                cluster = pid.split("_")[0] if "_" in pid else pid
                counts[cluster] = counts.get(cluster, 0) + 1
            except Exception:
                pass
    print("\nCluster distribution:")
    for cluster, cnt in sorted(counts.items()):
        print(f"  {cluster:20s}: {cnt:>5,}")


if __name__ == "__main__":
    main()
