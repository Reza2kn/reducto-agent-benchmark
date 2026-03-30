#!/usr/bin/env python3
"""
gen_mcp_r3_gaps.py — Round-3 synthetic data generator targeting the 9 highest-priority
MCP gaps identified from the MCP race results and ReductoLoRA Q6K benchmark failures.

Design:
  - Single teacher: Claude Haiku 4.5 + thinking (21/21 — only perfect MCP score)
  - 9 focused scenario banks, one per gap, with adversarial prompts where applicable
  - Writes to benchmark/data/synthetic_training_r3/.checkpoint_r3.jsonl
  - Does NOT modify R2 checkpoint

Gap priority (matches scenario bank order):
  1. schema (not schema_json)                    — adversarial: prompts say "schema_json"
  2. filter_blocks + agentic_scopes as arrays    — adversarial: prompts imply string form
  3. table_cutoff="preserve" when prompt says "allow"  — adversarial trap
  4. 4-hop chain: upload→parse(persist)→get_job→extract(jobid://)
  5. Dual-doc fan-out: 2×parse, 2×get_job, 2 distinct jobid://
  6. array_extract=True for repeating-row schemas
  7. reducto_upload first for presigned/expiring URLs
  8. citations=True + deep_extract=True for compliance/legal
  9. Chain termination + classify-route — multi-turn: tool_result → final text, NO loop
     (fixes the #1 ReductoLoRA failure: 50-75 repeat calls after getting a valid result)

Usage:
    python gen_mcp_r3_gaps.py                  # full run, target=11069
    python gen_mcp_r3_gaps.py --target 200     # quick test
    python gen_mcp_r3_gaps.py --dry-run
    python gen_mcp_r3_gaps.py --resume
"""

import argparse
import json
import os
import random
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))
from models import PREMIUM_MODELS, ModelConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("benchmark/data/synthetic_training_r3")
CHECKPOINT = OUTPUT_DIR / ".checkpoint_r3.jsonl"

TEACHER_NAME   = "Claude Haiku 4.5 + thinking"
REPHRASE_MODEL = "claude-haiku-4-5-20251001"   # cheap + fast for variations

# ---------------------------------------------------------------------------
# Reuse MCP tool schemas + system prompt from gen_mcp_data.py
# ---------------------------------------------------------------------------

try:
    from gen_mcp_data import MCP_TOOL_SCHEMAS, MCP_SYSTEM_PROMPT
except ImportError:
    print("ERROR: could not import from gen_mcp_data.py. Run from benchmark/scripts/.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Gap scenario banks
# ---------------------------------------------------------------------------

# ── Gap 1: schema (NOT schema_json) ──────────────────────────────────────
# Adversarial: prompts explicitly say "schema_json" or use legacy API terminology.
# Correct: model must use `schema` parameter as defined in MCP tool definition.

GAP1_SCHEMA_PARAM = [
    ("gap1_schema_trap_explicit",
     "Extract vendor_name, invoice_number, and total_amount from https://example.com/invoice.pdf. "
     "Pass the JSON schema using the 'schema_json' parameter."),

    ("gap1_schema_trap_api_docs",
     "I'm following the Reducto REST API docs which say to use schema_json. "
     "Extract account_number and portfolio_value from https://cdn.reducto.ai/samples/fidelity-example.pdf "
     "using schema_json for the field definitions."),

    ("gap1_schema_trap_old_client",
     "My old Python client used schema_json to pass the extraction schema. "
     "Extract: contract_party_a, contract_party_b, effective_date, governing_law from "
     "https://example.com/contract.pdf using schema_json."),

    ("gap1_schema_no_hint",
     "Extract from https://example.com/statement.pdf: account_holder, statement_date, "
     "opening_balance, closing_balance, total_credits, total_debits."),

    ("gap1_schema_trap_both_names",
     "I've seen two parameter names mentioned: 'schema' and 'schema_json'. "
     "Use the correct one to extract vendor, po_number, and line_items from "
     "https://example.com/purchase_order.pdf."),

    ("gap1_schema_structured_output",
     "I need structured output from https://example.com/earnings.pdf. "
     "Using your schema_json parameter, extract: company_name, fiscal_quarter, "
     "revenue, net_income, eps."),

    ("gap1_schema_trap_curl_example",
     "Based on this curl example: `--data '{\"schema_json\": {...}}'`, "
     "extract patient_name, dob, diagnosis_code from https://example.com/medical_record.pdf."),

    ("gap1_schema_minimal",
     "Extract just two fields from https://example.com/receipt.pdf: merchant_name and total. "
     "Use a JSON schema."),

    ("gap1_schema_nested",
     "Extract a nested object from https://example.com/report.pdf: "
     "{ summary: { title, period, author }, financials: { revenue, expenses, profit } }. "
     "Use the schema_json argument for the schema definition."),

    ("gap1_schema_with_descriptions",
     "Extract from https://example.com/tax_form.pdf: "
     "taxpayer_name (string), ssn (string, last 4 only), gross_income (number), "
     "federal_tax_withheld (number). Add descriptions to each field in your schema."),

    ("gap1_schema_trap_verbose",
     "According to the Reducto API documentation, the extraction endpoint accepts a "
     "schema_json parameter that defines the output structure. Please extract: "
     "fund_name, nav_per_unit, total_assets, inception_date from "
     "https://example.com/fund_factsheet.pdf using schema_json."),

    ("gap1_schema_after_classify",
     "First classify https://example.com/doc.pdf as invoice, statement, or contract. "
     "Then extract the appropriate fields — use schema_json to define the output structure."),

    ("gap1_schema_complex_object",
     "Extract a complex object from https://example.com/loan_agreement.pdf: "
     "borrower (name, address), lender (name, address), loan_amount, interest_rate, "
     "maturity_date, collateral_description. Use schema_json."),

    ("gap1_schema_from_jobid",
     "I already parsed https://example.com/financials.pdf and got job_id='parse-abc-123'. "
     "Extract net_revenue, cost_of_goods, gross_profit from jobid://parse-abc-123. "
     "Use schema_json to define the fields."),

    ("gap1_schema_array_with_schema_json",
     "Extract all line items from the invoice at https://example.com/invoice.pdf. "
     "Pass the repeating-row schema using schema_json and set array_extract=true."),

    ("gap1_schema_no_trap_clean",
     "Pull these fields from https://example.com/property_deed.pdf: grantor, grantee, "
     "parcel_number, legal_description, sale_price, recording_date."),

    ("gap1_schema_trap_user_confirmed",
     "I confirmed with your support team that I should use schema_json. "
     "Extract: policy_number, policyholder, coverage_amount, premium, expiry_date "
     "from https://example.com/insurance_policy.pdf."),

    ("gap1_schema_deep_trap",
     "The Reducto Python SDK has a field called schema_json on the ExtractRequest object. "
     "Extract holding_name, isin, quantity, current_value, weight_pct from "
     "https://cdn.reducto.ai/samples/fidelity-example.pdf using schema_json."),

    ("gap1_schema_clean_no_confusion",
     "From https://example.com/payslip.pdf extract: employee_name, employee_id, "
     "pay_period, gross_pay, deductions, net_pay. No special instructions — "
     "just extract these fields cleanly."),

    ("gap1_schema_trap_soft",
     "When you call reducto_extract, the schema parameter — or is it schema_json? — "
     "should define my output structure. Extract: company, ticker, market_cap, pe_ratio "
     "from https://example.com/stock_report.pdf."),
]

# ── Gap 2: filter_blocks + agentic_scopes as native arrays ───────────────
# Adversarial: prompts use comma-separated strings or quoted JSON strings.
# Correct: native array ["Header","Footer"] not "Header,Footer" or '["Header","Footer"]'

GAP2_NATIVE_ARRAYS = [
    ("gap2_filter_comma_trap",
     "Parse https://example.com/report.pdf. Remove these block types: Header, Footer, Page Number. "
     "Pass them as a comma-separated string to filter_blocks."),

    ("gap2_filter_string_trap",
     "Parse https://example.com/annual.pdf. I want to filter out 'Header,Footer,Page Number' — "
     "that's the string value for filter_blocks."),

    ("gap2_filter_quoted_json_trap",
     "Parse https://example.com/doc.pdf and exclude these blocks: '[\"Header\",\"Footer\",\"Page Number\"]'. "
     "That's the filter_blocks value — it's a JSON string."),

    ("gap2_filter_natural_request",
     "Parse https://example.com/manual.pdf for clean LLM input. "
     "Remove headers, footers, page numbers, and titles."),

    ("gap2_filter_enumerated",
     "Parse https://example.com/whitepaper.pdf. I need just the body text — "
     "strip out: Header, Footer, Page Number, Title, Section Header."),

    ("gap2_agentic_scopes_comma_trap",
     "Parse https://example.com/portfolio.pdf with agentic correction. "
     "Use agentic_scopes='table,text' — that's a comma-separated string."),

    ("gap2_agentic_scopes_string_trap",
     "Parse https://example.com/financials.pdf. Set agentic_scopes to the string 'table' "
     "for column alignment correction on the portfolio table."),

    ("gap2_agentic_scopes_natural",
     "Parse https://example.com/charts.pdf. The tables have misaligned columns and there are "
     "figures with chart data I need. Use agentic correction for both tables and figures."),

    ("gap2_both_arrays_natural",
     "Parse https://example.com/report.pdf. Filter out headers, footers, and page numbers. "
     "Enable agentic correction for tables. Persist results for later reuse."),

    ("gap2_filter_two_items",
     "Parse https://example.com/clean_doc.pdf. Remove just Header and Footer — "
     "pass those two as a comma string: 'Header,Footer'."),

    ("gap2_agentic_all_scopes_trap",
     "Parse https://example.com/complex.pdf with all agentic scopes enabled. "
     "Pass 'text,table,figure' as the agentic_scopes string."),

    ("gap2_filter_natural_two",
     "Parse https://example.com/earnings.pdf and strip decorative headers and page number footers. "
     "I'm passing this to an LLM so I want clean text only."),

    ("gap2_filter_quoted_array_trap",
     "Parse https://example.com/form.pdf. filter_blocks should be '[\"Signature\",\"Comment\"]' — "
     "a JSON-encoded array string."),

    ("gap2_agentic_figure_only",
     "Parse https://example.com/slides.pdf. The revenue data is in bar charts. "
     "Use agentic_scopes with just figure extraction — pass 'figure' as the value."),

    ("gap2_both_no_trap",
     "Parse https://cdn.reducto.ai/samples/fidelity-example.pdf. "
     "Filter out headers, footers, and page numbers. "
     "Apply agentic correction for tables (complex layout). "
     "Persist results for downstream extract."),

    ("gap2_filter_six_types",
     "Parse https://example.com/legal.pdf. Exclude: Header, Footer, Page Number, Title, "
     "Section Header, Signature. "
     "The prompt says to pass these as filter_blocks='Header,Footer,Page Number,Title,Section Header,Signature'."),

    ("gap2_agentic_table_text",
     "Parse https://example.com/annual_report.pdf. Apply AI correction to: "
     "text (OCR cleanup) and tables (column alignment). "
     "Use agentic_scopes — pass them as a comma string like 'table,text'."),

    ("gap2_filter_for_rag",
     "Parse https://example.com/kb.pdf for RAG indexing. Remove noise: headers, footers, page numbers. "
     "Use variable chunk mode."),

    ("gap2_agentic_figure_table_trap",
     "Parse https://example.com/research.pdf. Use agentic_scopes='figure,table' (as a string) "
     "for both chart extraction and column correction."),

    ("gap2_both_persist",
     "Parse https://example.com/fund.pdf. Remove headers and footers. "
     "Run agentic correction on tables. Persist results."),
]

# ── Gap 3: table_cutoff="preserve" when prompt says "allow" ──────────────
# Adversarial: user says "allow tables to span" → correct value is "preserve" not "allow"

GAP3_TABLE_CUTOFF_PRESERVE = [
    ("gap3_allow_trap_direct",
     "Split https://example.com/report.pdf into Summary, Financials, Appendix. "
     "I want tables to be allowed to cross section boundaries — use table_cutoff='allow'."),

    ("gap3_allow_trap_verbose",
     "Split https://example.com/annual.pdf into: Cover, Executive Summary, Operations, "
     "Financial Statements, Notes to Financials, Appendix. "
     "For tables that span a section boundary, allow them to carry over — "
     "set table_cutoff to 'allow'."),

    ("gap3_allow_trap_natural",
     "Split https://cdn.reducto.ai/samples/fidelity-example.pdf into: "
     "Account Overview, Portfolio Holdings, Income Summary, Transactions. "
     "Don't cut tables in half — allow them to span the boundary."),

    ("gap3_preserve_explicit",
     "Split https://example.com/doc.pdf into Introduction, Body, Conclusion. "
     "Use table_cutoff='preserve' to keep tables whole."),

    ("gap3_allow_trap_user_knows",
     "I know the table_cutoff option — I want 'allow' mode so tables aren't truncated "
     "when they sit at a section boundary. "
     "Split https://example.com/fund_report.pdf into Fund Overview, Holdings, Risks, Disclosures."),

    ("gap3_allow_trap_paraphrase",
     "Split this into 4 sections: Summary, Methods, Results, Discussion. "
     "Document: https://example.com/research.pdf. "
     "When a table sits at a section boundary, let it flow into the next section."),

    ("gap3_allow_trap_misremember",
     "Split https://example.com/legal.pdf into Terms, Definitions, Signatures. "
     "The table_cutoff option — I think it's 'allow' — should prevent tables from being cut."),

    ("gap3_no_table_hint",
     "Split https://example.com/portfolio.pdf into three sections: Overview, Holdings, Performance. "
     "Make sure tables are not split across sections."),

    ("gap3_allow_trap_api_wording",
     "Split https://example.com/prospectus.pdf into five sections. "
     "The Reducto API has a table_cutoff parameter — set it to 'allow' to preserve table integrity."),

    ("gap3_allow_trap_colloquial",
     "Split https://example.com/10k.pdf: Management Discussion, Financial Statements, Footnotes, Exhibits. "
     "I want whole tables — if a table is at a split point, allow it to stay together."),

    ("gap3_preserve_then_extract",
     "Split https://example.com/fund.pdf into Overview, Holdings, Performance, Disclosures. "
     "Keep tables whole at boundaries. Then extract holding_name and current_value from Holdings."),

    ("gap3_allow_trap_with_split_rules",
     "Split https://example.com/agreement.pdf into Recitals, Obligations, Termination, Signatures. "
     "Set table_cutoff='allow' and add a split rule: never split a numbered list item."),

    ("gap3_allow_trap_soft",
     "Split https://cdn.reducto.ai/samples/fidelity-example.pdf into: "
     "Account Summary, Investment Portfolio, Tax Summary. "
     "If a table spans a section break — allow it, don't truncate."),

    ("gap3_allow_then_parse",
     "Split https://example.com/prospectus.pdf into Introduction, Risk Factors, Financial Data. "
     "Allow tables to cross boundaries. Then use the Financial Data page range to parse that section."),

    ("gap3_allow_trap_five_sections",
     "Split https://example.com/annual_report.pdf into: "
     "Letter to Shareholders, Business Overview, Financial Highlights, "
     "Management Discussion, Financial Statements. "
     "table_cutoff='allow' so no table is split mid-row."),

    ("gap3_jobid_preserve",
     "I parsed https://example.com/doc.pdf earlier, job_id='parse-777'. "
     "Split jobid://parse-777 into Summary, Details, Appendix. "
     "Keep tables whole — use the preserve option."),

    ("gap3_allow_trap_technical_user",
     "Split https://example.com/10q.pdf into Financials, Notes, Disclosures. "
     "Set table_cutoff to 'allow' — I need tables to be kept intact."),

    ("gap3_no_trap_keep_tables",
     "Split https://example.com/fund_factsheet.pdf into "
     "Fund Details, Performance, Holdings, Fees, Disclosures. "
     "Ensure that no table is divided across two sections."),

    ("gap3_allow_trap_with_rules",
     "Split https://example.com/policy.pdf into Definitions, Coverage, Exclusions, Claims Process. "
     "table_cutoff should be 'allow'. Add a split rule: each section must contain at least one page."),

    ("gap3_allow_trap_fidelity",
     "Split https://cdn.reducto.ai/samples/fidelity-example.pdf into: "
     "Portfolio Summary, Holdings Table, Income Summary, Transaction History. "
     "Use table_cutoff='allow' — I need the holdings table to stay intact."),
]

# ── Gap 4: 4-hop chain — upload→parse(persist)→get_job→extract(jobid://) ─

GAP4_FOUR_HOP_CHAIN = [
    ("gap4_chain_basic",
     "Process https://example.com/report.pdf: upload it, parse it with persist_results, "
     "verify the job with get_job, then extract title, author, and key_findings using the job ID."),

    ("gap4_chain_presigned_full",
     "Presigned URL (expires in 5 min): https://s3.amazonaws.com/bucket/invoice.pdf?X-Amz-Expires=300&sig=abc. "
     "Upload → parse (persist) → get_job → extract: vendor, invoice_number, total, line_items."),

    ("gap4_chain_persist_required",
     "I need to extract data from https://example.com/financials.pdf without calling the parse API twice. "
     "Use persist_results and job ID chaining to do it in one parse + one extract."),

    ("gap4_chain_verify_before_extract",
     "Parse https://cdn.reducto.ai/samples/fidelity-example.pdf with persist_results=true. "
     "Before extracting, verify the parse is complete using get_job. "
     "Then extract account_number, total_portfolio_value, and holdings."),

    ("gap4_chain_large_doc",
     "This 200-page document https://example.com/annual.pdf needs to be parsed once and reused. "
     "Parse with persist_results. Confirm job completion. Then extract: "
     "company_name, fiscal_year, revenue, net_income, eps."),

    ("gap4_chain_async_pattern",
     "Run the full async pattern on https://example.com/complex.pdf: "
     "1. Upload to get stable ID. 2. Parse with persist. 3. Poll with get_job until done. "
     "4. Extract financial_summary using jobid://."),

    ("gap4_chain_stable_id_first",
     "The document at https://storage.example.com/intake/doc.pdf?token=xyz123 "
     "needs a stable ID for downstream processing. "
     "Full chain: upload → parse(persist) → get_job → extract contract_parties and effective_date."),

    ("gap4_chain_no_reparse",
     "Parse https://example.com/ledger.pdf exactly once. Persist the result. "
     "Verify it's ready. Then extract all journal entries using jobid://."),

    ("gap4_chain_cost_efficient",
     "For cost efficiency on https://example.com/report.pdf: parse once, persist the result, "
     "verify it completed, then extract: section_count, page_count, summary using the job ID."),

    ("gap4_chain_fidelity",
     "Full chain on https://cdn.reducto.ai/samples/fidelity-example.pdf: "
     "upload, parse with persist_results=true, verify via get_job, "
     "then extract account_number, account_type, portfolio_value."),

    ("gap4_chain_with_filter",
     "Parse https://example.com/doc.pdf with filter_blocks=['Header','Footer','Page Number'] "
     "and persist_results=true. Verify the job. "
     "Then extract company, report_date, total_revenue using the job ID."),

    ("gap4_chain_medical",
     "Process this medical record https://example.com/patient.pdf: "
     "upload, parse (persist), verify parse job, extract patient_name, dob, diagnosis, medications."),

    ("gap4_chain_finance",
     "I need account_number, balance, and transaction_count from https://example.com/statement.pdf. "
     "Use the 4-step chain: upload, parse with persist, get_job to verify, extract with jobid://."),

    ("gap4_chain_with_agentic",
     "Parse https://example.com/complex_tables.pdf with agentic_scopes=['table'] and persist_results. "
     "Verify the parse completed. Then extract all table data using jobid://."),

    ("gap4_chain_minimal",
     "Parse https://example.com/simple.pdf, persist the result, check the job, "
     "extract the title. Use jobid:// for the extract input."),

    ("gap4_chain_with_page_range",
     "Parse only pages 5-20 of https://example.com/bigdoc.pdf. Persist. Verify. "
     "Extract: section_title, key_metrics, table_count from those pages using the job ID."),

    ("gap4_chain_upload_stable",
     "https://example.com/temp/report.pdf?exp=1700000000 expires soon. "
     "Full chain: upload (stable ID) → parse(persist) → get_job → "
     "extract net_assets, expense_ratio, inception_date."),

    ("gap4_chain_legal",
     "Process the contract at https://example.com/agreement.pdf. "
     "Parse with persist, verify completion, extract parties, obligations, and term_length."),

    ("gap4_chain_ocr",
     "Parse this scanned document https://example.com/scan.pdf in OCR mode. Persist results. "
     "Verify the job completed. Extract: form_type, submission_date, applicant_name."),

    ("gap4_chain_with_get_job_explicit",
     "I want to make sure my parse job finished before I extract. "
     "Parse https://example.com/report.pdf with persist_results=true. "
     "Call get_job with the returned job ID. Then extract revenue and profit using jobid://."),
]

# ── Gap 5: Dual-doc fan-out — 2×parse, 2×get_job, 2 distinct jobid:// ────

GAP5_DUAL_DOC_FANOUT = [
    ("gap5_dual_basic",
     "I have two documents: https://example.com/doc_a.pdf and https://example.com/doc_b.pdf. "
     "Parse both in parallel (persist_results), verify both jobs, "
     "then extract account_number and balance from each using their respective job IDs."),

    ("gap5_dual_compare",
     "Compare Q3 and Q4 earnings reports: "
     "https://example.com/q3_earnings.pdf and https://example.com/q4_earnings.pdf. "
     "Parse both, verify both, extract revenue and net_income from each."),

    ("gap5_dual_fidelity_two_accounts",
     "Process two account statements: "
     "https://cdn.reducto.ai/samples/fidelity-example.pdf and "
     "https://example.com/account_b.pdf. "
     "Parse both with persist. Get job results for both. "
     "Extract account_number, portfolio_value from each using separate job IDs."),

    ("gap5_dual_upload_first",
     "Both URLs are presigned: "
     "https://s3.example.com/a.pdf?X-Amz-Expires=120 and "
     "https://s3.example.com/b.pdf?X-Amz-Expires=120. "
     "Upload both. Parse both with persist. Verify both jobs. "
     "Extract contract_party and effective_date from each."),

    ("gap5_dual_vendor_comparison",
     "Extract vendor names and totals from two invoices: "
     "https://example.com/invoice_jan.pdf and https://example.com/invoice_feb.pdf. "
     "Process them in parallel — parse both, verify both, extract from each."),

    ("gap5_dual_legal_contracts",
     "I need to compare two contracts side by side. "
     "Parse https://example.com/contract_v1.pdf and https://example.com/contract_v2.pdf "
     "with persist_results. Verify both. Extract parties, term, and governing_law from each."),

    ("gap5_dual_portfolio_two",
     "Pull holdings data from two portfolio reports: "
     "https://example.com/portfolio_a.pdf and https://example.com/portfolio_b.pdf. "
     "Run parallel parses (persist both). Verify both. "
     "Extract top_5_holdings and total_value from each via their job IDs."),

    ("gap5_dual_different_schemas",
     "https://example.com/invoice.pdf and https://example.com/contract.pdf need processing. "
     "Parse both with persist. Verify both. "
     "From invoice: vendor, total, line_items. From contract: parties, term, jurisdiction."),

    ("gap5_dual_no_jobid_confusion",
     "Process two financial statements: doc1 at https://example.com/stmt_alice.pdf "
     "and doc2 at https://example.com/stmt_bob.pdf. "
     "Parse each, confirm each job, extract account_holder and closing_balance from each. "
     "Use distinct job IDs — don't mix them up."),

    ("gap5_dual_annual_reports",
     "Annual reports for two companies: "
     "https://example.com/company_a_2025.pdf and https://example.com/company_b_2025.pdf. "
     "Parallel parse (persist), parallel job verification, "
     "then extract company_name, revenue, net_income from each."),

    ("gap5_dual_fan_out_explicit",
     "Fan out across two documents: "
     "https://example.com/fund_a.pdf and https://example.com/fund_b.pdf. "
     "Parse both in parallel with persist_results=true. "
     "Call get_job for each. Extract nav, expense_ratio, inception_date from each using jobid://."),

    ("gap5_dual_tax_forms",
     "Process two W-2 forms: https://example.com/w2_employee_1.pdf and "
     "https://example.com/w2_employee_2.pdf. "
     "Parse both, verify both jobs, extract employee_name, ssn_last4, gross_wages from each."),

    ("gap5_dual_real_estate",
     "Two property deeds to process: https://example.com/deed_property_a.pdf and "
     "https://example.com/deed_property_b.pdf. "
     "Parse in parallel (persist). Verify. Extract grantor, grantee, parcel_number, sale_price from each."),

    ("gap5_dual_persist_both",
     "I need to run extract on both these docs later. First parse them both with persist_results. "
     "Doc A: https://example.com/report_a.pdf, Doc B: https://example.com/report_b.pdf. "
     "Verify both jobs. Then extract title and summary from each."),

    ("gap5_dual_three_hop_each",
     "Two customer files: https://example.com/customer_01.pdf and "
     "https://example.com/customer_02.pdf. "
     "For each: parse(persist) → get_job → extract(jobid://). "
     "Extract customer_name, account_id, and outstanding_balance."),
]

# ── Gap 6: array_extract=True for repeating rows ─────────────────────────

GAP6_ARRAY_EXTRACT = [
    ("gap6_array_basic",
     "Extract all line items from https://example.com/invoice.pdf. "
     "There are 50+ rows — use array_extract."),

    ("gap6_array_holdings",
     "Pull every holding from the portfolio table in https://cdn.reducto.ai/samples/fidelity-example.pdf. "
     "Schema: array of {holding_name, isin, quantity, current_value, weight_pct}. "
     "Use array_extract=true."),

    ("gap6_array_transactions",
     "Extract all 200+ bank transactions from https://example.com/bank_statement.pdf. "
     "Each row has: date, description, debit, credit, balance. "
     "Use array_extract for the repeating rows."),

    ("gap6_array_natural_no_hint",
     "Get every transaction from https://example.com/statement_march.pdf. "
     "Schema: [{date, merchant, amount, category}]. There are about 150 rows."),

    ("gap6_array_with_jobid",
     "I parsed https://example.com/ledger.pdf earlier, job_id='parse-ledger-001'. "
     "Extract all journal entries as an array from jobid://parse-ledger-001. "
     "Each entry: entry_date, account_code, debit_amount, credit_amount, description. "
     "Set array_extract=true."),

    ("gap6_array_fund_positions",
     "Extract all fund positions from https://example.com/portfolio_report.pdf. "
     "I expect 80-120 rows. Schema: array of {fund_name, ticker, shares, nav, market_value}. "
     "Use array extraction mode."),

    ("gap6_array_employees",
     "Pull the complete employee list from https://example.com/hr_report.pdf. "
     "Fields per row: employee_id, name, department, salary, start_date. Use array_extract."),

    ("gap6_array_inventory",
     "Extract inventory rows from https://example.com/warehouse.pdf. "
     "Each item: sku, description, quantity_on_hand, reorder_point, unit_cost. "
     "There are 500+ items — use array_extract."),

    ("gap6_array_natural_line_items",
     "Get all line items from https://example.com/purchase_order.pdf. "
     "PO has 30 items: description, part_number, quantity, unit_price, extended_price."),

    ("gap6_array_medical_records",
     "Extract all medication entries from https://example.com/prescription_list.pdf. "
     "Rows: medication_name, dosage, frequency, prescribing_doctor, start_date. "
     "Use array_extract=true."),

    ("gap6_array_trades",
     "Pull every trade execution from https://example.com/trade_blotter.pdf. "
     "Schema: [{trade_date, symbol, side, quantity, price, commission}]. "
     "Hundreds of rows — set array_extract."),

    ("gap6_array_natural_implicit",
     "From https://example.com/sales_report.pdf, get every sale: "
     "{salesperson, region, product, amount, close_date}. The report covers Q4 with ~300 deals."),

    ("gap6_array_with_deep",
     "Extract all loan entries from https://example.com/loan_portfolio.pdf. "
     "Fields: loan_id, borrower, principal, interest_rate, maturity. "
     "Use array_extract=true and deep_extract=true for accuracy."),

    ("gap6_array_from_split",
     "I split https://example.com/annual.pdf and got the Holdings section is pages 10-25. "
     "Extract all holdings from those pages: {name, cusip, shares, value, pct_of_portfolio}. "
     "Use array_extract."),

    ("gap6_array_repeating_sections",
     "This document https://example.com/checks.pdf has repeating check records. "
     "Extract each check: check_number, date, payee, amount, memo. "
     "Set array_extract since it's repeating rows."),
]

# ── Gap 7: reducto_upload first for presigned/expiring URLs ──────────────

GAP7_UPLOAD_FIRST = [
    ("gap7_upload_presigned_s3",
     "Presigned S3 URL (60 seconds to expiry): "
     "https://s3.amazonaws.com/my-bucket/contract.pdf?X-Amz-Expires=60&X-Amz-Signature=abc123. "
     "Process it before it expires. Extract: parties, effective_date, term."),

    ("gap7_upload_expiring_token",
     "This URL has an auth token that expires soon: "
     "https://storage.example.com/docs/invoice.pdf?token=eyJhbGc...&exp=1700000300. "
     "Upload it first to get a stable ID, then extract vendor, total, invoice_date."),

    ("gap7_upload_gcs_signed",
     "Google Cloud Storage signed URL (valid 5 minutes): "
     "https://storage.googleapis.com/bucket/report.pdf?X-Goog-Expires=300&X-Goog-Signature=xyz. "
     "Get a stable reducto:// ID before the signature expires, then parse it."),

    ("gap7_upload_azure_sas",
     "Azure Blob SAS URL expiring in 2 minutes: "
     "https://myaccount.blob.core.windows.net/docs/file.pdf?sv=2021&se=2026-04-01T00%3A02%3A00Z&sig=abc. "
     "Upload first, then parse and extract title and author."),

    ("gap7_upload_multi_op",
     "I need to run parse AND extract AND classify on https://example.com/doc.pdf?token=short_lived_xyz. "
     "The token is short-lived — upload first to get a stable reducto:// ID, "
     "then reuse it for all three operations."),

    ("gap7_upload_then_chain",
     "The document at https://s3.example.com/intake.pdf?X-Amz-Expires=120 needs to be: "
     "uploaded (short-lived URL), parsed, and then have three fields extracted. "
     "Do the full upload → parse → extract chain."),

    ("gap7_upload_natural_presigned",
     "My backend gave me a presigned URL to process: "
     "https://api.example.com/documents/temp/abc123.pdf. "
     "It'll expire in about a minute. Extract: contract_party_a, contract_party_b, jurisdiction."),

    ("gap7_upload_stable_reference",
     "I want a stable reducto:// ID for https://example.com/report.pdf?sig=temp&expires=1700001000 "
     "because I'll run multiple operations on it. "
     "Upload first, then parse and extract executive_summary."),

    ("gap7_upload_no_expiry_hint",
     "Process https://example.com/invoice.pdf and extract: "
     "vendor_name, invoice_number, due_date, total_amount. "
     "The URL is from our S3 presigned link generator."),

    ("gap7_upload_then_classify_extract",
     "Short-lived URL: https://files.example.com/upload/abc.pdf?token=xyz&exp=soon. "
     "Upload it. Classify as invoice/contract/report. "
     "Then extract the relevant fields based on doc type."),

    ("gap7_upload_expiry_warning",
     "URGENT: this URL expires in 30 seconds: https://cdn.example.com/doc.pdf?X-Amz-Expires=30. "
     "Get a permanent copy first, then parse and extract key_terms."),

    ("gap7_upload_multi_tool_reuse",
     "The document https://example.com/annual_report.pdf?signed=abc&valid_until=soon "
     "needs to go through: parse, split into sections, and extract financials. "
     "Upload first for a stable reference across all three operations."),

    ("gap7_upload_s3_pattern",
     "S3 presigned link with 5-minute window: "
     "https://bucket.s3.us-east-1.amazonaws.com/financials.pdf?X-Amz-Expires=300. "
     "Standard upload-first pattern, then parse with filter for clean output."),

    ("gap7_upload_then_persist",
     "This URL will expire: https://temp.storage.example.com/intake.pdf?ttl=60. "
     "Upload it. Then parse with persist_results. Then extract fields."),

    ("gap7_upload_fanout_reuse",
     "Upload https://files.example.com/report.pdf?token=short_lived_123. "
     "Use the returned reducto:// ID for: (1) classify, (2) parse for full text, "
     "(3) extract financial summary. Don't parse more than once."),
]

# ── Gap 8: citations=True + deep_extract=True for compliance/legal ────────

GAP8_CITATIONS_DEEP_EXTRACT = [
    ("gap8_citations_highlight",
     "Extract vendor, invoice_number, line_items, and total from https://example.com/invoice.pdf. "
     "Return bounding box coordinates for each field so my UI can highlight the values."),

    ("gap8_citations_legal_review",
     "Legal contract at https://example.com/contract.pdf. "
     "Extract: parties, effective_date, term, governing_law, termination_clause. "
     "Include citations (bbox per field) — I need to show auditors exactly where each value came from."),

    ("gap8_deep_extract_difficult",
     "Extract contract_parties, payment_terms, and liability_cap from "
     "https://example.com/complex_agreement.pdf. "
     "Initial extraction missed some nested clauses — use deep_extract for accuracy."),

    ("gap8_both_compliance",
     "Compliance extraction from https://example.com/prospectus.pdf. "
     "Fields: issuer_name, offering_amount, risk_factors_summary, underwriter. "
     "Need citations for audit trail AND deep_extract for accuracy — this is regulatory filing."),

    ("gap8_citations_medical",
     "Extract diagnosis_codes, prescribed_medications, and physician_signature from "
     "https://example.com/medical_record.pdf. "
     "Return field locations (bounding boxes) for each extracted value."),

    ("gap8_deep_extract_tables",
     "The financial tables in https://example.com/10k.pdf are complex with footnotes. "
     "Extract total_revenue, operating_income, net_income, eps. "
     "Use deep_extract=true — standard extraction missed the footnote-adjusted figures."),

    ("gap8_citations_verification",
     "Extract from https://example.com/deed.pdf: grantor, grantee, property_description, "
     "consideration_amount, recording_date. "
     "I need the page + bounding box for each field for verification."),

    ("gap8_both_legal",
     "Pull from https://example.com/nda.pdf: disclosing_party, receiving_party, "
     "confidential_info_definition, term, remedies. "
     "This is a legal review — use citations for traceability and deep_extract for completeness."),

    ("gap8_deep_extract_natural",
     "Extract executive_compensation, board_members, and material_risks from "
     "https://example.com/proxy_statement.pdf. "
     "The document is complex — use the deep extraction mode."),

    ("gap8_citations_natural",
     "I need to know exactly where each value appears in the document. "
     "Extract from https://example.com/appraisal.pdf: appraised_value, property_address, "
     "appraisal_date, appraiser_name. Include location info for each."),

    ("gap8_both_after_persist",
     "I parsed https://example.com/sec_filing.pdf earlier, job_id='parse-sec-001'. "
     "Extract from jobid://parse-sec-001: filer_name, filing_date, material_events, "
     "risk_factors using deep_extract=true and citations=true."),

    ("gap8_citations_esg",
     "ESG report at https://example.com/esg_2025.pdf. "
     "Extract: carbon_emissions, water_usage, diversity_metrics, board_independence. "
     "Return bounding boxes so analysts can cross-check against the source."),

    ("gap8_deep_natural",
     "This lease agreement at https://example.com/lease.pdf is dense. "
     "Extract: tenant_name, landlord_name, premises_address, monthly_rent, "
     "lease_start, lease_end, security_deposit. "
     "Use deep extraction — some fields are in addendum pages."),

    ("gap8_both_financial_audit",
     "Audit extract from https://example.com/financial_statements.pdf. "
     "Fields: auditor_name, audit_opinion, material_weaknesses, going_concern_flag. "
     "Use deep_extract for accuracy and citations for the audit trail."),

    ("gap8_citations_bbox_explicit",
     "Extract key_terms, parties, and obligations from https://example.com/service_agreement.pdf. "
     "I'm building a contract review tool — I need the page number and bounding box "
     "for each extracted field, not just the values."),
]

# ── Gap 9: Chain termination + classify-route ─────────────────────────────
# The #1 ReductoLoRA failure: model calls tools 50-75 times after getting a
# valid result. These examples teach: result received → output answer → STOP.
# Also covers classify→route chains where the model must pick the next tool
# based on the classification result (not just repeat classify).
#
# Data shape is DIFFERENT from gaps 1-8: full multi-turn conversation
# [system, user, assistant(tool_calls), tool_result(s), assistant(text, no tools)]

GAP9_TERMINATION = [
    # ── Termination: single tool, clean stop ──────────────────────────────
    ("gap9_term_extract_done",
     "Extract vendor_name, invoice_number, and total_amount from "
     "https://example.com/invoice.pdf. Give me the values."),

    ("gap9_term_parse_done",
     "Parse https://cdn.reducto.ai/samples/fidelity-example.pdf and give me "
     "the markdown content."),

    ("gap9_term_split_done",
     "Split https://example.com/report.pdf into Introduction, Methods, Results, "
     "Discussion. Tell me the page ranges for each section."),

    ("gap9_term_classify_done",
     "Classify https://example.com/unknown_doc.pdf into one of: Invoice, Contract, "
     "Financial Statement, Other. What type is it?"),

    ("gap9_term_upload_parse_done",
     "Upload https://s3.example.com/file.pdf?token=abc and parse it. "
     "What's in the document?"),

    # ── Termination: 2-step chains ─────────────────────────────────────────
    ("gap9_term_parse_extract_done",
     "Parse https://example.com/financials.pdf then extract "
     "company_name, revenue, net_income. Return those three fields."),

    ("gap9_term_four_hop_done",
     "Upload https://example.com/report.pdf, parse with persist, verify the job, "
     "then extract title and key_findings. Show me the extracted data."),

    ("gap9_term_array_done",
     "Extract all transactions from https://example.com/statement.pdf. "
     "Each row: date, description, amount. Return the full list."),

    ("gap9_term_dual_doc_done",
     "Parse https://example.com/doc_a.pdf and https://example.com/doc_b.pdf. "
     "Get a summary from each. Compare them."),

    ("gap9_term_edit_done",
     "Edit https://example.com/form.pdf to fill in the date as 2026-01-15 "
     "and the name as John Smith. Give me the output URL."),

    ("gap9_term_citations_done",
     "Extract account_number and total_value from "
     "https://cdn.reducto.ai/samples/fidelity-example.pdf with bounding boxes. "
     "Show me the values and their locations."),

    ("gap9_term_get_job_done",
     "I started a parse job earlier, job_id='parse-abc123'. "
     "Check if it's complete and give me the result."),

    # ── Classify-route: classify → pick the right next tool ───────────────
    ("gap9_classify_route_invoice_extract",
     "I have a document at https://example.com/mystery.pdf — I don't know what type it is. "
     "First classify it. If it's an invoice, extract vendor, invoice_number, total. "
     "If it's a contract, extract parties and effective_date. Do the right thing."),

    ("gap9_classify_route_financial_fields",
     "Classify https://example.com/upload_doc.pdf. If it's a financial statement, "
     "extract account_number and portfolio_value. If it's something else, just "
     "parse it for me. Proceed based on what you find."),

    ("gap9_classify_route_medical",
     "Document at https://example.com/patient_record.pdf. "
     "Classify it first. If medical record: extract patient_name, diagnosis, medications. "
     "If prescription: extract drug_name, dosage, prescriber. Route appropriately."),

    ("gap9_classify_route_then_split",
     "Is https://example.com/long_doc.pdf a report or a contract? Classify it. "
     "If report: split into sections Introduction, Body, Conclusion. "
     "If contract: extract parties, term, governing_law."),

    ("gap9_classify_route_unknown_type",
     "Process https://example.com/intake.pdf. Start by classifying it as one of: "
     "Invoice, W2, Bank Statement, Other. Based on the type, extract the most "
     "relevant fields for that document category."),

    ("gap9_classify_route_dual_path",
     "Classify https://example.com/doc.pdf. "
     "Invoices → vendor_name + total_amount. "
     "Contracts → party_a + party_b + jurisdiction. "
     "Annual reports → company_name + fiscal_year + revenue. "
     "Run the right extraction for whatever type it is."),

    ("gap9_classify_route_legal_or_financial",
     "Unknown document: https://example.com/submission.pdf. "
     "Classify it. Legal document → citations=true extract with parties/dates. "
     "Financial → extract with deep_extract=true for accuracy. Run the right one."),

    ("gap9_classify_route_simple",
     "What type of document is https://example.com/filed_doc.pdf? "
     "Once you know, extract the key fields that matter for that doc type."),
]

# ---------------------------------------------------------------------------
# Synthetic tool results for gap 9 multi-turn examples
# One realistic result per tool — varied enough to avoid the model memorizing
# a single fixture, but consistent enough to be plausible.
# ---------------------------------------------------------------------------

import random as _random

_EXTRACT_RESULTS = [
    {"account_number": "Z12345678", "account_type": "Individual Brokerage",
     "portfolio_value": "$1,247,832.50", "statement_date": "2025-12-31"},
    {"vendor_name": "Acme Supplies Inc.", "invoice_number": "INV-2025-04821",
     "invoice_date": "2025-11-15", "total_amount": "$4,320.00", "currency": "USD"},
    {"company_name": "Meridian Holdings LLC", "revenue": "$82,400,000",
     "net_income": "$9,100,000", "fiscal_year": "2025", "eps": "$3.12"},
    {"party_a": "Northgate Capital Partners", "party_b": "Solaris Tech Inc.",
     "effective_date": "2026-02-01", "term": "3 years", "governing_law": "Delaware"},
    {"patient_name": "Jane Doe", "dob": "1982-06-14",
     "diagnosis": "Type 2 Diabetes", "prescribed": "Metformin 500mg BID"},
]

_PARSE_JOB_IDS = [
    "parse-a1b2c3d4-5e6f-7890-abcd-ef1234567890",
    "parse-f9e8d7c6-b5a4-3210-fedc-ba9876543210",
    "parse-11223344-5566-7788-99aa-bbccddeeff00",
]

_CLASSIFY_RESULTS = [
    {"category": "Financial Statement", "confidence": 0.96},
    {"category": "Invoice",            "confidence": 0.94},
    {"category": "Contract",           "confidence": 0.91},
    {"category": "Medical Record",     "confidence": 0.88},
    {"category": "Annual Report",      "confidence": 0.93},
]

_SPLIT_RESULTS = [
    {"sections": [
        {"section": "Introduction",         "start_page": 1, "end_page": 2},
        {"section": "Financial Highlights", "start_page": 3, "end_page": 5},
        {"section": "Disclosures",          "start_page": 6, "end_page": 7},
    ]},
    {"sections": [
        {"section": "Account Summary",      "start_page": 1, "end_page": 2},
        {"section": "Investment Holdings",  "start_page": 3, "end_page": 4},
        {"section": "Transaction History",  "start_page": 5, "end_page": 5},
    ]},
]


def _synthetic_result(tool_name: str, args: dict) -> dict:
    """Return a plausible fake tool result for the given tool call."""
    if tool_name == "reducto_upload":
        return {"file_id": "reducto://c97e7a59-9d14-4c2e-b59e-f7b42447dc0d", "status": "ok"}
    if tool_name == "reducto_parse":
        jid = _random.choice(_PARSE_JOB_IDS)
        return {"job_id": jid, "status": "completed",
                "result": {"markdown": "# Document\n\nContent successfully parsed.\n",
                           "page_count": 5}}
    if tool_name == "reducto_get_job":
        return {"job_id": args.get("job_id", _PARSE_JOB_IDS[0]),
                "status": "completed",
                "result": {"markdown": "# Document\n\nReady for extraction.\n"}}
    if tool_name == "reducto_extract":
        return {"result": _random.choice(_EXTRACT_RESULTS), "status": "completed"}
    if tool_name == "reducto_classify":
        return _random.choice(_CLASSIFY_RESULTS)
    if tool_name == "reducto_split":
        return _random.choice(_SPLIT_RESULTS)
    if tool_name == "reducto_edit":
        return {"output_url": "reducto://edited-doc-abc123def456", "status": "completed"}
    return {"status": "completed", "result": {}}


# ---------------------------------------------------------------------------
# Combined scenario bank
# ---------------------------------------------------------------------------

GAP_SCENARIOS = {
    "gap1_schema":          GAP1_SCHEMA_PARAM,
    "gap2_arrays":          GAP2_NATIVE_ARRAYS,
    "gap3_preserve":        GAP3_TABLE_CUTOFF_PRESERVE,
    "gap4_four_hop":        GAP4_FOUR_HOP_CHAIN,
    "gap5_dual_doc":        GAP5_DUAL_DOC_FANOUT,
    "gap6_array_extract":   GAP6_ARRAY_EXTRACT,
    "gap7_upload_first":    GAP7_UPLOAD_FIRST,
    "gap8_citations_deep":  GAP8_CITATIONS_DEEP_EXTRACT,
    "gap9_termination":     GAP9_TERMINATION,
}

ALL_SCENARIOS = [
    (probe_id, prompt)
    for scenarios in GAP_SCENARIOS.values()
    for probe_id, prompt in scenarios
]

# ---------------------------------------------------------------------------
# Variation generation + teacher call (same pattern as gen_mcp_data.py)
# ---------------------------------------------------------------------------

REPHRASE_PROMPT = """\
You are generating synthetic training data for an AI agent using Reducto MCP tools. \
Rephrase the following user request in {n} distinct ways.
Rules:
- Same underlying task, different wording/framing/detail level
- Vary style: terse vs verbose, developer vs analyst, some mention MCP context (Claude Code, Cursor, Cline)
- For adversarial prompts (ones that use wrong param names like schema_json, or wrong enum values like 'allow'): \
  keep the wrong terms — the adversarial nature is intentional
- Be realistic — something a real user would actually write

Original:
{original}

Return ONLY a JSON array of {n} strings. No markdown, no explanation."""

_REPHRASE_BATCH = 20


def generate_variations(prompt: str, n: int, dry_run: bool = False) -> list[str]:
    if dry_run:
        return [f"[DRY RUN #{i+1}] {prompt[:80]}..." for i in range(n)]

    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(
        model=REPHRASE_MODEL,
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


def run_through_teacher(
    model: ModelConfig, prompt: str, probe_id: str, dry_run: bool = False
) -> Optional[dict]:
    if dry_run:
        # Return a minimal valid example
        example = {
            "messages": [
                {"role": "system", "content": MCP_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_dry", "type": "function",
                     "function": {"name": "reducto_parse",
                                  "arguments": json.dumps({"input": "https://example.com/doc.pdf"})}}
                ]},
            ],
            "tools": MCP_TOOL_SCHEMAS,
            "metadata": {"probe_id": probe_id, "source": "generated",
                         "teacher": model.display, "round": "mcp_r3_gaps"},
        }
        return example

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
        if not raw_calls:
            return None

        tool_calls_fmt = [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": (
                        json.dumps(tc.get("args", {}))
                        if isinstance(tc.get("args", {}), dict)
                        else tc.get("args", "{}")
                    ),
                },
            }
            for tc in raw_calls if tc.get("name")
        ]

        return {
            "messages": [
                {"role": "system", "content": MCP_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
                {"role": "assistant", "content": None, "tool_calls": tool_calls_fmt},
            ],
            "tools": MCP_TOOL_SCHEMAS,
            "metadata": {
                "probe_id": probe_id,
                "source": "generated",
                "teacher": model.display,
                "round": "mcp_r3_gaps",
            },
        }
    except Exception as e:
        print(f"    [teacher] error ({model.display}): {e}")
        return None


# ---------------------------------------------------------------------------
# Gap 9: multi-turn termination example generator
# ---------------------------------------------------------------------------

_FINAL_RESPONSE_PROMPT = """\
You are an AI assistant that has just completed a Reducto document processing task.
The user asked: {user_prompt}

You made the following tool call(s) and received these results:
{tool_summary}

Now write a short, clear final response to the user summarising what you found.
Rules:
- Do NOT call any more tools — the task is complete.
- Be concise: 1-4 sentences.
- Refer to the actual values from the tool results.
- Sound like a helpful assistant, not a robot.
Return ONLY the response text."""


def _get_final_text(prompt: str, tool_calls: list, results: list,
                    dry_run: bool = False) -> str:
    """Ask Gemini Flash Lite (cheap) to write the closing assistant turn."""
    if dry_run:
        return "I've completed the task. Here are the results from the document."

    summary_lines = []
    for tc, res in zip(tool_calls, results):
        name = tc.get("function", {}).get("name", "tool")
        summary_lines.append(f"  {name} → {json.dumps(res)[:200]}")
    tool_summary = "\n".join(summary_lines)

    try:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=REPHRASE_MODEL,
            temperature=0.7,
            max_tokens=256,
            timeout=30,
        )
        msg = llm.invoke(_FINAL_RESPONSE_PROMPT.format(
            user_prompt=prompt, tool_summary=tool_summary))
        return msg.content.strip() or "I've extracted the requested data from the document."
    except Exception as e:
        print(f"    [gap9 final text] error: {e}")
        return "I've completed the task. The results are shown above."


def generate_gap9_example(model: ModelConfig, prompt: str, probe_id: str,
                          dry_run: bool = False) -> Optional[dict]:
    """
    Build a full multi-turn termination example:
      [system, user, assistant(tool_calls), tool_result(s), assistant(text, NO tools)]

    The critical training signal is the last assistant turn — text only, no tool calls.
    This teaches the model to stop after receiving a valid result.
    """
    # Step 1: get teacher's tool calls for the user prompt (same as gaps 1-8)
    initial = run_through_teacher(model, prompt, probe_id, dry_run)
    if initial is None:
        return None

    tool_calls = initial["messages"][-1].get("tool_calls", [])
    if not tool_calls:
        return None

    # Step 2: build synthetic tool result messages
    tool_result_msgs = []
    results = []
    for tc in tool_calls:
        tool_name = tc.get("function", {}).get("name", "")
        args = {}
        try:
            args = json.loads(tc.get("function", {}).get("arguments", "{}"))
        except Exception:
            pass
        result = _synthetic_result(tool_name, args)
        results.append(result)
        tool_result_msgs.append({
            "role": "tool",
            "tool_call_id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
            "content": json.dumps(result),
        })

    # Step 3: get Gemini to write the closing text (no tools needed, cheap model)
    final_text = _get_final_text(prompt, tool_calls, results, dry_run)

    # Step 4: assemble the full multi-turn conversation
    messages = [
        initial["messages"][0],            # system
        initial["messages"][1],            # user
        initial["messages"][2],            # assistant (tool_calls)
        *tool_result_msgs,                 # tool results
        {"role": "assistant",              # final text turn — NO tool_calls
         "content": final_text,
         "tool_calls": []},
    ]

    return {
        "messages": messages,
        "tools": MCP_TOOL_SCHEMAS,
        "metadata": {
            "probe_id": probe_id,
            "source": "generated",
            "teacher": model.display,
            "round": "mcp_r3_gaps",
            "gap9_multiturn": True,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint: Path) -> set[str]:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",  type=int, default=17_669,
                        help="Total examples to generate (default: 17669 = 8000 gaps1-8 + 9669 gap9)")
    parser.add_argument("--resume",  action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers for teacher calls (default: 8)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load teacher
    all_models = {m.display: m for m in PREMIUM_MODELS}
    teacher = all_models.get(TEACHER_NAME)
    if not teacher:
        print(f"ERROR: teacher '{TEACHER_NAME}' not found in PREMIUM_MODELS")
        sys.exit(1)

    seen_prompts = load_checkpoint(CHECKPOINT) if args.resume else set()
    current = sum(1 for _ in CHECKPOINT.open()) if (args.resume and CHECKPOINT.exists()) else 0

    print(f"=== MCP R3 Gap-Focused Gen ===")
    print(f"  Teacher  : {TEACHER_NAME}")
    print(f"  Target   : {args.target}")
    print(f"  Workers  : {args.workers}")
    print(f"  Resume   : {current} existing examples")
    print(f"  Scenarios: {len(ALL_SCENARIOS)} base × variations each")
    print(f"  Gaps     : {list(GAP_SCENARIOS.keys())}")
    print()

    # Compute variations per scenario to hit target
    n_scenarios    = len(ALL_SCENARIOS)
    variations_per = max(3, (args.target - current) // n_scenarios)
    print(f"  ~{variations_per} variations per base scenario")

    generated  = current
    _lock      = threading.Lock()

    def process_one(probe_id: str, base_prompt: str) -> int:
        nonlocal generated

        with _lock:
            if generated >= args.target:
                return 0
            n_to_gen = min(variations_per, args.target - generated)

        variations = generate_variations(base_prompt, n_to_gen, dry_run=args.dry_run)
        if not variations:
            variations = [base_prompt]

        is_gap9 = probe_id.startswith("gap9_")
        added   = 0

        for variation in variations:
            with _lock:
                if generated >= args.target:
                    break
                if variation[:120] in seen_prompts:
                    continue
                # Reserve this slot before the slow API call
                seen_prompts.add(variation[:120])

            # ── slow part — no lock held ──────────────────────────────────
            if is_gap9:
                row = generate_gap9_example(teacher, variation, probe_id,
                                            dry_run=args.dry_run)
            else:
                row = run_through_teacher(teacher, variation, probe_id,
                                          dry_run=args.dry_run)

            if row is None:
                with _lock:
                    seen_prompts.discard(variation[:120])   # free the slot on failure
                continue

            with _lock:
                with CHECKPOINT.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                generated += 1
                added     += 1
                if generated % 50 == 0:
                    print(f"  [{generated}/{args.target}] probe={probe_id}")

        return added

    # Process all scenarios in parallel across args.workers threads
    scenarios = list(ALL_SCENARIOS)
    random.shuffle(scenarios)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, pid, prompt): pid
                   for pid, prompt in scenarios}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"  [worker error] {futures[fut]}: {e}")

    print(f"\n=== Done: {generated} examples in {CHECKPOINT} ===")

    # Print gap distribution
    counts: dict[str, int] = {}
    if CHECKPOINT.exists():
        for line in CHECKPOINT.read_text().splitlines():
            try:
                d = json.loads(line)
                pid = d.get("metadata", {}).get("probe_id", "unknown")
                gap = pid.rsplit("_", 1)[0] if "_" in pid else pid
                # Map to gap group
                for gname in GAP_SCENARIOS:
                    if any(pid == s[0] for s in GAP_SCENARIOS[gname]):
                        gap = gname
                        break
                counts[gap] = counts.get(gap, 0) + 1
            except Exception:
                pass
    for gap, cnt in sorted(counts.items()):
        print(f"  {gap:25s}: {cnt:>5,}")


if __name__ == "__main__":
    main()
