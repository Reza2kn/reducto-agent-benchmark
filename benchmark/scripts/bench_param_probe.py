#!/usr/bin/env python3
"""
bench_param_probe.py — Targeted probe of 10 non-obvious Reducto API params.

Unlike the main benchmark (which tests general API usage at the surface level),
this asks: when a task clearly warrants a specific advanced param, does the model
discover and use it correctly?

3 models × 10 probes = 30 runs.
Each probe scores 0–3: param_present (1) + param_correct (1) + api_accepted (1).

Scoring:
  param_present  — model included the param in a tool call at all
  param_correct  — param value was semantically right for the task
  api_accepted   — API returned 2xx (param didn't cause a 422 or error)

Run:
    python bench_param_probe.py
    python bench_param_probe.py --model qwen3-coder  # single model
    python bench_param_probe.py --probe ocr_mode     # single probe
"""

import argparse
import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests

sys.path.insert(0, os.path.dirname(__file__))
from models import ModelConfig, OPENROUTER_MODELS, PREMIUM_MODELS

REDUCTO_BASE = "https://platform.reducto.ai"
TEST_DOC = "https://cdn.reducto.ai/samples/fidelity-example.pdf"
RESULTS_DIR = "benchmark/results/param_probe"


# ---------------------------------------------------------------------------
# Probe models — top 3 from main benchmark
# ---------------------------------------------------------------------------

def get_probe_models() -> list[ModelConfig]:
    all_models = OPENROUTER_MODELS + PREMIUM_MODELS
    ids = {"xiaomi/mimo-v2-pro", "qwen/qwen3-coder-next", "o4-mini"}
    return [m for m in all_models if m.id in ids]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    model: str
    probe_id: str
    param: str
    param_present: bool = False   # model used the param at all
    param_correct: bool = False   # param value was semantically correct
    api_accepted: bool = False    # API returned 2xx
    tool_calls_made: list = field(default_factory=list)
    error: str = ""
    score: int = 0                # 0–3
    credit_error: bool = False    # True if run was skipped due to 402 (not a real 0)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _h() -> dict:
    return {
        "Authorization": f"Bearer {os.environ.get('REDUCTO_API_KEY', '')}",
        "Content-Type": "application/json",
    }


def _post(path: str, body: dict) -> tuple[bool, dict]:
    """POST to Reducto. Returns (ok, response). Never raises."""
    try:
        r = requests.post(f"{REDUCTO_BASE}{path}", headers=_h(), json=body, timeout=120)
        ct = r.headers.get("content-type", "")
        data = r.json() if "json" in ct else {"raw": r.text[:500]}
        return r.ok, data
    except Exception as e:
        return False, {"error": str(e)}


# ---------------------------------------------------------------------------
# Advanced probe tools — expose ALL params being tested
# Models that know these params will use them; models that don't won't.
# ---------------------------------------------------------------------------

def _build_probe_tools():
    from langchain_core.tools import tool

    @tool
    def reducto_parse(
        input: str,
        table_format: Optional[str] = None,
        agentic_scopes: Optional[str] = None,
        extraction_mode: Optional[str] = None,
        merge_tables: Optional[bool] = None,
        persist_results: Optional[bool] = None,
        chunk_mode: Optional[str] = None,
        chunk_size: Optional[int] = None,
        return_figure_images: Optional[bool] = None,
        return_images_types: Optional[str] = None,
        filter_blocks: Optional[str] = None,
        page_range_start: Optional[int] = None,
        page_range_end: Optional[int] = None,
        embedding_optimized: Optional[bool] = None,
    ) -> str:
        """
        Parse a document into structured text, tables, and figures.

        input: public URL, reducto:// file ID, or JSON array string of URLs for multi-doc input.
        table_format: 'html'|'md'|'json'|'csv'|'jsonbbox'|'dynamic'.
            Use 'jsonbbox' when caller needs exact cell coordinates/bounding boxes on the page.
        agentic_scopes: JSON array of AI-correction scopes e.g. '["table"]' or '["text","table","figure"]'.
            Use ["table"] when tables appear misaligned, garbled, or have merged-cell issues.
        extraction_mode: 'hybrid' (default) or 'ocr'.
            Use 'ocr' for scanned/photographed documents where native text layer is absent.
        persist_results: if True, caches result for 1 hour accessible as 'jobid://JOB_ID'.
            Use when you plan to run extract or split on the same document afterwards.
        chunk_mode: 'variable'|'section'|'page'|'block'|'disabled'. For RAG chunking.
            Use 'section' for semantic chunking by document section (best for RAG).
        chunk_size: target tokens per chunk (250-1500). Default 500.
        return_figure_images: if True, returns presigned image URLs for every chart/figure found.
        return_images_types: JSON array of image types to return: '["figure"]', '["table"]',
            or '["figure","table","page"]'. Use when you need specific image type subsets.
        filter_blocks: JSON array of block types to EXCLUDE from output.
            Options: 'Header','Footer','Title','Section Header','Page Number','List Item',
            'Figure','Table','Key Value','Text','Comment','Signature'.
            Use '["Header","Footer","Page Number"]' for clean body text without boilerplate.
        page_range_start: first page to process (1-indexed). Use to reduce cost on large docs.
        page_range_end: last page to process (inclusive). Combine with page_range_start.
        embedding_optimized: if True, strips markdown formatting for cleaner embedding input.
            Use when output will be embedded in a vector database.
        """
        # Input may be URL string or JSON array of URLs
        try:
            parsed_input = json.loads(input) if input.strip().startswith("[") else input
        except Exception:
            parsed_input = input

        body: dict = {"input": parsed_input}

        if agentic_scopes:
            try:
                scopes = json.loads(agentic_scopes) if isinstance(agentic_scopes, str) else agentic_scopes
            except Exception:
                scopes = [agentic_scopes]
            body["enhance"] = {"agentic": [{"scope": s} for s in scopes]}

        fmt: dict = {}
        if table_format:
            fmt["table_output_format"] = table_format
        if merge_tables is not None:
            fmt["merge_tables"] = merge_tables
        if fmt:
            body["formatting"] = fmt

        ret: dict = {}
        if chunk_mode:
            ret["chunking"] = {"chunk_mode": chunk_mode}
            if chunk_size:
                ret["chunking"]["chunk_size"] = chunk_size
        if embedding_optimized is not None:
            ret["embedding_optimized"] = embedding_optimized
        if filter_blocks:
            try:
                ret["filter_blocks"] = json.loads(filter_blocks)
            except Exception:
                ret["filter_blocks"] = [filter_blocks]
        if ret:
            body["retrieval"] = ret

        settings: dict = {}
        if extraction_mode:
            settings["extraction_mode"] = extraction_mode
        if persist_results is not None:
            settings["persist_results"] = persist_results
        if return_figure_images:
            settings["return_images"] = ["figure"]
        if return_images_types:
            try:
                settings["return_images"] = json.loads(return_images_types)
            except Exception:
                settings["return_images"] = [return_images_types]
        if page_range_start or page_range_end:
            settings["page_range"] = {}
            if page_range_start:
                settings["page_range"]["start"] = page_range_start
            if page_range_end:
                settings["page_range"]["end"] = page_range_end
        if settings:
            body["settings"] = settings

        ok, data = _post("/parse", body)
        if not ok:
            return f"Error: {json.dumps(data)[:300]}"

        chunks = data.get("result", {}).get("chunks", [])
        content = "\n\n".join(c.get("content", "") for c in chunks[:2]) if chunks else ""
        job_id = data.get("job_id", "")
        hint = (
            f"\n💡 Re-use without re-parsing: pass input='jobid://{job_id}' to reducto_extract or reducto_split"
            if persist_results and job_id else ""
        )
        return (
            f"Job ID: {job_id}\n"
            f"Pages: {data.get('usage', {}).get('num_pages', '?')}"
            f"{hint}\n\n{content[:800]}"
        )

    @tool
    def reducto_extract(
        input: str,
        schema_json: str,
        array_extract: Optional[bool] = None,
        deep_extract: Optional[bool] = None,
        citations: Optional[bool] = None,
        optimize_for_latency: Optional[bool] = None,
        system_prompt: Optional[str] = None,
        include_images: Optional[bool] = None,
    ) -> str:
        """
        Extract structured JSON from a document using a JSON Schema.

        input: public URL, reducto:// file ID, or 'jobid://JOB_ID' to reuse a cached parse result.
        schema_json: JSON Schema string with field descriptions.
        array_extract: True when extracting repeating rows (line items, holdings, transactions).
            Schema must include at least one top-level array property.
        deep_extract: True for difficult documents where initial extraction misses fields.
            Avoid when speed matters — it runs an additional agentic pass.
        citations: True to return bounding box location for each extracted field.
        optimize_for_latency: True when speed is the priority over extraction depth.
        system_prompt: custom extraction instructions for edge cases or unit conventions.
        include_images: True to include chart/figure image context during extraction.
            Use when the data you need is in a chart or graph, not in text or tables.
        """
        try:
            schema = json.loads(schema_json)
        except Exception:
            return "Error: schema_json must be valid JSON"

        body: dict = {"input": input, "instructions": {"schema": schema}}
        if system_prompt:
            body["instructions"]["system_prompt"] = system_prompt

        settings: dict = {}
        if array_extract is not None:
            settings["array_extract"] = array_extract
        if deep_extract is not None:
            settings["deep_extract"] = deep_extract
        if citations:
            settings["citations"] = {"enabled": True, "numerical_confidence": True}
        if optimize_for_latency is not None:
            settings["optimize_for_latency"] = optimize_for_latency
        if include_images is not None:
            settings["include_images"] = include_images
        if settings:
            body["settings"] = settings

        ok, data = _post("/extract", body)
        if not ok:
            return f"Error: {json.dumps(data)[:300]}"
        return (
            f"Job ID: {data.get('job_id', '?')}\n"
            + json.dumps(data.get("result", data), indent=2)[:600]
        )

    @tool
    def reducto_split(
        input: str,
        sections_json: str,
        split_rules: Optional[str] = None,
        table_cutoff: Optional[str] = None,
    ) -> str:
        """
        Divide a document into named sections by page range.

        input: URL, reducto:// file ID, or jobid://JOB_ID.
        sections_json: JSON array of [{name, description}] section definitions.
        split_rules: natural language rules that override default splitting behavior.
            Use this for constraints like 'never split a table across sections' or
            'if a section is missing mark it empty rather than merging with adjacent'.
        table_cutoff: 'truncate' (default) or 'preserve' when a boundary falls inside a table.
        """
        try:
            split_desc = json.loads(sections_json)
        except Exception:
            split_desc = [{"name": s.strip(), "description": s.strip()} for s in sections_json.split(",")]

        body: dict = {"input": input, "split_description": split_desc}
        if split_rules:
            body["split_rules"] = split_rules
        if table_cutoff:
            body["settings"] = {"table_cutoff": table_cutoff}

        ok, data = _post("/split", body)
        if not ok:
            return f"Error: {json.dumps(data)[:300]}"

        splits = data.get("result", {}).get("splits", [])
        lines = [f"• {s['name']}: pages {s.get('pages')} (conf {s.get('conf', '?')})" for s in splits]
        return f"Job ID: {data.get('job_id', '?')}\n" + "\n".join(lines)

    @tool
    def reducto_classify(
        input: str,
        categories_json: str,
        document_metadata: Optional[str] = None,
    ) -> str:
        """
        Classify a document into one of the provided categories.

        input: URL, reducto:// file ID, or jobid://JOB_ID.
        categories_json: JSON array of [{category, criteria}] objects.
        document_metadata: optional context about where this document came from or what
            it's expected to be. Providing source/pipeline context improves accuracy.
        """
        try:
            schema = json.loads(categories_json)
        except Exception:
            cats = [c.strip() for c in categories_json.split(",")]
            schema = [{"category": c, "criteria": [c]} for c in cats]

        body: dict = {"input": input, "classification_schema": schema}
        if document_metadata:
            body["document_metadata"] = document_metadata

        ok, data = _post("/classify", body)
        if not ok:
            return f"Error: {json.dumps(data)[:300]}"
        return (
            f"Classification: {data.get('result', {}).get('category', '?')}\n"
            + json.dumps(data.get("result", {}), indent=2)
        )

    @tool
    def reducto_upload(file_url: str) -> str:
        """
        Upload a document from a URL and get a stable reducto:// file ID.

        Use this FIRST when:
        - The source URL is a presigned/expiring URL (S3, GCS, Azure SAS)
        - You need to reuse the same document across multiple operations
        - The file requires authentication to access

        Returns a reducto:// file ID that stays valid for 24 hours and can be
        passed as input to parse, extract, split, classify instead of the original URL.
        """
        ok, data = _post("/upload", {"url": file_url})
        if not ok:
            return f"Error: {json.dumps(data)[:300]}"
        file_id = data.get("file_id", "")
        return f"Uploaded. Use this ID for all subsequent operations: {file_id}"

    @tool
    def reducto_edit(
        document_url: str,
        edit_instructions: str,
        flatten: Optional[bool] = None,
        form_schema_json: Optional[str] = None,
    ) -> str:
        """
        Edit or fill a PDF document using natural language instructions.

        document_url: URL or reducto:// file ID of the PDF to edit.
        edit_instructions: natural language description of edits (e.g. 'Fill Recipient with John Smith').
        flatten: if True, converts all form fields to static text — use when delivering
            a final non-editable PDF to a recipient.
        form_schema_json: JSON array of EditWidget objects for structured multi-field form filling.
            Each widget: {field_name, value, field_type: 'text'|'checkbox'|'date'|'signature'}.
            Use this instead of edit_instructions when filling multiple specific fields precisely.

        Returns a URL to the edited PDF.
        """
        body: dict = {"document_url": document_url, "edit_instructions": edit_instructions}
        opts: dict = {}
        if flatten is not None:
            opts["flatten"] = flatten
        if form_schema_json:
            try:
                opts["form_schema"] = json.loads(form_schema_json)
            except Exception:
                pass
        if opts:
            body["edit_options"] = opts
        ok, data = _post("/edit", body)
        if not ok:
            return f"Error: {json.dumps(data)[:300]}"
        return f"Edited PDF URL: {data.get('url', data.get('result', {}).get('url', str(data)[:200]))}"

    @tool
    def reducto_get_job(job_id: str) -> str:
        """
        Poll the status of an async job.

        job_id: the job UUID returned by any async operation.
        Returns status ('processing'|'completed'|'failed') and result when completed.
        Call repeatedly until status is 'completed' or 'failed'.
        """
        import requests as _req
        r = _req.get(
            f"{REDUCTO_BASE}/job/{job_id}",
            headers=_h(),
            timeout=30,
        )
        data = r.json()
        status = data.get("status", "unknown")
        if status == "completed":
            result = data.get("result", {})
            return f"Status: completed\n{json.dumps(result, indent=2)[:600]}"
        return f"Status: {status}\n{json.dumps(data, indent=2)[:300]}"

    return [reducto_parse, reducto_extract, reducto_split, reducto_classify,
            reducto_upload, reducto_edit, reducto_get_job]


# ---------------------------------------------------------------------------
# Tool call extraction from LangGraph message history
# ---------------------------------------------------------------------------

def _extract_tool_calls(messages) -> list[dict]:
    calls = []
    for msg in messages:
        tcs = getattr(msg, "tool_calls", None)
        if not tcs:
            continue
        for tc in tcs:
            name = tc.get("name", "")
            args = tc.get("args", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            calls.append({"tool": name, "args": args if isinstance(args, dict) else {}})
    return calls


def _pcalls(tc): return [c for c in tc if "parse" in c["tool"]]
def _ecalls(tc): return [c for c in tc if "extract" in c["tool"]]
def _scalls(tc): return [c for c in tc if "split" in c["tool"]]
def _ccalls(tc): return [c for c in tc if "classify" in c["tool"]]


# ---------------------------------------------------------------------------
# Probe definitions
# ---------------------------------------------------------------------------

@dataclass
class Probe:
    id: str
    param: str           # short label for the param being tested
    prompt: str          # user message sent to the agent
    check: Callable      # (tool_calls) → (param_present: bool, param_correct: bool)
    notes: str = ""      # what correct behavior looks like


def _is_array_input(val) -> bool:
    if isinstance(val, list):
        return len(val) > 1
    if isinstance(val, str) and val.strip().startswith("["):
        try:
            parsed = json.loads(val)
            return isinstance(parsed, list) and len(parsed) > 1
        except Exception:
            pass
    return False


PROBES: list[Probe] = [
    Probe(
        id="jobid_chaining",
        param="persist_results + jobid://",
        prompt=(
            f"I have a financial statement at {TEST_DOC}. "
            "First parse the full document to understand its structure, then extract "
            "the account number and portfolio values (beginning + ending). "
            "Do this efficiently — avoid processing the document twice."
        ),
        check=lambda tc: (
            any(c["args"].get("persist_results") for c in _pcalls(tc)),
            any(c["args"].get("persist_results") for c in _pcalls(tc))
            and any(
                str(c["args"].get("input", "")).startswith("jobid://")
                for c in _ecalls(tc)
            ),
        ),
        notes="parse(persist_results=True) → extract(input='jobid://...')",
    ),
    Probe(
        id="array_extract",
        param="array_extract=True",
        prompt=(
            f"Extract ALL individual stock holdings from this financial statement: {TEST_DOC}. "
            "I need every holding as a separate row with name, ticker symbol, market value, "
            "and percentage of portfolio."
        ),
        check=lambda tc: (
            any("array_extract" in c["args"] for c in _ecalls(tc)),
            any(c["args"].get("array_extract") is True for c in _ecalls(tc)),
        ),
        notes="array_extract=True required when extracting repeating rows",
    ),
    Probe(
        id="agentic_scopes",
        param="agentic_scopes=['table']",
        prompt=(
            f"Parse {TEST_DOC}. The document has complex multi-column financial tables that "
            "are likely to have misaligned columns and garbled numbers due to the PDF layout. "
            "Use AI-powered correction to fix the table structure before returning the content."
        ),
        check=lambda tc: (
            any("agentic_scopes" in c["args"] for c in _pcalls(tc)),
            any(
                "table" in str(c["args"].get("agentic_scopes", ""))
                for c in _pcalls(tc)
            ),
        ),
        notes="agentic_scopes must include 'table' for table correction",
    ),
    Probe(
        id="deep_extract_off",
        param="deep_extract absent/False",
        prompt=(
            f"This is time-sensitive — quickly extract just the account number and "
            f"ending portfolio value from {TEST_DOC}. Speed is the priority here."
        ),
        check=lambda tc: (
            # present = model showed latency awareness (used optimize_for_latency or set deep_extract=False)
            any(
                "optimize_for_latency" in c["args"] or c["args"].get("deep_extract") is False
                for c in _ecalls(tc)
            ),
            # correct = model did NOT set deep_extract=True
            not any(c["args"].get("deep_extract") is True for c in _ecalls(tc)),
        ),
        notes="deep_extract must NOT be True when speed is requested",
    ),
    Probe(
        id="jsonbbox_format",
        param="table_format='jsonbbox'",
        prompt=(
            f"Parse the tables from {TEST_DOC}. I need the exact pixel coordinates and "
            "bounding box positions of every table cell on each page so I can reconstruct "
            "the layout programmatically in my UI."
        ),
        check=lambda tc: (
            any("table_format" in c["args"] for c in _pcalls(tc)),
            any(c["args"].get("table_format") == "jsonbbox" for c in _pcalls(tc)),
        ),
        notes="table_format=jsonbbox required when cell coordinates are needed",
    ),
    Probe(
        id="split_rules",
        param="split_rules (NL constraint)",
        prompt=(
            f"Split {TEST_DOC} into these sections: Cover Page, Account Summary, "
            "Holdings, Income Summary. "
            "Important constraint: if a section boundary falls in the middle of a table, "
            "extend the section to include the complete table — never split a table across sections."
        ),
        check=lambda tc: (
            any("split_rules" in c["args"] for c in _scalls(tc)),
            any(
                c["args"].get("split_rules") and len(str(c["args"]["split_rules"])) > 10
                for c in _scalls(tc)
            ),
        ),
        notes="The table constraint must be passed as split_rules, not just implicit",
    ),
    Probe(
        id="ocr_mode",
        param="extraction_mode='ocr'",
        prompt=(
            "Parse this document — it's a scanned paper form that was photographed with a "
            f"mobile camera and then PDF'd, so there is no native text layer: {TEST_DOC}. "
            "Make sure to use full OCR mode for text extraction."
        ),
        check=lambda tc: (
            any("extraction_mode" in c["args"] for c in _pcalls(tc)),
            any(c["args"].get("extraction_mode") == "ocr" for c in _pcalls(tc)),
        ),
        notes="extraction_mode='ocr' required for scanned documents",
    ),
    Probe(
        id="document_metadata",
        param="document_metadata (context hint)",
        prompt=(
            f"Classify this document: {TEST_DOC}. "
            "It came from our accounts payable automation pipeline and is expected to be "
            "one of: invoice, purchase order, or financial statement. "
            "Use that pipeline context to improve your classification accuracy."
        ),
        check=lambda tc: (
            any("document_metadata" in c["args"] for c in _ccalls(tc)),
            any(
                c["args"].get("document_metadata") and len(str(c["args"]["document_metadata"])) > 10
                for c in _ccalls(tc)
            ),
        ),
        notes="Source/pipeline context should be passed as document_metadata",
    ),
    Probe(
        id="url_array",
        param="input as JSON array of URLs",
        prompt=(
            "Parse these three quarterly financial reports together as a single unified document "
            "so I can do cross-quarter analysis in one pass: "
            f"{TEST_DOC}, {TEST_DOC}, {TEST_DOC}. "
            "Combine them into one parse result."
        ),
        check=lambda tc: (
            # present = used array format input at all (not 3 separate calls)
            any(_is_array_input(c["args"].get("input", "")) for c in _pcalls(tc)),
            # correct = used array format AND fewer calls than number of URLs (3)
            any(_is_array_input(c["args"].get("input", "")) for c in _pcalls(tc)) and len(_pcalls(tc)) <= 2,
        ),
        notes="Should pass JSON array of URLs, not 3 separate parse calls",
    ),
    Probe(
        id="return_figure_images",
        param="return_figure_images=True",
        prompt=(
            f"Parse {TEST_DOC} and I need the actual image files for all charts and graphs "
            "in the document — not text descriptions of them, but the actual rendered images "
            "as URLs I can embed in my web app."
        ),
        check=lambda tc: (
            any("return_figure_images" in c["args"] for c in _pcalls(tc)),
            any(c["args"].get("return_figure_images") is True for c in _pcalls(tc)),
        ),
        notes="return_figure_images=True required to get presigned image URLs for figures",
    ),
]


# ---------------------------------------------------------------------------
# HARD probes — 23 probes × 3 pts = 69 pts
# These target Reducto-specific behaviours that are NOT in model training data.
# Expect frontier models to score 30-50%, tiny models much lower.
# ---------------------------------------------------------------------------

HARD_PROBES: list[Probe] = [
    # --- Batch A: Extract depth ---
    Probe(
        id="citations_bbox",
        param="citations=True → bbox in response",
        prompt=(
            f"Extract the account number and total portfolio value from {TEST_DOC}. "
            "I need to know EXACTLY where on the page each value appears so I can "
            "draw a highlight box over it in my PDF viewer. Give me the bounding box coordinates."
        ),
        check=lambda tc: (
            any(c["args"].get("citations") for c in _ecalls(tc)),
            any(c["args"].get("citations") for c in _ecalls(tc)),
        ),
        notes="citations=True returns bbox per extracted field. Models default to parse+describe.",
    ),
    Probe(
        id="citations_no_chunking",
        param="citations=True must NOT have chunk_mode",
        prompt=(
            f"Extract all section headings from {TEST_DOC} with their bounding boxes "
            "so I can link to them, AND chunk the output by page so I can stream it "
            "to my frontend as pages complete."
        ),
        check=lambda tc: (
            any(c["args"].get("citations") for c in _ecalls(tc)),
            # correct = citations used but no chunk_mode set (they're incompatible)
            any(c["args"].get("citations") for c in _ecalls(tc))
            and not any(c["args"].get("chunk_mode") for c in _pcalls(tc)),
        ),
        notes="citations and chunking are incompatible — model must choose one.",
    ),
    Probe(
        id="system_prompt_units",
        param="system_prompt for unit conversion",
        prompt=(
            f"Extract all dollar amounts from {TEST_DOC}. "
            "The document reports values in thousands (e.g. '$1,234' means $1,234,000). "
            "I need the actual full dollar values, not the abbreviated thousands format."
        ),
        check=lambda tc: (
            any("system_prompt" in c["args"] for c in _ecalls(tc)),
            any(
                c["args"].get("system_prompt") and
                any(w in c["args"]["system_prompt"].lower() for w in ["thousand", "multiply", "000", "million", "convert"])
                for c in _ecalls(tc)
            ),
        ),
        notes="system_prompt is the right lever for unit/format instructions during extraction.",
    ),
    Probe(
        id="optimize_latency_not_deep",
        param="optimize_for_latency=True, deep_extract absent",
        prompt=(
            f"Extract the account number and ending balance from {TEST_DOC} as fast as possible. "
            "This is a real-time API response — I need it under 2 seconds. "
            "Accuracy can be slightly lower if it means speed."
        ),
        check=lambda tc: (
            any(c["args"].get("optimize_for_latency") for c in _ecalls(tc)),
            any(c["args"].get("optimize_for_latency") for c in _ecalls(tc))
            and not any(c["args"].get("deep_extract") for c in _ecalls(tc)),
        ),
        notes="optimize_for_latency=True + NO deep_extract. Models love to always set deep_extract.",
    ),
    Probe(
        id="include_images_chart",
        param="include_images=True for chart data",
        prompt=(
            f"The pie chart on page 2 of {TEST_DOC} shows the asset allocation percentages. "
            "Extract those percentages for each asset class. "
            "The data is only in the chart, not in any table."
        ),
        check=lambda tc: (
            any("include_images" in c["args"] for c in _ecalls(tc)),
            any(c["args"].get("include_images") for c in _ecalls(tc)),
        ),
        notes="include_images=True provides chart context. Models default to parse+describe.",
    ),

    # --- Batch B: Parse depth ---
    Probe(
        id="merge_tables_crosspage",
        param="merge_tables=True",
        prompt=(
            f"Parse {TEST_DOC}. The portfolio holdings table is split across two pages — "
            "the header row is on page 3 and the data rows continue on page 4. "
            "I need a single unified table, not two fragments."
        ),
        check=lambda tc: (
            any("merge_tables" in c["args"] for c in _pcalls(tc)),
            any(c["args"].get("merge_tables") is True for c in _pcalls(tc)),
        ),
        notes="merge_tables=True stitches tables split across page breaks.",
    ),
    Probe(
        id="filter_blocks_clean",
        param="filter_blocks=['Header','Footer','Page Number']",
        prompt=(
            f"Parse {TEST_DOC} and return only the body content. "
            "Strip out all headers, footers, and page numbers — "
            "I'm feeding this into an LLM and want clean signal only."
        ),
        check=lambda tc: (
            any("filter_blocks" in c["args"] for c in _pcalls(tc)),
            any(
                c["args"].get("filter_blocks") and
                any(b in str(c["args"]["filter_blocks"]) for b in ["Header", "Footer", "Page Number"])
                for c in _pcalls(tc)
            ),
        ),
        notes="filter_blocks with exact type names. Models guess wrong enum values.",
    ),
    Probe(
        id="chunk_section_rag",
        param="chunk_mode='section' for semantic RAG",
        prompt=(
            f"I'm building a RAG system over {TEST_DOC}. "
            "Parse it and chunk the output semantically — I want chunks that respect "
            "document sections so each chunk is a coherent topic, not arbitrary page cuts."
        ),
        check=lambda tc: (
            any("chunk_mode" in c["args"] for c in _pcalls(tc)),
            any(c["args"].get("chunk_mode") == "section" for c in _pcalls(tc)),
        ),
        notes="chunk_mode='section' for semantic chunking. Models default to 'page'.",
    ),
    Probe(
        id="embedding_optimized_flag",
        param="embedding_optimized=True",
        prompt=(
            f"Parse {TEST_DOC} to build a vector search index. "
            "I'll be embedding all the text chunks — "
            "optimize the output format for embedding quality."
        ),
        check=lambda tc: (
            any("embedding_optimized" in c["args"] for c in _pcalls(tc)),
            any(c["args"].get("embedding_optimized") is True for c in _pcalls(tc)),
        ),
        notes="embedding_optimized strips markdown noise. Lives in retrieval, not settings.",
    ),
    Probe(
        id="page_range_cost",
        param="page_range to reduce credit cost",
        prompt=(
            f"This document is 50 pages but the financial summary I need is on pages 3 to 7. "
            f"Parse just those pages from {TEST_DOC} — I don't want to pay for the whole thing."
        ),
        check=lambda tc: (
            any("page_range_start" in c["args"] or "page_range" in str(c["args"]) for c in _pcalls(tc)),
            any(
                (c["args"].get("page_range_start") in (3, "3") and c["args"].get("page_range_end") in (7, "7"))
                or ('"start": 3' in str(c["args"]) or '"start":3' in str(c["args"]))
                for c in _pcalls(tc)
            ),
        ),
        notes="page_range reduces credits. Models parse full doc and filter in code.",
    ),
    Probe(
        id="return_images_array",
        param="return_images=['figure','table']",
        prompt=(
            f"Parse {TEST_DOC}. I need rendered image files for ALL tables AND all charts. "
            "Give me presigned URLs for both types separately — tables as images for my OCR "
            "fallback pipeline and charts as images for my visualization renderer."
        ),
        check=lambda tc: (
            any("return_images_types" in c["args"] or "return_figure_images" in c["args"] for c in _pcalls(tc)),
            any(
                c["args"].get("return_images_types") and
                "table" in str(c["args"]["return_images_types"]) and
                "figure" in str(c["args"]["return_images_types"])
                for c in _pcalls(tc)
            ),
        ),
        notes="return_images is an array ['figure','table']. Models pass True instead.",
    ),

    # --- Batch C: Multi-step pipelines ---
    Probe(
        id="upload_reuse",
        param="upload → reducto:// → reuse across calls",
        prompt=(
            "I have a financial report at this presigned S3 URL that expires in 3 minutes: "
            f"{TEST_DOC}. "
            "I need to: 1) parse the full document, 2) extract the portfolio summary table, "
            "3) classify whether it's a quarterly or annual report. "
            "Make sure you don't re-download the source file for each operation."
        ),
        check=lambda tc: (
            any("upload" in c["tool"] for c in tc),
            any("upload" in c["tool"] for c in tc) and
            any(
                str(c["args"].get("input", "")).startswith("reducto://")
                for c in _pcalls(tc) + _ecalls(tc) + _ccalls(tc)
            ),
        ),
        notes="upload first, reuse reducto:// ID. Models re-download source URL each time.",
    ),
    Probe(
        id="classify_then_route",
        param="classify → choose schema → extract",
        prompt=(
            f"I receive many different document types. For {TEST_DOC}, first figure out "
            "what type of document it is (financial statement vs invoice vs contract), "
            "then extract the fields appropriate for that document type."
        ),
        check=lambda tc: (
            len(_ccalls(tc)) > 0 and len(_ecalls(tc)) > 0,
            # classify must come before extract in the call sequence
            len(_ccalls(tc)) > 0 and len(_ecalls(tc)) > 0 and
            next((i for i, c in enumerate(tc) if "classify" in c["tool"]), 999) <
            next((i for i, c in enumerate(tc) if "extract" in c["tool"]), 999),
        ),
        notes="classify → route → extract. Models skip classification and go straight to extract.",
    ),
    Probe(
        id="split_then_extract_section",
        param="split → use page range → extract section only",
        prompt=(
            f"From {TEST_DOC}, I only want to extract data from the 'Portfolio Holdings' section. "
            "First identify which pages that section spans, then extract only from those pages."
        ),
        check=lambda tc: (
            len(_scalls(tc)) > 0,
            len(_scalls(tc)) > 0 and len(_ecalls(tc)) > 0,
        ),
        notes="split to find section pages, then targeted extract. Models parse whole doc.",
    ),
    Probe(
        id="edit_basic_fill",
        param="reducto_edit with edit_instructions",
        prompt=(
            f"Fill in this PDF form at {TEST_DOC}: set the account holder name to 'Jane Smith' "
            "and the date to today. Return the filled PDF."
        ),
        check=lambda tc: (
            any("edit" in c["tool"] for c in tc),
            any(
                "edit" in c["tool"] and c["args"].get("edit_instructions")
                for c in tc
            ),
        ),
        notes="Models parse+describe instead of using /edit. Edit endpoint almost never known.",
    ),
    Probe(
        id="edit_form_schema",
        param="form_schema_json with EditWidget objects",
        prompt=(
            f"Fill this PDF form at {TEST_DOC} with exactly these values: "
            "Name='John Smith', Date='2026-03-28', Amount='$45,231.00', "
            "Account='X-7842', Signature='approved'. "
            "Use structured field definitions for precision."
        ),
        check=lambda tc: (
            any("edit" in c["tool"] for c in tc),
            any(
                "edit" in c["tool"] and c["args"].get("form_schema_json")
                for c in tc
            ),
        ),
        notes="form_schema_json for multi-field structured fill. Completely Reducto-specific.",
    ),
    Probe(
        id="edit_flatten_lock",
        param="flatten=True to lock filled PDF",
        prompt=(
            f"Fill in the form at {TEST_DOC} with the recipient name 'Acme Corp' "
            "and then lock it so the recipient cannot modify the filled values — "
            "this is a legally binding document."
        ),
        check=lambda tc: (
            any("edit" in c["tool"] for c in tc),
            any(
                "edit" in c["tool"] and c["args"].get("flatten") is True
                for c in tc
            ),
        ),
        notes="flatten=True converts form fields to static text. Models won't know this param.",
    ),

    # --- Batch D: Error recovery ---
    Probe(
        id="empty_result_ocr_retry",
        param="empty parse → retry with ocr mode",
        prompt=(
            "I tried to parse a scanned PDF and got this result: "
            '{"job_id": "abc-123", "result": {"chunks": []}, "usage": {"num_pages": 5}}. '
            "The chunks array is empty — no content was extracted. "
            f"Now parse {TEST_DOC} using the appropriate fallback strategy."
        ),
        check=lambda tc: (
            any(c["args"].get("extraction_mode") for c in _pcalls(tc)),
            any(c["args"].get("extraction_mode") == "ocr" for c in _pcalls(tc)),
        ),
        notes="Empty chunks = scanned doc. Retry with extraction_mode='ocr'.",
    ),
    Probe(
        id="classify_page_limit",
        param="classify page_range max 10 pages",
        prompt=(
            f"Classify this 80-page contract at {TEST_DOC} — determine if it's an NDA, "
            "MSA, SOW, or employment agreement. Be cost-efficient."
        ),
        check=lambda tc: (
            len(_ccalls(tc)) > 0,
            any(
                "page_range" in str(c["args"]) or "page_range_start" in c["args"]
                for c in _ccalls(tc)
            ),
        ),
        notes="classify has 10-page max. Cost-efficient model sets page_range on large docs.",
    ),
    Probe(
        id="jobid_expiry_recovery",
        param="404 on jobid:// → re-parse original",
        prompt=(
            "I have a jobid from a parse I ran 2 hours ago: jobid://stale-job-id-abc123. "
            f"Run extract on it to get the account number. The original document is at {TEST_DOC}."
        ),
        check=lambda tc: (
            # model should attempt something (either tries jobid and then falls back, or goes direct to URL)
            len(tc) > 0,
            # correct = model parses the original URL (not just tries the stale jobid and gives up)
            any(TEST_DOC in str(c["args"].get("input", "")) for c in _pcalls(tc) + _ecalls(tc)),
        ),
        notes="Stale jobid:// (>1hr) returns 404. Model must re-parse original URL.",
    ),
    Probe(
        id="table_cutoff_preserve",
        param="table_cutoff='preserve' in split",
        prompt=(
            f"Split {TEST_DOC} into sections: 'Account Summary', 'Portfolio Holdings', 'Income'. "
            "Make sure no table gets cut in half at a section boundary — "
            "if a table spans a boundary, keep it whole in the section where it started."
        ),
        check=lambda tc: (
            len(_scalls(tc)) > 0,
            any(
                "table_cutoff" in c["args"] and c["args"]["table_cutoff"] == "preserve"
                for c in _scalls(tc)
            ),
        ),
        notes="table_cutoff='preserve' keeps tables whole at split boundaries.",
    ),
    Probe(
        id="get_job_poll",
        param="reducto_get_job for async polling",
        prompt=(
            "I submitted a large 100-page document for parsing. The job ID is "
            "'parse-job-abc123-def456'. Check if it's done and retrieve the result."
        ),
        check=lambda tc: (
            any("get_job" in c["tool"] for c in tc),
            any(
                "get_job" in c["tool"] and "abc123" in str(c["args"].get("job_id", ""))
                for c in tc
            ),
        ),
        notes="reducto_get_job for polling async results. Models try to re-call parse.",
    ),
]

# Combined set (standard 10 + hard 23 = 33 probes = 99 pts max)
ALL_PROBES = PROBES + HARD_PROBES


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _build_llm(model: ModelConfig):
    if model.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kwargs: dict = dict(model=model.id, max_tokens=16000)
        if model.reasoning:
            # Extended thinking: temperature must be 1, and budget_tokens sets depth.
            kwargs["temperature"] = 1
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": model.budget_tokens}
            kwargs["extra_headers"] = {"anthropic-beta": "interleaved-thinking-2025-05-14"}
        else:
            kwargs["temperature"] = 0
        return ChatAnthropic(**kwargs)

    if model.provider == "local":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model.id,
            base_url=model.local_base_url,
            api_key="local",
            temperature=0,
            max_tokens=8192,   # generous budget for thinking tokens + tool calls
            timeout=180,
        )

    if model.provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs: dict = dict(
            model=model.id,
            max_tokens=16384,   # generous for reasoning + full tool responses
        )
        if model.reasoning:
            kwargs["reasoning_effort"] = model.reasoning_effort
        else:
            kwargs["temperature"] = 0
        return ChatOpenAI(**kwargs)

    # OpenRouter
    from langchain_openai import ChatOpenAI
    extra_body: dict = {}
    if model.or_provider_order:
        extra_body["provider"] = {"order": model.or_provider_order, "allow_fallbacks": True}
    if model.or_quantization:
        extra_body.setdefault("provider", {})["quantizations"] = [model.or_quantization]
    return ChatOpenAI(
        model=model.id,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        temperature=0,
        max_tokens=16384,   # generous; some models emit long CoT before tool calls
        timeout=300,        # slow reasoning models (Qwen, Devstral) can take 4+ min per probe
        model_kwargs={"extra_body": extra_body} if extra_body else {},
    )


def run_probe(model: ModelConfig, probe: Probe) -> ProbeResult:
    from langchain.agents import create_agent

    result = ProbeResult(model=model.display, probe_id=probe.id, param=probe.param)
    print(f"  [{model.display[:30]}] {probe.id} …")
    start = time.time()

    try:
        tools = _build_probe_tools()
        llm = _build_llm(model)
        agent = create_agent(model=llm, tools=tools)

        raw = agent.invoke({"messages": [{"role": "user", "content": probe.prompt}]})
        messages = raw.get("messages", [])
        tool_calls = _extract_tool_calls(messages)
        result.tool_calls_made = tool_calls

        present, correct = probe.check(tool_calls)
        result.param_present = present
        result.param_correct = correct

        # api_accepted: at least one tool response didn't start with "Error:"
        tool_msgs = [m for m in messages if type(m).__name__ == "ToolMessage"]
        result.api_accepted = any(
            not str(getattr(m, "content", "")).startswith("Error:")
            for m in tool_msgs
        ) if tool_msgs else False

    except Exception as e:
        err_str = str(e)
        # Detect OpenRouter credit exhaustion (402) — NOT a real model failure
        if "402" in err_str or "Insufficient credits" in err_str:
            import time as _t
            print(f"    !! {model.display}/{probe.id}: Credit error (402) — retrying in 15s…")
            _t.sleep(15)
            # Retry once
            try:
                raw = agent.invoke({"input": probe.prompt})
                messages = raw.get("messages", [])
                tool_calls = _extract_tool_calls(messages)
                result.tool_calls_made = tool_calls
                present, correct = probe.check(tool_calls)
                result.param_present = present
                result.param_correct = correct
                tool_msgs = [m for m in messages if type(m).__name__ == "ToolMessage"]
                result.api_accepted = any(
                    not str(getattr(m, "content", "")).startswith("Error:")
                    for m in tool_msgs
                ) if tool_msgs else False
                result.error = ""
            except Exception as e2:
                err_str2 = str(e2)
                result.error = err_str2
                if "402" in err_str2 or "Insufficient credits" in err_str2:
                    result.credit_error = True
                    print(f"    !! {model.display}/{probe.id}: Still 402 after retry — marking as N/A (not a model failure)")
                else:
                    print(f"    !! {model.display}/{probe.id}: {e2}")
        else:
            result.error = err_str
            print(f"    !! {model.display}/{probe.id}: {e}")

    result.score = int(result.param_present) + int(result.param_correct) + int(result.api_accepted)
    elapsed = time.time() - start
    status = "N/A" if result.credit_error else f"{result.score}/3"
    print(f"    → {status} ({elapsed:.1f}s) | present={result.param_present} correct={result.param_correct} api={result.api_accepted}")
    return result


# ---------------------------------------------------------------------------
# Async parallel runner
# ---------------------------------------------------------------------------

async def run_all(models: list[ModelConfig], probes: list[Probe]) -> list[ProbeResult]:
    loop = asyncio.get_running_loop()
    # One thread per concurrent probe — semaphore is the real throttle
    n_concurrent = max(len(models) * 2, 20)
    executor = ThreadPoolExecutor(max_workers=n_concurrent)
    # Round-robin across models (probe-first order) so every model gets slots
    # immediately rather than one model monopolising the semaphore queue.
    sem = asyncio.Semaphore(n_concurrent)

    async def bounded(m: ModelConfig, p: Probe):
        async with sem:
            return await loop.run_in_executor(executor, run_probe, m, p)

    # Interleave: probe1 for all models, probe2 for all models, …
    tasks = [bounded(m, p) for p in probes for m in models]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)
    executor.shutdown(wait=False)
    return [r for r in all_results if isinstance(r, ProbeResult)]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list[ProbeResult], models: list[ModelConfig], probes: list[Probe]):
    print("\n" + "=" * 90)
    print("PARAM PROBE RESULTS  (✓ = point earned, · = missed)")
    print("Each probe: [param_present][param_correct][api_accepted] = 0–3 pts")
    print("=" * 90)

    col = 22
    header = f"{'Probe':<22} {'Param':<32}"
    for m in models:
        short = (m.display.split()[0] + " " + m.display.split()[1])[:col]
        header += f"  {short:>{col}}"
    print(header)
    print("-" * (54 + (col + 2) * len(models)))

    for probe in probes:
        row = f"{probe.id:<22} {probe.param[:31]:<32}"
        for model in models:
            r = next((x for x in results if x.model == model.display and x.probe_id == probe.id), None)
            if r:
                sym = (
                    ("✓" if r.param_present else "·")
                    + ("✓" if r.param_correct else "·")
                    + ("✓" if r.api_accepted else "·")
                    + f" ({r.score}/3)"
                )
                row += f"  {sym:>{col}}"
            else:
                row += f"  {'—':>{col}}"
        print(row)

    max_pts = len(probes) * 3
    print("-" * (54 + (col + 2) * len(models)))
    totrow = f"{'TOTAL (max ' + str(max_pts) + ')':<54}"
    for model in models:
        total = sum(r.score for r in results if r.model == model.display)
        totrow += f"  {str(total) + '/' + str(max_pts):>{col}}"
    print(totrow)

    # Per-probe insight summary
    print("\n--- Key Findings ---")
    for probe in probes:
        probe_results = [r for r in results if r.probe_id == probe.id]
        n_correct = sum(1 for r in probe_results if r.param_correct)
        n_total = len(probe_results)
        rating = "✅ all models" if n_correct == n_total else f"⚠️  {n_correct}/{n_total} models" if n_correct > 0 else "❌ no models"
        print(f"  {probe.id:<22} {rating}  — {probe.notes}")


def save_results(results: list[ProbeResult], probe_set: str = "standard"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Use separate subdirs for standard vs hard probe sets to prevent overwrites.
    if probe_set == "hard":
        subdir = "by_model_hard"
    elif probe_set == "all":
        subdir = "by_model_all"
    else:
        subdir = "by_model"

    # Group results by model so each model gets its own file (supports --all-models).
    from collections import defaultdict
    by_model: dict[str, list] = defaultdict(list)
    for r in results:
        # Redact schema_json from saved args (too verbose)
        clean_calls = [
            {"tool": tc["tool"], "args": {k: v for k, v in tc["args"].items() if k != "schema_json"}}
            for tc in r.tool_calls_made
        ]
        by_model[r.model].append({
            "model": r.model,
            "probe_id": r.probe_id,
            "param": r.param,
            "param_present": r.param_present,
            "param_correct": r.param_correct,
            "api_accepted": r.api_accepted,
            "score": r.score,
            "tool_calls": clean_calls,
            "error": r.error,
            "credit_error": r.credit_error,
        })

    for model_name, out in by_model.items():
        # Skip saving models where every probe was a credit error (not a real result)
        real_results = [r for r in out if not r.get("credit_error")]
        if not real_results:
            print(f"  ⚠ Skipping save for {model_name} — all probes were credit errors (402)")
            continue
        import re as _re
        slug = _re.sub(r'[^a-z0-9_]', '_', model_name.lower())[:40]
        path = f"{RESULTS_DIR}/{subdir}/{slug}.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(real_results, f, indent=2)
        print(f"Saved {model_name} → {path}")

    # Merge all per-model files into combined results JSON
    all_results = []
    for fn in sorted(os.listdir(f"{RESULTS_DIR}/{subdir}")):
        if fn.endswith(".json"):
            with open(f"{RESULTS_DIR}/{subdir}/{fn}") as f:
                all_results.extend(json.load(f))
    with open(f"{RESULTS_DIR}/probe_results_{probe_set}.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results → {RESULTS_DIR}/probe_results_{probe_set}.json ({len(all_results)} entries)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reducto advanced param probe")
    parser.add_argument("--model", help="Filter to model ID substring (e.g. 'mimo', 'qwen3-coder', 'o4')")
    parser.add_argument("--all-models", action="store_true", help="Run all models (PREMIUM + OPENROUTER)")
    parser.add_argument("--probe", help="Run single probe by ID (e.g. 'ocr_mode', 'array_extract')")
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip models that already have a complete result file in by_model_hard/")
    parser.add_argument(
        "--probe-set",
        choices=["standard", "hard", "all"],
        default="standard",
        help="Which probe set to run: standard (10 probes), hard (23 probes), all (33 probes)",
    )
    args = parser.parse_args()

    if not os.environ.get("REDUCTO_API_KEY"):
        print("Error: REDUCTO_API_KEY not set")
        sys.exit(1)

    try:
        from langchain.agents import create_agent  # noqa
        from langchain_openai import ChatOpenAI    # noqa
    except ImportError:
        print("Missing deps: pip install langchain langchain-openai requests")
        sys.exit(1)

    from models import LOCAL_MODELS
    if args.all_models:
        models = PREMIUM_MODELS + OPENROUTER_MODELS
    elif args.model:
        # Search full registry when a filter is given
        all_models = OPENROUTER_MODELS + PREMIUM_MODELS + LOCAL_MODELS
        models = [m for m in all_models if args.model.lower() in m.id.lower() or args.model.lower() in m.display.lower()]
    else:
        models = get_probe_models()

    # --skip-done: drop models that already have a clean (non-credit-error) result file
    if args.skip_done:
        subdir = "by_model_hard" if (args.probe_set == "hard") else "by_model"
        done_slugs = set()
        done_dir = f"{RESULTS_DIR}/{subdir}"
        if os.path.isdir(done_dir):
            for fn in os.listdir(done_dir):
                if fn.endswith(".json"):
                    done_slugs.add(fn[:-5])
        def _slug(name: str) -> str:
            import re
            return re.sub(r'[^a-z0-9_]', '_', name.lower())[:40]
        before = len(models)
        models = [m for m in models if _slug(m.display) not in done_slugs]
        skipped = before - len(models)
        if skipped:
            print(f"  --skip-done: skipping {skipped} model(s) with existing results.")

    if not models:
        print(f"No models matched '{args.model}'.")
        sys.exit(1)

    # Select probe set
    if args.probe_set == "hard":
        probe_pool = HARD_PROBES
    elif args.probe_set == "all":
        probe_pool = ALL_PROBES
    else:
        probe_pool = PROBES

    probes = probe_pool
    if args.probe:
        probes = [p for p in probe_pool if p.id == args.probe]
    if not probes:
        print(f"No probe matched '{args.probe}'. Available: {[p.id for p in probe_pool]}")
        sys.exit(1)

    print(f"\nParam probe: {len(models)} models × {len(probes)} probes = {len(models) * len(probes)} runs\n")
    for m in models:
        print(f"  • {m.display}")
    print()

    results = asyncio.run(run_all(models, probes))
    print_report(results, models, probes)
    save_results(results, probe_set=args.probe_set)


if __name__ == "__main__":
    main()
