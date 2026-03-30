"""
Reducto Agent Benchmark — Core framework.

Shared scoring model, test documents, prompts, ground truth, and evaluators
used by every per-platform and per-integration script.
"""

import json
import time
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Test documents
# ---------------------------------------------------------------------------

TEST_DOC_URL = "https://cdn.reducto.ai/samples/fidelity-example.pdf"  # backward compat

TEST_DOCS = {
    "financial": "https://cdn.reducto.ai/samples/fidelity-example.pdf",
    # Add more as Reducto publishes sample docs:
    # "invoice":   "https://cdn.reducto.ai/samples/sample-invoice.pdf",
    # "contract":  "https://cdn.reducto.ai/samples/sample-contract.pdf",
}

# ---------------------------------------------------------------------------
# Ground truth (from our actual API run on 2026-03-27)
# ---------------------------------------------------------------------------

GROUND_TRUTH = {
    "financial": {
        "url": TEST_DOCS["financial"],
        "document_type": "financial_statement",
        "expected_pages": 3,
        "expected_tables": 5,
        "portfolio_beginning": 253221.83,
        "portfolio_ending": 274222.20,
        "account_number": "111-111111",
        "account_type": "GENERAL INVESTMENTS",
        "top_holding_name": "Johnson & Johnson",
        "top_holding_ticker": "JNJ",
        "top_holding_value": 47113.80,
        "top_holding_pct": 17,
        "second_holding_ticker": "AAPL",
        "second_holding_value": 28892.05,
        "expected_split_sections": ["Cover Page", "Account Summary", "Holdings", "Income Summary"],
    }
}

EXTRACT_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "portfolio_value": {
            "type": "object",
            "description": "Beginning and ending portfolio values from the summary table",
            "properties": {
                "beginning": {"type": "number", "description": "Beginning portfolio value in dollars"},
                "ending":    {"type": "number", "description": "Ending portfolio value in dollars"},
            }
        },
        "account_number": {"type": "string", "description": "The brokerage account number"},
        "account_type":   {"type": "string", "description": "Account type e.g. GENERAL INVESTMENTS"},
        "income_summary": {
            "type": "object",
            "description": "Income breakdown by tax category (taxable, tax-exempt, etc.)"
        },
        "top_holdings": {
            "type": "array",
            "description": "Top holdings by value",
            "items": {
                "type": "object",
                "properties": {
                    "name":       {"type": "string"},
                    "value":      {"type": "number", "description": "Market value in dollars"},
                    "percentage": {"type": "number", "description": "Portfolio percentage 0-100"},
                }
            }
        }
    }
})

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkScore:
    """
    Score a single (platform, integration_path, capability) combination.

    8 dimensions × 1-5 scale = 40 max.
    Dimensions scored 0 are excluded from the total (not applicable).

    Scoring rubrics
    ---------------
    discovery          5 = found Reducto zero-shot; 3 = needed endpoint name; 1 = needed full example
    setup_friction     5 = one command; 4 = one config file; 3 = 2-3 steps; 2 = 4-6 steps; 1 = 7+
    integration_complexity  5 = ≤5 LOC/1 call; 4 = 6-15/2-3; 3 = 16-30/4-6; 2 = 31-60/7-12; 1 = 60+
    feature_coverage   5 = all 7 capabilities; 4 = 5-6; 3 = 3-4; 2 = 1-2; 1 = none without workaround
    error_recovery     5 = auto-retry + clean error; 3 = surfaced error, needed human; 1 = silent fail
    output_quality     5 = all ground truth fields correct; 4 = 4/5 correct; 3 = 3/5; etc.
    token_efficiency   5 = minimal; 3 = reasonable; 1 = excessive (0 = not measurable, e.g. Ollama)
    mcp_compatibility  5 = correct discovery + invocation; 0 = N/A (non-MCP path)
    """

    # Identity — forms the result filename
    platform: str
    model: str
    integration_path: str   # mcp | python_sdk | node_sdk | rest_api | cli | workflow | framework
    capability: str         # parse | extract | split | classify | edit | upload
                            # | jobid_chaining | async_polling | error_handling

    # Scored dimensions (1-5 or 0 = N/A)
    discovery:              int = 0
    setup_friction:         int = 0
    integration_complexity: int = 0
    feature_coverage:       int = 0
    error_recovery:         int = 0
    output_quality:         int = 0
    token_efficiency:       int = 0
    mcp_compatibility:      int = 0

    # Raw metadata
    total_tokens:       int   = 0
    wall_time_seconds:  float = 0.0
    tool_calls_count:   int   = 0
    lines_of_code:      int   = 0
    errors_encountered: list  = field(default_factory=list)
    notes:              str   = ""

    @property
    def scored_dimensions(self) -> list[int]:
        return [v for v in [
            self.discovery, self.setup_friction, self.integration_complexity,
            self.feature_coverage, self.error_recovery, self.output_quality,
            self.token_efficiency, self.mcp_compatibility,
        ] if v > 0]

    @property
    def total_score(self) -> int:
        return sum(self.scored_dimensions)

    @property
    def max_score(self) -> int:
        return len(self.scored_dimensions) * 5

    def result_key(self) -> str:
        # "/" in platform (e.g. "LangChain/ModelName") becomes "__" so it stays flat
        plat = (self.platform.lower()
                .replace("/", "__")
                .replace(" ", "_")
                .replace(".", "_")
                .replace("(", "")
                .replace(")", ""))
        return f"{plat}__{self.integration_path}__{self.capability}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_score"] = self.total_score
        d["max_score"] = self.max_score
        d["result_key"] = self.result_key()
        return d


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_URL = TEST_DOCS["financial"]
_SCHEMA = EXTRACT_SCHEMA

PROMPTS = {
    # --- Discovery (no hints — agent must figure out Reducto exists) ---
    "discovery": (
        f"A financial statement PDF is available at {_URL}. "
        "Extract the portfolio values, account number, account type, and top holdings. "
        "Return structured JSON."
    ),

    # --- Guided (endpoint named, no implementation details) ---
    "parse": (
        f"Use Reducto to parse this document: {_URL}\n"
        "Show all extracted text and identify every table in the output."
    ),
    "extract": (
        f"Use Reducto to extract structured data from: {_URL}\n\n"
        "Fields needed:\n"
        "- portfolio_value (beginning and ending amounts in dollars)\n"
        "- account_number\n"
        "- account_type\n"
        "- top_holdings (array with name, value, percentage)\n\n"
        f"Use this JSON Schema:\n{_SCHEMA}\n\n"
        "Return the extracted JSON."
    ),
    "split": (
        f"Use Reducto to split {_URL} into logical sections.\n"
        "Sections to find: Cover Page, Account Summary, Holdings, Income Summary.\n"
        "Return page ranges per section."
    ),
    "classify": (
        f"Use Reducto to classify {_URL} as one of: "
        "financial_statement, invoice, contract, form, other.\n"
        "Explain what signals led to the classification."
    ),
    "edit": (
        f"Use Reducto to fill the form at {_URL}.\n"
        "Instructions: Fill Name: John Doe, Date: 2026-01-01\n"
        "Return the URL of the edited document."
    ),
    "jobid_chaining": (
        f"Use Reducto to process {_URL} efficiently:\n"
        "1. Parse it with persist_results=true — note the job_id\n"
        "2. Run extract using jobid://JOB_ID (not the original URL) "
        f"with schema: {_SCHEMA}\n"
        "3. Run split on the same jobid://JOB_ID, sections: Cover Page, Account Summary, Holdings\n\n"
        "The point: parse once, reuse the result twice. "
        "Show the job IDs used and confirm you didn't re-parse."
    ),
    "async_polling": (
        f"Use Reducto's async mode to parse {_URL}.\n"
        "Start the job (POST /parse_async), poll GET /job/{{job_id}} until complete, "
        "then print the extracted content."
    ),
    "error_handling": (
        "Test Reducto's error handling with three scenarios:\n"
        "1. Call POST /parse with Authorization: Bearer totally_invalid_key_xyz — note the 401\n"
        f"2. Call POST /parse with input: 'https://this-does-not-exist.example.com/fake.pdf' — note the error\n"
        "3. Call POST /extract with a malformed JSON schema (missing closing brace)\n\n"
        "Document the exact error messages returned and how you recovered from each."
    ),

    # --- MCP-specific (for MCP-supporting agents) ---
    "mcp_parse": (
        f"Use the reducto_parse MCP tool on {_URL}.\n"
        "Set merge_tables=true and filter_blocks=['Header', 'Footer', 'Page Number'].\n"
        "Report how many tables and figures were found."
    ),
    "mcp_extract": (
        f"Use reducto_parse on {_URL} with persist_results=true.\n"
        "Then call reducto_extract with input='jobid://JOB_ID' (use the actual job ID) "
        f"and schema={_SCHEMA}\n"
        "Show that jobid:// avoids re-parsing the document."
    ),
    "mcp_full_pipeline": (
        f"Use the Reducto MCP tools to run a full pipeline on {_URL}:\n"
        "1. reducto_parse with persist_results=true, merge_tables=true\n"
        f"2. reducto_extract (jobid://) with schema: {_SCHEMA}\n"
        "3. reducto_split (jobid://) into: Cover Page, Account Summary, Holdings, Income Summary\n"
        "4. reducto_classify on the original URL\n\n"
        "Report all results and confirm steps 2+3 used jobid:// not the original URL."
    ),
}

# ---------------------------------------------------------------------------
# Evaluators — return (quality_score 1-5, details_dict)
# ---------------------------------------------------------------------------

def _try_extract_json(text: str) -> Optional[dict]:
    """Pull the first valid JSON object out of a string."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


def evaluate_parse_output(output: str) -> tuple[int, dict]:
    """Score parse output quality against ground truth."""
    gt = GROUND_TRUTH["financial"]
    lo = output.lower()

    details = {
        "has_portfolio_value": any(x in lo for x in ["274,222", "253,221", "portfolio value"]),
        "has_table_content":   any(x in lo for x in ["beginning", "ending", "additions"]),
        "has_account_info":    any(x in lo for x in ["brokerage", "fidelity", "111-111111"]),
        "has_holdings":        any(x in lo for x in ["johnson", "apple", "aapl", "jnj"]),
        "has_markdown_structure": "#" in output or "---" in output or "##" in output,
        "table_count_mentioned": any(str(n) in output for n in range(3, 8)),
    }
    score = sum(1 for v in details.values() if v)
    return max(1, min(5, score)), details


def evaluate_extract_output(output: str) -> tuple[int, dict]:
    """Score extract output against ground truth numeric fields."""
    gt = GROUND_TRUTH["financial"]
    lo = output.lower()

    # Try to find numeric values within 1% tolerance
    def near(text: str, value: float, tol: float = 0.01) -> bool:
        # Look for the number formatted as integer or decimal
        patterns = [
            str(int(value)),
            f"{value:.2f}",
            f"{value:,.2f}",
            f"{value:,.0f}",
        ]
        return any(p in text for p in patterns)

    details = {
        "portfolio_beginning_correct": near(output, gt["portfolio_beginning"]),
        "portfolio_ending_correct":    near(output, gt["portfolio_ending"]),
        "account_number_found":        gt["account_number"] in output,
        "account_type_found":          "general investments" in lo or "general" in lo,
        "top_holding_found":           "johnson" in lo or "jnj" in lo,
        "is_structured_json":          _try_extract_json(output) is not None,
    }
    score = sum(1 for v in details.values() if v)
    return max(1, min(5, round(score * 5 / len(details)))), details


def evaluate_split_output(output: str) -> tuple[int, dict]:
    gt = GROUND_TRUTH["financial"]
    lo = output.lower()
    details = {
        s.lower(): s.lower() in lo
        for s in gt["expected_split_sections"]
    }
    details["has_page_numbers"] = bool(re.search(r"page[s]?\s*\d+", lo) or re.search(r"\[?\d+\]?", output))
    score = sum(1 for v in details.values() if v)
    return max(1, min(5, round(score * 5 / len(details)))), details


def evaluate_classify_output(output: str) -> tuple[int, dict]:
    lo = output.lower()
    correct = "financial" in lo or "statement" in lo
    details = {
        "correct_category": correct,
        "has_reasoning": len(output) > 100,
        "mentions_fidelity": "fidelity" in lo,
    }
    score = 5 if correct and details["has_reasoning"] else (3 if correct else 1)
    return score, details


def evaluate_error_handling(output: str) -> tuple[int, dict]:
    lo = output.lower()
    details = {
        "caught_401": "401" in output or "unauthorized" in lo or "invalid" in lo,
        "caught_bad_url": any(x in lo for x in ["not found", "error", "failed", "404", "invalid"]),
        "caught_schema_error": any(x in lo for x in ["schema", "json", "invalid", "error"]),
        "shows_recovery": any(x in lo for x in ["retry", "handled", "recovered", "fallback", "caught"]),
    }
    score = max(1, sum(1 for v in details.values() if v) + 1)  # 2-5
    return min(5, score), details


def evaluate_output(output: str, capability: str) -> tuple[int, dict]:
    """Dispatch to the right evaluator."""
    if capability == "parse":
        return evaluate_parse_output(output)
    if capability == "extract":
        return evaluate_extract_output(output)
    if capability == "split":
        return evaluate_split_output(output)
    if capability == "classify":
        return evaluate_classify_output(output)
    if capability == "error_handling":
        return evaluate_error_handling(output)
    # Default: keyword presence check
    lo = output.lower()
    found = sum(1 for kw in ["reducto", "parse", "extract", "fidelity", "portfolio"]
                if kw in lo)
    return max(1, min(5, found)), {"keyword_hits": found}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent.parent / "results"


def save_result(score: BenchmarkScore, subdir: str = "", raw_output: str = "") -> Path:
    """Save score to results/{subdir}/{result_key}.json."""
    out_dir = RESULTS_DIR / subdir if subdir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{score.result_key()}.json"

    data = score.to_dict()
    if raw_output:
        data["raw_output"] = raw_output[:5000]

    out_path.write_text(json.dumps(data, indent=2))
    print(f"  ✅  {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

class PhaseTimer:
    def __init__(self, name: str, phases: list):
        self.name = name
        self.phases = phases

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *_):
        self.phases.append({"phase": self.name, "duration_s": round(time.time() - self._start, 2)})


# ---------------------------------------------------------------------------
# Standard prompts kept for backward compat
# ---------------------------------------------------------------------------

STANDARD_PROMPT = PROMPTS["extract"]
STANDARD_PROMPT_WITH_MCP = PROMPTS["mcp_full_pipeline"]
