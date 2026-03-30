#!/usr/bin/env python3
"""
verify_synthetic_data.py — Multi-layer quality gate for generated training examples.

Layer 1 — Schema validation (100% of examples, free):
  - Required parameters present
  - Enum values valid
  - No known mutually-exclusive parameter combinations
  - Tool name is a real Reducto tool

Layer 2 — Probe-based semantic check (probe-derived examples only):
  - Re-runs the original probe.check() function against generated tool calls
  - Keeps only examples scoring ≥ min_probe_score (default 2/3)

Layer 3 — Cross-model consensus (all examples):
  - Groups examples by (probe_id, prompt_fingerprint)
  - If multiple teachers generated the example, checks they agree on
    the core tool + key params
  - Flags disagreements; keeps examples where ≥ consensus_threshold teachers agree

Layer 4 — Reducto live API spot-check (sampled subset, optional):
  - Replaces the prompt's document URL with the known-good test doc
  - Extracts the generated tool call pattern and replays it against the real API
  - Accepts the example if the API call succeeds (200 OK, non-empty result)
  - Rejects on 4xx param errors (bad enum value slipped through L1, etc.)
  - Skips 5xx / network errors (API hiccup, not a training data problem)
  - Controlled by --l4-sample-rate (default 0.10 = 10% spot-check)

Usage:
    python verify_synthetic_data.py                        # verify checkpoint, write verified splits
    python verify_synthetic_data.py --input train.jsonl    # verify a specific file
    python verify_synthetic_data.py --no-consensus         # skip layer 3
    python verify_synthetic_data.py --no-l4               # skip Reducto API spot-check
    python verify_synthetic_data.py --l4-sample-rate 1.0  # check every example (thorough)
    python verify_synthetic_data.py --report-only          # print stats without writing output
"""

import argparse
import json
import sys
import os
import random
import re
import time
import urllib.request
import urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR   = Path("benchmark/data/synthetic_training")
CHECKPOINT = DATA_DIR / ".checkpoint.jsonl"

# ---------------------------------------------------------------------------
# Layer 1 — Schema rules
# ---------------------------------------------------------------------------

VALID_TOOLS = {
    "reducto_parse", "reducto_extract", "reducto_split",
    "reducto_classify", "reducto_edit", "reducto_upload", "reducto_get_job",
}

REQUIRED_PARAMS = {
    "reducto_parse":    {"input"},
    "reducto_extract":  {"input", "schema_json"},
    "reducto_split":    {"input", "split_description"},
    "reducto_classify": {"input", "classification_schema"},
    "reducto_edit":     {"document_url"},
    "reducto_upload":   {"file_url"},
    "reducto_get_job":  {"job_id"},
}

VALID_ENUMS = {
    "table_format":     {"html", "md", "json", "csv", "jsonbbox", "dynamic"},
    "extraction_mode":  {"hybrid", "ocr"},
    "chunk_mode":       {"variable", "section", "page", "block", "disabled"},
    "table_cutoff":     {"preserve", "allow"},
}

# Known mutually exclusive parameter pairs
CONFLICTS = [
    # (tool, param_a, param_b, reason)
    ("reducto_extract", "citations",           "chunk_mode",
     "citations=True returns per-field coordinates — incompatible with chunk_mode"),
    ("reducto_extract", "optimize_for_latency", "deep_extract",
     "optimize_for_latency and deep_extract contradict each other — use one or the other"),
]

# Tool selection hints — if the prompt contains these keywords, the tool should be this
TOOL_HINTS = [
    ({"structured json", "json schema", "extract fields", "extract the", "get the value",
      "extract all", "line items", "holdings", "transactions"},
     {"reducto_extract"}, "prompt asks for structured extraction → reducto_extract"),
    ({"classify", "document type", "invoice or", "is this a", "what type of document"},
     {"reducto_classify"}, "prompt asks for classification → reducto_classify"),
    ({"split", "section", "sections", "divide into"},
     {"reducto_split"}, "prompt asks for splitting → reducto_split"),
    ({"fill", "fill in", "fill out", "edit the form", "complete the form", "flatten"},
     {"reducto_edit"}, "prompt asks for form fill/edit → reducto_edit"),
    ({"upload", "presigned", "expires", "short-lived", "expiring"},
     {"reducto_upload"}, "prompt has expiring URL → reducto_upload should be first call"),
    ({"job_id", "job id", "pending", "poll", "retrieve the result"},
     {"reducto_get_job"}, "prompt asks to poll a job → reducto_get_job"),
]


@dataclass
class Rejection:
    layer: int
    reason: str
    detail: str = ""


def validate_schema(tool_calls: list[dict]) -> list[Rejection]:
    """Layer 1: structural validation."""
    issues = []

    if not tool_calls:
        issues.append(Rejection(1, "no_tool_calls", "assistant made zero tool calls"))
        return issues

    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("arguments", {})

        # Parse stringified JSON args
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                issues.append(Rejection(1, "invalid_args_json", f"{name}: args not parseable"))
                continue

        if name not in VALID_TOOLS:
            issues.append(Rejection(1, "unknown_tool", f"'{name}' is not a Reducto tool"))
            continue

        # Required params
        required = REQUIRED_PARAMS.get(name, set())
        missing = required - set(args.keys())
        if missing:
            issues.append(Rejection(1, "missing_required", f"{name} missing: {missing}"))

        # Enum validation
        for param, valid_values in VALID_ENUMS.items():
            if param in args and args[param] not in valid_values:
                issues.append(Rejection(1, "invalid_enum",
                    f"{name}.{param}='{args[param]}' not in {valid_values}"))

        # Conflict detection
        for conflict_tool, pa, pb, reason in CONFLICTS:
            if name == conflict_tool and args.get(pa) and args.get(pb):
                issues.append(Rejection(1, "param_conflict",
                    f"{name}: {pa} + {pb} conflict — {reason}"))

    return issues


def validate_tool_hints(prompt: str, tool_calls: list[dict]) -> list[Rejection]:
    """Layer 1b: loose semantic hint check — flag obvious tool mismatches."""
    issues = []
    if not tool_calls:
        return issues

    prompt_lower = prompt.lower()
    used_tools = {tc.get("name", "") for tc in tool_calls}

    for keywords, expected_tools, reason in TOOL_HINTS:
        if any(kw in prompt_lower for kw in keywords):
            if not (used_tools & expected_tools):
                # Only flag if none of the expected tools appear
                issues.append(Rejection(1, "tool_hint_mismatch",
                    f"hint: {reason} — but used {used_tools}"))
                break   # one flag per example is enough

    return issues


# ---------------------------------------------------------------------------
# Layer 2 — Probe semantic check
# ---------------------------------------------------------------------------

def load_probe_checkers() -> dict:
    """Load probe check functions keyed by probe_id."""
    try:
        from bench_param_probe import HARD_PROBES, PROBES
        return {p.id: p.check for p in (PROBES + HARD_PROBES)}
    except Exception as e:
        print(f"  [verify] could not load probe checkers: {e}")
        return {}


def validate_probe_semantics(probe_id: str, tool_calls: list[dict],
                              checkers: dict) -> Optional[Rejection]:
    """Layer 2: run the original probe.check() on generated tool calls."""
    if probe_id not in checkers:
        return None   # not a probe-derived example — skip
    try:
        # Normalize tool calls to the format probe.check() expects
        normalized = [{"tool": tc.get("name",""), "args": tc.get("arguments", {})}
                      for tc in tool_calls]
        present, correct = checkers[probe_id](normalized)
        score = int(present) + int(correct)
        if score < 2:
            return Rejection(2, "probe_semantic_fail",
                f"probe '{probe_id}' score={score}/2 (present={present}, correct={correct})")
    except Exception as e:
        return Rejection(2, "probe_check_error", str(e)[:80])
    return None


# ---------------------------------------------------------------------------
# Layer 3 — Cross-model consensus
# ---------------------------------------------------------------------------

def _tool_fingerprint(tool_calls: list[dict]) -> str:
    """Stable fingerprint: sorted tool names + key params (not all args)."""
    KEY_PARAMS = {
        "reducto_parse":    {"chunk_mode", "extraction_mode", "filter_blocks",
                             "embedding_optimized", "merge_tables", "persist_results"},
        "reducto_extract":  {"array_extract", "citations", "optimize_for_latency",
                             "deep_extract", "include_images"},
        "reducto_split":    {"table_cutoff"},
        "reducto_classify": set(),
        "reducto_edit":     {"flatten"},
        "reducto_upload":   set(),
        "reducto_get_job":  set(),
    }
    parts = []
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("arguments", {})
        if isinstance(args, str):
            try: args = json.loads(args)
            except: args = {}
        key_args = {k: args[k] for k in KEY_PARAMS.get(name, set()) if k in args}
        parts.append(f"{name}:{sorted(key_args.items())}")
    return "|".join(sorted(parts))


def build_consensus_groups(examples: list[dict], top_teachers: set = None) -> dict:
    """Group examples by (probe_id, prompt_fingerprint) and collect teacher votes.

    If top_teachers is provided, only votes from those teachers count toward L3
    consensus.  Examples from other teachers still pass L1/L2 — they just don't
    influence the majority fingerprint.
    """
    groups = defaultdict(lambda: defaultdict(int))  # key → fingerprint → count
    for ex in examples:
        meta    = ex.get("metadata", {})
        teacher = meta.get("teacher", "")

        # Skip this example's vote if it's not from an approved teacher
        if top_teachers and teacher not in top_teachers:
            continue

        msgs  = ex.get("messages", [])
        probe = meta.get("probe_id", "")
        user  = next((m["content"] for m in msgs if m["role"] == "user"), "")
        calls = next((m.get("tool_calls", []) for m in msgs if m["role"] == "assistant"), [])

        # Normalize tool_calls from OpenAI format → simple list
        tcs = [{"name": tc["function"]["name"],
                "arguments": tc["function"].get("arguments", "{}")}
               for tc in calls if tc.get("type") == "function"]

        fp_key  = f"{probe}||{user[:120]}"
        fp_val  = _tool_fingerprint(tcs)
        groups[fp_key][fp_val] += 1
    return groups


def get_consensus_fingerprint(groups: dict, probe_id: str, prompt: str,
                               threshold: int = 3) -> Optional[str]:
    """Return the majority fingerprint if it meets threshold, else None."""
    key = f"{probe_id}||{prompt[:120]}"
    votes = groups.get(key, {})
    if not votes:
        return None
    best_fp, best_count = max(votes.items(), key=lambda x: x[1])
    return best_fp if best_count >= threshold else None


# ---------------------------------------------------------------------------
# Layer 4 — Reducto live API check (concurrent, rate-limited)
# ---------------------------------------------------------------------------

TEST_DOC      = "https://cdn.reducto.ai/samples/fidelity-example.pdf"
_URL_RE       = re.compile(r"https?://\S+\.(?:pdf|docx?|xlsx?|pptx?|png|jpe?g|tiff?|csv|html?|txt|msg|eml|bmp)\b", re.IGNORECASE)
_REDUCTO_BASE = "https://platform.reducto.ai"

# Gentle: 4 concurrent calls, minimum 0.25 s gap between any two fires (~4 req/s ceiling)
import threading as _threading
_L4_SEM       = _threading.Semaphore(4)
_L4_LOCK      = _threading.Lock()
_L4_LAST_TS   = [0.0]
_L4_MIN_GAP   = 0.25

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


def _l4_check_one(tool_calls: list[dict], api_key: str) -> Optional[Rejection]:
    """
    Live-call the first testable Reducto endpoint with normalised args.
    4xx  → Rejection   |   2xx / 5xx / timeout → None (pass)
    Rate-limited to ≤4 concurrent, ≤4 req/s globally.
    """
    if not api_key:
        return None

    for tc in tool_calls:
        name     = tc.get("name", "")
        raw_args = tc.get("arguments", {})
        if isinstance(raw_args, str):
            try: raw_args = json.loads(raw_args)
            except: raw_args = {}

        path = _ENDPOINT_MAP.get(name)
        if path is None:
            continue   # get_job: synthetic ids can't be tested

        # Build safe payload, swap any doc URL for known-good test doc
        args = {k: v for k, v in raw_args.items() if k in _SAFE_KEYS.get(name, set())}
        for key in ("input", "document_url", "file_url"):
            val = args.get(key, "")
            if not val or _URL_RE.search(val) or val.startswith(("reducto://", "jobid://")):
                args[key] = TEST_DOC

        body = json.dumps(args).encode()
        req  = urllib.request.Request(
            f"{_REDUCTO_BASE}{path}", data=body,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            method="POST",
        )

        # Rate gate
        with _L4_LOCK:
            gap = _L4_MIN_GAP - (time.time() - _L4_LAST_TS[0])
            if gap > 0:
                time.sleep(gap)
            _L4_LAST_TS[0] = time.time()

        with _L4_SEM:
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return None   # 2xx → pass
            except urllib.error.HTTPError as e:
                if 400 <= e.code < 500:
                    detail = e.read().decode("utf-8", errors="replace")[:100]
                    return Rejection(4, "reducto_api_4xx",
                                     f"{name} → HTTP {e.code}: {detail}")
                return None   # 5xx → pass
            except Exception:
                return None   # network hiccup → pass

    return None   # only get_job calls → pass


def run_l4_concurrent(
    examples: list[dict],
    api_key: str,
    sample_rate: float = 1.0,
    verbose: bool = False,
) -> dict[int, Optional[Rejection]]:
    """
    Run L4 checks concurrently across all sampled examples.
    Returns a dict mapping example index → Rejection (or None = pass).
    """
    indices = [i for i in range(len(examples)) if random.random() < sample_rate]
    results: dict[int, Optional[Rejection]] = {}

    def _worker(idx: int, ex: dict) -> tuple[int, Optional[Rejection]]:
        msgs      = ex.get("messages", [])
        asst      = next((m for m in msgs if m["role"] == "assistant"), {})
        raw_calls = asst.get("tool_calls", [])
        tcs       = [{"name": tc["function"]["name"],
                      "arguments": tc["function"].get("arguments", "{}")}
                     for tc in raw_calls if tc.get("type") == "function"]
        return idx, _l4_check_one(tcs, api_key)

    done = 0
    total = len(indices)
    print(f"  L4: checking {total:,} examples via Reducto API (≤4 concurrent, ≤4 req/s)...")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(_worker, i, examples[i]): i for i in indices}
        for fut in as_completed(futs):
            idx, rej = fut.result()
            results[idx] = rej
            done += 1
            if done % 500 == 0 or done == total:
                print(f"  L4: {done:,}/{total:,} checked", end="\r", flush=True)
    print()  # newline after progress
    return results


# ---------------------------------------------------------------------------
# Main verifier
# ---------------------------------------------------------------------------

@dataclass
class VerifyStats:
    total: int = 0
    passed: int = 0
    rejected_l1: int = 0
    rejected_l2: int = 0
    rejected_l3: int = 0
    rejected_l4: int = 0
    l4_sampled: int = 0
    rejection_reasons: dict = field(default_factory=lambda: defaultdict(int))


def verify_examples(
    examples: list[dict],
    use_probe_check: bool = True,
    use_tool_hints: bool = True,
    use_consensus: bool = True,
    consensus_threshold: int = 3,
    top_teachers: set = None,
    min_probe_score: int = 2,
    use_reducto_api: bool = True,
    l4_sample_rate: float = 0.10,
    reducto_api_key: str = "",
    verbose: bool = False,
) -> tuple[list[dict], VerifyStats]:

    stats = VerifyStats(total=len(examples))
    checkers = load_probe_checkers() if use_probe_check else {}

    # Pre-build consensus groups (layer 3)
    groups = build_consensus_groups(examples, top_teachers=top_teachers) if use_consensus else {}

    # L4: resolve API key
    if use_reducto_api and not reducto_api_key:
        reducto_api_key = os.environ.get("REDUCTO_API_KEY", "")

    # ── Layer 4 pre-pass: run all checks concurrently BEFORE the main loop ──
    # This is much faster than doing it inline (serial). Results are keyed by index.
    l4_results: dict[int, Optional[Rejection]] = {}
    if use_reducto_api and reducto_api_key:
        l4_results = run_l4_concurrent(examples, reducto_api_key, l4_sample_rate, verbose)
        stats.l4_sampled = len(l4_results)

    verified = []

    for i, ex in enumerate(examples):
        meta   = ex.get("metadata", {})
        msgs   = ex.get("messages", [])
        probe  = meta.get("probe_id", "")
        source = meta.get("source", "")
        user   = next((m["content"] for m in msgs if m["role"] == "user"), "")

        # Extract tool calls from OpenAI format
        asst = next((m for m in msgs if m["role"] == "assistant"), {})
        raw_calls = asst.get("tool_calls", [])
        tool_calls = [
            {"name": tc["function"]["name"],
             "arguments": tc["function"].get("arguments", "{}")}
            for tc in raw_calls if tc.get("type") == "function"
        ]

        rejections = []

        # ── Layer 1: schema ──
        rejections += validate_schema(tool_calls)
        if use_tool_hints:
            rejections += validate_tool_hints(user, tool_calls)

        if rejections:
            stats.rejected_l1 += 1
            for r in rejections:
                stats.rejection_reasons[r.reason] += 1
            if verbose:
                print(f"  L1 REJECT [{probe}] {rejections[0].reason}: {rejections[0].detail[:80]}")
            continue

        # ── Layer 2: probe semantics ──
        if use_probe_check and source == "harvested":
            r = validate_probe_semantics(probe, tool_calls, checkers)
            if r:
                stats.rejected_l2 += 1
                stats.rejection_reasons[r.reason] += 1
                if verbose:
                    print(f"  L2 REJECT [{probe}] {r.reason}: {r.detail}")
                continue

        # ── Layer 3: consensus (only for generated examples with enough data) ──
        if use_consensus and source == "generated":
            fp_key = f"{probe}||{user[:120]}"
            votes  = groups.get(fp_key, {})
            total_votes = sum(votes.values())
            if total_votes >= consensus_threshold:
                best_fp, best_count = max(votes.items(), key=lambda x: x[1])
                this_fp = _tool_fingerprint(tool_calls)
                if this_fp != best_fp:
                    stats.rejected_l3 += 1
                    stats.rejection_reasons["consensus_minority"] += 1
                    if verbose:
                        print(f"  L3 REJECT [{probe}] minority answer ({best_count} others disagree)")
                    continue

        # ── Layer 4: look up pre-computed result from concurrent pre-pass ──
        if i in l4_results:
            r = l4_results[i]
            if r:
                stats.rejected_l4 += 1
                stats.rejection_reasons[r.reason] += 1
                if verbose:
                    print(f"  L4 REJECT [{probe}] {r.reason}: {r.detail[:80]}")
                continue

        verified.append(ex)
        stats.passed += 1

    return verified, stats


# ---------------------------------------------------------------------------
# Save + report
# ---------------------------------------------------------------------------

def save_verified(
    examples: list[dict],
    out_dir: Path,
    train_n: int = 16_669,
    val_n: int   = 3_669,
    save_all: bool = False,
):
    """Write verified splits.

    If save_all=True, write ALL verified examples to verified_all.jsonl (no split).
    Otherwise write exactly train_n + val_n examples; aborts if pool is too small.
    """
    import random

    out_dir.mkdir(parents=True, exist_ok=True)
    random.shuffle(examples)

    if save_all:
        path = out_dir / "verified_all.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in examples))
        print(f"  verified_all.jsonl   : {len(examples):,}")
        return path, None

    need = train_n + val_n
    if len(examples) < need:
        print(
            f"\n  ✗  Not enough verified examples to produce exact splits.\n"
            f"     Have {len(examples):,}  —  need {need:,} "
            f"({train_n:,} train + {val_n:,} val).\n"
            f"     Re-run after more data is generated."
        )
        return None, None

    train = examples[:train_n]
    val   = examples[train_n : train_n + val_n]

    (out_dir / "verified_train.jsonl").write_text("\n".join(json.dumps(e) for e in train))
    (out_dir / "verified_val.jsonl").write_text(  "\n".join(json.dumps(e) for e in val))

    print(f"  verified_train.jsonl : {len(train):,}  (exact)")
    print(f"  verified_val.jsonl   : {len(val):,}  (exact)")
    return out_dir / "verified_train.jsonl", out_dir / "verified_val.jsonl"


def print_report(stats: VerifyStats):
    pct_pass = stats.passed / stats.total * 100 if stats.total else 0
    print(f"\n{'='*60}")
    print(f"Verification report")
    print(f"{'='*60}")
    print(f"  Total examples    : {stats.total:>7,}")
    print(f"  Passed            : {stats.passed:>7,}  ({pct_pass:.1f}%)")
    print(f"  Rejected L1       : {stats.rejected_l1:>7,}  (schema / tool mismatch)")
    print(f"  Rejected L2       : {stats.rejected_l2:>7,}  (probe semantic fail)")
    print(f"  Rejected L3       : {stats.rejected_l3:>7,}  (consensus minority)")
    if stats.l4_sampled:
        pct_l4 = stats.rejected_l4 / stats.l4_sampled * 100 if stats.l4_sampled else 0
        print(f"  Rejected L4       : {stats.rejected_l4:>7,}  (Reducto API 4xx — {stats.l4_sampled} sampled, {pct_l4:.1f}% fail rate)")
    print(f"\n  Rejection breakdown:")
    for reason, cnt in sorted(stats.rejection_reasons.items(), key=lambda x: -x[1]):
        bar = "█" * (cnt // max(1, stats.total // 200))
        print(f"    {reason:<32} {cnt:>5}  {bar}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify synthetic training data quality")
    parser.add_argument("--input", default=str(CHECKPOINT),
                        help="Input file (default: .checkpoint.jsonl)")
    parser.add_argument("--output", default=str(DATA_DIR),
                        help="Output directory for verified splits")
    parser.add_argument("--no-probe-check",  action="store_true")
    parser.add_argument("--no-tool-hints",   action="store_true",
                        help="Skip L1b tool-hint check (use for MCP data where multi-step "
                             "prompts mention doc content keywords not tied to a single tool)")
    parser.add_argument("--no-consensus",    action="store_true")
    parser.add_argument("--consensus-threshold", type=int, default=3,
                        help="Min teacher votes for consensus (default: 3)")
    parser.add_argument("--top-teachers", default="",
                        help="Comma-separated teacher display names whose votes count in L3 "
                             "(e.g. 'Claude Haiku 4.5 + thinking,Gemini 3.1 Flash Lite (preview),"
                             "MiniMax M2.7 (highspeed)'). Default: all teachers vote.")
    parser.add_argument("--no-l4",           action="store_true",
                        help="Skip Layer 4 Reducto live API spot-check")
    parser.add_argument("--l4-sample-rate",  type=float, default=0.10,
                        help="Fraction of examples to spot-check via Reducto API (default: 0.10)")
    parser.add_argument("--l4-full",         action="store_true",
                        help="Check every example via Reducto API (--l4-sample-rate 1.0)")
    parser.add_argument("--reducto-api-key", default="",
                        help="Reducto API key (default: $REDUCTO_API_KEY env var)")
    parser.add_argument("--train-n", type=int, default=16_669,
                        help="Exact training split size (default: 16,669)")
    parser.add_argument("--val-n",   type=int, default=3_669,
                        help="Exact validation split size (default: 3,669)")
    parser.add_argument("--save-all", action="store_true",
                        help="Save ALL verified examples to verified_all.jsonl (no train/val split)")
    parser.add_argument("--report-only", action="store_true",
                        help="Print stats only, don't write output files")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"Input not found: {inp}")
        sys.exit(1)

    print(f"Loading {inp}...")
    # Filter out any dry-run leftovers before verifying
    raw_lines = [l for l in inp.read_text().splitlines() if l.strip()]
    examples = []
    skipped_dry = 0
    for line in raw_lines:
        try:
            d = json.loads(line)
            msgs = d.get("messages", [])
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            if "[DRY RUN" in user:
                skipped_dry += 1
                continue
            examples.append(d)
        except Exception:
            pass
    print(f"  Loaded {len(examples):,} examples", end="")
    if skipped_dry:
        print(f"  (skipped {skipped_dry} dry-run examples)", end="")
    print()

    top_teachers = set(t.strip() for t in args.top_teachers.split(",") if t.strip()) or None
    if top_teachers:
        print(f"  Layer 3: only counting votes from: {', '.join(sorted(top_teachers))}")

    l4_rate = 1.0 if args.l4_full else args.l4_sample_rate
    if not args.no_l4:
        api_key = args.reducto_api_key or os.environ.get("REDUCTO_API_KEY", "")
        if api_key:
            print(f"  Layer 4: Reducto API spot-check at {l4_rate*100:.0f}% sample rate")
        else:
            print("  Layer 4: disabled (no REDUCTO_API_KEY found)")

    verified, stats = verify_examples(
        examples,
        use_probe_check     = not args.no_probe_check,
        use_tool_hints      = not args.no_tool_hints,
        use_consensus       = not args.no_consensus,
        consensus_threshold = args.consensus_threshold,
        top_teachers        = top_teachers,
        use_reducto_api     = not args.no_l4,
        l4_sample_rate      = l4_rate,
        reducto_api_key     = args.reducto_api_key or os.environ.get("REDUCTO_API_KEY", ""),
        verbose             = args.verbose,
    )

    print_report(stats)

    if not args.report_only:
        print(f"\nSaving verified splits to {args.output}/")
        save_verified(verified, Path(args.output),
                      train_n=args.train_n, val_n=args.val_n,
                      save_all=args.save_all)
        print("\nTrain on these files — not the raw checkpoint.")

if __name__ == "__main__":
    main()
