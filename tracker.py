#!/usr/bin/env python3
"""Hard probe tracker — compact header grid + per-probe interactive scatter sections."""

import re, time, os, json
from collections import defaultdict
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler

LOG_PATH = "/tmp/hard_probe_run.log"
TOTAL_RUNS = 308
PROBES_PER_MODEL = 22

RE_START     = re.compile(r"^\s+\[(.+?)\]\s+(\S+)\s+…")
RE_RESULT    = re.compile(r"→\s+(\d|N/A)/3\s+\(([0-9.]+)s\)")  # N/A = credit error
RE_HEADER    = re.compile(r"Param probe:\s+(\d+)\s+models.*?(\d+)\s+probes\s+=\s+(\d+)\s+runs")
RE_MODEL_LST = re.compile(r"^\s+•\s+(.+)$")
RE_ERROR     = re.compile(r"!!\s+(.+?)/(\S+):")
RE_CREDIT    = re.compile(r"Still 402|Credit error \(402\)|marking as N/A")

PROBE_METHODOLOGY = {
  # How each probe is measured: what was given, what was asked, how we score it.
  # Scoring = 3 points per probe: (1) did the model use the right parameter at all?
  # (2) did it use it correctly? (3) did the Reducto API actually accept the call?
  "get_job_poll":
    "Models are told a large document was already submitted and given a job ID. "
    "We ask them to retrieve the result. Score: (1) did they call the get_job tool? "
    "(2) did they pass the correct job ID? (3) did the API accept it? "
    "We're testing whether the model knows the async retrieval tool even exists.",
  "upload_reuse":
    "Models get a PDF URL described as expiring in 3 minutes, then asked to parse it, "
    "extract a table, AND classify it — three operations. Score: (1) did they upload first? "
    "(2) did they reuse the resulting reducto:// reference across all three calls? "
    "We're testing whether the model knows to cache the document instead of re-downloading it three times.",
  "citations_bbox":
    "Models are asked to extract two specific values and told 'I need to know exactly where on the page each one appears so I can highlight it.' "
    "Score: (1) did they request location coordinates in the extract call? (2) did they set it correctly? "
    "(3) did the API accept it? We're testing whether the model knows coordinates are a separate flag to request.",
  "citations_no_chunking":
    "Models are given a conflicting request: extract with page coordinates AND chunk the output by page. "
    "These two options are mutually exclusive in Reducto. Score: (1) did they request coordinates? "
    "(2) did they correctly drop the chunking option (the right tradeoff)? "
    "We're testing whether the model understands API constraints and resolves conflicts correctly.",
  "system_prompt_units":
    "Models are asked to extract dollar amounts from a document that reports values in thousands. "
    "Score: (1) did they pass a system prompt instructing the unit conversion? "
    "(2) did that prompt actually mention multiplying by 1000? "
    "We're testing whether the model knows to use the system_prompt field for format instructions rather than post-processing.",
  "optimize_latency_not_deep":
    "Models are told 'extract this as fast as possible, under 2 seconds, accuracy can be slightly lower.' "
    "Score: (1) did they turn on the speed optimization flag? (2) did they also leave the deep extraction mode off? "
    "We're testing whether the model knows these two settings conflict and that 'fast' means turning off 'thorough'.",
  "include_images_chart":
    "Models are told the data they need is only in a chart — not in any table or text. "
    "Score: (1) did they turn on image inclusion? (2) did they set it correctly? "
    "We're testing whether the model knows chart data requires a separate flag since images are excluded by default.",
  "merge_tables_crosspage":
    "Models are told a portfolio table has its header on page 3 and data rows continuing onto page 4, and asked for a single unified table. "
    "Score: (1) did they set the merge tables option? (2) did they set it to true? "
    "We're testing whether the model knows page-spanning tables need explicit stitching.",
  "filter_blocks_clean":
    "Models are asked to parse a document and strip headers, footers, and page numbers for a clean LLM input. "
    "Score: (1) did they pass the filter_blocks parameter? (2) did they name the correct block types exactly? "
    "We're testing whether the model knows this filter exists and uses the exact enum strings Reducto expects.",
  "chunk_section_rag":
    "Models are told 'I'm building a RAG system and want chunks that respect document sections, not page boundaries.' "
    "Score: (1) did they set the chunk mode? (2) did they set it to 'section' specifically? "
    "We're testing whether the model knows non-default chunking modes exist.",
  "embedding_optimized_flag":
    "Models are told the output will be embedded for vector search and to optimize for embedding quality. "
    "Score: (1) did they set the embedding optimization flag? (2) did they set it to true? "
    "We're testing knowledge of a rarely-documented Reducto flag. Almost no public training data covers it.",
  "page_range_cost":
    "Models are told the document is 50 pages but the content they need is on pages 3-7, and explicitly asked not to pay for the rest. "
    "Score: (1) did they set a page range? (2) did they set it to exactly pages 3-7? "
    "We're testing whether the model knows page ranges exist as a cost-saving feature.",
  "return_images_array":
    "Models are asked for rendered image files for all tables AND all charts separately, with presigned URLs for each. "
    "Score: (1) did they use the return_images parameter? (2) did they pass it as a list with both 'figure' and 'table'? "
    "We're testing whether the model knows images require a typed list, not just a boolean.",
  "classify_then_route":
    "Models are told they receive many document types and need to figure out what this one is before extracting from it. "
    "Score: (1) did they call classify AND extract? (2) did classify happen before extract in the sequence? "
    "We're testing whether the model understands the two-step classify-then-act workflow.",
  "split_then_extract_section":
    "Models are asked to extract data from just the 'Portfolio Holdings' section — but must first figure out which pages that section covers. "
    "Score: (1) did they call split first? (2) did they then also call extract? "
    "We're testing whether the model knows to use split to locate sections before targeted extraction.",
  "edit_basic_fill":
    "Models are asked to fill in two form fields in a PDF and return the filled document. "
    "Score: (1) did they call the edit tool (not parse or extract)? (2) did they pass editing instructions? "
    "We're testing whether the model knows a separate edit tool exists for writing to documents.",
  "edit_form_schema":
    "Models are asked to fill 5 specific named fields with exact values using structured definitions for precision. "
    "Score: (1) did they call the edit tool? (2) did they use the structured form schema format? "
    "We're testing knowledge of Reducto's structured form-fill API, which is completely specific to Reducto.",
  "edit_flatten_lock":
    "Models are asked to fill a form and then lock it so the recipient can't change the values. "
    "Score: (1) did they call the edit tool? (2) did they set flatten=True to convert fields to static text? "
    "We're testing a niche Reducto parameter with essentially no public training data.",
  "empty_result_ocr_retry":
    "Models are shown a real parse response where the content array came back empty (a scanned image PDF with no text layer). "
    "They're then asked to parse a document using the right fallback strategy. "
    "Score: (1) did they set an extraction mode? (2) did they specifically use OCR mode? "
    "We're testing whether the model can diagnose a known failure pattern and apply the correct fix.",
  "classify_page_limit":
    "Models are asked to classify an 80-page document and told to be cost-efficient. "
    "Score: (1) did they call classify? (2) did they set a page range limit to avoid processing all 80 pages? "
    "We're testing whether the model knows classification has a cost-saving page cap option.",
  "jobid_expiry_recovery":
    "Models are given a job reference from 2 hours ago and asked to extract data from it. "
    "Score: (1) did they attempt anything? (2) did they ultimately parse the original document URL instead of giving up on the stale ID? "
    "We're testing whether the model knows job references expire and can recover gracefully.",
  "table_cutoff_preserve":
    "Models are asked to split a document into 3 sections and told 'don't cut any table in half at a section boundary.' "
    "Score: (1) did they call split? (2) did they set table_cutoff='preserve'? "
    "We're testing knowledge of a split option that prevents partial table extraction.",
  # Standard probes
  "jobid_chaining":
    "Models parse a document, then immediately extract from it — two sequential calls. "
    "Score: (1) did they pass persist_results on the parse? (2) did they use the resulting jobid:// reference in the extract? "
    "We're testing whether the model knows to chain calls via job references instead of re-uploading.",
  "array_extract":
    "Models are asked to extract all individual stock holdings — a repeating list of rows. "
    "Score: (1) did they set array_extract? (2) did they set it to true? "
    "We're testing whether the model knows list extraction needs a different mode than single-object extraction.",
  "agentic_scopes":
    "Models are asked to parse a document with complex multi-column financial tables and ensure the tables are correct. "
    "Score: (1) did they set agentic_scopes? (2) did they include 'table' in the scope list? "
    "We're testing whether the model knows the AI table-correction pass exists.",
  "deep_extract_off":
    "Models are asked to extract just two fields and told speed is the priority. "
    "Score: (1) did they omit deep_extract or set it false? That IS the point — the correct answer is to NOT use it. "
    "We're testing whether the model resists the instinct to always use 'thorough' mode.",
  "jsonbbox_format":
    "Models are asked to parse tables and told they need the exact coordinates of each cell for a UI overlay. "
    "Score: (1) did they set table_format? (2) did they set it to 'jsonbbox'? "
    "We're testing knowledge of a non-default table output format.",
  "split_rules":
    "Models are asked to split a document into 3 named sections with a specific rule about where boundaries fall. "
    "Score: (1) did they use split_rules to pass that rule? (2) was the rule actually in the parameter? "
    "We're testing whether the model knows split accepts natural language constraints.",
  "ocr_mode":
    "Models are told the document was scanned from a physical copy and may not have a text layer. "
    "Score: (1) did they set extraction_mode? (2) did they set it to 'ocr'? "
    "We're testing whether the model knows scanned documents need a different reading mode.",
  "document_metadata":
    "Models are asked to parse a document and told to include context about where it came from and which pipeline it belongs to. "
    "Score: (1) did they include a document_metadata field? (2) did it contain relevant context? "
    "We're testing whether the model knows this field exists for passing pipeline context.",
  "url_array":
    "Models are given three separate PDF URLs and asked to process all of them together. "
    "Score: (1) did they pass input as a list/array? (2) did they format it correctly as a JSON array? "
    "We're testing whether the model knows Reducto accepts batches, not just individual documents.",
  "return_figure_images":
    "Models are asked to parse a document and get presigned image URLs for all charts and figures. "
    "Score: (1) did they set return_figure_images? (2) did they set it to true? "
    "We're testing whether the model knows figure images are excluded by default.",
}

PROBE_EXPLAIN = {
  # ── Hard probes ─────────────────────────────────────────────────────
  "get_job_poll":
    "Starting a task and checking back later. Like placing a takeout order and coming back when it's ready, "
    "instead of standing at the counter blocking everyone. Most models don't know this option exists and just wait inline.",

  "upload_reuse":
    "Send a document once, get a reference ticket, reuse that ticket for every follow-up question. "
    "Most models re-send the whole PDF every single time — like faxing the same 100 pages repeatedly instead of saying 'see my earlier fax'.",

  "citations_bbox":
    "When you need to know the exact physical location of data on the page — like 'that number is in the top-right corner' — "
    "you have to ask for coordinates. Models usually pull the data correctly but forget to ask where on the page it lives.",

  "citations_no_chunking":
    "By default Reducto slices documents into chunks. This probe checks whether the model knows to turn that off "
    "and get the whole document back as one piece. Matters when something downstream can't handle a split document.",

  "system_prompt_units":
    "If you want numbers in miles instead of kilometers, or pounds instead of kilograms, you need to say so upfront. "
    "Models almost never think to specify units — they just grab whatever the document has and call it done.",

  "optimize_latency_not_deep":
    "When someone says 'be fast', models often still reach for the most thorough extraction mode. "
    "These two goals conflict — the 'deep' mode is slower. Models confuse 'get it right' with 'get it fast'.",

  "include_images_chart":
    "Charts and graphs are excluded from results by default. If you want them, you have to ask. "
    "Models almost always forget this switch exists and silently return text-only results.",

  "merge_tables_crosspage":
    "A table that starts on page 3 and finishes on page 4 is really one table, not two. "
    "Models process pages separately and return two broken fragments instead of one joined table.",

  "filter_blocks_clean":
    "Headers, footers, and page numbers are noise. Reducto can strip them before extraction, "
    "but you have to ask. Models skip this and dump all the junk into the output alongside the real content.",

  "chunk_section_rag":
    "When building a search index, you want content grouped by topic or section, not by page. "
    "Models default to page-by-page slicing, which produces fragments that span unrelated topics and make search worse.",

  "embedding_optimized_flag":
    "There's a special mode that tweaks how the document is sliced to make AI search work better. "
    "It's barely documented and almost no model knows it exists — pure knowledge gap.",

  "page_range_cost":
    "If the answer is on pages 1-5, there's no reason to process all 200 pages and pay for it. "
    "Models almost always process the entire document even when told exactly where to look.",

  "return_images_array":
    "Figures and diagrams come back as image links only if you ask for them. "
    "Models routinely return text-only results and leave all the visuals behind.",

  "classify_then_route":
    "First figure out what kind of document this is, then decide what to do with it. "
    "Models often try to skip the classification step and jump straight to extraction, guessing the document type wrong.",

  "split_then_extract_section":
    "Divide the document into named sections first, then extract data from just one section. "
    "Two separate steps, two separate calls. Most models try to do it all in one shot and get confused.",

  "edit_basic_fill":
    "Reducto has a separate tool specifically for modifying documents — filling in fields, making changes. "
    "Models almost always reach for the data-reading tool instead, which can't write anything.",

  "edit_form_schema":
    "Filling out a form and getting the result back in a structured format requires a specific editing tool. "
    "Models confuse it with the data extraction tool, which is a completely different operation.",

  "edit_flatten_lock":
    "Converting a form into plain, locked, uneditable content is a niche operation. "
    "Virtually no training data covers this, so models have essentially no idea it's possible.",

  "empty_result_ocr_retry":
    "If the document comes back empty, it's probably a scanned image with no readable text layer. "
    "The fix is to retry using optical character recognition. Models almost never think to check and retry — they just return the empty result.",

  "classify_page_limit":
    "You can tell Reducto to only look at the first few pages when classifying a document. "
    "Models ignore this and send everything, paying more and taking longer than necessary.",

  "jobid_expiry_recovery":
    "Reference tickets for uploaded documents expire after about an hour. "
    "If a ticket has gone stale, the model needs to recognize the error and start fresh. Most don't — they just throw the error.",

  "table_cutoff_preserve":
    "A table that gets cut off at the bottom of a page often loses its last few rows silently. "
    "Telling Reducto to be careful about this requires an extra instruction most models don't know to give.",

  "merge_tables_crosspage":
    "Same as above — a table that spans multiple pages should come back as one table, not several fragments. "
    "Without explicit instruction, models get fragments and don't notice the join is missing.",

  # ── Standard probes ──────────────────────────────────────────────────
  "jobid_chaining":
    "Parse a document once, save the result, then reference that saved result in the next call instead of re-uploading. "
    "Models usually re-send the document every time, wasting time and money.",

  "array_extract":
    "When a document has repeating rows — like a list of transactions — you need to tell Reducto you want a list back, not just one item. "
    "Models often return only the first row and call it done.",

  "agentic_scopes":
    "Complex or messy tables can be corrected using an AI pass before extraction. "
    "You have to ask for it. Models skip this and return garbled table data from documents with unusual layouts.",

  "deep_extract_off":
    "There's a 'go deep' mode that's thorough but slow. When speed is requested, it must stay off. "
    "Models often turn it on anyway because they think 'more thorough' is always better.",

  "jsonbbox_format":
    "Same idea as citations_bbox — asking for the physical coordinates of each table cell on the page. "
    "Models extract the data but forget to also return where on the page each cell lives.",

  "split_rules":
    "When splitting a document, you can add custom rules in plain English about how to divide it. "
    "Models ignore the rules field and just use whatever default splitting logic Reducto picks.",

  "ocr_mode":
    "Scanned documents are images — there's no real text to read. You have to turn on OCR explicitly. "
    "Models try to read the document normally, get nothing, and don't think to switch modes.",

  "document_metadata":
    "You can attach context to a request — where the document came from, which customer it belongs to, etc. "
    "Models almost never use this field, even when that context would make the result more accurate.",

  "url_array":
    "When processing three documents at once, you can pass them all in a single request as a list. "
    "Models make three separate requests instead, tripling the overhead.",

  "return_figure_images":
    "Charts, graphs, and figures are excluded by default. You have to ask for them. "
    "Models return text-only results and silently leave all the visual content behind.",
}

# Models dropped mid-run — listed in old log headers but should not appear in tracker
EXCLUDED_MODELS = {
    "Alibaba Tongyi DeepResearch 30B-A3B",
    "GLM-4.6 (Novita bf16)",
    "Inception Mercury Coder",
    "Codex Mini Latest",
    "Mistral Devstral Small",
    "Qwen3.5 Flash (02-23)",
    "Qwen3 Coder Next (ionstream fp8)",
    "Arcee Trinity Large (prime)",
    "Nemotron Super 120B (Nebius bf16)",
}

def _is_excluded(name: str) -> bool:
    """Prefix-match so truncated log names are caught too."""
    return any(name.startswith(ex[:25]) or ex.startswith(name[:25])
               for ex in EXCLUDED_MODELS)

def parse_log():
    if not os.path.exists(LOG_PATH):
        return {"error": "Log not found — is the hard probe run active?"}
    total_runs = TOTAL_RUNS
    probes_per_model = PROBES_PER_MODEL
    pending, completed, errors = [], [], []
    credit_errors = set()   # (model, probe) pairs that were 402 — not real failures
    all_models_listed = []
    total_results = 0   # raw count of "→ X/3" lines — ground truth for done
    last_start = (None, None)   # track last started (model, probe) for credit error tagging
    try:
        with open(LOG_PATH) as f:
            lines = f.readlines()
    except Exception as e:
        return {"error": str(e)}
    try:
        start_t = os.stat(LOG_PATH).st_birthtime
    except AttributeError:
        start_t = os.stat(LOG_PATH).st_mtime
    for line in lines:
        m = RE_HEADER.search(line)
        if m:
            total_runs = int(m.group(3)); probes_per_model = int(m.group(2))
        m = RE_MODEL_LST.match(line)
        if m:
            name = m.group(1).strip()
            if not _is_excluded(name):
                all_models_listed.append(name)
        m = RE_START.match(line)
        if m:
            last_start = (m.group(1), m.group(2))
            pending.append(last_start)
        # Credit error detection — tag the last (model,probe) pair
        if RE_CREDIT.search(line) and last_start[0]:
            credit_errors.add(last_start)
        m = RE_RESULT.search(line)
        if m:
            score_raw = m.group(1)
            is_credit = (score_raw == "N/A")
            if is_credit:
                total_results += 1      # still counts as done (the run finished, just invalid)
                if pending:
                    model, probe = pending.pop(0)
                    credit_errors.add((model, probe))
            else:
                total_results += 1
                if pending:
                    model, probe = pending.pop(0)
                    if (model, probe) not in credit_errors:
                        completed.append({"model": model, "probe": probe,
                                           "score": int(score_raw), "secs": float(m.group(2))})
        m = RE_ERROR.search(line)
        if m:
            errors.append(f"{m.group(1)}/{m.group(2)}")
    elapsed  = time.time() - start_t
    # live_done = probes finished in the active log (ground truth for ETA)
    live_done = total_results
    # rolling rate (last 80) to avoid slow-model skew
    if len(completed) >= 20:
        w = completed[-80:]
        rate = len(w) / max(sum(c["secs"] for c in w) / 6, 1)
    else:
        rate = live_done / elapsed if elapsed > 0 else 0
    remaining_s = (total_runs - live_done) / rate if rate > 0 else 0

    # ── Inject saved JSON results for models NOT in the live log ────────
    # This merges the "archive" models (complete from previous run) with
    # the live log (currently re-running models) so the tracker shows all 26.
    JSON_DIR = os.path.join(os.path.dirname(__file__),
                            "benchmark/results/param_probe/by_model_hard")
    if os.path.isdir(JSON_DIR):
        # Collect model names seen in the live log
        log_model_names = set(m for m, _ in pending)
        for line in lines:
            ms = RE_START.match(line)
            if ms: log_model_names.add(ms.group(1))
        for fn in sorted(os.listdir(JSON_DIR)):
            if not fn.endswith(".json"): continue
            try:
                with open(os.path.join(JSON_DIR, fn)) as jf:
                    rows = json.load(jf)
            except Exception:
                continue
            if not rows: continue
            model_name = rows[0].get("model", "")
            # Skip if this model is actively being tracked from the log
            if any(model_name.startswith(lm[:20]) or lm.startswith(model_name[:20])
                   for lm in log_model_names):
                continue
            # Skip excluded or already-listed models
            if _is_excluded(model_name):
                continue
            if any(model_name == n for n in all_models_listed):
                continue
            if model_name not in all_models_listed:
                all_models_listed.append(model_name)
            for row in rows:
                if row.get("credit_error"): continue
                completed.append({
                    "model": model_name,
                    "probe": row.get("probe_id", ""),
                    "score": row.get("score", 0),
                    "secs": 1.0,
                })
                total_results += 1
    # ────────────────────────────────────────────────────────────────────
    # total_results now includes both live log + JSON-injected probes
    combined_done = total_results

    # Build canonical name map: truncated/corrupted log names → full names
    # (parallel stdout interleaving can cut off last chars of model name in [...] blocks)
    canonical_names = {}
    listed_set = list(dict.fromkeys(all_models_listed))  # deduped, order preserved
    def _canonicalize(name):
        if name in canonical_names:
            return canonical_names[name]
        if name in listed_set:
            canonical_names[name] = name
            return name
        # try prefix match — the truncated name should be a prefix of the real one
        matches = [m for m in listed_set if m.startswith(name)]
        if len(matches) == 1:
            canonical_names[name] = matches[0]
            return matches[0]
        canonical_names[name] = name
        return name

    model_pts  = defaultdict(int)
    model_done = defaultdict(int)
    probe_results = defaultdict(dict)   # {probe: {model: score}}
    probe_order   = []
    for c in completed:
        c["model"] = _canonicalize(c["model"])
        model_pts[c["model"]]  += c["score"]
        model_done[c["model"]] += 1
        if c["probe"] not in probe_results:
            probe_order.append(c["probe"])
        probe_results[c["probe"]][c["model"]] = c["score"]
    # Also canonicalize credit_errors set
    credit_errors = {(_canonicalize(m), p) for m, p in credit_errors}

    # Strip excluded models from completed results too (they may appear via probe lines)
    completed = [c for c in completed if not _is_excluded(c["model"])]

    seen_order, seen_set = [], set()
    for name in listed_set:      # already deduped
        if name not in seen_set:
            seen_order.append(name); seen_set.add(name)
    for c in completed:
        if c["model"] not in seen_set and not _is_excluded(c["model"]):
            seen_order.append(c["model"]); seen_set.add(c["model"])

    # Count credit errors per model
    credit_by_model = defaultdict(int)
    for (mod, _) in credit_errors:
        credit_by_model[mod] += 1

    max_pts = probes_per_model * 3
    models_out = []
    for name in seen_order:
        n = model_done[name]; pts = model_pts[name]
        n_credit = credit_by_model[name]
        # Consider a model finished if no process is actively running probes for it
        active_pids = [p for p in __import__('os').popen("pgrep -f bench_param_probe").read().split() if p]
        run_active = len(active_pids) > 0
        fin = (n + n_credit) >= probes_per_model or (not run_active and n > 0)
        all_credit = (n == 0 and n_credit > 0)    # every probe was a credit error
        pct = round(pts / (n * 3) * 100, 1) if n > 0 else 0
        models_out.append({"name": name, "pts": pts, "max": max_pts,
                            "done": n, "total": probes_per_model,
                            "finished": fin, "pct": pct,
                            "credit_fail": all_credit, "n_credit": n_credit})
    models_out.sort(key=lambda m: (-m["pts"], -m["done"]))

    # Recompute done from models_out so both sides count the same model set
    # (total_results can include orphaned JSON rows that don't land in seen_order)
    correct_done = sum(m["done"] + m["n_credit"] for m in models_out)
    # For finished models, their "total" is what they actually completed — not the
    # theoretical max — so a model that crashed at 20/22 doesn't drag pct below 100%.
    # For in-progress models, use probes_per_model as the target.
    total_all = sum(
        (m["done"] + m["n_credit"]) if m["finished"] else probes_per_model
        for m in models_out
    )
    eta_dt = datetime.now() + timedelta(seconds=remaining_s)
    return {
        "done": correct_done, "total": total_all,
        "pct": round(correct_done / total_all * 100, 1) if total_all > 0 else 0,
        "elapsed": str(timedelta(seconds=int(elapsed))),
        "eta_secs": int(remaining_s),
        "eta_clock": eta_dt.strftime("%H:%M:%S"),
        "rate": round(rate * 60, 1),
        "in_flight": len(pending),
        "errors": len(errors),
        "models": models_out,
        "probe_results": {p: probe_results[p] for p in probe_order},
        "probe_order": probe_order,
        "probe_explain": PROBE_EXPLAIN,
        "probe_methodology": PROBE_METHODOLOGY,
    }


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Hard Probe Tracker</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
<style>
/* ── Reducto brand palette (dark) ───────────────────────────────────────
   Brand purple  #9d17a0   Light purple  #c44ac7   Highlight #dcbffb
   Deep bg       #0c0710   Card bg       #160d1b   Card+     #1e1428
   Border dim    #2e1a37   Border act    #9d17a0
   Text          #f2eaf4   Muted         #9b85a0   Dimmed    #4e3358
──────────────────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background-color: #0c0710;
  background-image: radial-gradient(#2a1533 1px, transparent 1px);
  background-size: 22px 22px;
  color: #f2eaf4;
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  font-size: 16px; font-weight: 500;
  -webkit-font-smoothing: antialiased;
  overflow-y: auto; overflow-x: hidden;
  padding: 0;
}

/* ─── NAVBAR ─── */
.navbar {
  background: #160d1b; border-bottom: 1px solid #2e1a37;
  padding: 12px 20px; display: flex; align-items: center; gap: 14px;
  position: sticky; top: 0; z-index: 100;
  backdrop-filter: blur(8px);
}
.navbar-logo {
  display: flex; align-items: center; gap: 8px;
  font-size: 18px; font-weight: 800; color: #f2eaf4; letter-spacing: -.3px;
}
.navbar-logo .logo-mark {
  width: 22px; height: 22px; background: #9d17a0; border-radius: 4px;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; line-height: 1; color: #fff; font-weight: 900;
}
.navbar-badge {
  font-size: 12px; font-weight: 600; padding: 3px 10px;
  background: rgba(157,23,160,.18); color: #c44ac7;
  border: 1px solid rgba(157,23,160,.35); border-radius: 20px;
}
.navbar-title {
  font-size: 15px; color: #9b85a0; font-weight: 500; margin-left: 4px;
}

/* ─── PAGE BODY ─── */
.page { padding: 16px 20px; }

/* ─── TOP HEADER ─── */
.title {
  font-size: 24px; font-weight: 800; color: #c44ac7;
  letter-spacing: -.3px; margin-bottom: 12px;
}

/* stats row — horizontal, no wrap */
.stats-row {
  display: flex; flex-direction: row; flex-wrap: nowrap;
  gap: 8px; align-items: flex-start;
  overflow-x: auto; padding-bottom: 4px; margin-bottom: 12px;
}
.stat {
  background: #160d1b; border: 1px solid #2e1a37; border-radius: 10px;
  padding: 8px 18px; white-space: nowrap; flex-shrink: 0; text-align: center;
}
.stat .v { display: block; color: #f2eaf4; font-weight: 800; font-size: 40px; line-height: 1.1; }
.stat .l { display: block; color: #9b85a0; font-size: 16px; text-transform: uppercase; letter-spacing: .5px; margin-top: 2px; }
.err-v    { color: #f85149 !important; }

/* progress bar */
.bar-wrap { background: #1e1428; border-radius: 6px; height: 12px; margin-bottom: 16px; overflow: hidden; }
.bar { height: 100%; background: linear-gradient(90deg, #6b0d6e, #9d17a0, #c44ac7); transition: width 1s; }

/* model grid */
.grid-hdr { font-size: 14px; color: #9b85a0; text-transform: uppercase; letter-spacing: .8px; font-weight: 700; margin-bottom: 8px; }
.grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 5px; margin-bottom: 22px; }
.mrow {
  background: #160d1b; border: 1px solid #2e1a37; border-radius: 8px;
  padding: 8px 12px; display: flex; align-items: center; gap: 10px;
  transition: border-color .15s;
}
.mrow.done   { border-color: #238636; }
.mrow.active { border-color: #7a2080; }
.mdot  { width: 14px; height: 14px; border-radius: 50%; flex-shrink: 0; }
.mname { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 15px; }
.mbar  { width: 54px; height: 5px; background: #2e1a37; border-radius: 3px; overflow: hidden; flex-shrink: 0; }
.mfill { height: 100%; border-radius: 3px; }
.mpts  { font-size: 16px; font-weight: 700; min-width: 42px; text-align: right; flex-shrink: 0; }
.mtag  { font-size: 12px; padding: 3px 8px; border-radius: 8px; flex-shrink: 0; min-width: 48px; text-align: center; font-weight: 600; }
.mtag.done    { background: #0d3016; color: #3fb950; border: 1px solid #1a5c2a; }
.mtag.active  { background: #2e0d35; color: #c44ac7; border: 1px solid #5a1a65; }
.mtag.pending { background: #1e1428; color: #4e3358; border: 1px solid #2e1a37; }
.mtag.credit  { background: #1e1428; color: #4e3358; border: 1px solid #2e1a37; font-size: 11px; }
.mrow.credit  { opacity: 0.45; }
.mrow.done, .mrow.active { cursor: pointer; }
.mrow.done:hover, .mrow.active:hover {
  border-color: #c44ac7; background: #1e1030; transform: translateY(-1px);
  box-shadow: 0 4px 16px rgba(157,23,160,.25); transition: all .15s;
}

/* ─── MODEL MODAL ─── */
#model-modal-overlay {
  display: none; position: fixed; inset: 0; z-index: 1000;
  background: rgba(0,0,0,.72); backdrop-filter: blur(4px);
  align-items: center; justify-content: center;
}
#model-modal-overlay.open { display: flex; }
#model-modal {
  background: #160d1b; border: 1px solid #5a1a65; border-radius: 14px;
  width: min(680px, 94vw); max-height: 82vh; overflow-y: auto;
  padding: 28px 30px; position: relative;
  box-shadow: 0 20px 60px rgba(0,0,0,.7), 0 0 0 1px rgba(157,23,160,.2);
}
#model-modal::-webkit-scrollbar { width: 6px; }
#model-modal::-webkit-scrollbar-thumb { background: #3a1040; border-radius: 3px; }
.modal-close {
  position: absolute; top: 16px; right: 18px; background: none; border: none;
  color: #6e4a7a; font-size: 22px; cursor: pointer; line-height: 1;
  transition: color .15s;
}
.modal-close:hover { color: #c44ac7; }
.modal-model-name {
  font-size: 22px; font-weight: 700; color: #f2eaf4; margin-bottom: 4px;
  padding-right: 32px;
}
.modal-score-row {
  display: flex; align-items: center; gap: 14px; margin-bottom: 20px;
  padding-bottom: 16px; border-bottom: 1px solid #2e1a37;
}
.modal-score-big {
  font-size: 36px; font-weight: 800; line-height: 1;
}
.modal-score-sub { font-size: 14px; color: #9b85a0; line-height: 1.6; }
.modal-pct-bar {
  flex: 1; height: 7px; background: #2e1a37; border-radius: 4px; overflow: hidden;
}
.modal-pct-fill { height: 100%; border-radius: 4px; transition: width .4s; }
.modal-section-title {
  font-size: 13px; font-weight: 700; color: #9b85a0;
  text-transform: uppercase; letter-spacing: .8px;
  margin: 18px 0 10px;
}
.modal-missed-item {
  background: #0c0710; border: 1px solid #2e1a37; border-radius: 8px;
  padding: 12px 14px; margin-bottom: 8px;
}
.modal-missed-header {
  display: flex; align-items: center; gap: 10px; margin-bottom: 6px;
}
.modal-probe-id {
  font-size: 14px; font-weight: 700; color: #c44ac7; font-family: monospace;
}
.modal-probe-score {
  margin-left: auto; font-size: 13px; font-weight: 700;
  padding: 2px 8px; border-radius: 6px; background: #2e1a37;
}
.modal-probe-what {
  font-size: 13px; color: #dcbffb; margin-bottom: 6px; font-weight: 600;
}
.modal-probe-why {
  font-size: 13px; color: #9b85a0; line-height: 1.55;
}
.modal-perfect { color: #3fb950; font-size: 14px; padding: 10px 0; }
.modal-stat-row {
  display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin-top: 4px;
}
.modal-stat-row { perspective: 600px; }
.modal-stat {
  background: #0c0710; border: 1px solid #2e1a37; border-radius: 8px;
  padding: 10px 12px; text-align: center;
  position: relative; overflow: hidden;
  transform-style: preserve-3d;
  transition: transform .08s ease-out, border-color .2s, box-shadow .2s;
  will-change: transform;
}
/* The sheen layer — a linear gradient whose angle tracks the tilt */
.modal-stat::before {
  content: '';
  position: absolute; inset: -1px; border-radius: 8px; pointer-events: none; z-index: 0;
  background: linear-gradient(
    calc(var(--shine-angle, 135deg)),
    transparent 25%,
    rgba(255,255,255, calc(var(--shine-str, 0) * .13)) 45%,
    rgba(220,191,251, calc(var(--shine-str, 0) * .22)) 50%,
    rgba(196, 74,199, calc(var(--shine-str, 0) * .10)) 55%,
    transparent 75%
  );
  transition: opacity .15s;
}
.modal-stat-val { font-size: 20px; font-weight: 700; color: #dcbffb; position: relative; z-index: 1; }
.modal-stat-lbl { font-size: 12px; color: #6e4a7a; margin-top: 2px; position: relative; z-index: 1; }

/* ─── PROBE SECTIONS ─── */
.section-hdr {
  font-size: 20px; font-weight: 700; color: #c44ac7;
  margin: 16px 0 10px; padding-bottom: 8px;
  border-bottom: 1px solid #2e1a37;
  letter-spacing: -.2px;
}

details {
  background: #160d1b;
  border: 1px solid #2e1a37; border-radius: 10px;
  margin-bottom: 6px; overflow: hidden;
  transition: border-color .15s;
}
details[open] { border-color: #9d17a0; }
details[open] summary { background: rgba(157,23,160,.08); }
summary {
  cursor: pointer; padding: 14px 18px; font-size: 18px;
  display: flex; align-items: center; gap: 12px;
  list-style: none; user-select: none; transition: background .15s;
}
summary:hover { background: rgba(157,23,160,.06); }
summary::-webkit-details-marker { display: none; }
summary::before {
  content: '▶'; font-size: 11px; color: #4e3358;
  transition: transform .2s, color .15s; flex-shrink: 0;
}
details[open] summary::before { transform: rotate(90deg); color: #9d17a0; }
.sum-probe { color: #f2eaf4; font-weight: 700; }
.sum-n     { font-size: 15px; color: #9b85a0; }
.sum-avg   { margin-left: auto; font-size: 18px; font-weight: 700; }
.sum-exp   { font-size: 14px; color: #9b85a0; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.plot-wrap { padding: 8px 12px 12px; background: #0c0710; }
.plot-div  { width: 100%; height: 380px; }

/* ─── INSIGHTS ─── */
.insights-grid {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 10px; margin-bottom: 20px;
}
.ins-card {
  background: #160d1b; border: 1px solid #2e1a37; border-radius: 10px;
  padding: 14px 16px; display: flex; flex-direction: column; gap: 8px;
}
.ins-title {
  font-size: 15px; font-weight: 700; color: #c44ac7;
  text-transform: uppercase; letter-spacing: .7px;
  padding-bottom: 8px; border-bottom: 1px solid #2e1a37;
  display: flex; align-items: center; gap: 8px;
}
.ins-subtitle {
  font-size: 13px; color: #9b85a0; margin-top: -4px; margin-bottom: 4px; line-height: 1.5;
}
.ins-row {
  display: flex; align-items: flex-start; gap: 10px;
  padding: 8px 0; border-bottom: 1px solid #1e1428;
  font-size: 15px;
}
.ins-row:last-child { border-bottom: none; }
.ins-rank {
  font-size: 20px; font-weight: 800; min-width: 32px; line-height: 1;
  flex-shrink: 0; color: #4e3358;
}
.ins-rank.gold   { color: #dcbffb; }
.ins-rank.silver { color: #9b85a0; }
.ins-rank.bronze { color: #7a5080; }
.ins-body { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 3px; }
.ins-probe-name { font-weight: 700; color: #f2eaf4; font-size: 15px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ins-model-name { font-weight: 700; color: #f2eaf4; font-size: 15px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ins-explain    { font-size: 13px; color: #9b85a0; line-height: 1.5; }
.ins-meta       { font-size: 13px; }
.score-pill {
  font-size: 13px; font-weight: 700; padding: 3px 10px;
  border-radius: 10px; white-space: nowrap; flex-shrink: 0; margin-top: 1px;
}
.pill-red    { background: #3d0000; color: #f85149; }
.pill-yellow { background: #2e0d35; color: #c44ac7; }
.pill-green  { background: #0d3016; color: #3fb950; }
.contrast-badges { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px; }
.cbadge {
  font-size: 12px; padding: 3px 8px; border-radius: 6px; white-space: nowrap;
  cursor: default; position: relative; font-weight: 600;
}
.cbadge-hi  { background: #0d3016; color: #3fb950; border: 1px solid #1a5c2a; }
.cbadge-lo  { background: #3d0000; color: #f85149; border: 1px solid #6e1b1b; }
.cbadge[data-tip]:hover::after {
  content: attr(data-tip);
  position: absolute; bottom: calc(100% + 6px); left: 0;
  background: #160d1b; border: 1px solid #9d17a0; border-radius: 8px;
  padding: 8px 12px; font-size: 13px; color: #f2eaf4; line-height: 1.5;
  width: 320px; white-space: normal; z-index: 999;
  box-shadow: 0 4px 20px rgba(0,0,0,.7);
  pointer-events: none;
}

/* 👀 methodology button */
.eye-btn {
  margin-left: auto; font-size: 18px; cursor: pointer;
  opacity: 0.5; transition: opacity .2s; flex-shrink: 0;
  background: none; border: none; color: inherit; padding: 0 2px;
  line-height: 1;
}
.eye-btn:hover { opacity: 1; }
.eye-btn.eye-active { opacity: 1 !important; background: rgba(157,23,160,.22); border-radius: 4px; }

/* methodology tooltip (click-toggle) */
.mtip {
  position: fixed; background: #160d1b; border: 1px solid #9d17a0;
  border-radius: 10px; padding: 16px 18px; pointer-events: auto; z-index: 9999;
  font-size: 15px; line-height: 1.65; color: #f2eaf4;
  box-shadow: 0 8px 32px rgba(0,0,0,.8), 0 0 0 1px rgba(157,23,160,.15);
  max-width: 420px; display: none;
}
.mtip-title { font-weight: 700; color: #dcbffb; font-size: 15px; margin-bottom: 8px; padding-right: 20px; }
.mtip-close {
  position: absolute; top: 10px; right: 12px; background: none; border: none;
  color: #4e3358; font-size: 16px; cursor: pointer; line-height: 1; padding: 0 2px;
  transition: color .15s;
}
.mtip-close:hover { color: #c44ac7; }

#ts { font-size: 13px; color: #2e1a37; text-align: right; margin-top: 8px; }
</style>
</head>
<body>

<nav class="navbar">
  <div class="navbar-logo">
    <div class="logo-mark">r</div>
    reducto
  </div>
  <span class="navbar-badge">internal</span>
  <span class="navbar-title">Agent Benchmark — Hard Probe Tracker</span>
</nav>

<div class="page">
<div class="title">⚡ 27 models × 22 probes = 594 runs</div>
<div class="stats-row" id="stats">
  <div class="stat"><span class="v">…</span><span class="l">loading</span></div>
</div>
<div class="bar-wrap"><div class="bar" id="bar" style="width:0%"></div></div>
<div class="grid-hdr">Leaderboard — sorted by score ↓  <span style="font-size:11px;font-weight:400;color:#6e4a7a;text-transform:none;letter-spacing:0">click any card for details</span></div>
<div class="grid" id="grid">
  <div class="mrow pending"><span class="mname" style="color:#484f58">Waiting for data…</span></div>
</div>

<!-- Model detail modal -->
<div id="model-modal-overlay">
  <div id="model-modal">
    <button class="modal-close" onclick="closeModal()">✕</button>
    <div id="modal-content"></div>
  </div>
</div>

<div class="section-hdr">Per-Probe Breakdown — click to expand</div>
<div id="probes"></div>

<div class="section-hdr" style="margin-top:24px">Insights</div>
<div class="insights-grid" id="insights-grid">
  <div class="ins-card"><div class="ins-title">Loading…</div></div>
</div>
<div id="ts"></div>
</div><!-- /.page -->
<div class="mtip" id="mtip">
  <button class="mtip-close" id="mtip-close">✕</button>
  <div class="mtip-title" id="mtip-title"></div>
  <div id="mtip-body"></div>
</div>

<script>
const COLORS = Array.from({length:30},(_,i)=>{
  const h=Math.round((i*137.508)%360), s=65+(i%3)*10, l=52+(i%2)*12;
  return `hsl(${h},${s}%,${l}%)`;
});
const clr = p => p>=85?'#3fb950':p>=65?'#2ea043':p>=40?'#c44ac7':p>=20?'#9d17a0':'#f85149';

let colorMap = {}, plotRendered = new Set(), lastData = null;

// ─── Stats row ────────────────────────────────────────────────────────
function renderStats(d) {
  const eta = d.eta_secs>0 ? `${Math.floor(d.eta_secs/60)}m ${String(d.eta_secs%60).padStart(2,'0')}s` : '—';
  document.getElementById('stats').innerHTML = [
    ['RUNS',     `${d.done}/${d.total}`],
    ['DONE',     `${d.pct}%`],
    ['RATE',     `${d.rate}<small style="font-size:18px">/min</small>`],
    ['ETA',      eta],
    ['FINISH @', d.eta_clock],
    ['IN FLIGHT',d.in_flight],
    ...(d.errors>0?[['ERRORS',`<span class="err-v">${d.errors}</span>`]]:[]),
  ].map(([l,v])=>`<div class="stat"><span class="v">${v}</span><span class="l">${l}</span></div>`).join('');
  document.getElementById('bar').style.width = d.pct + '%';
}

// ─── Model grid ───────────────────────────────────────────────────────
function renderGrid(models) {
  // assign colors once, stably
  models.forEach(m => { if(!colorMap[m.name]) colorMap[m.name]=COLORS[Object.keys(colorMap).length%COLORS.length]; });
  document.getElementById('grid').innerHTML = models.map((m, i) => {
    const col   = colorMap[m.name];
    const state = m.credit_fail ? 'credit'
                : m.finished ? 'done'
                : m.done>0   ? 'active'
                :               'pending';
    const tag   = m.credit_fail ? `<span class="mtag credit">💳 N/A</span>`
                : m.finished    ? `<span class="mtag done">${m.pts}/${m.max}</span>`
                : m.done>0      ? `<span class="mtag active">${m.done}/${m.total}</span>`
                :                 `<span class="mtag pending">—</span>`;
    const fill  = m.done>0 ? `<div class="mfill" style="width:${m.pct}%;background:${col}"></div>` : '';
    const pts   = m.credit_fail ? '' : m.done>0 ? m.pts : '';
    const ptsCol = m.credit_fail ? '#4e3358' : m.done>0 ? col : '#4e3358';
    const clickable = (m.finished || m.done>0) && !m.credit_fail;
    const rank = i+1;
    const medal = rank===1?'🥇':rank===2?'🥈':rank===3?'🥉':'';
    return `<div class="mrow ${state}" ${clickable?`onclick="openModelModal('${m.name.replace(/'/g,"\\'")}')"`:''}
      title="${clickable?'Click for details':''}">
      <div class="mdot" style="background:${m.credit_fail?'#2e1a37':col}"></div>
      <span class="mname" style="${m.credit_fail?'color:#4e3358':''}" title="${m.name}">${medal?medal+' ':''}${m.name}</span>
      <div class="mbar">${fill}</div>
      <span class="mpts" style="color:${ptsCol}">${pts}</span>
      ${tag}
    </div>`;
  }).join('');
}

// ─── Model detail modal ────────────────────────────────────────────────
let _lastData = null;

function openModelModal(name) {
  const d = _lastData;
  if (!d) return;
  const m = d.models.find(x => x.name === name);
  if (!m) return;

  const col = colorMap[name] || '#9d17a0';
  const methodology = d.probe_methodology || {};
  const probeResults = d.probe_results || {};
  const rank = d.models.indexOf(m) + 1;
  const medal = rank===1?'🥇 ':rank===2?'🥈 ':rank===3?'🥉 ':`#${rank} `;

  // Find missed probes (score < 3) for this model
  const missed = [];
  const perfect = [];
  for (const [probe, scores] of Object.entries(probeResults)) {
    const s = scores[name];
    if (s === undefined) continue;
    if (s < 3) missed.push({probe, score: s});
    else perfect.push(probe);
  }
  missed.sort((a,b) => a.score - b.score);

  // Stats
  const probeScores = Object.values(probeResults)
    .map(s => s[name]).filter(s => s !== undefined);
  const avgPer = probeScores.length ? (probeScores.reduce((a,b)=>a+b,0)/probeScores.length).toFixed(2) : '—';
  const zeros  = probeScores.filter(s=>s===0).length;
  const threes = probeScores.filter(s=>s===3).length;

  // Score color
  const scoreColor = clr(m.pct);

  // Missed probes HTML
  const missedHTML = missed.length === 0
    ? `<div class="modal-perfect">✅ Perfect score — all probes 3/3</div>`
    : missed.map(({probe, score}) => {
        const meth = methodology[probe] || '';
        // Split methodology: first sentence is "what", rest is "why"
        const sentences = meth.split(/(?<=\.)\s+/);
        const what = sentences[0] || '';
        const why  = sentences.slice(1).join(' ');
        const lostPts = 3 - score;
        const scoreColor = score===0?'#f85149':score===1?'#9d17a0':'#c44ac7';
        return `<div class="modal-missed-item">
          <div class="modal-missed-header">
            <span class="modal-probe-id">${probe}</span>
            <span class="modal-probe-score" style="color:${scoreColor}">${score}/3 &minus;${lostPts}pt${lostPts>1?'s':''}</span>
          </div>
          ${what ? `<div class="modal-probe-what">${what}</div>` : ''}
          ${why  ? `<div class="modal-probe-why">${why}</div>`   : ''}
        </div>`;
      }).join('');

  document.getElementById('modal-content').innerHTML = `
    <div class="modal-model-name">${medal}${name}</div>
    <div class="modal-score-row">
      <div>
        <div class="modal-score-big" style="color:${scoreColor}">${m.pts}<span style="font-size:20px;color:#6e4a7a">/${m.max}</span></div>
        <div class="modal-score-sub">${m.pct}% accuracy &nbsp;·&nbsp; ${m.done}/${m.total} probes run${m.finished?'':' (in progress)'}</div>
      </div>
      <div class="modal-pct-bar">
        <div class="modal-pct-fill" style="width:${m.pct}%;background:${scoreColor}"></div>
      </div>
    </div>

    <div class="modal-stat-row">
      <div class="modal-stat">
        <div class="modal-stat-val">${threes}</div>
        <div class="modal-stat-lbl">Perfect (3/3)</div>
      </div>
      <div class="modal-stat">
        <div class="modal-stat-val" style="color:${missed.length?'#c44ac7':'#3fb950'}">${missed.length}</div>
        <div class="modal-stat-lbl">Missed</div>
      </div>
      <div class="modal-stat">
        <div class="modal-stat-val">${avgPer}</div>
        <div class="modal-stat-lbl">Avg / probe</div>
      </div>
    </div>

    <div class="modal-section-title">${missed.length ? `${missed.length} probe${missed.length>1?'s':''} with lost points` : 'probe results'}</div>
    ${missedHTML}
  `;

  document.getElementById('model-modal-overlay').classList.add('open');
}

function closeModal() {
  document.getElementById('model-modal-overlay').classList.remove('open');
}
// Close on overlay click (outside modal box)
document.getElementById('model-modal-overlay').addEventListener('click', e => {
  if (e.target === document.getElementById('model-modal-overlay')) closeModal();
});
// Close on Escape
document.addEventListener('keydown', e => { if(e.key==='Escape') closeModal(); });

// Metallic card tilt — stat cards rotate in 3D as cursor moves over them.
// The sheen gradient angle is perpendicular to the tilt direction, so the
// light glimmers across whichever edge is angled toward the virtual light source.
function attachStatTilt() {
  document.querySelectorAll('.modal-stat').forEach(card => {
    card.addEventListener('mousemove', e => {
      const r  = card.getBoundingClientRect();
      const cx = r.left + r.width  / 2;
      const cy = r.top  + r.height / 2;
      const dx = (e.clientX - cx) / (r.width  / 2);  // -1 … 1
      const dy = (e.clientY - cy) / (r.height / 2);  // -1 … 1

      const rotX   = -dy * 6;            // tilt up/down
      const rotY   =  dx * 6;            // tilt left/right
      const dist   = Math.sqrt(dx*dx + dy*dy);   // 0 at center, ~1.4 at corner
      const str    = Math.min(dist, 1.2).toFixed(3);

      // Sheen sweeps perpendicular to the tilt axis —
      // atan2 gives direction of cursor from center, +90° flips it to edge normal
      const angle  = Math.atan2(dy, dx) * (180 / Math.PI) + 90;

      card.style.transform = `perspective(400px) rotateX(${rotX}deg) rotateY(${rotY}deg) scale(1.04)`;
      card.style.borderColor = `rgba(196,74,199,${(dist * 0.5).toFixed(2)})`;
      card.style.boxShadow   = `0 ${6+dist*8}px ${16+dist*20}px rgba(157,23,160,${(dist*.18).toFixed(2)})`;
      card.style.setProperty('--shine-angle', angle + 'deg');
      card.style.setProperty('--shine-str',   str);
    });

    card.addEventListener('mouseleave', () => {
      card.style.transform   = '';
      card.style.borderColor = '';
      card.style.boxShadow   = '';
      card.style.setProperty('--shine-str', '0');
    });
  });
}

// Re-attach tilt listeners every time the modal opens (cards are re-rendered)
const _origOpenModal = openModelModal;
openModelModal = name => { _origOpenModal(name); setTimeout(attachStatTilt, 0); };

// ─── Probe sections ───────────────────────────────────────────────────
function renderProbeSections(d) {
  const container = document.getElementById('probes');
  const probes = d.probe_order;
  const results = d.probe_results;
  const explain = d.probe_explain || {};
  const methodology = d.probe_methodology || {};
  const models  = d.models;

  // Overall score per model for X axis
  const overallScore = {};
  models.forEach(m => overallScore[m.name] = m.pts);

  probes.forEach(probe => {
    const pr = results[probe] || {};
    const modelNames = Object.keys(pr);
    const n = modelNames.length;
    const avg = n>0 ? (Object.values(pr).reduce((a,b)=>a+b,0)/n).toFixed(2) : '?';
    const avgColor = n>0 ? clr(parseFloat(avg)/3*100) : '#8b949e';
    const exp = explain[probe] || '';
    const meth = methodology[probe] || '';

    // Create details element if it doesn't exist
    let details = document.getElementById(`det-${probe}`);
    if (!details) {
      details = document.createElement('details');
      details.id = `det-${probe}`;
      details.innerHTML = `
        <summary>
          <span class="sum-probe">${probe}</span>
          <span class="sum-exp">${exp}</span>
          <span class="sum-n" id="sn-${probe}">${n} models</span>
          <span class="sum-avg" id="sa-${probe}" style="color:${avgColor}">avg ${avg}</span>
          ${meth ? `<button class="eye-btn" data-probe="${probe}" data-method="${meth.replace(/"/g,'&quot;')}">👀</button>` : ''}
        </summary>
        <div class="plot-wrap">
          <div class="plot-div" id="plot-${probe}"></div>
        </div>`;
      // Render plot when first opened
      details.addEventListener('toggle', () => {
        if (details.open && !plotRendered.has(probe)) {
          renderPlot(probe, lastData);
          plotRendered.add(probe);
        }
      });
      container.appendChild(details);
    } else {
      // Update summary stats only
      const sn = document.getElementById(`sn-${probe}`);
      const sa = document.getElementById(`sa-${probe}`);
      if (sn) sn.textContent = `${n} models`;
      if (sa) { sa.textContent = `avg ${avg}`; sa.style.color = avgColor; }
      // Re-render plot if open and data changed
      if (details.open && n > 0) {
        renderPlot(probe, lastData);
      }
    }
  });
}

function renderPlot(probe, d) {
  const div = document.getElementById(`plot-${probe}`);
  if (!div) return;
  const pr      = (d.probe_results||{})[probe] || {};
  const models  = d.models;
  const explain = (d.probe_explain||{})[probe] || '';

  const xs=[], ys=[], texts=[], colors=[], sizes=[];
  models.forEach(m => {
    if (!(m.name in pr)) return;
    xs.push(m.pts);
    ys.push(pr[m.name]);
    texts.push(m.name);
    colors.push(colorMap[m.name]||'#888');
    sizes.push(16);
  });

  const layout = {
    paper_bgcolor: '#160d1b', plot_bgcolor: '#0c0710',
    font: { color: '#9b85a0', family: "'Inter', system-ui, sans-serif", size: 13 },
    margin: { t: 36, r: 20, b: 60, l: 50 },
    xaxis: {
      title: { text: 'Overall Score (pts / 66)', font:{size:13} },
      range: [-2, 68], gridcolor: '#2e1a37', zerolinecolor: '#2e1a37',
      tickfont: {size:12}, tickcolor: '#4e3358',
    },
    yaxis: {
      title: { text: 'Probe Score (0–3)', font:{size:13} },
      range: [-0.3, 3.3], tickvals:[0,1,2,3],
      gridcolor: '#2e1a37', zerolinecolor: '#2e1a37',
      tickfont: {size:12}, tickcolor: '#4e3358',
    },
    title: { text: probe, font:{color:'#c44ac7',size:15}, x:0.02 },
    hovermode: 'closest',
    showlegend: false,
    dragmode: 'zoom',
  };

  // Reference lines at score = 0,1,2,3
  layout.shapes = [0,1,2,3].map(s=>({
    type:'line', xref:'paper', yref:'y',
    x0:0, x1:1, y0:s, y1:s,
    line:{color:s===3?'#1a5c2a':s===0?'#6e1b1b':'#2e1a37', width:1, dash:'dot'},
  }));

  const data = [{
    type: 'scatter', mode: 'markers',
    x: xs, y: ys, text: texts,
    marker: { color: colors, size: sizes, line:{color:'#0c0710',width:1.5}, opacity:0.9 },
    hovertemplate: '<b>%{text}</b><br>Overall: %{x}/66<br>This probe: %{y}/3<extra></extra>',
  }];

  const config = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['select2d','lasso2d','autoScale2d'],
    toImageButtonOptions: { format:'svg', filename: probe },
  };

  Plotly.react(div, data, layout, config);
}

// ─── Insights ─────────────────────────────────────────────────────────
// ─── Methodology tooltip (click-toggle) ───────────────────────────────
const mtip = document.getElementById('mtip');
let openBtn = null;

function positionMtip(btn) {
  const r = btn.getBoundingClientRect();
  const tw = mtip.offsetWidth, th = mtip.offsetHeight;
  // prefer to open to the right of the button, fall back to left
  let x = r.right + 14, y = r.top - 8;
  if (x + tw > window.innerWidth  - 8) x = r.left - tw - 14;
  if (y + th > window.innerHeight - 8) y = window.innerHeight - th - 8;
  if (y < 8) y = 8;
  mtip.style.left = x + 'px'; mtip.style.top = y + 'px';
}

function showMtip(btn) {
  document.getElementById('mtip-title').textContent = btn.dataset.probe + ' — how we measure it';
  document.getElementById('mtip-body').textContent  = btn.dataset.method;
  mtip.style.display = 'block';
  positionMtip(btn);
  openBtn = btn;
  btn.classList.add('eye-active');
}

function hideMtip() {
  mtip.style.display = 'none';
  if (openBtn) { openBtn.classList.remove('eye-active'); openBtn = null; }
}

// toggle on 👀 click; close on outside click or ✕
document.addEventListener('click', e => {
  if (e.target.closest('#mtip-close')) { hideMtip(); return; }
  const btn = e.target.closest('.eye-btn');
  if (btn) {
    e.stopPropagation();
    if (openBtn === btn) { hideMtip(); }
    else { hideMtip(); showMtip(btn); }
    return;
  }
  if (!mtip.contains(e.target)) hideMtip();
});

function renderInsights(d) {
  const results   = d.probe_results    || {};
  const probes    = d.probe_order      || [];
  const models    = d.models           || [];
  const explain   = d.probe_explain    || {};
  const method    = d.probe_methodology || {};
  const el = document.getElementById('insights-grid');

  // ── Card 1: Hardest probes ──────────────────────────────────────────
  // Probes with the lowest average score across all models that attempted them
  const probeStats = probes.map(p => {
    const scores = Object.values(results[p]||{});
    if (scores.length < 2) return null;
    const avg  = scores.reduce((a,b)=>a+b,0)/scores.length;
    const zero = scores.filter(s=>s===0).length;
    const full = scores.filter(s=>s===3).length;
    const part = scores.filter(s=>s===1||s===2).length;
    return {probe:p, avg, n:scores.length, zero, full, part};
  }).filter(Boolean).sort((a,b)=>a.avg-b.avg);

  const pillCls = avg => avg>=2?'pill-green':avg>=1?'pill-yellow':'pill-red';
  const ranks = ['gold','silver','bronze'];

  const hardestHTML = probeStats.slice(0,15).map((ps,i)=>{
    const rankCls = ranks[i]||'';
    const nums    = ['①','②','③','④','⑤','⑥','⑦','⑧','⑨','⑩','⑪','⑫','⑬','⑭','⑮'];
    const avgStr  = ps.avg.toFixed(2);
    const c       = pillCls(ps.avg);
    const exp     = explain[ps.probe] || '—';
    const meth    = (method[ps.probe]||'').replace(/'/g,'&#39;');
    return `<div class="ins-row">
      <div class="ins-rank ${rankCls}">${nums[i]||i+1}</div>
      <div class="ins-body">
        <span class="ins-probe-name">
          ${ps.probe}
          <button class="eye-btn" data-probe="${ps.probe}" data-method="${meth}">👀</button>
        </span>
        <span class="ins-explain">${exp}</span>
        <span class="ins-meta" style="color:#8b949e">
          ${ps.n} models &nbsp;·&nbsp;
          <span style="color:#f85149">${ps.zero} zero</span> &nbsp;
          <span style="color:#d29922">${ps.part} partial</span> &nbsp;
          <span style="color:#2ea043">${ps.full} perfect</span>
        </span>
      </div>
      <span class="score-pill ${c}">${avgStr}/3</span>
    </div>`;
  }).join('');

  // ── Card 2: Contradictory models ───────────────────────────────────
  // Models that score great on some probes and terrible on others (high variance)
  const modelVariance = models.map(m => {
    const scores = probes.map(p=>(results[p]||{})[m.name]).filter(s=>s!==undefined);
    if (scores.length < 4) return null;
    const avg = scores.reduce((a,b)=>a+b,0)/scores.length;
    const variance = scores.reduce((a,s)=>a+(s-avg)**2,0)/scores.length;
    // find best and worst probes for this model
    const probeScores = probes
      .map(p=>({p, s:(results[p]||{})[m.name]}))
      .filter(x=>x.s!==undefined)
      .sort((a,b)=>b.s-a.s);
    const best  = probeScores.slice(0,3).filter(x=>x.s===3).map(x=>x.p);
    const worst = probeScores.slice(-3).filter(x=>x.s===0).map(x=>x.p);
    return {name:m.name, pts:m.pts, done:m.done, variance, avg, best, worst};
  }).filter(Boolean).sort((a,b)=>b.variance-a.variance);

  const contraHTML = modelVariance.slice(0,5).map((mv,i)=>{
    const col = colorMap[mv.name]||'#888';
    const bestBadges  = mv.best.map(p=>`<span class="cbadge cbadge-hi" data-tip="${(explain[p]||p).replace(/"/g,'&quot;')}">✓ ${p.slice(0,14)}</span>`).join('');
    const worstBadges = mv.worst.map(p=>`<span class="cbadge cbadge-lo" data-tip="${(explain[p]||p).replace(/"/g,'&quot;')}">✗ ${p.slice(0,14)}</span>`).join('');
    return `<div class="ins-row">
      <div class="ins-rank ${ranks[i]||''}">${['①','②','③','④','⑤'][i]}</div>
      <div class="ins-body">
        <span class="ins-model-name" style="color:${col}">${mv.name}</span>
        <div class="contrast-badges">${bestBadges}${worstBadges}</div>
        <span class="ins-explain">σ²=${mv.variance.toFixed(2)} across ${mv.done} probes — avg ${mv.avg.toFixed(2)}/3. Excels at some API patterns, blind spots on others.</span>
      </div>
      <span class="score-pill ${pillCls(mv.avg)}">${mv.pts}/${mv.done*3}</span>
    </div>`;
  }).join('');

  // ── Card 3: Split-result probes ─────────────────────────────────────
  // Probes where SOME models nail it (3/3) and OTHERS totally bomb (0/3)
  // These reveal which API features separate "knows Reducto" from "doesn't"
  const splitProbes = probeStats.map(ps => {
    const scores = Object.values(results[ps.probe]||{});
    const zeros  = scores.filter(s=>s===0).length;
    const fulls  = scores.filter(s=>s===3).length;
    const split  = Math.min(zeros, fulls);
    return {...ps, split};
  }).filter(ps=>ps.split>=1).sort((a,b)=>b.split-a.split);

  const splitHTML = splitProbes.slice(0,15).map((ps,i)=>{
    const exp = explain[ps.probe]||'';
    // names of models that aced it
    const nums2   = ['①','②','③','④','⑤','⑥','⑦','⑧','⑨','⑩','⑪','⑫','⑬','⑭','⑮'];
    const acers  = Object.entries(results[ps.probe]||{}).filter(([,s])=>s===3).map(([m])=>m.replace(/ \(.+/,'').replace(' + thinking','').replace(' (reasoning=high',''));
    const bombers= Object.entries(results[ps.probe]||{}).filter(([,s])=>s===0).map(([m])=>m.replace(/ \(.+/,'').replace(' + thinking','').replace(' (reasoning=high',''));
    const acerStr  = acers.slice(0,3).join(', ')+(acers.length>3?` +${acers.length-3}`:'');
    const bombStr  = bombers.slice(0,3).join(', ')+(bombers.length>3?` +${bombers.length-3}`:'');
    const meth2    = (method[ps.probe]||'').replace(/'/g,'&#39;');
    return `<div class="ins-row">
      <div class="ins-rank ${ranks[i]||''}">${nums2[i]||i+1}</div>
      <div class="ins-body">
        <span class="ins-probe-name">
          ${ps.probe}
          <button class="eye-btn" data-probe="${ps.probe}" data-method="${meth2}">👀</button>
        </span>
        <span class="ins-explain">${exp}</span>
        <span class="ins-meta">
          <span style="color:#2ea043">✓ ${acerStr}</span><br>
          <span style="color:#f85149">✗ ${bombStr}</span>
        </span>
      </div>
      <span class="score-pill ${pillCls(ps.avg)}">${ps.full}✓ ${ps.zero}✗</span>
    </div>`;
  }).join('');

  const waiting = '<div class="ins-row"><span style="color:#484f58;font-size:12px">Need more data…</span></div>';

  const scoring = 'Each probe scores 0–3 points: (1) did the model use the right parameter at all? (2) was it set correctly? (3) did the Reducto API actually accept the call? Every model gets the same prompt and the same set of Reducto tools available.';
  el.innerHTML = `
    <div class="ins-card">
      <div class="ins-title">🔥 Hardest Probes
        <button class="eye-btn" data-probe="Scoring methodology" data-method="${scoring}">👀</button>
      </div>
      <div class="ins-subtitle">Ranked by avg score across all models — lowest first. Hover 👀 on any probe for exact methodology.</div>
      ${hardestHTML || waiting}
    </div>
    <div class="ins-card">
      <div class="ins-title">⚡ Contradictory Models
        <button class="eye-btn" data-probe="Variance scoring" data-method="We compute each model's score variance across all probes it has completed. High variance means the model is great at some API patterns and blind on others — not uniformly good or bad. Green badges = probes it aced (3/3). Red badges = probes it bombed (0/3). Hover any badge to see what that probe tests.">👀</button>
      </div>
      <div class="ins-subtitle">Ranked by score variance. Hover the green/red probe badges to see what each one tests.</div>
      ${contraHTML || waiting}
    </div>
    <div class="ins-card">
      <div class="ins-title">🎯 Discriminator Probes
        <button class="eye-btn" data-probe="Discriminator scoring" data-method="A discriminator probe is one where at least one model scores 3/3 (perfect) and at least one scores 0/3 (complete miss) on the same task. These are the most revealing probes — they show exactly which Reducto-specific knowledge separates models that know the API from those that don't. Ranked by how many models fall on each extreme.">👀</button>
      </div>
      <div class="ins-subtitle">Probes where some models ace it and others bomb — the clearest signal of Reducto-specific knowledge. Top 15.</div>
      ${splitHTML || waiting}
    </div>`;
}

// ─── Main tick ────────────────────────────────────────────────────────
async function tick() {
  try {
    const d = await fetch('/api/status').then(r=>r.json());
    if (d.error) {
      document.getElementById('stats').innerHTML = `<div class="stat"><span class="v err-v">⚠</span><span class="l">${d.error}</span></div>`;
      return;
    }
    lastData = d;
    _lastData = d;   // for model modal
    renderStats(d);
    renderGrid(d.models);
    renderProbeSections(d);
    renderInsights(d);
    document.getElementById('ts').textContent = '↻ ' + new Date().toLocaleTimeString();
  } catch(e) {
    document.getElementById('stats').innerHTML = `<div class="stat"><span class="v err-v">⚠</span><span class="l">fetch error</span></div>`;
  }
}

tick();
setInterval(tick, 3000);
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == '/api/status':
            body = json.dumps(parse_log()).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers(); self.wfile.write(body)
        else:
            body = HTML.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers(); self.wfile.write(body)

if __name__ == '__main__':
    port = 7842
    print(f"Tracker → http://localhost:{port}")
    HTTPServer(('0.0.0.0', port), Handler).serve_forever()
