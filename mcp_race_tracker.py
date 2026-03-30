#!/usr/bin/env python3
"""MCP Race tracker — identical design to tracker.py, 7 MCP enterprise probes."""

import re, time, os, json
from collections import defaultdict
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler

LOG_PATH         = "/tmp/mcp_race.log"
TOTAL_RUNS       = 189      # 27 models × 7 probes (26 + ReductoLoRA baseline)
PROBES_PER_MODEL = 7

RE_START     = re.compile(r"\[(.+?)\]\s+(\S+)\s+…")           # findall over each line
RE_RESULT    = re.compile(r"^\s+\[(.+?)\]\s+(\S+)\s+→\s+(\d|N/A)/3\s+\(([0-9.]+)s\)")  # captures model+probe
RE_HEADER    = re.compile(r"MCP Probe:\s+(\d+)\s+models\s+x\s+(\d+)\s+probes\s+=\s+(\d+)\s+runs")
RE_MODEL_LST = re.compile(r"^\s+\*\s+(.+)$")
RE_ERROR     = re.compile(r"!!\s+(.+?)/(\S+):")
RE_CREDIT    = re.compile(r"Still 402|Credit error \(402\)|marking as N/A")

MCP_PROBE_METHODOLOGY = {
  "classify_route_extract":
    "Models are given a document and told to classify it first, then extract structured fields "
    "using 'schema_json' — the correct MCP parameter is 'schema'. "
    "Score: (1) did they call classify then extract in the right order? "
    "(2) did they use 'schema' not 'schema_json' in the extract call? "
    "(3) did the Reducto API actually accept the call? "
    "We're testing whether the model reads the MCP tool definition "
    "or hallucinates the REST API parameter name.",

  "upload_persist_array_extract":
    "Models get an 'expiring' presigned URL and must execute a 4-hop chain: "
    "upload the document, parse it with filter_blocks as a native array and persist the result, "
    "verify with get_job, then extract with array_extract=True and the 'schema' parameter. "
    "Score: (1) did they call upload and set persist_results? "
    "(2) did all 4 hops fire with correct param types? "
    "(3) did the API accept? "
    "We're testing multi-hop pipeline execution, upload-first discipline, "
    "and native array type safety vs. passing filter_blocks as a JSON string.",

  "split_preserve_extract_range":
    "Models must split a document into sections with table_cutoff. "
    "The prompt deliberately says 'allow' — the correct MCP enum is 'preserve'. "
    "They must also persist the parse result and use page_range from the split output for extraction. "
    "Score: (1) did they call split with table_cutoff at all? "
    "(2) did they use 'preserve' not 'allow', set persist, and target the page range? "
    "(3) did the API accept? "
    "We're testing whether models hallucinate the REST API enum or read the MCP tool definition carefully.",

  "dual_doc_fan_out":
    "Models must independently process two documents: parse both with persist_results, "
    "verify both with get_job, then extract from each using its own jobid:// reference. "
    "Six tool calls minimum, two separate job pipelines running in parallel. "
    "Score: (1) did they parse both docs with persist? "
    "(2) did they get_job for both and use distinct jobid:// references in each extract? "
    "(3) did the API accept? "
    "We're testing parallel job management and fan-out discipline — "
    "failure is reusing the same job ID or skipping one verification.",

  "agentic_parse_citations":
    "Models must parse with agentic_scopes passed as a native array (not a comma-separated string), "
    "set return_figure_images and persist_results, then extract with citations=True and deep_extract=True. "
    "Score: (1) did they use agentic_scopes at all in the parse call? "
    "(2) did they pass it as an array and include citations and deep_extract in extract? "
    "(3) did the API accept? "
    "We're testing agentic scope configuration, native array type safety, "
    "and whether the model knows bounding-box citations are a separate extract flag.",

  "extract_then_edit_form":
    "Models extract five fields from a financial document, then fill a form template "
    "with highlight_color='#FFFF00' and enable_overflow_pages=True. "
    "Score: (1) did they include highlight_color in the edit call? "
    "(2) did they use a yellow hex value plus the overflow flag, with correct 'schema' (not 'schema_json') in extract? "
    "(3) did the API accept? "
    "We're testing the two-tool extract→edit pipeline and niche edit parameters "
    "that have essentially no public training data.",

  "full_enterprise_pipeline":
    "Maximum difficulty: all 7 tools in sequence with every adversarial trap active simultaneously. "
    "upload → parse(filter_blocks[] array, agentic_scopes[] array, persist, OCR mode) "
    "→ get_job → split(table_cutoff='preserve', NOT 'allow') "
    "→ extract(schema, array_extract, deep_extract). "
    "Score: (1) did they use at least 4 of the 5 required tool types? "
    "(2) did they get all array params, the 'preserve' enum, and 'schema' correct across all 5 tools? "
    "(3) did the API accept? "
    "Expected: 0–1/3 for most models — only the best MCP-aware models reach 2–3/3.",
}

MCP_PROBE_EXPLAIN = {
  "classify_route_extract":
    "First figure out what type of document you're dealing with, then extract specific data from it. "
    "A two-step workflow that also requires using the correct MCP parameter name. "
    "Models trained on REST API docs will hallucinate 'schema_json' — the MCP tool uses 'schema'.",

  "upload_persist_array_extract":
    "A 4-step chain: rescue an expiring document by uploading it, parse with noise filtering and job persistence, "
    "verify the job completed, then extract a list of structured rows. "
    "Every link in the chain must work. The filter_blocks must be a real array, not a JSON string.",

  "split_preserve_extract_range":
    "Divide a document into named sections, then extract from just one section "
    "using the exact page numbers returned by the split. "
    "The prompt says 'allow' mode — the correct value is 'preserve'. "
    "Only models that actually read the MCP tool definition will get this right.",

  "dual_doc_fan_out":
    "Process two client documents fully in parallel. Each needs its own upload, "
    "its own parse job, its own job verification, and its own extraction. "
    "Crossing the wires between them, or skipping a verification step, is a failure.",

  "agentic_parse_citations":
    "Parse a document with AI correction for multiple content types (text AND tables) as a proper list, "
    "then extract with exact bounding-box locations for every field. "
    "Chain-of-custody compliance — every extracted value must be traceable to its position on the page.",

  "extract_then_edit_form":
    "Extract data from a financial statement, then use those values to fill a compliance form — "
    "with yellow highlighting and overflow page support. "
    "Two completely different operations using two completely different tools. "
    "Most models don't know the edit tool's niche parameters exist.",

  "full_enterprise_pipeline":
    "The full kitchen sink: upload an expiring document, parse it with four simultaneous settings "
    "(noise stripping, AI correction, OCR mode, job persistence), verify it completed, "
    "split into sections without cutting tables, then extract a list of holdings from one section only. "
    "All adversarial traps active at once. The only way to pass is to actually know the MCP surface.",
}


def _is_excluded(name: str) -> bool:
    return False   # no exclusions for MCP race


def parse_log():
    if not os.path.exists(LOG_PATH):
        return {"error": "Log not found — is the MCP race active?"}
    total_runs = TOTAL_RUNS
    probes_per_model = PROBES_PER_MODEL
    completed, errors = [], []
    credit_errors = set()
    all_models_listed = []
    total_results = 0
    seen = set()   # (model, probe) dedup
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
            # Take the largest run (first full run dominates; ignore smaller appended runs)
            if int(m.group(3)) > total_runs or total_runs == TOTAL_RUNS:
                total_runs = int(m.group(3)); probes_per_model = int(m.group(2))
        m = RE_MODEL_LST.match(line)
        if m:
            name = m.group(1).strip()
            all_models_listed.append(name)
        # RE_RESULT now anchors model+probe directly — no FIFO queue needed
        m = RE_RESULT.match(line)
        if m:
            model     = m.group(1)
            probe     = m.group(2)
            score_raw = m.group(3)
            secs      = float(m.group(4))
            key       = (model, probe)
            if key in seen:
                continue   # duplicate line (log re-read)
            seen.add(key)
            total_results += 1
            if score_raw == "N/A":
                credit_errors.add(key)
            else:
                if key not in credit_errors:
                    completed.append({"model": model, "probe": probe,
                                      "score": int(score_raw), "secs": secs})
        m = RE_ERROR.search(line)
        if m:
            errors.append(f"{m.group(1)}/{m.group(2)}")
    elapsed = time.time() - start_t

    # ── Inject saved JSON results ────────────────────────────────────────
    JSON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "benchmark/results/mcp_probe/by_model")
    if os.path.isdir(JSON_DIR):
        log_model_names = set(m for m, _ in seen)
        for line in lines:
            for ms in RE_START.finditer(line):
                log_model_names.add(ms.group(1))
        for fn in sorted(os.listdir(JSON_DIR)):
            if not fn.endswith(".json"): continue
            try:
                with open(os.path.join(JSON_DIR, fn)) as jf:
                    rows = json.load(jf)
            except Exception:
                continue
            if not rows: continue
            model_name = rows[0].get("model", "")
            if any(model_name.startswith(lm[:20]) or lm.startswith(model_name[:20])
                   for lm in log_model_names):
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

    live_done = total_results
    if len(completed) >= 10:
        w = completed[-40:]
        rate = len(w) / max(sum(c["secs"] for c in w) / 6, 1)
    else:
        rate = live_done / elapsed if elapsed > 0 else 0
    remaining_s = (total_runs - live_done) / rate if rate > 0 else 0

    canonical_names = {}
    listed_set = list(dict.fromkeys(all_models_listed))
    def _canonicalize(name):
        if name in canonical_names: return canonical_names[name]
        if name in listed_set:
            canonical_names[name] = name; return name
        matches = [m for m in listed_set if m.startswith(name)]
        if len(matches) == 1:
            canonical_names[name] = matches[0]; return matches[0]
        canonical_names[name] = name; return name

    model_pts  = defaultdict(int)
    model_done = defaultdict(int)
    probe_results = defaultdict(dict)
    probe_order   = []
    for c in completed:
        c["model"] = _canonicalize(c["model"])
        model_pts[c["model"]]  += c["score"]
        model_done[c["model"]] += 1
        if c["probe"] not in probe_results:
            probe_order.append(c["probe"])
        probe_results[c["probe"]][c["model"]] = c["score"]
    credit_errors = {(_canonicalize(m), p) for m, p in credit_errors}

    seen_order, seen_set = [], set()
    for name in listed_set:
        if name not in seen_set:
            seen_order.append(name); seen_set.add(name)
    for c in completed:
        if c["model"] not in seen_set:
            seen_order.append(c["model"]); seen_set.add(c["model"])

    credit_by_model = defaultdict(int)
    for (mod, _) in credit_errors:
        credit_by_model[mod] += 1

    max_pts = probes_per_model * 3
    models_out = []
    for name in seen_order:
        n = model_done[name]; pts = model_pts[name]
        n_credit = credit_by_model[name]
        active_pids = [p for p in __import__('os').popen("pgrep -f bench_mcp_probe").read().split() if p]
        run_active = len(active_pids) > 0
        fin = (n + n_credit) >= probes_per_model or (not run_active and n > 0)
        all_credit = (n == 0 and n_credit > 0)
        pct = round(pts / (n * 3) * 100, 1) if n > 0 else 0
        models_out.append({"name": name, "pts": pts, "max": max_pts,
                            "done": n, "total": probes_per_model,
                            "finished": fin, "pct": pct,
                            "credit_fail": all_credit, "n_credit": n_credit})
    models_out.sort(key=lambda m: (-m["pts"], -m["done"]))

    correct_done = sum(m["done"] + m["n_credit"] for m in models_out)
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
        "in_flight": total_runs - correct_done,
        "errors": len(errors),
        "models": models_out,
        "probe_results": {p: probe_results[p] for p in probe_order},
        "probe_order": probe_order,
        "probe_explain": MCP_PROBE_EXPLAIN,
        "probe_methodology": MCP_PROBE_METHODOLOGY,
    }


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MCP Race · AgentReducto</title>
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
  <span class="navbar-badge">MCP Race</span>
  <span class="navbar-title">🏎️ 27 models × 7 hard enterprise probes</span>
</nav>

<div class="page">
<div class="title">🏎️ MCP Race — 27 models × 7 enterprise probes = 189 runs</div>
<div style="text-align:center;font-size:12px;color:#6e4a7a;margin:-10px 0 12px;letter-spacing:0.03em">
  Tool definitions sourced from our open-source
  <a href="https://github.com/Reza2kn/reducto-agent-benchmark/tree/main/mcp-server" target="_blank"
     style="color:#c44ac7;text-decoration:none;border-bottom:1px solid #6e2a6e">Reducto MCP server</a>
  — exact parameter names, types, and descriptions. Models that read the tool spec pass; models that hallucinate REST API params fail.
</div>
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
    ...(d.errors>0?[['ERRORS',`<span class="err-v">${d.errors}</span>`]]:[]),
  ].map(([l,v])=>`<div class="stat"><span class="v">${v}</span><span class="l">${l}</span></div>`).join('');
  document.getElementById('bar').style.width = d.pct + '%';
}

// ─── Model grid ───────────────────────────────────────────────────────
function renderGrid(models) {
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

  const missed = [];
  const perfect = [];
  for (const [probe, scores] of Object.entries(probeResults)) {
    const s = scores[name];
    if (s === undefined) continue;
    if (s < 3) missed.push({probe, score: s});
    else perfect.push(probe);
  }
  missed.sort((a,b) => a.score - b.score);

  const probeScores = Object.values(probeResults)
    .map(s => s[name]).filter(s => s !== undefined);
  const avgPer = probeScores.length ? (probeScores.reduce((a,b)=>a+b,0)/probeScores.length).toFixed(2) : '—';
  const zeros  = probeScores.filter(s=>s===0).length;
  const threes = probeScores.filter(s=>s===3).length;

  const scoreColor = clr(m.pct);

  const missedHTML = missed.length === 0
    ? `<div class="modal-perfect">✅ Perfect score — all probes 3/3</div>`
    : missed.map(({probe, score}) => {
        const meth = methodology[probe] || '';
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
document.getElementById('model-modal-overlay').addEventListener('click', e => {
  if (e.target === document.getElementById('model-modal-overlay')) closeModal();
});
document.addEventListener('keydown', e => { if(e.key==='Escape') closeModal(); });

function attachStatTilt() {
  document.querySelectorAll('.modal-stat').forEach(card => {
    card.addEventListener('mousemove', e => {
      const r  = card.getBoundingClientRect();
      const cx = r.left + r.width  / 2;
      const cy = r.top  + r.height / 2;
      const dx = (e.clientX - cx) / (r.width  / 2);
      const dy = (e.clientY - cy) / (r.height / 2);

      const rotX   = -dy * 6;
      const rotY   =  dx * 6;
      const dist   = Math.sqrt(dx*dx + dy*dy);
      const str    = Math.min(dist, 1.2).toFixed(3);
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
      details.addEventListener('toggle', () => {
        if (details.open && !plotRendered.has(probe)) {
          renderPlot(probe, lastData);
          plotRendered.add(probe);
        }
      });
      container.appendChild(details);
    } else {
      const sn = document.getElementById(`sn-${probe}`);
      const sa = document.getElementById(`sa-${probe}`);
      if (sn) sn.textContent = `${n} models`;
      if (sa) { sa.textContent = `avg ${avg}`; sa.style.color = avgColor; }
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
      title: { text: 'Overall Score (pts / 21)', font:{size:13} },
      range: [-1, 22], gridcolor: '#2e1a37', zerolinecolor: '#2e1a37',
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

  layout.shapes = [0,1,2,3].map(s=>({
    type:'line', xref:'paper', yref:'y',
    x0:0, x1:1, y0:s, y1:s,
    line:{color:s===3?'#1a5c2a':s===0?'#6e1b1b':'#2e1a37', width:1, dash:'dot'},
  }));

  const data = [{
    type: 'scatter', mode: 'markers',
    x: xs, y: ys, text: texts,
    marker: { color: colors, size: sizes, line:{color:'#0c0710',width:1.5}, opacity:0.9 },
    hovertemplate: '<b>%{text}</b><br>Overall: %{x}/21<br>This probe: %{y}/3<extra></extra>',
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
const mtip = document.getElementById('mtip');
let openBtn = null;

function positionMtip(btn) {
  const r = btn.getBoundingClientRect();
  const tw = mtip.offsetWidth, th = mtip.offsetHeight;
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
  const modelVariance = models.map(m => {
    const scores = probes.map(p=>(results[p]||{})[m.name]).filter(s=>s!==undefined);
    if (scores.length < 3) return null;
    const avg = scores.reduce((a,b)=>a+b,0)/scores.length;
    const variance = scores.reduce((a,s)=>a+(s-avg)**2,0)/scores.length;
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
        <span class="ins-explain">σ²=${mv.variance.toFixed(2)} across ${mv.done} probes — avg ${mv.avg.toFixed(2)}/3. Excels at some MCP patterns, blind spots on others.</span>
      </div>
      <span class="score-pill ${pillCls(mv.avg)}">${mv.pts}/${mv.done*3}</span>
    </div>`;
  }).join('');

  // ── Card 3: Discriminator probes ────────────────────────────────────
  const splitProbes = probeStats.map(ps => {
    const scores = Object.values(results[ps.probe]||{});
    const zeros  = scores.filter(s=>s===0).length;
    const fulls  = scores.filter(s=>s===3).length;
    const split  = Math.min(zeros, fulls);
    return {...ps, split};
  }).filter(ps=>ps.split>=1).sort((a,b)=>b.split-a.split);

  const splitHTML = splitProbes.slice(0,15).map((ps,i)=>{
    const exp = explain[ps.probe]||'';
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

  const scoring = 'Each MCP probe scores 0–3 points: (1) did the model use the right parameter at all? (2) was it set correctly — right value, right type? (3) did the Reducto API actually accept the call? Every model gets the same prompt and the same tool definitions — sourced directly from our open-source Reducto MCP server (github.com/Reza2kn/reducto-agent-benchmark/tree/main/mcp-server). Adversarial prompts deliberately use REST API param names (e.g. "schema_json", "allow") to surface models that hallucinate instead of reading the tool spec.';
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
        <button class="eye-btn" data-probe="Variance scoring" data-method="We compute each model's score variance across all probes it has completed. High variance means the model is great at some MCP patterns and blind on others — not uniformly good or bad. Green badges = probes it aced (3/3). Red badges = probes it bombed (0/3). Hover any badge to see what that probe tests.">👀</button>
      </div>
      <div class="ins-subtitle">Ranked by score variance. Hover the green/red probe badges to see what each one tests.</div>
      ${contraHTML || waiting}
    </div>
    <div class="ins-card">
      <div class="ins-title">🎯 Discriminator Probes
        <button class="eye-btn" data-probe="Discriminator scoring" data-method="A discriminator probe is one where at least one model scores 3/3 (perfect) and at least one scores 0/3 (complete miss) on the same task. These are the most revealing probes — they show exactly which MCP-specific knowledge separates models that read the tool definition from those that hallucinate REST API params. Ranked by how many models fall on each extreme.">👀</button>
      </div>
      <div class="ins-subtitle">Probes where some models ace it and others bomb — the clearest signal of real MCP knowledge vs. REST API hallucination. Top 7.</div>
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
    _lastData = d;
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
    port = 8082
    print(f"MCP Race tracker → http://localhost:{port}")
    HTTPServer(('0.0.0.0', port), Handler).serve_forever()
