#!/usr/bin/env python3
"""Synthetic data generation tracker — live progress dashboard (R1 + R2)."""

import re, time, os, json
from collections import defaultdict
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

# ── Round 1 (direct API) ───────────────────────────────────────────────────────
R1_CHECKPOINT   = "benchmark/data/synthetic_training/.checkpoint.jsonl"
R1_GEN_LOG      = "/tmp/synth_gen.log"
R1_WATCHDOG_LOG = "/tmp/synth_watchdog.log"
R1_VERIFIED_TRAIN = "benchmark/data/synthetic_training/verified_train.jsonl"
R1_VERIFIED_VAL   = "benchmark/data/synthetic_training/verified_val.jsonl"
R1_TARGET_RAW   = 36_669
R1_TARGET_FINAL = 20_338   # 16,669 train + 3,669 val after ~57% verification

# ── Round 2 (MCP format) ──────────────────────────────────────────────────────
R2_CHECKPOINT   = "benchmark/data/synthetic_training_mcp/.checkpoint_mcp.jsonl"
R2_GEN_LOG      = "/tmp/synth_gen_r2.log"
R2_WATCHDOG_LOG = "/tmp/synth_watchdog_r2.log"
R2_TARGET_RAW   = 36_669
R2_TARGET_FINAL = 33_002   # 90% of raw (no verification)

# ── Round 3 (gap-focused, Haiku only) ─────────────────────────────────────────
R3_CHECKPOINT   = "benchmark/data/synthetic_training_r3/.checkpoint_r3.jsonl"
R3_TARGET_RAW   = 17_669
R3_GAPS         = ["gap1_","gap2_","gap3_","gap4_",
                   "gap5_","gap6_","gap7_","gap8_","gap9_"]
R3_GAP_LABELS   = ["gap1 schema","gap2 arrays","gap3 preserve","gap4 four-hop",
                   "gap5 dual-doc","gap6 array_extract","gap7 upload","gap8 citations",
                   "gap9 termination"]

# ── DPO (termination preference pairs) ────────────────────────────────────────
DPO_CHECKPOINT  = "benchmark/data/dpo_termination/.checkpoint_dpo.jsonl"
DPO_TARGET      = 1_669

PORT = 8081

TEACHER_SHORT = {
    "Kimi K2.5 via Fireworks":         "Kimi K2.5",
    "MiniMax M2.7 (highspeed)":        "MiniMax M2.7",
    "Gemini 3.1 Flash Lite (preview)": "Gemini Flash Lite",
    "Qwen3.5-122B-A10B":               "Qwen3.5-122B",
    "Inception Mercury 2":             "Mercury 2",
    "GLM-5 Turbo":                     "GLM-5 Turbo",
    "GPT-5.4 Nano":                    "GPT-5.4 Nano",
    "StepFun Step-3.5 Flash":          "StepFun Flash",
}

# ── Parsers ────────────────────────────────────────────────────────────────────

def parse_checkpoint(path):
    teacher_counts = defaultdict(int)
    probe_counts   = defaultdict(int)
    total = 0
    if not os.path.exists(path):
        return 0, teacher_counts, probe_counts
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                d    = json.loads(line)
                meta = d.get("metadata", {})
                teacher = TEACHER_SHORT.get(meta.get("teacher", ""), meta.get("teacher", "unknown")[:18])
                probe   = meta.get("probe_id", "unknown")
                teacher_counts[teacher] += 1
                probe_counts[probe]     += 1
            except Exception:
                pass
    return total, teacher_counts, probe_counts


def parse_watchdog_log(path):
    """Return (last_heartbeat_count, last_restart_time, verified_announced)."""
    last_count    = 0
    last_restart  = None
    verified_done = False
    if not os.path.exists(path):
        return last_count, last_restart, verified_done
    with open(path, errors="replace") as f:
        for line in f:
            m = re.search(r"checkpoint=\s*([\d,]+)\s*lines", line)
            if m:
                last_count = int(m.group(1).replace(",", ""))
            if "gen started" in line.lower() or "Started PID" in line:
                ts = re.search(r"\[(\d{2}:\d{2}:\d{2})\]", line)
                if ts:
                    try:
                        last_restart = datetime.strptime(ts.group(1), "%H:%M:%S").replace(
                            year=datetime.now().year, month=datetime.now().month, day=datetime.now().day)
                    except Exception:
                        pass
            if "verification complete" in line.lower() or "verified_train" in line.lower():
                verified_done = True
    return last_count, last_restart, verified_done


def file_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def proc_alive(pattern):
    return bool(os.popen(f"pgrep -f '{pattern}'").read().strip())


# ── ETA tracking ──────────────────────────────────────────────────────────────

_hist_r1  = []
_hist_r2  = []
_hist_r3  = []
_hist_dpo = []

def update_history(hist, count):
    now = time.time()
    hist.append((now, count))
    cutoff = now - 600
    while hist and hist[0][0] < cutoff:
        hist.pop(0)

def epm(hist):
    if len(hist) < 2:
        return 0.0
    dt = hist[-1][0] - hist[0][0]
    dn = hist[-1][1] - hist[0][1]
    return (dn / (dt / 60)) if dt > 0 else 0.0

def eta_str(remaining, rate):
    if rate <= 0 or remaining <= 0:
        return "—"
    mins = remaining / rate
    if mins < 60:
        return f"{int(mins)}m"
    elif mins < 1440:
        return f"{int(mins/60)}h {int(mins%60)}m"
    return f"{int(mins/1440)}d {int((mins%1440)/60)}h"


# ── HTML ──────────────────────────────────────────────────────────────────────

def pill(label, alive, color=None):
    if alive:
        bg, fg, bd = ("#1a2a1a", "#6fcf7a", "#2a4a2a")
    else:
        bg, fg, bd = ("#2a1a1a", "#cf6f6f", "#4a2a2a")
    if color and alive:
        bg, fg, bd = (f"#1a1a2a", color, "#2a2a4a")
    sym = "●" if alive else "○"
    return f'<span style="background:{bg};color:{fg};border:1px solid {bd};padding:3px 10px;border-radius:20px;font-size:11px">{sym} {label}</span>'

def progress_bar(current, target, color="#7c4dff"):
    pct = min(100, current / max(1, target) * 100)
    return f"""
    <div style="background:#1e1e2a;border-radius:4px;height:10px;margin-top:8px">
      <div style="background:{color};width:{pct:.2f}%;height:10px;border-radius:4px;transition:width 0.4s"></div>
    </div>"""

def teacher_table(teacher_counts):
    total = max(1, sum(teacher_counts.values()))
    rows = ""
    for short in sorted(teacher_counts, key=lambda k: -teacher_counts[k]):
        n   = teacher_counts[short]
        pct = n / total * 100
        bw  = int(pct * 1.6)
        rows += f"""<tr>
          <td style="color:#c8c0e0;padding:3px 6px">{short}</td>
          <td style="color:#b8a0ff;font-weight:600;text-align:right;padding:3px 6px">{n:,}</td>
          <td style="color:#888;padding:3px 6px">{pct:.1f}%</td>
          <td style="padding:3px 6px"><div style="background:#7c4dff;height:7px;border-radius:3px;width:{bw}px;min-width:2px"></div></td>
        </tr>"""
    return rows or '<tr><td colspan="4" style="color:#444;padding:8px">No data yet…</td></tr>'


def build_page(d):
    # ── Round 1 ───────────────────────────────────────────────────────────
    r1_total      = d["r1_total"]
    r1_epm        = d["r1_epm"]
    r1_alive      = d["r1_alive"]
    r1_dog_alive  = d["r1_dog_alive"]
    r1_verified   = d["r1_verified"]
    r1_restart    = d["r1_restart"]
    r1_tc         = d["r1_teacher_counts"]

    r1_pct        = min(100, r1_total / R1_TARGET_RAW * 100)
    r1_eta        = eta_str(R1_TARGET_RAW - r1_total, r1_epm)
    r1_done       = r1_total >= R1_TARGET_RAW
    r1_restart_s  = r1_restart.strftime("%H:%M") if r1_restart else "—"

    # ── Round 2 ───────────────────────────────────────────────────────────
    r2_total      = d["r2_total"]
    r2_epm        = d["r2_epm"]
    r2_alive      = d["r2_alive"]
    r2_dog_alive  = d["r2_dog_alive"]
    r2_tc         = d["r2_teacher_counts"]

    r2_pct        = min(100, r2_total / R2_TARGET_RAW * 100)
    r2_eta        = eta_str(R2_TARGET_RAW - r2_total, r2_epm)
    r2_done       = r2_total >= R2_TARGET_RAW

    # ── Round 3 ───────────────────────────────────────────────────────────
    r3_total      = d["r3_total"]
    r3_epm        = d["r3_epm"]
    r3_alive      = d["r3_alive"]
    r3_gc         = d["r3_gap_counts"]

    r3_pct        = min(100, r3_total / R3_TARGET_RAW * 100)
    r3_eta        = eta_str(R3_TARGET_RAW - r3_total, r3_epm)
    r3_done       = r3_total >= R3_TARGET_RAW

    # ── Combined ──────────────────────────────────────────────────────────
    combined_train = r1_verified[0] + (int(r2_total * 0.9) if r2_done else 0)
    combined_val   = r1_verified[1] + (r2_total - int(r2_total * 0.9) if r2_done else 0)

    r1_status = "✅ Done" if r1_done else (f"⚙️ {r1_epm:.1f}/m · ETA {r1_eta}" if r1_alive else "⚠️ Stopped")
    r2_status = ("✅ Done" if r2_done else
                 (f"⚙️ {r2_epm:.1f}/m · ETA {r2_eta}" if r2_alive else "⚠️ Stopped"))
    r3_status = ("✅ Done" if r3_done else
                 (f"⚙️ {r3_epm:.1f}/m · ETA {r3_eta}" if r3_alive else "⚠️ Stopped"))

    # ── DPO ───────────────────────────────────────────────────────────────
    dpo_total = d["dpo_total"]
    dpo_epm   = d["dpo_epm"]
    dpo_alive = d["dpo_alive"]
    dpo_pct   = min(100, dpo_total / DPO_TARGET * 100)
    dpo_eta   = eta_str(DPO_TARGET - dpo_total, dpo_epm)
    dpo_done  = dpo_total >= DPO_TARGET
    dpo_status = ("✅ Done" if dpo_done else
                  (f"⚙️ {dpo_epm:.1f}/m · ETA {dpo_eta}" if dpo_alive else "⚠️ Stopped"))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="15">
<title>Reducto Synth Gen</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:#0e0e12; color:#d0d0e0; font-family:'Helvetica Neue',sans-serif; font-size:13px; padding:24px; }}
  h1 {{ font-size:18px; color:#e8e0ff; margin-bottom:4px; }}
  .sub {{ color:#555; font-size:11px; margin-bottom:20px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-bottom:16px; }}
  .box {{ background:#17171f; border:1px solid #2a2a3a; border-radius:8px; padding:16px 18px; }}
  .box h2 {{ font-size:11px; text-transform:uppercase; letter-spacing:0.6px; color:#555; margin-bottom:12px; }}
  .cards {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px; }}
  .card {{ background:#0e0e12; border:1px solid #2a2a3a; border-radius:6px; padding:10px 14px; min-width:100px; }}
  .card .val {{ font-size:20px; font-weight:700; color:#b8a0ff; }}
  .card .lbl {{ font-size:10px; color:#444; text-transform:uppercase; margin-top:2px; }}
  .status {{ font-size:12px; margin-bottom:10px; color:#aaa; }}
  .pills {{ display:flex; gap:6px; margin-bottom:12px; flex-wrap:wrap; }}
  .combined {{ background:#17171f; border:1px solid #2a2a3a; border-radius:8px; padding:16px 18px; margin-bottom:16px; }}
  table {{ width:100%; border-collapse:collapse; }}
  td,th {{ padding:3px 6px; }}
  tr:hover td {{ background:#1e1e2a; }}
  .footer {{ margin-top:20px; color:#333; font-size:10px; text-align:center; }}
  .tag {{ display:inline-block; font-size:10px; padding:1px 6px; border-radius:4px; margin-left:6px; }}
  .tag-r1 {{ background:#1a1a3a; color:#8888ff; border:1px solid #3a3a6a; }}
  .tag-r2 {{ background:#1a2a1a; color:#88cc88; border:1px solid #3a6a3a; }}
  .tag-mcp {{ background:#2a1a2a; color:#cc88cc; border:1px solid #6a3a6a; }}
  .tag-r3  {{ background:#1a2a2a; color:#88cccc; border:1px solid #3a6a6a; }}
</style>
</head>
<body>

<h1>Reducto Synth Gen Tracker</h1>
<p class="sub">Auto-refreshes every 15s &nbsp;·&nbsp; {datetime.now().strftime("%H:%M:%S")}</p>

<div class="grid">

  <!-- Round 1 -->
  <div class="box">
    <h2>Round 1 — Direct API <span class="tag tag-r1">R1</span></h2>
    <div class="pills">
      {pill("Gen", r1_alive)}
      {pill("Watchdog", r1_dog_alive)}
    </div>
    <p class="status">{r1_status}{"&nbsp;· last restart " + r1_restart_s if r1_restart_s != "—" else ""}</p>
    <div class="cards">
      <div class="card"><div class="val">{r1_total:,}</div><div class="lbl">Generated</div></div>
      <div class="card"><div class="val">{R1_TARGET_RAW:,}</div><div class="lbl">Target</div></div>
      <div class="card"><div class="val">{r1_pct:.1f}%</div><div class="lbl">Progress</div></div>
    </div>
    <div style="font-size:11px;color:#666;margin-bottom:6px">→ after L1+L3+L4 verification (~57%): <b style="color:#b8a0ff">{R1_TARGET_FINAL:,}</b> verified</div>
    <div style="font-size:11px;color:#666;margin-bottom:8px">→ split: <b style="color:#b8a0ff">16,669</b> train + <b style="color:#b8a0ff">3,669</b> val</div>
    {progress_bar(r1_total, R1_TARGET_RAW, "#7c4dff")}
    {"" if not r1_verified[0] else f'<div style="font-size:11px;color:#6fcf7a;margin-top:8px">✅ Verified: {r1_verified[0]:,} train + {r1_verified[1]:,} val</div>'}
    <div style="margin-top:14px">
      <table>
        <tr><th style="text-align:left;color:#444;padding:2px 6px">Teacher</th>
            <th style="text-align:right;color:#444;padding:2px 6px">Examples</th>
            <th style="color:#444;padding:2px 6px">Share</th><th></th></tr>
        {teacher_table(r1_tc)}
      </table>
    </div>
  </div>

  <!-- Round 2 -->
  <div class="box">
    <h2>Round 2 — MCP Format <span class="tag tag-r2">R2</span><span class="tag tag-mcp">MCP</span></h2>
    <div class="pills">
      {pill("Gen", r2_alive, "#88cc88")}
      {pill("Watchdog", r2_dog_alive, "#88cc88")}
    </div>
    <p class="status">{r2_status}</p>
    <div class="cards">
      <div class="card"><div class="val">{r2_total:,}</div><div class="lbl">Generated</div></div>
      <div class="card"><div class="val">{R2_TARGET_RAW:,}</div><div class="lbl">Target</div></div>
      <div class="card"><div class="val">{r2_pct:.1f}%</div><div class="lbl">Progress</div></div>
    </div>
    <div style="font-size:11px;color:#666;margin-bottom:6px">→ no verification (L4 only caught 68/4762 = 1.4% in R1)</div>
    <div style="font-size:11px;color:#666;margin-bottom:8px">→ split: <b style="color:#88cc88">~33,002</b> train + <b style="color:#88cc88">~3,667</b> val (90/10 raw)</div>
    <div style="font-size:11px;color:#666;margin-bottom:8px">→ 7 MCP tools · all params match mcp-server/src/index.ts exactly</div>
    {progress_bar(r2_total, R2_TARGET_RAW, "#52d9a0")}
    <div style="margin-top:14px">
      <table>
        <tr><th style="text-align:left;color:#444;padding:2px 6px">Teacher</th>
            <th style="text-align:right;color:#444;padding:2px 6px">Examples</th>
            <th style="color:#444;padding:2px 6px">Share</th><th></th></tr>
        {teacher_table(r2_tc) if r2_total > 0 else '<tr><td colspan="4" style="color:#444;padding:8px">Not started yet…</td></tr>'}
      </table>
    </div>
    </div>

  <!-- Round 3 -->
  <div class="box">
    <h2>Round 3 — Gap-Focused <span class="tag tag-r3">R3</span><span class="tag tag-mcp">Haiku</span></h2>
    <div class="pills">
      {pill("Gen", r3_alive, "#88cccc")}
    </div>
    <p class="status">{r3_status}</p>
    <div class="cards">
      <div class="card"><div class="val">{r3_total:,}</div><div class="lbl">Generated</div></div>
      <div class="card"><div class="val">{R3_TARGET_RAW:,}</div><div class="lbl">Target</div></div>
      <div class="card"><div class="val">{r3_pct:.1f}%</div><div class="lbl">Progress</div></div>
    </div>
    <div style="font-size:11px;color:#666;margin-bottom:8px">→ 9 gap probes · single teacher: Claude Haiku 4.5</div>
    {progress_bar(r3_total, R3_TARGET_RAW, "#38b2ac")}
    <div style="margin-top:14px">
      <table>
        <tr><th style="text-align:left;color:#444;padding:2px 6px">Gap</th>
            <th style="text-align:right;color:#444;padding:2px 6px">Examples</th></tr>
        {''.join(f'<tr><td style="color:#88cccc;padding:2px 6px;font-size:11px">{lbl}</td><td style="color:#b8a0ff;font-weight:600;text-align:right;padding:2px 6px">{r3_gc.get(pfx,0):,}</td></tr>' for pfx, lbl in zip(R3_GAPS, R3_GAP_LABELS)) or '<tr><td colspan="2" style="color:#444;padding:8px">No data yet…</td></tr>'}
      </table>
    </div>
  </div>

</div>

<!-- DPO -->
<div class="combined" style="margin-top:12px">
  <h2 style="font-size:11px;text-transform:uppercase;letter-spacing:0.6px;color:#555;margin-bottom:12px">
    DPO Preference Pairs — Termination Fix
    <span class="tag tag-r3" style="margin-left:6px">DPO</span>
    <span class="tag tag-mcp">Haiku</span>
  </h2>
  <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin-bottom:10px">
    <div class="card"><div class="val">{dpo_total:,}</div><div class="lbl">Pairs</div></div>
    <div class="card"><div class="val">{DPO_TARGET:,}</div><div class="lbl">Target</div></div>
    <div class="card"><div class="val">{dpo_pct:.1f}%</div><div class="lbl">Progress</div></div>
    <p class="status" style="margin:0">{dpo_status}</p>
  </div>
  <div style="font-size:11px;color:#666;margin-bottom:8px">
    chosen = final text (stop) · rejected = repeat last tool call (loop) · no extra teacher call for rejected
  </div>
  {progress_bar(dpo_total, DPO_TARGET, "#f6ad55")}
</div>

<!-- Combined -->
<div class="combined">
  <h2 style="font-size:11px;text-transform:uppercase;letter-spacing:0.6px;color:#555;margin-bottom:12px">Combined Training Dataset</h2>
  <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center">
    <div class="card"><div class="val">{combined_train:,}</div><div class="lbl">Train total</div></div>
    <div class="card"><div class="val">{combined_val:,}</div><div class="lbl">Val total</div></div>
    <div class="card"><div class="val">{combined_train + combined_val:,}</div><div class="lbl">Grand total</div></div>
    <div style="color:#555;font-size:11px;line-height:1.8">
      R1 verified: <b style="color:#b8a0ff">{r1_verified[0]:,}</b> train + <b style="color:#b8a0ff">{r1_verified[1]:,}</b> val<br>
      R2 raw: <b style="color:#88cc88">{int(r2_total*0.9):,}</b> train + <b style="color:#88cc88">{r2_total - int(r2_total*0.9):,}</b> val<br>
      → <code style="color:#aaa">benchmark/data/combined_training/</code>
    </div>
  </div>
</div>

<p class="footer">gen_tracker.py · port {PORT}</p>
</body>
</html>"""


# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        r1_total, r1_tc, r1_pc = parse_checkpoint(R1_CHECKPOINT)
        r2_total, r2_tc, r2_pc = parse_checkpoint(R2_CHECKPOINT)
        r3_total,  r3_tc,  r3_pc  = parse_checkpoint(R3_CHECKPOINT)
        dpo_total, _dpo_tc, _     = parse_checkpoint(DPO_CHECKPOINT)
        _, r1_restart, r1_ver_done = parse_watchdog_log(R1_WATCHDOG_LOG)

        update_history(_hist_r1,  r1_total)
        update_history(_hist_r2,  r2_total)
        update_history(_hist_r3,  r3_total)
        update_history(_hist_dpo, dpo_total)

        r1_verified_train = file_lines(R1_VERIFIED_TRAIN)
        r1_verified_val   = file_lines(R1_VERIFIED_VAL)

        # Build R3 gap counts from probe_id prefix
        r3_gap_counts = {}
        for probe_id, cnt in r3_pc.items():
            for g in R3_GAPS:
                if probe_id.startswith(g):
                    r3_gap_counts[g] = r3_gap_counts.get(g, 0) + cnt
                    break

        page = build_page({
            "r1_total":         r1_total,
            "r1_epm":           epm(_hist_r1),
            "r1_alive":         proc_alive("gen_synthetic_data.py"),
            "r1_dog_alive":     proc_alive("watchdog_synth.sh"),
            "r1_verified":      (r1_verified_train, r1_verified_val),
            "r1_restart":       r1_restart,
            "r1_teacher_counts": r1_tc,
            "r2_total":         r2_total,
            "r2_epm":           epm(_hist_r2),
            "r2_alive":         proc_alive("gen_mcp_data.py"),
            "r2_dog_alive":     proc_alive("watchdog_synth_r2.sh"),
            "r2_teacher_counts": r2_tc,
            "r3_total":         r3_total,
            "r3_epm":           epm(_hist_r3),
            "r3_alive":         proc_alive("gen_mcp_r3_gaps.py"),
            "r3_gap_counts":    r3_gap_counts,
            "dpo_total":        dpo_total,
            "dpo_epm":          epm(_hist_dpo),
            "dpo_alive":        proc_alive("gen_dpo_termination.py"),
        })

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(page.encode())


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = HTTPServer(("", PORT), Handler)
    print(f"Gen tracker → http://localhost:{PORT}  (auto-refresh 15s)")
    server.serve_forever()
