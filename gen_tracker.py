#!/usr/bin/env python3
"""Synthetic data generation tracker — live progress dashboard."""

import re, time, os, json
from collections import defaultdict
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler

CHECKPOINT  = "benchmark/data/synthetic_training/.checkpoint.jsonl"
GEN_LOG     = "/tmp/synth_gen.log"
WATCHDOG_LOG = "/tmp/synth_watchdog.log"
PORT        = 8081

TARGET_CHECKPOINT = 30_000   # checkpoint target
TARGET_FINAL      = 18_000   # after consensus verification (~60% pass rate)

TEACHERS = [
    "Kimi K2.5 via Fireworks",
    "MiniMax M2.7 (highspeed)",
    "Gemini 3.1 Flash Lite (preview)",
    "Qwen3.5-122B-A10B",
    "Inception Mercury 2",
]

TEACHER_SHORT = {
    "Kimi K2.5 via Fireworks":         "Kimi K2.5",
    "MiniMax M2.7 (highspeed)":        "MiniMax M2.7",
    "Gemini 3.1 Flash Lite (preview)": "Gemini Flash Lite",
    "Qwen3.5-122B-A10B":               "Qwen3.5-122B",
    "Inception Mercury 2":             "Mercury 2",
}

RE_SEED = re.compile(
    r"\[\s*(\d+)/116\]\s+(\S+)\s+\+\s*(\d+)(?:\s+L4✗(\d+))?\s+total=([\d,]+)/30,000"
)
RE_TIMEOUT = re.compile(r"\[timeout\]\s+(.+?)\s+/\s+(\S+)")
RE_L4_DROP = re.compile(r"L4✗(\d+)")

# ── Parse gen log ──────────────────────────────────────────────────────────────

def parse_gen_log():
    seeds_done = []
    timeouts   = defaultdict(int)   # teacher → count
    total_l4_rejected = 0
    last_restart = None

    if not os.path.exists(GEN_LOG):
        return seeds_done, timeouts, total_l4_rejected, last_restart

    with open(GEN_LOG, errors="replace") as f:
        for line in f:
            # Seed completion line
            m = RE_SEED.search(line)
            if m:
                idx, name, added, l4x, total = m.groups()
                l4x = int(l4x) if l4x else 0
                total_l4_rejected += l4x
                seeds_done.append({
                    "idx": int(idx), "name": name,
                    "added": int(added), "l4_dropped": l4x,
                    "total": int(total.replace(",", "")),
                })

            # Per-teacher timeout
            m = RE_TIMEOUT.search(line)
            if m:
                teacher, _ = m.groups()
                # normalise to short name
                for full, short in TEACHER_SHORT.items():
                    if full[:20] in teacher or teacher in full:
                        timeouts[short] += 1
                        break
                else:
                    timeouts[teacher[:20]] += 1

            # Track last restart timestamp
            if "Synthetic data gen started:" in line:
                ts_str = line.split("Synthetic data gen started:")[-1].strip()
                try:
                    last_restart = datetime.strptime(ts_str, "%a %b %d %H:%M:%S %Z %Y")
                except Exception:
                    pass

    return seeds_done, timeouts, total_l4_rejected, last_restart


# ── Parse checkpoint ───────────────────────────────────────────────────────────

def parse_checkpoint():
    teacher_counts = defaultdict(int)
    probe_counts   = defaultdict(int)
    total = 0

    if not os.path.exists(CHECKPOINT):
        return total, teacher_counts, probe_counts

    with open(CHECKPOINT, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                d = json.loads(line)
                meta    = d.get("metadata", {})
                teacher = meta.get("teacher", "unknown")
                probe   = meta.get("probe_id", "unknown")
                short   = TEACHER_SHORT.get(teacher, teacher[:18])
                teacher_counts[short] += 1
                probe_counts[probe] += 1
            except Exception:
                pass

    return total, teacher_counts, probe_counts


# ── Check process health ───────────────────────────────────────────────────────

def gen_is_alive():
    out = os.popen("pgrep -f gen_synthetic_data").read().strip()
    return bool(out)

def watchdog_is_alive():
    out = os.popen("pgrep -f watchdog_synth").read().strip()
    return bool(out)

def hf_sync_is_alive():
    out = os.popen("pgrep -f sync_to_hf").read().strip()
    return bool(out)


# ── Throughput estimate ────────────────────────────────────────────────────────

_history = []   # [(timestamp, line_count), ...]

def update_history(count):
    now = time.time()
    _history.append((now, count))
    # keep last 10 minutes of history
    cutoff = now - 600
    while _history and _history[0][0] < cutoff:
        _history.pop(0)

def examples_per_minute():
    if len(_history) < 2:
        return 0.0
    dt = _history[-1][0] - _history[0][0]
    dn = _history[-1][1] - _history[0][1]
    if dt <= 0:
        return 0.0
    return dn / (dt / 60)


# ── HTML page ─────────────────────────────────────────────────────────────────

def build_page(data: dict) -> str:
    cp_total        = data["cp_total"]
    teacher_counts  = data["teacher_counts"]
    probe_counts    = data["probe_counts"]
    seeds_done      = data["seeds_done"]
    timeouts        = data["timeouts"]
    l4_rejected     = data["l4_rejected"]
    gen_alive       = data["gen_alive"]
    watchdog_alive  = data["watchdog_alive"]
    hf_alive        = data["hf_alive"]
    epm             = data["epm"]
    last_restart    = data["last_restart"]

    cp_pct   = min(100, cp_total / TARGET_CHECKPOINT * 100)
    n_seeds  = len(set(s["name"] for s in seeds_done))
    eta_str  = "—"
    if epm > 0:
        remaining = TARGET_CHECKPOINT - cp_total
        mins = remaining / epm
        if mins < 60:
            eta_str = f"{int(mins)}m"
        elif mins < 1440:
            eta_str = f"{int(mins/60)}h {int(mins%60)}m"
        else:
            eta_str = f"{int(mins/1440)}d {int((mins%1440)/60)}h"

    # L4 stats
    total_attempted = sum(s["added"] + s["l4_dropped"] for s in seeds_done)
    l4_accept_rate  = (
        (total_attempted - l4_rejected) / total_attempted * 100
        if total_attempted > 0 else 0
    )

    # per-teacher rows
    teacher_rows = ""
    total_counted = max(1, sum(teacher_counts.values()))
    for t in TEACHER_SHORT.values():
        n   = teacher_counts.get(t, 0)
        pct = n / total_counted * 100
        tw  = TEACHER_SHORT.get(t, t)
        bar_w = int(pct * 1.8)   # max ~180px for 100%
        to  = timeouts.get(t, 0)
        teacher_rows += f"""
        <tr>
          <td class="tname">{t}</td>
          <td class="tnum">{n:,}</td>
          <td class="tpct">{pct:.1f}%</td>
          <td class="tbar"><div class="bar-fill" style="width:{bar_w}px"></div></td>
          <td class="tto">{('⏱ ' + str(to)) if to else '—'}</td>
        </tr>"""

    # seed log (last 20)
    seed_rows = ""
    for s in reversed(seeds_done[-20:]):
        accept = s["added"]
        reject = s["l4_dropped"]
        total_s = accept + reject
        ar = f"{accept/total_s*100:.0f}%" if total_s else "—"
        seed_rows += f"""
        <tr>
          <td class="snum">[{s['idx']:>3}/116]</td>
          <td class="sname">{s['name']}</td>
          <td class="sadd">+{s['added']:,}</td>
          <td class="sl4">L4✗{s['l4_dropped']:,}</td>
          <td class="sar">{ar} pass</td>
        </tr>"""

    # status pills
    def pill(label, alive):
        cls = "pill-ok" if alive else "pill-dead"
        sym = "●" if alive else "○"
        return f'<span class="{cls}">{sym} {label}</span>'

    gen_pill = pill("Gen", gen_alive)
    dog_pill = pill("Watchdog", watchdog_alive)
    hf_pill  = pill("HF sync", hf_alive)

    restart_str = last_restart.strftime("%H:%M:%S") if last_restart else "unknown"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="15">
<title>Reducto Synth Gen Tracker</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0e0e12; color: #d0d0e0; font-family: 'Helvetica Neue', sans-serif; font-size: 13px; padding: 20px; }}
  h1 {{ font-size: 18px; color: #e8e0ff; margin-bottom: 4px; }}
  .sub {{ color: #666; font-size: 11px; margin-bottom: 20px; }}

  /* stat cards */
  .cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }}
  .card {{ background: #17171f; border: 1px solid #2a2a3a; border-radius: 8px; padding: 14px 18px; min-width: 140px; }}
  .card .val {{ font-size: 26px; font-weight: 700; color: #b8a0ff; letter-spacing: -0.5px; }}
  .card .lbl {{ font-size: 10px; color: #555; text-transform: uppercase; margin-top: 2px; letter-spacing: 0.5px; }}

  /* progress bar */
  .prog-wrap {{ background: #17171f; border: 1px solid #2a2a3a; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px; }}
  .prog-label {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 12px; color: #888; }}
  .prog-track {{ background: #1e1e2a; border-radius: 4px; height: 10px; }}
  .prog-fill  {{ background: linear-gradient(90deg, #7c4dff, #b47aff); border-radius: 4px; height: 10px; transition: width 0.4s; }}

  /* pills */
  .pills {{ display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }}
  .pill-ok   {{ background: #1a2a1a; color: #6fcf7a; border: 1px solid #2a4a2a; padding: 3px 10px; border-radius: 20px; font-size: 11px; }}
  .pill-dead {{ background: #2a1a1a; color: #cf6f6f; border: 1px solid #4a2a2a; padding: 3px 10px; border-radius: 20px; font-size: 11px; }}

  /* tables */
  .section {{ background: #17171f; border: 1px solid #2a2a3a; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; }}
  .section h2 {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.6px; color: #666; margin-bottom: 10px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  td {{ padding: 4px 6px; }}
  tr:hover td {{ background: #1e1e2a; }}

  .tname, .sname {{ color: #c8c0e0; }}
  .tnum, .sadd {{ color: #b8a0ff; font-weight: 600; }}
  .tpct, .sar  {{ color: #888; }}
  .sl4         {{ color: #cf8060; }}
  .snum        {{ color: #555; font-family: monospace; }}
  .tto         {{ color: #888; font-size: 11px; }}
  .bar-fill {{ background: #7c4dff; height: 8px; border-radius: 3px; min-width: 2px; }}

  .footer {{ margin-top: 20px; color: #333; font-size: 10px; text-align: center; }}
</style>
</head>
<body>

<h1>Reducto Synth Gen</h1>
<p class="sub">Student: Qwen3.5-35B-A3B &nbsp;·&nbsp; Last restart: {restart_str} &nbsp;·&nbsp; Refreshes every 15s</p>

<div class="pills">
  {gen_pill}
  {dog_pill}
  {hf_pill}
</div>

<div class="cards">
  <div class="card"><div class="val">{cp_total:,}</div><div class="lbl">Checkpoint entries</div></div>
  <div class="card"><div class="val">{TARGET_CHECKPOINT:,}</div><div class="lbl">Target entries</div></div>
  <div class="card"><div class="val">{n_seeds}/116</div><div class="lbl">Seeds completed</div></div>
  <div class="card"><div class="val">{epm:.1f}/m</div><div class="lbl">Examples / min</div></div>
  <div class="card"><div class="val">{eta_str}</div><div class="lbl">ETA to target</div></div>
  <div class="card"><div class="val">{l4_accept_rate:.0f}%</div><div class="lbl">L4 accept rate</div></div>
</div>

<div class="prog-wrap">
  <div class="prog-label">
    <span>Checkpoint progress → {cp_pct:.1f}%</span>
    <span>{cp_total:,} / {TARGET_CHECKPOINT:,} &nbsp;→&nbsp; ~{TARGET_FINAL:,} verified final</span>
  </div>
  <div class="prog-track"><div class="prog-fill" style="width:{cp_pct:.2f}%"></div></div>
</div>

<div class="section">
  <h2>Teacher Distribution</h2>
  <table>
    <tr><th style="text-align:left;color:#444;padding:2px 6px">Teacher</th>
        <th style="text-align:right;color:#444;padding:2px 6px">Examples</th>
        <th style="text-align:right;color:#444;padding:2px 6px">Share</th>
        <th style="color:#444;padding:2px 6px">Bar</th>
        <th style="color:#444;padding:2px 6px">Timeouts</th>
    </tr>
    {teacher_rows}
  </table>
</div>

<div class="section">
  <h2>Seed Log (last 20)</h2>
  <table>
    <tr><th style="color:#444;padding:2px 6px">#</th>
        <th style="text-align:left;color:#444;padding:2px 6px">Seed</th>
        <th style="color:#444;padding:2px 6px">Added</th>
        <th style="color:#444;padding:2px 6px">L4 drops</th>
        <th style="color:#444;padding:2px 6px">Pass rate</th>
    </tr>
    {seed_rows if seed_rows else '<tr><td colspan="5" style="color:#444;padding:8px">No seeds completed yet…</td></tr>'}
  </table>
</div>

<p class="footer">gen_tracker.py · port {PORT}</p>
</body>
</html>"""


# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        seeds_done, timeouts, l4_rejected, last_restart = parse_gen_log()
        cp_total, teacher_counts, probe_counts = parse_checkpoint()
        update_history(cp_total)
        epm = examples_per_minute()

        page = build_page({
            "cp_total": cp_total,
            "teacher_counts": teacher_counts,
            "probe_counts": probe_counts,
            "seeds_done": seeds_done,
            "timeouts": timeouts,
            "l4_rejected": l4_rejected,
            "gen_alive": gen_is_alive(),
            "watchdog_alive": watchdog_is_alive(),
            "hf_alive": hf_sync_is_alive(),
            "epm": epm,
            "last_restart": last_restart,
        })

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(page.encode())


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = HTTPServer(("", PORT), Handler)
    print(f"Gen tracker → http://localhost:{PORT}  (auto-refresh 15s)")
    server.serve_forever()
