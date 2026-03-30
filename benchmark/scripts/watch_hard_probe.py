#!/usr/bin/env python3
"""
watch_hard_probe.py — Real-time dashboard for the hard probe run.

Tails /tmp/hard_probe_run.log, parses scores as they arrive, and redraws
a live leaderboard + progress bar every --interval seconds.

Usage:
    python watch_hard_probe.py
    python watch_hard_probe.py --log /tmp/hard_probe_run.log --interval 5
"""

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------

# Matches:  [Model Name] probe_id …
RE_START  = re.compile(r"^\s+\[(.+?)\]\s+(\S+)\s+…")
# Matches:    → 3/3 (12.3s) | present=True correct=True api=True
RE_RESULT = re.compile(r"→\s+(\d)/3\s+\(([0-9.]+)s\)\s+\|.*present=(\w+).*correct=(\w+).*api=(\w+)")
# Matches: Param probe: 27 models × 22 probes = 594 runs
RE_HEADER = re.compile(r"Param probe:\s+(\d+)\s+models.*?(\d+)\s+probes\s+=\s+(\d+)\s+runs")

SCORE_COLORS = {3: "\033[92m", 2: "\033[93m", 1: "\033[93m", 0: "\033[91m"}
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"


def parse_log(log_path: str) -> dict:
    """
    Read the log file and extract all completed probe results plus in-flight probes.
    Returns a dict with structured data for display.
    """
    state = {
        "total_models": 27,
        "total_probes_per_model": 22,
        "total_runs": 594,
        "completed": [],       # list of (model, probe_id, score, seconds)
        "in_flight": [],       # list of (model, probe_id)  — started but no result yet
        "start_time": None,
        "last_line_time": None,
    }

    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return state

    if not lines:
        return state

    st = os.stat(log_path)
    # birthtime = file creation on macOS; fallback to mtime if unavailable
    state["start_time"] = getattr(st, "st_birthtime", st.st_mtime)
    pending: list[tuple[str, str]] = []   # (model, probe_id) started but not resolved

    for line in lines:
        # Header line — update totals
        m = RE_HEADER.search(line)
        if m:
            state["total_models"] = int(m.group(1))
            state["total_probes_per_model"] = int(m.group(2))
            state["total_runs"] = int(m.group(3))
            continue

        # Probe started
        m = RE_START.match(line)
        if m:
            pending.append((m.group(1), m.group(2)))
            continue

        # Probe result
        m = RE_RESULT.search(line)
        if m:
            score   = int(m.group(1))
            elapsed = float(m.group(2))
            if pending:
                model, probe_id = pending.pop(0)
                state["completed"].append((model, probe_id, score, elapsed))
            continue

    # Whatever is still in pending is in-flight
    state["in_flight"] = list(pending)
    return state


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def bar(fraction: float, width: int = 40) -> str:
    filled = int(fraction * width)
    pct = fraction * 100
    b = "█" * filled + "░" * (width - filled)
    color = GREEN if fraction >= 0.9 else YELLOW if fraction >= 0.5 else CYAN
    return f"{color}{b}{RESET} {pct:5.1f}%"


def score_str(s: int) -> str:
    c = SCORE_COLORS.get(s, "")
    return f"{c}{s}/3{RESET}"


def eta_str(elapsed_s: float, done: int, total: int) -> str:
    if done == 0:
        return "calculating..."
    if done >= total:
        return f"{GREEN}DONE{RESET}"
    rate = done / elapsed_s          # probes per second
    remaining = (total - done) / rate
    eta = timedelta(seconds=int(remaining))
    finish = datetime.now() + timedelta(seconds=remaining)
    return f"{YELLOW}{eta}{RESET}  (done ~{finish.strftime('%H:%M:%S')})"


def render(state: dict, elapsed_s: float):
    total    = state["total_runs"]
    done     = len(state["completed"])
    inflight = len(state["in_flight"])
    remaining = total - done

    # Per-model score aggregation
    model_scores: dict[str, dict] = defaultdict(lambda: {"done": 0, "score": 0, "probes": []})
    for model, probe_id, score, secs in state["completed"]:
        model_scores[model]["done"]   += 1
        model_scores[model]["score"]  += score
        model_scores[model]["probes"].append((probe_id, score))

    probes_per_model = state["total_probes_per_model"]
    max_pts = probes_per_model * 3

    os.system("clear 2>/dev/null || cls 2>/dev/null || print('\033[H\033[2J', end='')")
    print(f"{BOLD}{CYAN}━━━ Hard Probe Live Dashboard ━━━{RESET}  {DIM}{datetime.now().strftime('%H:%M:%S')}{RESET}")
    print()

    # Overall progress
    print(f"  {BOLD}Progress:{RESET}  {bar(done / total if total else 0)}  {done}/{total} probes")
    print(f"  {BOLD}In-flight:{RESET} {inflight} concurrent    {BOLD}Elapsed:{RESET} {timedelta(seconds=int(elapsed_s))}")
    print(f"  {BOLD}ETA:      {RESET} {eta_str(elapsed_s, done, total)}")
    print()

    # Currently running
    if state["in_flight"]:
        print(f"  {BOLD}Currently running:{RESET}")
        seen = set()
        for model, probe_id in state["in_flight"][-12:]:
            key = (model, probe_id)
            if key not in seen:
                seen.add(key)
                short_model = model.split()[0] + " " + (model.split()[1] if len(model.split()) > 1 else "")
                print(f"    {DIM}▶ {short_model[:30]:<30}  {probe_id}{RESET}")
        print()

    # Per-model leaderboard
    print(f"  {BOLD}{'Model':<44} {'Score':>8}  {'Progress':<28} {'Avg/probe'}{RESET}")
    print(f"  {'─'*90}")

    sorted_models = sorted(
        model_scores.items(),
        key=lambda x: (-x[1]["score"], -x[1]["done"])
    )

    for model, data in sorted_models:
        n      = data["done"]
        pts    = data["score"]
        pct    = n / probes_per_model if probes_per_model else 0
        avg    = pts / n if n else 0

        # Mini progress bar per model
        filled = int(pct * 20)
        mini   = "█" * filled + "░" * (20 - filled)
        color  = GREEN if pct >= 1.0 else YELLOW if pct >= 0.5 else CYAN

        score_color = GREEN if avg >= 2.5 else YELLOW if avg >= 1.5 else RED
        score_display = f"{score_color}{pts:>3}/{max_pts}{RESET}" if n > 0 else f"{DIM}  —/{max_pts}{RESET}"

        status = "✓" if n >= probes_per_model else f"{n}/{probes_per_model}"
        print(f"  {model:<44} {score_display}   {color}{mini}{RESET} {pct*100:4.0f}%  {score_color}{avg:.2f}{RESET}")

    # Models not yet started
    all_models_started = {m for m, *_ in state["completed"]}
    all_models_started |= {m for m, _ in state["in_flight"]}

    # Score distribution for completed probes
    if state["completed"]:
        dist = {0: 0, 1: 0, 2: 0, 3: 0}
        for _, _, score, _ in state["completed"]:
            dist[score] += 1
        print()
        print(f"  {BOLD}Score distribution:{RESET}", end="")
        for s in [3, 2, 1, 0]:
            c = SCORE_COLORS.get(s, "")
            print(f"  {c}{s}/3:{RESET} {dist[s]}", end="")
        overall_pct = sum(s * c for s, c in dist.items()) / (done * 3) * 100 if done > 0 else 0
        print(f"   {BOLD}Overall accuracy: {GREEN if overall_pct >= 70 else YELLOW}{overall_pct:.1f}%{RESET}")

    # Per-probe leaderboard (probes where all completed models scored)
    if done >= 10:
        probe_stats: dict[str, list[int]] = defaultdict(list)
        for _, probe_id, score, _ in state["completed"]:
            probe_stats[probe_id].append(score)

        print()
        print(f"  {BOLD}Hardest probes (by avg score across completed models):{RESET}")
        ranked = sorted(probe_stats.items(), key=lambda x: sum(x[1]) / len(x[1]))
        for probe_id, scores in ranked[:8]:
            avg = sum(scores) / len(scores)
            n   = len(scores)
            c   = GREEN if avg >= 2.5 else YELLOW if avg >= 1.5 else RED
            mini_bar = "█" * int(avg / 3 * 15) + "░" * (15 - int(avg / 3 * 15))
            print(f"    {probe_id:<28} {c}{mini_bar}{RESET}  {c}{avg:.2f}/3{RESET}  ({n} models)")

    print()
    print(f"  {DIM}Log: /tmp/hard_probe_run.log  |  Ctrl+C to exit{RESET}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Real-time hard probe progress dashboard")
    parser.add_argument("--log", default="/tmp/hard_probe_run.log")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds")
    args = parser.parse_args()

    print(f"Watching {args.log}  (refresh every {args.interval}s)")
    time.sleep(1)

    start_wall = None

    try:
        while True:
            state = parse_log(args.log)
            if state["start_time"] and start_wall is None:
                start_wall = state["start_time"]

            elapsed = time.time() - start_wall if start_wall else 0
            render(state, elapsed)

            done  = len(state["completed"])
            total = state["total_runs"]
            if done >= total and total > 0:
                print(f"\n  {GREEN}{BOLD}All {total} probes complete!{RESET}")
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
