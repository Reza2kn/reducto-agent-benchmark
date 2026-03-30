#!/usr/bin/env python3
"""Generation health check — called by cron monitor every 3 min."""
import re, subprocess

log     = open("/tmp/synth_watchdog.log").read()
gen_log = open("/tmp/synth_gen.log").read()

lines_data = []
for line in log.splitlines():
    m = re.search(r'\[(\d+):(\d+):(\d+)\].*Heartbeat.*checkpoint=\s*(\d+)', line)
    if m:
        h, mi, s, ckpt = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        lines_data.append((h*3600 + mi*60 + s, ckpt))

seen = []
for t, c in lines_data:
    if not seen or seen[-1][1] != c:
        seen.append((t, c))

recent = seen[-10:]
t0, c0 = recent[0]
t1, c1 = recent[-1]
delta_t = (t1 - t0) / 60
epm = (c1 - c0) / delta_t if delta_t > 0 else 0
GEN_TARGET = 36_669
remaining = GEN_TARGET - c1
eta_mins = remaining / epm if epm > 0 else 999999
eta_h = int(eta_mins // 60)
eta_m = int(eta_mins % 60)

print(f"Checkpoint : {c1:,} / {GEN_TARGET:,}  ({c1/GEN_TARGET*100:.1f}%)")
print(f"Rate       : {epm:.1f} epm  (last {delta_t:.0f} min)")
print(f"ETA        : {eta_h}h {eta_m}m")

alive = subprocess.run(["pgrep", "-f", "gen_synthetic_data.py"], capture_output=True).returncode == 0
print(f"Process    : {'✓ running' if alive else '✗ DEAD'}")

print(f"\nLast 5 intervals:")
intervals = []
prev_t, prev_c = seen[-6] if len(seen) >= 6 else seen[0]
for t, c in seen[-5:]:
    dt = (t - prev_t) / 60
    dc = c - prev_c
    rate = dc / dt if dt > 0 else 0
    intervals.append(rate)
    print(f"  +{dc:3d}  →  {rate:.0f} epm")
    prev_t, prev_c = t, c

low_count = sum(1 for r in intervals if r < 20)
if low_count >= 4:
    print(f"\n⚠ STALL: {low_count}/5 intervals below 20 epm")

clean = [l for l in gen_log.splitlines() if not any(x in l for x in
    ["UserWarning","should be specified","model_kwargs","pydantic","return meta","shadows","result = self"])]
e429  = sum(1 for l in clean if "429" in l)
etout = sum(1 for l in clean if "timeout" in l.lower())
ereph = sum(1 for l in clean if "rephrase" in l and "error" in l)
print(f"\nErrors (cumulative): 429={e429}  timeout={etout}  rephrase_err={ereph}")
