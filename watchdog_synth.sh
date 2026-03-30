#!/bin/bash
# Watchdog for gen_synthetic_data.py — restarts if the process dies.
# Logs to /tmp/synth_watchdog.log

source "$(dirname "$0")/.env"
SCRIPT="/Users/reducto-reza/AI/reducto-agent-benchmark/run_synth.sh"
LOG="/tmp/synth_gen.log"
WATCHDOG_LOG="/tmp/synth_watchdog.log"
CHECK_INTERVAL=60   # seconds between heartbeat checks
STALL_LIMIT=1200    # seconds with no new checkpoint lines before declaring stall (8 parallel seeds need ~17 min to land)



CHECKPOINT="/Users/reducto-reza/AI/reducto-agent-benchmark/benchmark/data/synthetic_training/.checkpoint.jsonl"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$WATCHDOG_LOG"; }

log "Watchdog started. Checking every ${CHECK_INTERVAL}s, stall threshold ${STALL_LIMIT}s."

last_line_count=0
last_progress_time=$(date +%s)
child_pid=""

start_run() {
    log "Starting gen_synthetic_data.py (--resume)..."
    bash "$SCRIPT" &
    child_pid=$!
    log "Started PID $child_pid"
}

start_run

while true; do
    sleep "$CHECK_INTERVAL"

    # Check if any gen process is alive
    if ! pgrep -f "gen_synthetic_data.py" > /dev/null 2>&1; then
        target_lines=$(wc -l < "$CHECKPOINT" 2>/dev/null || echo 0)
        if [ "$target_lines" -ge 36669 ]; then
            log "Target reached ($target_lines lines). Running verifier..."

            cd /Users/reducto-reza/AI/reducto-agent-benchmark
            python3 benchmark/scripts/verify_synthetic_data.py \
                --input "$CHECKPOINT" \
                --output benchmark/data/synthetic_training \
                --l4-full \
                --verbose \
                2>&1 | tee -a "$WATCHDOG_LOG"

            VERIFY_EXIT=$?
            if [ $VERIFY_EXIT -eq 0 ]; then
                log "✓ Verification complete. verified_train.jsonl + verified_val.jsonl written."
            else
                log "✗ Verifier exited with code $VERIFY_EXIT — check log."
            fi

            log "Watchdog done. Exiting."
            exit 0
        fi
        log "Process died (checkpoint at $target_lines lines). Restarting..."
        start_run
        last_progress_time=$(date +%s)
        continue
    fi

    # Stall detection — if checkpoint hasn't grown in STALL_LIMIT seconds, kill and restart
    current_lines=$(wc -l < "$CHECKPOINT" 2>/dev/null || echo 0)
    now=$(date +%s)

    if [ "$current_lines" -gt "$last_line_count" ]; then
        last_line_count=$current_lines
        last_progress_time=$now
        log "Heartbeat ✓  checkpoint=$current_lines lines (~$((current_lines * 100 / 36669))%)"
    else
        stalled_for=$(( now - last_progress_time ))
        log "No progress for ${stalled_for}s (checkpoint=$current_lines)..."
        if [ "$stalled_for" -ge "$STALL_LIMIT" ]; then
            log "Stall detected! Killing and restarting..."
            pkill -f "gen_synthetic_data.py"
            sleep 3
            start_run
            last_progress_time=$(date +%s)
        fi
    fi
done
