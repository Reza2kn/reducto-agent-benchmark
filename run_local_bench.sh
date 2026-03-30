#!/usr/bin/env bash
# run_local_bench.sh — start llama-server + run all 3 framework benchmarks for LOCAL_MODELS
# Usage: ./run_local_bench.sh [model-path-override]
#
# The model path is read from models.py LOCAL_MODELS[0].local_model_path by default.
# Override with: ./run_local_bench.sh /path/to/other.gguf

set -euo pipefail
set -a; source .env; set +a

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/Users/reducto-reza/miniforge3/bin/python
SERVER_PORT=18080
SERVER_PID=""

MODEL_PATH="${1:-}"
if [[ -z "$MODEL_PATH" ]]; then
    # Read from models.py
    MODEL_PATH=$(${PYTHON} -c "
from benchmark.scripts.models import LOCAL_MODELS
print(LOCAL_MODELS[0].local_model_path)
" 2>/dev/null || echo "")
fi

if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    exit 1
fi

cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        echo "Stopping llama-server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting llama-server for: $(basename $MODEL_PATH)"
llama-server \
    --model "$MODEL_PATH" \
    --port $SERVER_PORT \
    --host 127.0.0.1 \
    --ctx-size 32768 \
    --n-gpu-layers 99 \
    -np 4 \
    --log-disable \
    &
SERVER_PID=$!
echo "llama-server PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server..."
for i in $(seq 1 60); do
    if curl -s "http://127.0.0.1:${SERVER_PORT}/health" | grep -q "ok"; then
        echo "Server ready."
        break
    fi
    sleep 1
    if [[ $i -eq 60 ]]; then
        echo "ERROR: Server did not start within 60s"
        exit 1
    fi
done

cd "$SCRIPT_DIR/benchmark/scripts"

echo ""
echo "=== Running LangChain (local models) ==="
${PYTHON} bench_langchain.py --local-only 2>&1

echo ""
echo "=== Running smolagents (local models) ==="
${PYTHON} bench_smolagents.py --local-only 2>&1

echo ""
echo "=== Running LlamaIndex (local models) ==="
${PYTHON} bench_llamaindex.py --local-only 2>&1

echo ""
echo "=== Regenerating report ==="
${PYTHON} generate_report.py 2>&1

echo ""
echo "Local benchmark complete."
