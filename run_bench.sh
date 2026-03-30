#!/bin/bash
# Wrapper: loads .env and runs a benchmark script with all args forwarded.
set -a
source "$(dirname "$0")/.env"
set +a
cd "$(dirname "$0")/benchmark/scripts"
exec python "$@"
