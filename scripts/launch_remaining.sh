#!/usr/bin/env bash
set -euo pipefail
# Wait for any running training to finish, then launch phases 5, 6, 7, 8.

cd "$(dirname "$0")/.."

echo "Waiting for GPU to be free..."
while pgrep -f "torchrun" >/dev/null 2>&1; do
    sleep 10
done

echo "GPU free. Launching phases 8 5 6 7..."
./scripts/run_experiments_scaled.sh 8 5 6 7
