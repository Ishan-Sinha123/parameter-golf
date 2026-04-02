#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_all_records.sh [--num-gpus N] [--gpus-per-run K]
# --num-gpus N     Total GPUs to use (default: auto-detect via torch)
# --gpus-per-run K GPUs allocated per experiment (default: 1)
#                  Each run gets K GPUs; up to N/K experiments run in parallel.

NUM_GPUS=""
GPUS_PER_RUN=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus)    NUM_GPUS="$2";    shift 2 ;;
        --gpus-per-run) GPUS_PER_RUN="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$NUM_GPUS" ]]; then
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
    echo "Auto-detected $NUM_GPUS GPU(s)"
fi

if (( GPUS_PER_RUN > NUM_GPUS )); then
    echo "Error: --gpus-per-run $GPUS_PER_RUN exceeds --num-gpus $NUM_GPUS"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$REPO_ROOT/run_logs"
VENV="$REPO_ROOT/.venv/bin/activate"
mkdir -p "$LOG_DIR"

[[ -f "$VENV" ]] && source "$VENV"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_PATH="$REPO_ROOT/data/datasets/fineweb10B_sp1024/"
export TOKENIZER_PATH="$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024
export MAX_WALLCLOCK_SECONDS=90
export VAL_LOSS_EVERY=10

echo "Running with NUM_GPUS=$NUM_GPUS, GPUS_PER_RUN=$GPUS_PER_RUN (max $(( NUM_GPUS / GPUS_PER_RUN )) parallel runs)"

# pid -> "gpu_ids|log|label"  (gpu_ids is comma-separated)
declare -A pids
# free GPU pool (all GPUs start available)
free_gpus=()
for (( i=0; i<NUM_GPUS; i++ )); do free_gpus+=("$i"); done

total=0; passed=0; failed=0
base_port=29500
parallel_slots=$(( NUM_GPUS / GPUS_PER_RUN ))

# For parallel runs (>1 slot): background jobs tracked via pids array.
# Wait for any one to finish using 'wait -n -p' (bash 5.1+).
reap_one() {
    local waited_pid excode
    # Poll until we find a completed job from our pids map
    while true; do
        for waited_pid in "${!pids[@]}"; do
            if ! kill -0 "$waited_pid" 2>/dev/null; then
                wait "$waited_pid"; excode=$?
                local info="${pids[$waited_pid]}"
                local gpu_ids="${info%%|*}"
                local rest="${info#*|}"
                local _log="${rest%%|*}"
                local label="${rest#*|}"
                [[ $excode -eq 0 ]] && local s=ok || local s="FAILED (exit $excode)"
                echo "    [$label] $s"
                [[ "$s" == ok ]] && ((passed++)) || ((failed++)) || true
                unset "pids[$waited_pid]"
                IFS=',' read -ra released <<< "$gpu_ids"
                free_gpus+=("${released[@]}")
                return
            fi
        done
        sleep 0.5
    done
}

mapfile -t scripts < <(find "$REPO_ROOT/records" -name "train_gpt.py" | sort | awk -F'/' '{
    dir=$(NF-1); if (match(dir, /^[0-9]{4}-[0-9]{2}-[0-9]{2}/)) {
        date=substr(dir, RSTART, RLENGTH); if (date >= "2026-03-21") print
    }
}')

run_experiment() {
    local dir="$1" log="$2" run_id="$3" cuda_visible="$4" port="$5"
    (
        cd "$dir"
        export RUN_ID="$run_id"
        CUDA_VISIBLE_DEVICES=$cuda_visible \
        timeout 600 torchrun --standalone --master_port=$port --nproc_per_node=$GPUS_PER_RUN train_gpt.py
    ) >"$log" 2>&1
}

for script in "${scripts[@]}"; do
    dir="$(dirname "$script")"
    name="$(basename "$dir")"
    track="$(basename "$(dirname "$dir")")"
    run_id="${track}__${name}"
    log="$LOG_DIR/${run_id}.log"
    ((total++)) || true

    # Take GPUS_PER_RUN GPUs from the pool (wait if needed)
    while (( ${#free_gpus[@]} < GPUS_PER_RUN )); do
        reap_one
    done
    assigned=("${free_gpus[@]:0:$GPUS_PER_RUN}")
    free_gpus=("${free_gpus[@]:$GPUS_PER_RUN}")
    cuda_visible=$(IFS=','; echo "${assigned[*]}")
    port=$(( base_port + assigned[0] ))

    echo "━━━ [$total] $track/$name  (GPU(s) $cuda_visible, port $port)"
    echo "    log → $log"

    if (( parallel_slots > 1 )); then
        # Parallel mode: background + track
        run_experiment "$dir" "$log" "$run_id" "$cuda_visible" "$port" &
        pids[$!]="${cuda_visible}|${log}|${total}:${name}"
    else
        # Sequential mode: run inline, no background process shenanigans
        run_experiment "$dir" "$log" "$run_id" "$cuda_visible" "$port" && s=ok || s="FAILED (exit $?)"
        echo "    [$total:$name] $s"
        [[ "$s" == ok ]] && ((passed++)) || ((failed++)) || true
        free_gpus+=("${assigned[@]}")
    fi
done

# Drain remaining parallel jobs
while (( ${#pids[@]} > 0 )); do
    reap_one
done

echo ""
echo "Runs complete: $passed/$total passed, $failed failed"
echo "Generating chart..."

python3 "$REPO_ROOT/scripts/plot_val_bpb.py"
