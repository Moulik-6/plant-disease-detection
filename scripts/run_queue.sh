#!/usr/bin/env bash
# ==============================================================================
# scripts/run_queue.sh
#
# Sequential execution queue runner reading experiment jobs from a text file.
#
# Usage:
#   ./scripts/run_queue.sh <queue_file>
#
# Example:
#   export S3_BUCKET="my-ml-experiments-bucket"
#   ./scripts/run_queue.sh scripts/queues/stage2_main.txt
#
# Queue File Format:
#   <experiment_name>:<architecture> [optional extra --set arguments]
#   Lines starting with '#' or empty lines are ignored.
#
# Features:
#   - Skips runs marked by .done
#   - Enforces per-run execution timeout (PER_RUN_TIMEOUT, default: 4h)
#   - Syncs runs/ to S3 bucket after each completed run, excluding last.pth
#   - Traps EXIT: if SHUTDOWN=1, initiates an automatic shutdown of the instance
# ==============================================================================

set -euo pipefail

# Validate command line argument
if [ "$#" -lt 1 ]; then
    echo "ERROR: Queue file argument is required." >&2
    echo "Usage: $0 <queue_file>" >&2
    echo "Example: $0 scripts/queues/stage2_main.txt" >&2
    exit 1
fi

QUEUE_FILE="$1"
if [ ! -f "$QUEUE_FILE" ]; then
    echo "ERROR: Specified queue file does not exist: ${QUEUE_FILE}" >&2
    exit 1
fi

# Ensure S3_BUCKET is set
if [ -z "${S3_BUCKET:-}" ]; then
    echo "ERROR: S3_BUCKET environment variable must be specified." >&2
    echo "Example: export S3_BUCKET=my-ml-experiments-bucket" >&2
    exit 1
fi

PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-4h}"
SHUTDOWN="${SHUTDOWN:-0}"

# Shutdown trap handler on EXIT
cleanup() {
    exit_status=$?
    if [ "$SHUTDOWN" = "1" ]; then
        echo "[*] Queue finished (exit code: ${exit_status}). SHUTDOWN=1 set. Initiating poweroff..."
        sudo shutdown -h now || true
    else
        echo "[*] Queue finished (exit code: ${exit_status}). SHUTDOWN is disabled."
    fi
}
trap cleanup EXIT

echo "================================================================================"
echo "Starting Experiment Queue"
echo "Queue File:       ${QUEUE_FILE}"
echo "Target S3 Bucket: s3://${S3_BUCKET}/runs/"
echo "Per-run Timeout:  ${PER_RUN_TIMEOUT}"
echo "Auto Shutdown:    ${SHUTDOWN}"
echo "================================================================================"

while IFS= read -r line || [ -n "$line" ]; do
    # Strip leading/trailing whitespace
    line="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    # Skip empty lines and comment lines
    if [ -z "$line" ] || [[ "$line" =~ ^# ]]; then
        continue
    fi

    # Parse first token (experiment:architecture) and optional remaining args
    exp_arch="${line%% *}"
    extra_sets="${line#"$exp_arch"}"
    extra_sets="$(echo "$extra_sets" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    if [[ "$exp_arch" != *:* ]]; then
        echo "ERROR: Malformed queue line: '${line}'. Expected format: 'experiment:architecture [extra_sets]'" >&2
        exit 1
    fi

    exp="${exp_arch%%:*}"
    arch="${exp_arch##*:}"

    marker_file="runs/${exp}_${arch}/.done"

    # Check if already completed (at run level or inside timestamp subdirectories)
    if [ -f "$marker_file" ] || compgen -G "runs/${exp}_${arch}/**/.done" > /dev/null 2>&1; then
        echo "[*] [SKIP] Experiment ${exp} (${arch}) already completed (.done marker found)."
        continue
    fi

    echo "--------------------------------------------------------------------------------"
    echo "[*] [RUN] Starting: ${exp} | Architecture: ${arch}"
    if [ -n "$extra_sets" ]; then
        echo "[*] [RUN] Additional parameters: ${extra_sets}"
    fi
    echo "[*] Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "--------------------------------------------------------------------------------"

    mkdir -p "runs/${exp}_${arch}"

    # Build command arguments
    cmd_args=(
        python ml/train.py
        --exp "ml/configs/experiment/${exp}.yaml"
        --set "model.architecture=${arch}"
    )

    if [ -n "$extra_sets" ]; then
        read -r -a extra_arr <<< "$extra_sets"
        cmd_args+=("${extra_arr[@]}")
    fi

    # Execute training with timeout
    timeout "$PER_RUN_TIMEOUT" "${cmd_args[@]}"

    # Touch completion marker
    touch "$marker_file"
    echo "[*] [DONE] Completed ${exp} (${arch}). Marker created: ${marker_file}"

    # Sync to S3, excluding heavy optimizer/scheduler checkpoint last.pth
    echo "[*] [SYNC] Syncing runs/ to s3://${S3_BUCKET}/runs/ (excluding *last.pth)..."
    aws s3 sync runs/ "s3://${S3_BUCKET}/runs/" --exclude "*last.pth"

done < "$QUEUE_FILE"

echo "================================================================================"
echo "[*] All experiments in queue '${QUEUE_FILE}' processed."
echo "================================================================================"
