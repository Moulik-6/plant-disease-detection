#!/usr/bin/env bash
# ==============================================================================
# scripts/run_queue.sh
#
# Sequential execution queue for the 9 core Stage-II experiments across 3 architectures:
#   - a_pv_only (resnet50, efficientnet_v2_s, vit_b_16)
#   - b_mixed   (resnet50, efficientnet_v2_s, vit_b_16)
#   - c_finetune (resnet50, efficientnet_v2_s, vit_b_16)
#
# Features:
#   - Skips runs marked by .done
#   - Enforces per-run execution timeout (default: 4h)
#   - Syncs runs/ to S3 bucket after each completed run, excluding last.pth
#   - Traps EXIT: if SHUTDOWN=1, initiates an automatic shutdown of the instance
# ==============================================================================

set -euo pipefail

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

EXPERIMENTS=("a_pv_only" "b_mixed" "c_finetune")
ARCHITECTURES=("resnet50" "efficientnet_v2_s" "vit_b_16")

echo "================================================================================"
echo "Starting Experiment Queue"
echo "Target S3 Bucket: s3://${S3_BUCKET}/runs/"
echo "Per-run Timeout:  ${PER_RUN_TIMEOUT}"
echo "Auto Shutdown:    ${SHUTDOWN}"
echo "================================================================================"

for exp in "${EXPERIMENTS[@]}"; do
    for arch in "${ARCHITECTURES[@]}"; do
        marker_file="runs/${exp}_${arch}/.done"

        # Check if already completed (at run level or inside timestamp subdirectories)
        if [ -f "$marker_file" ] || compgen -G "runs/${exp}_${arch}/**/.done" > /dev/null 2>&1; then
            echo "[*] [SKIP] Experiment ${exp} (${arch}) already completed (.done marker found)."
            continue
        fi

        echo "--------------------------------------------------------------------------------"
        echo "[*] [RUN] Starting: ${exp} | Architecture: ${arch}"
        echo "[*] Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        echo "--------------------------------------------------------------------------------"

        mkdir -p "runs/${exp}_${arch}"

        # Execute training with timeout
        timeout "$PER_RUN_TIMEOUT" python ml/train.py \
            --exp "ml/configs/experiment/${exp}.yaml" \
            --set model.architecture="${arch}"

        # Touch completion marker
        touch "$marker_file"
        echo "[*] [DONE] Completed ${exp} (${arch}). Marker created: ${marker_file}"

        # Sync to S3, excluding heavy optimizer/scheduler checkpoint last.pth
        echo "[*] [SYNC] Syncing runs/ to s3://${S3_BUCKET}/runs/ (excluding *last.pth)..."
        aws s3 sync runs/ "s3://${S3_BUCKET}/runs/" --exclude "*last.pth"
    done
done

echo "================================================================================"
echo "[*] All 9 queue experiments successfully processed."
echo "================================================================================"
