#!/bin/bash
# HPatches evaluation. Two protocols:
#   1. Baselines: every (matcher x refiner) on the homography task.
#   2. Perturbed-point protocol (paper Table 4): MMA at {0.5, 1, 2, 3, 5} px
#      after deliberately deflecting GT keypoints.
#
# Pass a refiner checkpoint as $1; defaults to $SUBPIXR_CHECKPOINT otherwise.
#
# Usage: ./eval_hpatches.sh [refiner_path]

set -e

cd "$(dirname "$0")/.."
REFINER_PATH="${1:-${SUBPIXR_CHECKPOINT:-}}"

METHODS=(
    superpoint-lightglue
    superpoint-superglue
    superpoint-nn
    disk-lightglue
    disk-nn
    loftr
    patch2pix
    superpoint-caps
    aspanformer
)

echo ">>> HPatches baselines"
for METHOD in "${METHODS[@]}"; do
    echo "--- $METHOD ---"
    PYTHONPATH=. python3 eval/hpatches_baselines.py \
        --method "$METHOD" \
        --max_edge 9999 \
        --skip_s2dnet \
        ${REFINER_PATH:+--refiner_path "$REFINER_PATH"}
done

echo ""
echo ">>> HPatches perturbed-point MMA (paper Table 4)"
PYTHONPATH=. python3 eval/hpatches_perturbed.py \
    ${REFINER_PATH:+--refiner_path "$REFINER_PATH"} \
    --skip_baselines --skip_patch2pix --skip_caps --skip_s2dnet --skip_loftr
