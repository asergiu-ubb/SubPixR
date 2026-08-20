#!/usr/bin/env bash
# Two-view pose evaluation on MegaDepth-1500 and ScanNet-1500.
# Sweeps all five matchers and runs both baselines (no refinement) and the
# SubPixR refiner on each. Optional --sota flag also runs S2DNet / Patch2Pix /
# COTR for comparison (each needs its own upstream repo + weights set up — see
# release/subpixr/refiners_compared/*.py for the per-method clone instructions).
#
# Usage:
#   ./eval_pose_all.sh                       # SubPixR only
#   ./eval_pose_all.sh --sota                # SubPixR + S2DNet + Patch2Pix + COTR

set -euo pipefail

cd "$(dirname "$0")/.."

RUN_SOTA=0
for arg in "$@"; do
    [ "$arg" = "--sota" ] && RUN_SOTA=1
done

DATASETS=(megadepth scannet)
MATCHERS=(
    "superpoint+lightglue-official"
    "superpoint+superglue-official"
    "superpoint+NN"
    "disk+lightglue-official"
    "aliked+lightglue-official"
)

for DS in "${DATASETS[@]}"; do
    for M in "${MATCHERS[@]}"; do
        echo "=== ${DS} / ${M} ==="
        PYTHONPATH=. python3 eval/pose_megadepth_scannet.py \
            --dataset "$DS" --matcher "$M"

        if [ $RUN_SOTA -eq 1 ]; then
            for WRAPPER in refiners_compared/s2dnet_eval_wrapper.py \
                           refiners_compared/patch2pix_wrapper.py \
                           refiners_compared/cotr_wrapper.py; do
                [ -f "$WRAPPER" ] && PYTHONPATH=. python3 "$WRAPPER" \
                    --dataset "$DS" --matcher "$M" || true
            done
        fi
    done
done
