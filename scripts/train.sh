#!/bin/bash
# Train SubPixR on the synthetic patch dataset. Edit the EXPERIMENTS array to
# queue multiple architectural variants on the same machine — the loop runs
# them sequentially with a short cool-down in between.
#
# Each row: ENC XCORR COST CONF BS JITTER TRAIN_ITERS VAL_ITERS SPATIAL MULTI SCALE ATTN HYBRID
# The defaults below reproduce the final SubPixR recipe (ResNet-34, depthwise
# xcorr, cross-attention, hybrid fusion, train_iters=4 — matches the released
# checkpoint config.json).

set -e

cd "$(dirname "$0")/.."

EXPERIMENTS=(
    "resnet34 True False False 32 15.0 4 8 True True 8.0 True True"
)

for EXP in "${EXPERIMENTS[@]}"; do
    read -r ENC XCORR COST CONF BS JITTER TI VI SPATIAL MULTI SCALE ATTN HYBRID <<< "$EXP"
    [[ -z "$ENC" || "$ENC" == \#* ]] && continue

    echo "=================================================================="
    echo "Training: $ENC | xcorr=$XCORR cost=$COST attn=$ATTN hybrid=$HYBRID"
    echo "  scale=$SCALE jit=$JITTER ti=$TI vi=$VI bs=$BS"
    echo "=================================================================="

    PYTHONPATH=. python3 train.py \
        --encoder_type             "$ENC"    \
        --use_depthwise_xcorr      "$XCORR"  \
        --use_local_cost_volume    "$COST"   \
        --predict_confidence       "$CONF"   \
        --batch_size               "$BS"     \
        --max_jitter               "$JITTER" \
        --train_iters              "$TI"     \
        --val_iters                "$VI"     \
        --use_spatial_head         "$SPATIAL"\
        --use_multi_stage_features "$MULTI"  \
        --scale_factor             "$SCALE"  \
        --use_attention            "$ATTN"   \
        --use_hybrid_fusion        "$HYBRID"

    sleep 5
done
