#!/bin/bash
# Run all three TAP-Vid trackers (CoTracker3, LocoTrack, TAPIR) through the
# unified closed-loop refiner evaluation (eval/tapvid_closed_loop.py). Logs go
# next to the checkpoint if one is supplied, otherwise to the current directory.
#
# Usage: ./eval_tapvid_all.sh <davis|kinetics|rgb_stacking> [mode] [ema_alpha] [refiner_path]
#   mode:      online (default, causal sliding window) | offline (full-video)
#   ema_alpha: closed-loop EMA template momentum (default 0.9, matches the paper)

set -e

if [ -z "$1" ]; then
    echo "Usage: ./eval_tapvid_all.sh <davis|kinetics|rgb_stacking> [mode] [ema_alpha] [refiner_path]"
    exit 1
fi

DATASET=$1
MODE=${2:-online}
EMA_ALPHA=${3:-0.9}
REFINER_PATH=$4

cd "$(dirname "$0")/../eval"

if [ -n "$REFINER_PATH" ] && [ -f "$REFINER_PATH" ]; then
    CKPT_DIR=$(dirname "$REFINER_PATH")
    LOGFILE="${CKPT_DIR}/eval_tapvid_${DATASET}_${MODE}_ema${EMA_ALPHA}.log"
    REFINER_FLAG="--refiner_path $REFINER_PATH"
else
    LOGFILE="eval_tapvid_${DATASET}_${MODE}_ema${EMA_ALPHA}.log"
    REFINER_FLAG=""
fi

echo "== TAP-Vid eval: $DATASET ($MODE, ema=$EMA_ALPHA) ==" | tee -a "$LOGFILE"
echo "   refiner: ${REFINER_PATH:-<from SUBPIXR_CHECKPOINT env>}" | tee -a "$LOGFILE"

for TRACKER in cotracker locotrack tapir; do
    echo -e "\n>>> $TRACKER on $DATASET ($MODE) <<<" | tee -a "$LOGFILE"
    PYTHONPATH=.. python3 -u tapvid_closed_loop.py "$DATASET" \
        --tracker "$TRACKER" --mode "$MODE" --ema_alpha "$EMA_ALPHA" \
        $REFINER_FLAG 2>&1 | tee -a "$LOGFILE"
done

echo -e "\n== Done. Log: $LOGFILE ==" | tee -a "$LOGFILE"
