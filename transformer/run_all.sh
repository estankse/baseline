#!/usr/bin/env bash
# ============================================================
# run_all.sh  — run FFN ablation for all K values and 3 seeds
# ============================================================
# Usage:
#   bash run_all.sh            # sequential (safe)
#   bash run_all.sh --parallel # parallel (needs multi-GPU)
#
# Each K × seed run saves to results/K{K}_seed{seed}/summary.json
# After all runs complete, call:  python analyze.py
# ============================================================

set -e

K_VALUES=(1 2 4 8 16)
SEEDS=(42 123 456)
EPOCHS=20
BATCH_SIZE=64
OUT_DIR="results"

PARALLEL=false
for arg in "$@"; do
  [[ "$arg" == "--parallel" ]] && PARALLEL=true
done

pids=()

for K in "${K_VALUES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    CMD="python train.py \
      --K $K \
      --seed $SEED \
      --epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --out_dir $OUT_DIR"

    echo "▶ Starting K=$K seed=$SEED"

    if $PARALLEL; then
      $CMD &
      pids+=($!)
    else
      $CMD
    fi
  done
done

if $PARALLEL; then
  echo "Waiting for ${#pids[@]} jobs..."
  for pid in "${pids[@]}"; do
    wait $pid && echo "Job $pid done" || echo "Job $pid FAILED"
  done
fi

echo ""
echo "✓ All runs complete. Now run:  python analyze.py"
