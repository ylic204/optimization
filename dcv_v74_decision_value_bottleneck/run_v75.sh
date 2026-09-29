#!/usr/bin/env bash
set -euo pipefail

BASE="${1:?Usage: bash run_v75.sh /absolute/path/to/project}"

python train_joint_token_selector_v75.py \
  --data "$BASE/data_v7/train" \
  --val "$BASE/data_v7/val" \
  --perception "$BASE/checkpoints/perception_v7.pt" \
  --out checkpoints/v75_joint_decision_bottleneck.pt \
  --metrics results/v75_metrics.json \
  --budget 0.15 \
  --epochs 40 \
  --batch 32 \
  --hidden 256 \
  --latent-dim 64 \
  --layers 4 \
  --heads 8 \
  --lambda-task 1.0 \
  --lambda-task-kl 0.25 \
  --lambda-dgd 0.25 \
  --lambda-set-value 0.25 \
  --lambda-set-rank 0.10

