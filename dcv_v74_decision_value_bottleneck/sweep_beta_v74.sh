#!/usr/bin/env bash
set -euo pipefail

BASE="${1:?Usage: bash sweep_beta_v74.sh /absolute/path/to/original/V7/project}"

for BETA in 0 0.00001 0.00003 0.0001 0.0003 0.001
do
  TAG=$(echo "$BETA" | tr '.' 'p')
  echo "===== beta_ib=$BETA ====="

  python train_decision_value_bottleneck_v74.py \
    --data "$BASE/data_v7/train" \
    --val "$BASE/data_v7/val" \
    --teacher "$BASE/teacher_v7/train_15" \
    --teacher-val "$BASE/teacher_v7/val_15" \
    --perception "$BASE/checkpoints/perception_v7.pt" \
    --out "checkpoints/v74_dv_beta_${TAG}.pt" \
    --budget 0.15 \
    --epochs 40 \
    --batch 32 \
    --hidden 256 \
    --latent-dim 32 \
    --layers 4 \
    --heads 8 \
    --hard-gain-weight 0.5 \
    --lambda-value-reg 1.0 \
    --lambda-value-kl 0.5 \
    --lambda-value-rank 0.25 \
    --lambda-decision 1.0 \
    --lambda-grad-aux 0.10 \
    --beta-ib "$BETA" \
    --onpolicy-start 0.25 \
    --onpolicy-end 0.80 \
    --rollout-epsilon 0.10
done
