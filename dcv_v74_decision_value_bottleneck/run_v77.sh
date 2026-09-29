#!/usr/bin/env bash
set -euo pipefail

BASE="${1:?Usage: VLM_MODEL=/path/to/blip bash run_v77.sh /absolute/path/to/project}"
VLM_MODEL="${VLM_MODEL:?Set VLM_MODEL to a local BLIP model directory or model ID}"

if [[ ! -d "$BASE/data_v77/train" || ! -d "$BASE/data_v77/val" ]]; then
  python generate_dataset_v77.py --out "$BASE/data_v77" --split train --n 5000 --seed 7
  python generate_dataset_v77.py --out "$BASE/data_v77" --split val --n 1000 --seed 8
fi

python train_aligned_vlm_selector_v77.py \
  --data "$BASE/data_v77/train" \
  --val "$BASE/data_v77/val" \
  --vlm-model "$VLM_MODEL" \
  --vlm-local-files-only \
  --out "$BASE/checkpoints/v77_aligned_vlm_selector.pt" \
  --metrics "$BASE/results/v77_metrics.json" \
  --budget 0.15 \
  --epochs 40 \
  --batch 16 \
  --hidden 256 \
  --latent-dim 64 \
  --position-dim 32 \
  --layers 4 \
  --heads 8 \
  --lambda-task 1.0 \
  --lambda-task-kl 0.25 \
  --lambda-hard-task 1.0 \
  --lambda-dgd 0.25 \
  --lambda-set-value 0.25 \
  --lambda-set-rank 0.10
