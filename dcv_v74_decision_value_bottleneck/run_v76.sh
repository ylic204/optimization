#!/usr/bin/env bash
set -euo pipefail

BASE="${1:?Usage: TEXT_MODEL=/path/to/llm bash run_v76.sh /absolute/path/to/project}"
TEXT_MODEL="${TEXT_MODEL:?Set TEXT_MODEL to a local Hugging Face LLM directory or model ID}"

python generate_dataset_v76.py --out "$BASE/data_v76" --split train --n 5000 --seed 7
python generate_dataset_v76.py --out "$BASE/data_v76" --split val --n 1000 --seed 8

python train_position_c_selector_v76.py \
  --data "$BASE/data_v76/train" \
  --val "$BASE/data_v76/val" \
  --perception "$BASE/checkpoints/perception_v7.pt" \
  --text-model "$TEXT_MODEL" \
  --text-local-files-only \
  --out "$BASE/checkpoints/v76_position_c_selector.pt" \
  --metrics "$BASE/results/v76_metrics.json" \
  --budget 0.15 \
  --epochs 40 \
  --batch 32 \
  --hidden 256 \
  --latent-dim 64 \
  --task-hidden 128 \
  --position-dim 32 \
  --layers 4 \
  --heads 8 \
  --lambda-task 1.0 \
  --lambda-task-kl 0.25 \
  --lambda-dgd 0.25 \
  --lambda-set-value 0.25 \
  --lambda-set-rank 0.10
