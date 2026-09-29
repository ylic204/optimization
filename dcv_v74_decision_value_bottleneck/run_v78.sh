#!/usr/bin/env bash
set -euo pipefail

BASE="${1:?Usage: VLM_MODEL=/path/to/Qwen3-VL-8B-Instruct bash run_v78.sh /absolute/path/to/project}"
VLM_MODEL="${VLM_MODEL:?Set VLM_MODEL to the local Qwen3-VL-8B checkpoint directory}"

if [[ ! -d "$BASE/data_v77/train" || ! -d "$BASE/data_v77/val" ]]; then
  python generate_dataset_v77.py --out "$BASE/data_v77" --split train --n 5000 --seed 7
  python generate_dataset_v77.py --out "$BASE/data_v77" --split val --n 1000 --seed 8
fi

EXTRA_ARGS=()
if [[ "${LOAD_4BIT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--load-4bit)
fi

python train_qwen3vl_selector_v78.py \
  --data "$BASE/data_v77/train" \
  --val "$BASE/data_v77/val" \
  --vlm-model "$VLM_MODEL" \
  --vlm-local-files-only \
  --prune-layer "${PRUNE_LAYER:-4}" \
  --out "$BASE/checkpoints/v78_qwen3vl_selector.pt" \
  --metrics "$BASE/results/v78_metrics.json" \
  --budget 0.15 \
  --epochs 40 \
  --batch "${BATCH_SIZE:-1}" \
  --aligned-dim 256 \
  --hidden 256 \
  --latent-dim 64 \
  --position-dim 32 \
  --layers 4 \
  --heads 8 \
  --lambda-task 1.0 \
  --lambda-task-kl 0.25 \
  --lambda-hard-task 0.5 \
  --lambda-dgd 0.25 \
  --lambda-set-value 0.25 \
  --lambda-set-rank 0.10 \
  "${EXTRA_ARGS[@]}"
