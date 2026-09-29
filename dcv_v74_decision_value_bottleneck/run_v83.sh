#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?usage: bash run_v83.sh /path/to/dcv_v74_decision_value_bottleneck}"
shift

python "$ROOT/train_mapped_vlm_optimizer_v83.py" \
  --data "${TRAIN_DATA:?set TRAIN_DATA}" \
  --val "${VAL_DATA:?set VAL_DATA}" \
  --nuplan-task "${NUPLAN_TASK:-$ROOT/tasks/nuplan_safe_progress.json}" \
  --pointnav-task "${POINTNAV_TASK:-$ROOT/tasks/pointnav_safe_short.json}" \
  --vlm-model "${VLM_MODEL:-/data/lyi/models/Qwen3-VL-8B-Instruct}" \
  --device "cuda:${DEVICE:-0}" \
  --budget "${BUDGET:-0.15}" \
  --epochs "${EPOCHS:-30}" \
  --batch "${BATCH_SIZE:-1}" \
  "$@"
