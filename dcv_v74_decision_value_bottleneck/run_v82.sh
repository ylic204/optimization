#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?usage: bash run_v82.sh /path/to/dcv_v74_decision_value_bottleneck}"
shift

EXTRA_ARGS=()
if [[ "${LOAD_4BIT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--load-4bit)
fi
if [[ -n "${RESUME:-}" ]]; then
  EXTRA_ARGS+=(--resume "$RESUME")
fi

python "$ROOT/train_mapped_vlm_planner_v82.py" \
  --data "${TRAIN_DATA:?set TRAIN_DATA to the V8.2 train directory}" \
  --val "${VAL_DATA:?set VAL_DATA to the V8.2 validation directory}" \
  --vlm-model "${VLM_MODEL:-/data/lyi/models/Qwen3-VL-8B-Instruct}" \
  --device "${DEVICE:-0}" \
  --budget "${BUDGET:-0.15}" \
  --epochs "${EPOCHS:-30}" \
  --batch "${BATCH_SIZE:-1}" \
  --workers "${WORKERS:-2}" \
  --vlm-local-files-only \
  "${EXTRA_ARGS[@]}" \
  "$@"
