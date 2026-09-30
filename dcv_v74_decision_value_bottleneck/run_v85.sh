#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?usage: bash run_v85.sh /path/to/dcv_v74_decision_value_bottleneck}"
shift

EXTRA_ARGS=()
if [[ "${LOAD_4BIT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--load-4bit)
fi
if [[ -n "${RESUME:-}" ]]; then
  EXTRA_ARGS+=(--resume "$RESUME")
fi

python "$ROOT/train_gradient_flow_selector_v85.py" \
  --data "${TRAIN_DATA:?set TRAIN_DATA}" \
  --val "${VAL_DATA:?set VAL_DATA}" \
  --nuplan-task "${NUPLAN_TASK:-$ROOT/tasks/nuplan_safe_progress.json}" \
  --pointnav-task "${POINTNAV_TASK:-$ROOT/tasks/pointnav_safe_short.json}" \
  --nuplan-text "${NUPLAN_TEXT:-}" \
  --pointnav-text "${POINTNAV_TEXT:-}" \
  --vlm-model "${VLM_MODEL:-/data/lyi/models/Qwen3-VL-8B-Instruct}" \
  --device "cuda:${DEVICE:-0}" \
  --budget "${BUDGET:-0.15}" \
  --epochs "${EPOCHS:-30}" \
  --batch "${BATCH_SIZE:-1}" \
  --workers "${WORKERS:-2}" \
  --mask-steps "${MASK_STEPS:-4}" \
  --mask-step-size "${MASK_STEP_SIZE:-0.25}" \
  --teacher-step-size "${TEACHER_STEP_SIZE:-0.25}" \
  --lambda-gfm "${LAMBDA_GFM:-1.0}" \
  --lambda-latent "${LAMBDA_LATENT:-0.25}" \
  --lambda-value "${LAMBDA_VALUE:-0.25}" \
  --lambda-improve "${LAMBDA_IMPROVE:-0.10}" \
  --improve-margin "${IMPROVE_MARGIN:-0.0}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
