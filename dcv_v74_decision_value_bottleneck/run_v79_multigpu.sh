#!/usr/bin/env bash
set -euo pipefail

BASE="${1:?Usage: VLM_MODEL=/path/to/Qwen3-VL-8B-Instruct DEVICE=1 bash run_v79_multigpu.sh /absolute/path/to/project}"
VLM_MODEL="${VLM_MODEL:?Set VLM_MODEL to the local Qwen3-VL-8B checkpoint directory}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REQUIRED_FILES=(
  config.py
  dataset.py
  graph_utils.py
  render_utils_v7.py
  render_utils_v79.py
  spatial_paths_v79.py
  generate_dataset_v79_standalone.py
  qwen3vl_selector_v79.py
  train_qwen3vl_selector_v79_multigpu.py
)
MISSING_FILES=()
for required_file in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$SCRIPT_DIR/$required_file" ]]; then
    MISSING_FILES+=("$required_file")
  fi
done
if (( ${#MISSING_FILES[@]} > 0 )); then
  echo "Missing V7.9 files:" >&2
  printf '  %s\n' "${MISSING_FILES[@]}" >&2
  exit 2
fi

# The visible four-stage, three-branch corridor needs a 9x9 merge-region grid.
export GRID="${GRID:-9}"
if [[ "$GRID" -lt 9 ]]; then
  echo "V7.9 requires GRID>=9" >&2
  exit 2
fi

if [[ ! -d "$BASE/data_v79/train" || ! -d "$BASE/data_v79/val" ]]; then
  python generate_dataset_v79_standalone.py --out "$BASE/data_v79" --split train --n 5000 --seed 7
  python generate_dataset_v79_standalone.py --out "$BASE/data_v79" --split val --n 1000 --seed 8
fi

EXTRA_ARGS=()
if [[ "${LOAD_4BIT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--load-4bit)
fi

python train_qwen3vl_selector_v79_multigpu.py \
  --data "$BASE/data_v79/train" \
  --val "$BASE/data_v79/val" \
  --vlm-model "$VLM_MODEL" \
  --device "${DEVICE:-auto}" \
  --vlm-local-files-only \
  --task-text "${TASK_TEXT:-Find a spatially continuous path from the green S hub to the red G hub that minimizes total traversal cost. Account for rough terrain, hazards, and blocked regions.}" \
  --risk-normal "${RISK_NORMAL:-0.0}" \
  --risk-rough "${RISK_ROUGH:-0.35}" \
  --risk-hazard "${RISK_HAZARD:-1.0}" \
  --risk-blocked "${RISK_BLOCKED:-20.0}" \
  --prune-layer "${PRUNE_LAYER:-4}" \
  --out "$BASE/checkpoints/v79_qwen3vl_spatial_selector.pt" \
  --metrics "$BASE/results/v79_spatial_metrics.json" \
  --budget "${BUDGET:-0.075}" \
  --epochs "${EPOCHS:-40}" \
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
