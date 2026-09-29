#!/usr/bin/env bash
set -euo pipefail

BASE="${1:?Usage: bash generate_v81_splits.sh /absolute/path/to/project}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${OUT:-$BASE/data_v81}"

N_TRAIN="${N_TRAIN:-5000}"
N_VAL="${N_VAL:-1000}"
N_TEST="${N_TEST:-1000}"
TRAIN_SEED="${TRAIN_SEED:-7}"
VAL_SEED="${VAL_SEED:-10007}"
TEST_SEED="${TEST_SEED:-20007}"

if [[ "$TRAIN_SEED" == "$VAL_SEED" || "$TRAIN_SEED" == "$TEST_SEED" || "$VAL_SEED" == "$TEST_SEED" ]]; then
  echo "TRAIN_SEED, VAL_SEED and TEST_SEED must be different." >&2
  exit 2
fi

for value in "$N_TRAIN" "$N_VAL" "$N_TEST"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "N_TRAIN, N_VAL and N_TEST must be positive integers." >&2
    exit 2
  fi
done

OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

COMMON_ARGS=(
  --out "$OUT"
  --risk-normal "${RISK_NORMAL:-0.0}"
  --risk-rough "${RISK_ROUGH:-0.55}"
  --risk-hazard "${RISK_HAZARD:-2.0}"
  --risk-blocked "${RISK_BLOCKED:-25.0}"
  "${OVERWRITE_ARGS[@]}"
)

cd "$SCRIPT_DIR"

"$PYTHON_BIN" generate_camera_dataset_v81.py \
  "${COMMON_ARGS[@]}" \
  --split train --n "$N_TRAIN" --seed "$TRAIN_SEED" \
  --preview "$OUT/train_preview.png"

"$PYTHON_BIN" generate_camera_dataset_v81.py \
  "${COMMON_ARGS[@]}" \
  --split val --n "$N_VAL" --seed "$VAL_SEED" \
  --preview "$OUT/val_preview.png"

"$PYTHON_BIN" generate_camera_dataset_v81.py \
  "${COMMON_ARGS[@]}" \
  --split test --n "$N_TEST" --seed "$TEST_SEED" \
  --preview "$OUT/test_preview.png"

echo "Generated V8.1 IID camera splits under: $OUT"
echo "train=$N_TRAIN (seed=$TRAIN_SEED)"
echo "val=$N_VAL (seed=$VAL_SEED)"
echo "test=$N_TEST (seed=$TEST_SEED)"
