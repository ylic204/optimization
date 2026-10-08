#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?usage: bash run_v85_rl.sh /path/to/dcv_v74_decision_value_bottleneck}"
shift
: "${RESUME:?set RESUME to a V8.5 distilled BEV checkpoint}"
STAGE=rl bash "$ROOT/run_v85.sh" "$ROOT" "$@"
