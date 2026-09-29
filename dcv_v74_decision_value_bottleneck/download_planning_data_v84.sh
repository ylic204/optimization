#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:?usage: bash download_planning_data_v84.sh /data/planning_sources}"
mkdir -p "$DATA_ROOT"

# Public source code for the two benchmark environments.
if [[ ! -d "$DATA_ROOT/nuplan-devkit" ]]; then
  git clone https://github.com/motional/nuplan-devkit.git "$DATA_ROOT/nuplan-devkit"
fi
if [[ ! -d "$DATA_ROOT/PointNav-VO" ]]; then
  git clone https://github.com/Xiaoming-Zhao/PointNav-VO.git "$DATA_ROOT/PointNav-VO"
fi

# PointNav episode descriptions are public and can be downloaded directly.
POINTNAV_EPISODES="$DATA_ROOT/PointNav-VO/dataset/habitat_datasets/pointnav/gibson/v2"
mkdir -p "$POINTNAV_EPISODES"
if [[ ! -f "$DATA_ROOT/pointnav_gibson_v2.zip" ]]; then
  wget -O "$DATA_ROOT/pointnav_gibson_v2.zip" \
    https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v2/pointnav_gibson_v2.zip
fi
unzip -n "$DATA_ROOT/pointnav_gibson_v2.zip" -d "$POINTNAV_EPISODES"

cat <<EOF

Public downloads finished.

Two licensed assets still require your account and acceptance of their terms:
1. nuPlan: download maps, mini DBs and mini camera blobs from nuplan.org.
2. Gibson: download Habitat-compatible .glb and .navmesh scene files.

Read README_V84_COMPLETE_PIPELINE.md for the exact directory structures.
EOF
