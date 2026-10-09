# V8.5 BEV input and timestamped nuPlan targets (revision 3)

## Input boundary

One fixed-metric ego-centric RGB raster is rendered from map/route, current tracked actors,
traffic lights, past 2-second histories, ego and map reference/goal. The exporter uses
`include_cameras=False`. There is no first-person image or camera-to-BEV network.
The renderer never receives expert future poses.

| Property | Default |
|---|---|
| RGB | 288×288 |
| Bounds | x∈[-16,64]m, y∈[-40,40]m |
| Frame | current ego rear axle; x forward, y left |
| Raster layout | rows x descending, columns y descending |
| Regions | 9×9=81; 32px per region |
| History | 2s, 5 samples |
| Future target | 8s, 16 future timestamps; t0 is excluded from ADE |

Geometry is fixed across splits and never resized. Region centers from `region_world_bounds`
condition the field. Semantic layers/counts are audit labels, not policy features.
Early ViT sees all regions; retained tokens can contain early attention context from other regions.
Compression removes groups from later ViT/merger/DeepStack/LLM, not from sensor acquisition.

## GT and schema

nuPlan supplies recorded expert future rear-axle poses, not optimization-model JSON labels.
Exporter `_future_expert` stores `state.time_point.time_us - anchor.time_us` in seconds and
checks sample count/order/horizon coverage. Missing or insufficient future is skipped.
Processed revision3 interpolates by **time**, with the current pose (0,0,0) at t0;
stationary GT is valid. There is no arc-length resampling or extrapolation of future GT.

| Field | Meaning / use |
|---|---|
| raw `expert_trajectory_ego [N,3]` | recorded future x,y,yaw; target only |
| raw `expert_time_s [N]` | actual future timestamps relative to anchor |
| raw `planning_time_s [16]` | protocol grid, default 0.5..8.0 seconds |
| processed `expert_trajectory [16,2]` | time-interpolated GT xy |
| `expert_heading [16]` | time-interpolated wrapped yaw |
| `trajectory_gt_available` | true only when recorded timestamps/poses exist |
| `expert_is_fallback` | missing GT marker; fallback never allowed in solver RL |
| `gt_contract` | `ego_rear_axle_future_seconds_v1` |
| `schema_version / schema_revision` | 85 / 3 |
| `metric_raster_layout` | `bev_x_rows_y_cols` |
| `bev_rgb / region_world_bounds` | visual/policy input and public fixed geometry |
| `traversable / dynamic_obstacle` | current static occupancy reward labels only |
| candidate fields | proxy warm-up labels/features, excluded from primary modeling |

Timestamp-free old raw can still produce proxy-only revision3 samples marked unavailable,
but primary RL/evaluation preflight rejects them. Re-export these records for primary runs.
Old processed revision1/2 and checkpoints are rejected. Do not fabricate times from arc length.
The raster adapter retains the revision2 geometry correction: transpose/flip metric rasters to
V8.4 orientation before proxy labeling and convert SDF back; RGB is not rotated.

## Export and process

Use a nuPlan devkit environment with the dataset and maps installed:

```bash
python export_nuplan_bev_v85.py \
  --data-root /data/nuplan/dataset --map-root /data/nuplan/maps \
  --db-files /data/nuplan/splits/train/*.db \
  --output-root /data/v85_bev/raw_r3/train \
  --frame-stride 10 --future-horizon 8 --future-samples 16 \
  --history-seconds 2 --history-samples 5
python process_planning_bev_v85.py \
  --raw-root /data/v85_bev/raw_r3/train --output-root /data/v85_bev/r3/train \
  --min-candidates 8 --max-candidates 32
```

Keep complete logs disjoint across train/val/test; preserve log/scenario/iteration identity.
Use the same geometry and horizon for all splits. Current processed model family requires exactly
16 future points. Smoke export can use `--limit-scenarios 20 --frame-stride 20` on mini logs.
Inspect route direction, current actor location, traffic lines and metric bounds before training.

[Root commands](../README.md) cover all stages and evaluation.
`PlanningDatasetBEVV85(..., require_trajectory_gt=True)` checks GT and geometry before primary
training loads Qwen. Expert labels/occupancy can be in a batch for reward; the modeling function
uses an explicit allowlist and never forwards that batch to the LLM.

The reported rear-axle/16-point ADE/FDE are this repository's open-loop protocol. Official nuPlan
planner-expert metrics interpolate at common timestamps but use their own center/horizon/sampling
conventions. Apply that protocol explicitly before comparing official benchmark numbers.
See [official metric implementation](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/metrics/evaluation_metrics/common/planner_expert_average_l2_error_within_bound.py).
