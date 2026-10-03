# V8.5 ego-centric BEV input and nuPlan processing

This branch replaces the first-person camera image with one fixed-metric,
ego-centric bird's-eye-view raster.  It keeps the V8.5 Teacher-gradient mask
field, fixed-step neurodynamics, and endpoint `L_improve` objective unchanged.

The data design follows the useful input boundary of the ICLR 2026 BIRDriver
work: use an ego-centric BEV that exposes road structure, route, traffic
controls, dynamic actors, and short motion history instead of surround-camera
frames.  This repository does **not** reproduce BIRDriver's VLM keypoint
generation or scene-type SFT pipeline; the BEV is consumed by the existing
decision-related token selector.

## 1. Input contract

The default frame is:

| Property | Default |
|---|---:|
| Raster | `288 x 288` RGB |
| Longitudinal range | `x in [-16, 64] m` |
| Lateral range | `y in [-40, 40] m` |
| Convention | `x` forward, `y` left, origin at ego rear axle |
| Selector regions | `9 x 9 = 81` |
| Region size | `32 x 32 px`, about `8.89 x 8.89 m` |
| Actor history | 2 seconds, 5 samples |

The extent and raster size are fixed for every sample.  The dataset rejects
mixed geometries and never resizes BEV images, because resizing would change
the metric meaning of each visual region.

The mask field receives region-center positions derived from
`region_world_bounds` in physical `x-forward/y-left` order, normalized over
the fixed support.  It no longer uses generic image `column/row` coordinates.

The RGB BEV contains drivable area, route roadblocks, lane centerlines,
crosswalks, the route reference, traffic-light control lines, tracked agents,
their past tracks, ego, and the route goal.  A channel-first semantic raster is
saved beside RGB for auditing; it is not provided to the Student as a
privileged input.

## 2. Leakage boundary

Only information available at the current timestamp or in the past can enter
the raster:

- current map, route IDs, and traffic-light status;
- current tracked objects;
- up to two seconds of past tracked-object positions;
- the route reference assembled from map roadblocks.

`get_ego_future_trajectory` is used only to write
`expert_trajectory_ego`, which supervises candidate construction and metrics.
The renderer does not accept this future trajectory, so it cannot leak into
`bev_rgb` or the semantic layers.

## 3. Processing graph

```text
nuPlan DB + maps
  -> export_nuplan_bev_v85.py
  -> raw BEV NPZ (schema 85, camera-free)
  -> process_planning_bev_v85.py
  -> processed BEV NPZ + candidates/metrics/SDF
  -> log-disjoint train/val/test
  -> PlanningDatasetBEVV85
  -> V8.5 selector + shared mask field + fixed-step ND
```

The exporter requests scenarios with `include_cameras=False`; no camera frame
is loaded or serialized.

## 4. Export raw nuPlan BEV records

Run the exporter in the environment containing the pinned nuPlan devkit and
dataset dependencies:

```bash
cd /path/to/optimization/dcv_v74_decision_value_bottleneck

python export_nuplan_bev_v85.py \
  --data-root /data/nuplan/dataset \
  --map-root /data/nuplan/maps \
  --db-files /data/nuplan/splits/train/*.db \
  --output-root /data/v85_bev/raw/train \
  --frame-stride 10 \
  --future-horizon 8.0 \
  --future-samples 16 \
  --history-seconds 2.0 \
  --history-samples 5
```

Useful smoke-test options:

```bash
python export_nuplan_bev_v85.py \
  --data-root /data/nuplan/dataset \
  --map-root /data/nuplan/maps \
  --db-files /data/nuplan/splits/mini/*.db \
  --output-root /data/v85_bev/raw_smoke \
  --limit-scenarios 20 \
  --frame-stride 20
```

Records missing a valid route reference, expert future, or traversable raster
are skipped and counted.  Directory identity preserves
`log_name/scenario_token/iteration.npz`, which supports log-disjoint splits.

## 5. Build candidate-planning samples

Run this step in the existing planning-data environment:

```bash
python process_planning_bev_v85.py \
  --raw-root /data/v85_bev/raw/train \
  --output-root /data/v85_bev/processed/train \
  --min-candidates 8 \
  --max-candidates 32
```

`planning_data_bev_v85.py` reuses the V8.4 candidate generator, eight task
metrics, and SDF construction, then writes schema 85 metadata.  The temporary
`image` compatibility key exists only inside preprocessing; the stored and
loaded model input is `bev_rgb`, and the raw record never contains a generic
camera `image` field.

Split by complete `log_name` before reporting results.  Do not randomly split
individual frames from the same log across train, validation, and test.

## 6. Processed schema additions

| Field | Shape | Meaning |
|---|---|---|
| `input_mode` | scalar | exactly `ego_centric_bev_v85` |
| `bev_rgb` | `[288,288,3]` | only visual input consumed by the VLM |
| `bev_semantic` | `[15,288,288]` | audit-only semantic channels |
| `bev_config_json` | scalar string | metric bounds, size, grid, history config |
| `region_world_bounds` | `[81,4]` | `[xmin,xmax,ymin,ymax]` per visual region |
| `region_semantic_counts` | `[81,15]` | audit-only semantic occupancy per region |
| `scene_type` | scalar string | nuPlan scenario type for stratified analysis |

The existing candidate fields remain: `candidate_trajectories`,
`candidate_features`, `candidate_metrics`, `candidate_valid`, `goal_state`,
`ego_state`, `sdf`, and task text.

## 7. Train and evaluate

```bash
TRAIN_DATA=/data/v85_bev/processed/train \
VAL_DATA=/data/v85_bev/processed/val \
VLM_MODEL=/data/lyi/models/Qwen3-VL-8B-Instruct \
DEVICE=1 \
LOAD_4BIT=1 \
bash run_v85.sh "$PWD" \
  --output checkpoints/v85_bev_gradient_flow.pt \
  --metrics results/v85_bev_training_metrics.json
```

```bash
python evaluate_gradient_flow_selector_v85.py \
  --checkpoint checkpoints/v85_bev_gradient_flow.pt \
  --data /data/v85_bev/processed/test \
  --nuplan-task tasks/nuplan_safe_progress.json \
  --pointnav-task tasks/pointnav_safe_short.json \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct \
  --device cuda:1 \
  --load-4bit
```

Checkpoints record both `input_mode` and the exact `bev_config_json`.
Evaluation refuses checkpoints or data with a different view/geometry.  To
initialize training from old first-person weights, opt in explicitly with
`ALLOW_CROSS_VIEW_RESUME=1`; a fresh BEV run is the safer default.

## 8. Validation

Camera-free tests that do not load Qwen weights:

```bash
python -m unittest test_v85_bev.py -v
python -m py_compile \
  bev_renderer_v85.py raw_record_bev_v85.py \
  planning_data_bev_v85.py process_planning_bev_v85.py \
  planning_dataset_bev_v85.py export_nuplan_bev_v85.py
bash -n run_v85.sh
```

Before a full export, visually inspect samples from each nuPlan map and verify
the ego marker, route direction, red-light control line, actor headings, and
physical extent.  Also compare selected-region statistics by `scene_type` to
detect a selector that overfits to route paint while ignoring dynamic actors.
