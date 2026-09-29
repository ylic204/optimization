# V8.1 first-person camera dataset

V8.1 replaces the rendered top-down graph/BEV image with a first-person
camera observation.  The default Qwen3-VL input is **RGB + task text only**.
Depth, semantic segmentation, local BEV, trajectory costs and camera-to-BEV
correspondences are privileged labels.

## Data flow

1. Generate a procedural 72x72 local world with occupied cells and normal,
   rough and hazardous terrain.
2. Extrude occupied cells into 1.8 m obstacles and ray-cast a 288x288
   first-person camera.
3. Save aligned RGB, metric depth and semantic labels.
4. Evaluate 11 fixed local motion primitives, canonically ordered from left
   to right.  Their identity is stable across samples, unlike a newly sorted
   A* candidate list.
5. Use exact ray hits to map the 9x9 Qwen selector regions back to BEV cells
   and to the trajectory cost features that those regions reveal.

The camera pose is necessary metadata for rendering and correspondence.  It
is fixed and is not passed to the student as a pose or graph feature.

## Important fields

| Field | Shape | Default role |
|---|---:|---|
| `rgb`, `image` | `[288,288,3]` | Student VLM image |
| `depth_gt` | `[288,288]` | Optional sensor / auxiliary supervision |
| `semantic_gt` | `[288,288]` | Auxiliary supervision |
| `local_bev_gt` | `[72,72]` | Privileged planning/debug target |
| `trajectory_points_robot` | `[11,64,2]` | Fixed local motion primitives |
| `trajectory_features` | `[11,4]` | Length/rough/hazard/blocked amounts |
| `trajectory_costs` | `[11]` | Downstream task costs |
| `candidate_path_patch_relation` | `[11,81]` | Visible patch-to-path relation |
| `trajectory_region_features` | `[11,81,4]` | Cost features revealed per patch |
| `patch_to_bev` | `[81,2]` | Representative visible BEV cell |
| `optimal_path_idx` | scalar | Minimum-cost local trajectory |
| `split`, `dataset_seed` | scalar | Split provenance |
| `sample_seed`, `sample_index` | scalar | Individually reproducible sample |

`depth_vlm_image` and `semantic_vlm_image` are colorized views for a future
multi-image ablation.  The current single-image Qwen pruning path must not
silently concatenate them or treat them as default inputs.

## Generate data

Preview one sample:

```bash
cd dcv_v74_decision_value_bottleneck
python generate_camera_dataset_v81.py \
  --out data_v81 --split preview --n 1 \
  --preview camera_v81_preview.png
```

Generate splits:

```bash
N_TRAIN=5000 N_VAL=1000 N_TEST=1000 \
  bash generate_v81_splits.sh /absolute/path/to/project
```

The default seeds are `7`, `10007` and `20007`.  Train, validation and test
must use different seeds.  Existing split files are preserved by default;
set `OVERWRITE=1` only when the old `.npz` files in all three split
directories should be replaced.

For a quick pipeline check before the full generation:

```bash
N_TRAIN=100 N_VAL=20 N_TEST=20 OUT=/tmp/data_v81_smoke \
  bash generate_v81_splits.sh /absolute/path/to/project
```

Use `CameraTrajectoryDataset` from `camera_dataset_v81.py` to load samples.
Its `image` key is always RGB.  Dense per-pixel camera-to-BEV mapping and the
colorized depth/semantic images are opt-in to avoid wasting training memory.

## Real-world interpretation

- RGB: directly available from a monocular or RGB-D camera.
- Depth: directly available from an RGB-D camera; otherwise estimated by
  stereo, projected from LiDAR, or predicted by a depth network.
- Semantic segmentation: normally a model prediction, not a raw sensor
  measurement.  Simulator labels are suitable as privileged training GT but
  should not be assumed perfect at deployment.

The recommended main experiment is RGB-only inference.  RGB+depth can be a
separate sensor-rich ablation.  Semantic input should be reported as a
perception-assisted ablation because it includes another model's errors and
compute cost.

## Scope: IID frames versus continuous episodes

V8.1 intentionally generates independent local-planning scenes.  It tests the
core question "which RGB regions are necessary for the current trajectory
decision?" but it is not yet a temporal embodied-navigation dataset.

For a continuous V8.2 benchmark, split by complete scene/episode rather than
by frame.  A recommended episode record is:

| Field | Shape | Student or teacher |
|---|---:|---|
| `rgb` | `[T,288,288,3]` | Student visual input |
| `task_text` | string | Student task input |
| `goal_vector` | `[T,2]` | Student measurable local subgoal |
| `proprioception` | `[T,d_s]` | Student speed/previous-action state |
| `trajectory_costs` | `[T,11]` | Downstream supervision |
| `optimal_path_idx` | `[T]` | Downstream supervision |
| `depth_gt`, `semantic_gt` | `[T,...]` | Privileged supervision |
| `local_bev_gt`, `pose_gt` | `[T,...]` | Teacher/debug only |
| `candidate_path_patch_relation` | `[T,11,81]` | Patch-value teacher |

The recommended model input is a short RGB window (initially one current
frame, then 2--4 frames), task text, a relative local-goal vector and ordinary
robot state.  Ground-truth pose, BEV and graph features must remain outside
the student input.  Without a visible target or a relative local goal, a
single forward camera image cannot define a unique navigation objective.
