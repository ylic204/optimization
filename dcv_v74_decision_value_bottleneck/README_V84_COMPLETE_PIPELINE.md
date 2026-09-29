# V8.4-r2 complete experiment pipeline

V8.4 implements the complete research hypothesis:

1. an LLM task agent maps language to optimization weights and limits;
2. a Qwen3-VL selector retains a fixed budget of decision-relevant regions;
3. a full-token online teacher distills decision gradients and candidate policy;
4. conditional Flow Matching transports a dynamic warm start toward an
   independent expert/geodesic trajectory;
5. unrolled neurodynamic (ND) refinement reduces collision, goal, length and
   smoothness energy;
6. the final planning loss supplies the token-level decision-gradient teacher.

## A. Files

Data:

- `download_planning_data_v84.sh`: public downloads and licensed-data notice;
- `raw_record_v84.py`: source-environment export format;
- `split_planning_data_v84.py`: log/scene-level deterministic splitting;
- `planning_data_v84.py`: candidates, metric targets and signed-distance map;
- `process_planning_data_v84.py`: raw-to-training conversion CLI;
- `planning_dataset_v84.py`: Qwen/PyTorch loader.

Model and experiments:

- `flow_nd_planner_v84.py`: Qwen selector, coefficient head, flow and ND;
- `train_decision_flow_nd_v84.py`: end-to-end training and validation;
- `evaluate_decision_flow_nd_v84.py`: full/random/learned evaluation;
- `infer_decision_flow_nd_v84.py`: one-sample deployment-style inference;
- `run_v84.sh`: training launcher.

## B. Download

```bash
bash download_planning_data_v84.sh /data/planning_sources
```

The script clones nuPlan-devkit and PointNav-VO and downloads the public
PointNav-Gibson-v2 episode archive. It cannot accept data licenses for you.

For nuPlan, sign in at the official site, accept the terms and download maps,
mini DB files and mini camera sensor blobs. Arrange them as:

```text
/data/planning_sources/nuplan/
├── maps/
└── nuplan-v1.1/
    ├── splits/mini/*.db
    └── sensor_blobs/<log>/CAM_F0/*.jpg
```

For PointNav, accept the Gibson terms and put `.glb` and `.navmesh` files in:

```text
/data/planning_sources/PointNav-VO/dataset/Gibson/gibson/
```

Use three environments:

1. the official nuPlan environment for nuPlan raw export;
2. the pinned PointNav-VO/Habitat environment for indoor raw export and VO;
3. the current Qwen environment for V8.4 processing/training.

This prevents the old simulator dependencies from downgrading Transformers,
PyTorch or CUDA packages required by Qwen3-VL-8B.

## C. Export source data to raw records

Both simulators write the same small record with `save_raw_record(...)`:

```python
from raw_record_v84 import save_raw_record

save_raw_record(
    output_path="/data/v84_raw/train/nuplan/frame_000001.npz",
    image=front_rgb,
    reference_path_ego=route_xy_or_xyyaw,
    expert_trajectory_ego=future_ego_xy_or_geodesic,
    traversable=local_drivable_raster,
    dynamic_obstacle=local_dynamic_obstacle_raster,
    map_bounds=[-5.0, 35.0, -20.0, 20.0],
    goal_xy=[30.0, 0.0],
    ego_state=[vx, vy, ax, ay, yaw_rate, steering, 0.0, 0.0],
    task_text="Drive safely to the goal while following the mapped route.",
    source_id=0,
    sample_id="log-token/iteration-12",
)
```

The same call is used in Habitat with `source_id=1`, egocentric RGB, navmesh
traversability and the VO-updated PointGoal.

Coordinate convention is fixed:

- x forward;
- y left;
- raster row zero is `ymax`;
- raster column zero is `xmin`;
- `map_bounds=[xmin,xmax,ymin,ymax]`.

### nuPlan extraction loop

Inside an official nuPlan scenario loop:

1. retrieve `CAM_F0` at the current iteration;
2. rasterize lane/drivable map layers in the ego frame;
3. rasterize current tracked vehicles/pedestrians as dynamic obstacles;
4. obtain the route polyline and transform it into the ego frame;
5. choose the route goal or local horizon endpoint;
6. call `save_raw_record`.

The devkit's sensor tutorial should be treated as the authoritative camera
loader because sensor interfaces and blob layouts differ between devkit
releases.

### PointNav extraction loop

Inside the PointNav-VO agent loop:

1. obtain `obs["rgb"]` and `obs["depth"]`;
2. call `_compute_local_delta_states_from_vo(prev_obs, obs, action)`;
3. update the local PointGoal with the VO delta;
4. query the known navmesh shortest path;
5. sample `pathfinder.is_navigable()` over an ego-frame raster;
6. mark observed collision cells as dynamic obstacles;
7. call `save_raw_record`.

For the first controlled experiment, export both oracle-pose and VO-pose
versions. The difference measures localization sensitivity.

## D. Convert raw records to V8.4

First split by complete log or scene, not by individual frames. If the first
directory below each source root is the nuPlan log or Gibson scene, run:

```bash
python split_planning_data_v84.py \
  --input-root /data/v84_raw_all/nuplan \
  --output-root /data/v84_raw_split/nuplan \
  --val-ratio 0.1 --test-ratio 0.1 --seed 2026 --group-depth 1

python split_planning_data_v84.py \
  --input-root /data/v84_raw_all/pointnav \
  --output-root /data/v84_raw_split/pointnav \
  --val-ratio 0.1 --test-ratio 0.1 --seed 2026 --group-depth 1
```

The generated `split_manifest.json` records the group allocation and exact
sample counts. Keep it with every reported experiment.

```bash
python process_planning_data_v84.py \
  --raw-root /data/v84_raw_split/nuplan/train \
  --output-root /data/v84/train/nuplan \
  --source nuplan

python process_planning_data_v84.py \
  --raw-root /data/v84_raw_split/pointnav/train \
  --output-root /data/v84/train/pointnav \
  --source pointnav
```

Repeat for validation and test. Split nuPlan by complete log/scenario and
PointNav by complete Gibson scene before conversion.

The processor performs these deterministic steps:

1. resize traversability/obstacle rasters to 128x128;
2. create a signed-distance field, positive in free space;
3. allocate 8--32 candidates from route curvature and static-map complexity;
4. generate lateral, terminal-heading and progress-profile variations;
5. keep a diverse subset and pad it to 32 with `candidate_valid=0`;
6. sample map/obstacle values and generate eight normalized metric targets;
7. resample the independent expert trajectory to sixteen points;
8. save the RGB, map, candidates and supervision in one NPZ.

If `expert_trajectory_ego` is omitted, conversion falls back to the reference
path and writes `expert_is_fallback=1`. This supports a smoke test, but final
nuPlan/PointNav experiments should export true future/geodesic supervision.

## E. Qwen input

The image becomes a `[3,288,288]` float tensor. Qwen's processor creates the
vision patches and text tokens. After the early vision blocks, a 9x9 grid gives
81 selectable merge regions. The selector receives visual regions, text
embeddings, region coordinates and the token budget.

Candidate trajectories and the SDF are not inserted as fake image tokens:

- candidates enter the trajectory/coefficient branch;
- SDF enters ND constraint energy;
- only real RGB and language enter Qwen.

## F. Train in two stages

Before training, the language agent can turn a new instruction into the fixed
optimization JSON used by the deterministic teacher and the task loss:

```bash
python optimization_task_v83.py \
  --model /data/lyi/models/Qwen3-VL-8B-Instruct \
  --domain nuplan \
  --task-text "Follow the mapped route, avoid collisions, and make progress." \
  --output tasks/my_nuplan_task.json \
  --device cuda:1
```

Review the generated weights and limits once, then keep that JSON fixed for a
controlled experiment. The agent never writes or executes solver code.

Stage 1 learns coefficient prediction and the trajectory generator with all
visual regions:

```bash
TRAIN_DATA=/data/v84/train \
VAL_DATA=/data/v84/val \
VLM_MODEL=/data/lyi/models/Qwen3-VL-8B-Instruct \
DEVICE=1 BUDGET=1.0 EPOCHS=10 \
LAMBDA_DGD=0 LAMBDA_LATENT=0 LAMBDA_POLICY_DISTILL=0 \
bash run_v84.sh /home/lyi/neurodynamic/dcv_v74_decision_value_bottleneck \
  --output checkpoints/v84_warmup.pt
```

Stage 2 enables the target budget and decision-gradient distillation:

```bash
TRAIN_DATA=/data/v84/train \
VAL_DATA=/data/v84/val \
VLM_MODEL=/data/lyi/models/Qwen3-VL-8B-Instruct \
DEVICE=1 BUDGET=0.15 EPOCHS=30 \
RESUME=checkpoints/v84_warmup.pt \
NUPLAN_TEXT="Follow the mapped route safely and make progress." \
POINTNAV_TEXT="Reach the goal safely using the known map." \
bash run_v84.sh /home/lyi/neurodynamic/dcv_v74_decision_value_bottleneck \
  --output checkpoints/v84_decision_flow_nd.pt
```

With a 9x9 region grid, a budget of 0.15 retains about 12 of 81 regions.
For controlled ablations, `--flow-steps 0 --lambda-flow 0` removes Flow
Matching at inference/training, and `--nd-steps 0` removes ND refinement.

## G. Losses

The full-information online teacher sends all 81 visual regions through the
remaining Qwen vision blocks and LLM. The Student uses only the hard Top-K
regions. DGD differentiates the teacher's final planning loss with respect to
the all-one visual mask, so it estimates the loss increase caused by removing
each region from the full-information state.

The variable candidate bank supplies a warm start and discrete-regret target.
It no longer supplies the continuous Flow target. The Student predicts its
metrics from pruned RGB/text features, while Flow Matching targets the exported
expert future/geodesic trajectory.

Flow Matching learns the vector field between a candidate warm start and the
teacher trajectory. ND unrolls gradient flow on signed-distance safety,
smoothness, length and goal energy.

The total loss is:

```text
final planning task loss
+ candidate metric calibration
+ Flow Matching velocity loss
+ full-information decision-gradient token distillation
+ selected/full decision-latent cosine alignment
+ selected-set downstream-value prediction
+ full/student candidate-policy KL distillation
```

DGD uses only the final planning terms, not coefficient reconstruction loss.
The latent objective is explicit: the selected-region latent must approximate
the full-region decision code and predict the downstream planning loss of the
selected information set. This prevents `z` from being trained only as an
incidental input to the token score head.

## H. Evaluate

```bash
python evaluate_decision_flow_nd_v84.py \
  --data /data/v84/test \
  --checkpoint checkpoints/v84_decision_flow_nd.pt \
  --nuplan-task tasks/nuplan_safe_progress.json \
  --pointnav-task tasks/pointnav_safe_short.json \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct \
  --device cuda:1 \
  --budget 0.15 \
  --output results/v84_test_metrics.json
```

The evaluator runs three controlled modes with the same Flow/ND planner:

- `full`: all 81 regions;
- `random`: random K regions;
- `learned`: selector Top-K regions.

It reports candidate regret, expert-trajectory error, ND energy, collision
rate, goal error, candidate count, expert-fallback rate, retained region count
and retained-token ratio. A final experiment should have zero fallback rate.

Run one sample and export its selected image regions, candidate decision, Flow
trajectory and ND-refined trajectory:

```bash
python infer_decision_flow_nd_v84.py \
  --data /data/v84/test \
  --index 0 \
  --checkpoint checkpoints/v84_decision_flow_nd.pt \
  --nuplan-task tasks/nuplan_safe_progress.json \
  --pointnav-task tasks/pointnav_safe_short.json \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct \
  --device cuda:1 \
  --output results/v84_prediction.npz
```

For a paper, additionally measure wall-clock
latency, peak GPU memory, Qwen FLOPs, Flow ODE time, ND time and official
nuPlan/PointNav metrics.
