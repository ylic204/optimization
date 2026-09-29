# V8.3: Qwen3-VL task agent, visual token pruning and mathematical solver

V8.3 implements the research pipeline described as:

> an agent recognizes a natural-language planning task, produces a structured
> optimization specification, a pruned VLM predicts scene-dependent
> coefficients, and a deterministic solver returns the path.

Flow Matching and neurodynamic optimization are not used.

## 1. What changed from V8.2

V8.2 directly predicts eleven candidate logits. That is a useful baseline but
it is not a generated mathematical optimization model. V8.3 predicts eight
interpretable cost coefficients for every candidate:

1. collision;
2. non-traversable area;
3. safety risk;
4. route deviation;
5. lack of progress;
6. path length;
7. discomfort;
8. goal error.

Qwen converts the task sentence into weights and constraint limits. A
deterministic builder creates and solves the following MILP:

\[
\begin{aligned}
\min_{x,s}\quad &
\sum_i\sum_k w_k\hat m_{ik}x_i+\rho\sum_j s_j \\
\text{s.t.}\quad &\sum_i x_i=1,\\
&\sum_i\hat m_{ij}x_i-s_j\leq u_j,\\
&x_i\leq v_i,\quad x_i\in\{0,1\},\quad s_j\geq0.
\end{aligned}
\]

Here, \(x_i\) selects candidate trajectory \(i\), \(\hat m_{ik}\) is a VLM-
predicted metric, \(w_k\) and \(u_j\) come from the language agent, and \(v_i\)
is map feasibility. Slack variables keep the model feasible while reporting
constraint violations.

The agent is not allowed to generate Python code. It emits a fixed JSON
schema. This makes task generation testable and prevents prompt-dependent
arbitrary solver programs.

## 2. Download nuPlan

Use a separate `nuplan` environment. The official devkit is old and should not
be installed into the Qwen training environment.

```bash
git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit
conda env create -f environment.yml
conda activate nuplan
pip install -e .
```

Go to the official nuPlan download page, create an account, accept its terms,
and download these three parts first:

1. `nuplan-maps-v1.0`;
2. `nuplan-v1.1_mini` database files;
3. mini camera sensor blobs.

Extract them into:

```text
/data/nuplan/
├── maps/
│   ├── nuplan-maps-v1.0.json
│   └── ... map.gpkg files
└── nuplan-v1.1/
    ├── splits/mini/*.db
    └── sensor_blobs/<log-name>/CAM_F0/*.jpg
```

Set the official devkit variables:

```bash
export NUPLAN_DATA_ROOT=/data/nuplan
export NUPLAN_MAPS_ROOT=/data/nuplan/maps
export NUPLAN_EXP_ROOT=/data/nuplan/exp
```

Run the official tutorials before exporting:

```bash
jupyter notebook tutorials/nuplan_scenario_visualization.ipynb
jupyter notebook tutorials/nuplan_sensor_data_tutorial.ipynb
```

Inside the nuPlan iteration loop:

1. obtain `CAM_F0` through `scenario.get_sensors_at_iteration(iteration)`;
2. transform the route polyline to the current ego frame;
3. generate the same eleven candidates with `make_frenet_candidates`;
4. evaluate them with nuPlan map/agent/comfort metrics;
5. normalize every exported metric to `[0,1]` using train-set statistics;
6. call `planning_adapters_v83.export_nuplan(...)`.

Do not split adjacent frames randomly. Split complete logs/scenarios into
train, validation and test.

## 3. Download PointNav-VO and Gibson

For exact reproduction of the ICCV 2021 PointNav-VO setup:

```bash
git clone https://github.com/Xiaoming-Zhao/PointNav-VO.git
cd PointNav-VO
conda env create -f environment.yml
conda activate pointnav-vo
```

The paper repository was tested with:

```text
habitat-lab d0db1b55be57abbacc5563dca2ca14654c545552
habitat-sim 020041d75eaf3c70378a9ed0774b5c67b9d3ce99
```

Download PointNav Gibson v2 episodes:

```bash
mkdir -p dataset/habitat_datasets/pointnav/gibson/v2
wget -O /tmp/pointnav_gibson_v2.zip \
  https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v2/pointnav_gibson_v2.zip
unzip /tmp/pointnav_gibson_v2.zip \
  -d dataset/habitat_datasets/pointnav/gibson/v2
```

Gibson scene files require accepting the Gibson terms. Download the
Habitat-compatible `.glb` and `.navmesh` files, then arrange them as:

```text
PointNav-VO/dataset/
├── Gibson/gibson/
│   ├── Adrian.glb
│   ├── Adrian.navmesh
│   └── ...
└── habitat_datasets/pointnav/gibson/v2/
    ├── train/
    ├── val/
    └── valmini/
```

The paper's VO integration uses
`BaseRLTrainerWithVO._compute_local_delta_states_from_vo(prev_obs, cur_obs,
action)` and updates the current goal from that relative pose. During export:

1. render the current RGB observation;
2. update pose/PointGoal with VO;
3. use the known navmesh to find a shortest reference path;
4. generate eleven local candidates;
5. sample candidates against the navmesh and simulator;
6. normalize the eight metric targets;
7. call `planning_adapters_v83.export_pointnav(...)`.

Split by Gibson scene, not by frames or episodes from the same scene.

## 4. Exported V8.3 sample

Each `.npz` contains:

| Key | Shape | Used at inference | Purpose |
|---|---:|:---:|---|
| `image` | `H,W,3` | yes | front/egocentric RGB |
| `task_text` | string | yes | language request |
| `source_id` | scalar | yes | 0 nuPlan, 1 PointNav |
| `candidate_trajectories` | `11,16,3` | yes | ego-frame x, y, yaw |
| `candidate_features` | `11,8` | yes | path geometry |
| `candidate_metrics` | `11,8` | training label | normalized target metrics |
| `candidate_valid` | `11` | yes | map feasibility |
| `goal_state` | `4` | yes | dx, dy, distance, bearing |
| `ego_state` | `8` | yes | observable motion state |

`candidate_metrics` is not passed to Qwen or the prediction head. It only
supervises coefficient prediction and task regret.

## 5. Exactly how the sample enters Qwen3-VL

The call chain is:

```text
MappedOptimizationDataset.__getitem__
  image uint8 HWC -> resize 288x288 -> float CHW in [0,1]
  task_text       -> Python string
        |
        v
MappedVLMOptimizerV83.encode_and_select
        |
        v
FrozenQwen3VLPrunableBackbone.encode_inputs
  image -> Qwen image processor -> ViT patch embedding -> early vision blocks
  text  -> Qwen tokenizer -> Qwen text embeddings
        |
        v
DecisionRelatedQwenSelector
  visual regions + text tokens + 2-D positions + budget -> 81 region logits
        |
        v
Top-K regions -> remaining Qwen vision blocks -> Qwen LLM fusion
        |
        v
CandidateMetricHead
  fused Qwen state + candidate geometry + goal/ego state -> 11x8 metrics
        |
        v
SciPy MILP -> selected continuous trajectory
```

Candidate trajectories do not become Qwen image tokens. They are structured
optimization variables and enter `CandidateTrajectoryEncoder`. This avoids
forcing geometric paths into a visual-token representation.

## 6. Generate or use optimization tasks

Two fixed baseline tasks are included under `tasks/`. Generate a new task with
the Qwen agent as follows:

```bash
python optimization_task_v83.py \
  --model /data/lyi/models/Qwen3-VL-8B-Instruct \
  --device cuda:1 \
  --domain nuplan \
  --task-text "Reach the destination safely, never leave the drivable area, and prefer comfort." \
  --output tasks/my_nuplan_task.json
```

Inspect and save the resulting JSON. During controlled experiments, cache task
JSON rather than generating it again for every frame.

## 7. Train

Install the Qwen environment dependencies:

```bash
pip install -r requirements.txt
```

Then train:

```bash
TRAIN_DATA=/data/v83/train \
VAL_DATA=/data/v83/val \
VLM_MODEL=/data/lyi/models/Qwen3-VL-8B-Instruct \
DEVICE=1 BUDGET=0.15 BATCH_SIZE=1 EPOCHS=30 \
bash run_v83.sh /home/lyi/neurodynamic/dcv_v74_decision_value_bottleneck
```

The training loss contains:

- task regret from language-weighted downstream path cost;
- optimal-candidate cross entropy;
- coefficient regression for solver calibration;
- DGD from the task loss gradient with respect to visual masks.

The metric-regression loss is not used to construct the DGD teacher. DGD is
derived only from downstream decision terms.

## 8. Agent + solver inference

```bash
python infer_agent_solver_v83.py \
  --sample-dir /data/v83/test/one_scenario \
  --checkpoint checkpoints/v83_optimizer.pt \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct \
  --device cuda:1 \
  --domain nuplan \
  --task-text "Drive safely to the goal while staying in the drivable area." \
  --budget 0.15 \
  --output solver_result.json \
  --trajectory-output selected_trajectory.npy
```

This runs the complete sequence: task JSON generation, visual token selection,
metric prediction, MILP construction, solver call and trajectory export.

## 9. Required paper experiments

At minimum report:

1. full 81 regions;
2. random K regions;
3. attention-only Top-K;
4. learned selector without DGD;
5. learned selector with DGD;
6. oracle target metrics plus solver;
7. predicted metrics plus solver;
8. fixed task weights versus Qwen-generated task weights;
9. PointNav oracle pose versus VO pose;
10. cross-scene and cross-city generalization.

Measure visual token count, latency, peak memory, solver time, path regret,
constraint violation and official downstream metrics. FLOPs or token count
alone is not enough; end-to-end latency must improve.
