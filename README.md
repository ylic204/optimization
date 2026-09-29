# Decision-Related Visual Token Selection for VLM Planning

本仓库研究面向机器人导航与路径规划的视觉 token 压缩：在固定视觉预算下，
让 Qwen3-VL-8B 只保留对最终优化决策有用的图像区域，同时使用求解器监督、
决策梯度蒸馏（Decision-Gradient Distillation, DGD）、Flow Matching 和固定步数
神经动力学（Neurodynamic, ND）完成轨迹预测与约束细化。

当前主线版本是 **V8.4-r2**（动态候选与 full-information distillation）。
历史 V7.x 文件保留用于消融和演化过程复现。

## 1. 方法概览

```text
任务文本 ──> Qwen 任务 Agent ──> 优化权重、约束和惩罚系数
                                      │
第一视角 RGB ──> Qwen3-VL early ViT ──> 81 个视觉区域
                                      │
任务文本 embedding ──────────────────> Decision Token Selector
                                      │ Top-K，预算 K
                                      ▼
                         Qwen3-VL 后续 ViT + LLM 融合
                                      │
已知地图 ──> 8–32 条自适应连续候选 ──> 候选指标预测与优化选择
                                      │
                                      ▼
专家/测地连续轨迹 ─────> Conditional Flow Matching
                                      │
地图 SDF ─────────────────────────────> Fixed-Step ND Refiner
                                      │
                                      ▼
                最终任务损失 + 决策梯度蒸馏 token selector
```

关键边界：

- Student **不输入 `graph_feat`**。
- Qwen3-VL 只接收真实 RGB 与任务文本。
- 已知地图产生的候选轨迹进入独立的 trajectory branch。
- Signed Distance Field（SDF）只进入 ND 约束能量，不伪装成视觉 token。
- full-information teacher 保留全部视觉区域；student 只保留预算内 Top-K 区域。
- teacher 的候选指标、专家轨迹和约束代价只作为训练监督，不作为 student 输入。
- 默认 `region_grid=9`，共有 `9×9=81` 个可选视觉区域；`budget=0.15`
  时保留约 12 个区域。

V8.4 的总损失为：

```text
L = L_task
  + lambda_metric * L_metric
  + lambda_flow   * L_flow
  + lambda_dgd    * L_dgd
  + lambda_latent * L_latent
  + lambda_value  * L_value
  + lambda_policy * L_policy

L_task = candidate_regret
       + lambda_trajectory * trajectory_imitation
       + lambda_constraint * ND_energy
```

其中 DGD 的 token teacher 来自 **full-token planning forward** 的最终任务损失
对全信息 mask 的梯度，而不是 Student 自身 mask 的重建梯度。`L_latent` 对齐稀疏
Student 与全信息 Teacher 的 decision latent，`L_value` 使 latent 预测所选信息集的
下游规划损失，`L_policy` 蒸馏两者的候选决策分布。

## 2. 主要文件

所有实验代码位于 `dcv_v74_decision_value_bottleneck/`。

| 文件 | 功能 |
|---|---|
| `download_planning_data_v84.sh` | 下载公开 devkit、PointNav-VO 与 PointNav episode |
| `raw_record_v84.py` | nuPlan/Habitat 共用的原始数据导出接口 |
| `split_planning_data_v84.py` | 按完整 log/scene 划分 train/val/test |
| `planning_data_v84.py` | 构造候选轨迹、8 个指标和 SDF |
| `process_planning_data_v84.py` | raw NPZ 到 V8.4 NPZ 的转换入口 |
| `planning_dataset_v84.py` | PyTorch 数据集 |
| `optimization_task_v83.py` | Qwen 文本 Agent：任务文本到优化 JSON |
| `candidate_milp_solver_v83.py` | 有限候选 MILP teacher/审计求解器 |
| `qwen3vl_selector_v79.py` | 可在 ViT 内真正删除视觉区域的 Qwen3-VL backbone |
| `flow_nd_planner_v84.py` | token selector、指标头、Flow Matching、ND |
| `train_decision_flow_nd_v84.py` | 两阶段训练与验证主程序 |
| `run_v84.sh` | V8.4 训练启动脚本 |
| `evaluate_decision_flow_nd_v84.py` | full/random/learned 三种 token 模式测试 |
| `infer_decision_flow_nd_v84.py` | 单样本推理及轨迹导出 |
| `README_V84_COMPLETE_PIPELINE.md` | V8.4 补充技术说明 |

## 3. 环境准备

建议使用三个独立环境，避免旧版 Habitat/nuPlan 依赖覆盖 Qwen 所需的
PyTorch、Transformers 和 CUDA：

1. nuPlan 官方环境：读取 DB、地图和相机数据，导出 raw NPZ；
2. PointNav-VO/Habitat 固定环境：运行 Gibson 场景和 VO，导出 raw NPZ；
3. Qwen 训练环境：数据转换、V8.4 训练、验证和推理。

### 3.1 Qwen 训练环境

```bash
git clone https://github.com/ylic204/optimization.git
cd optimization/dcv_v74_decision_value_bottleneck

conda create -n dcv-v84 python=3.12 -y
conda activate dcv-v84
pip install -r requirements.txt
```

主要依赖包括：

- PyTorch；
- `transformers>=4.57,<5.18`；
- `accelerate>=1.0`；
- `bitsandbytes>=0.45`；
- NumPy、SciPy、Pillow、tqdm 和 pytest。

Qwen3-VL-8B 权重建议放在数据盘，例如：

```text
/data/lyi/models/Qwen3-VL-8B-Instruct
```

代码默认只读取本地权重，不会在训练时自动联网下载模型。

## 4. 数据介绍

### 4.1 nuPlan

nuPlan 用于已知道路地图条件下的自动驾驶局部路径规划。每个样本使用：

- 前视相机 `CAM_F0` RGB；
- ego-frame route/reference path；
- 局部 drivable-area raster；
- 当前车辆和行人形成的 dynamic-obstacle raster；
- 局部目标点和 8 维 ego state。

建议先用 mini split 验证完整流程，再扩展到 trainval。必须按照完整 log 或
scenario 划分数据，不能随机打散相邻帧，否则会产生严重的数据泄漏。

### 4.2 Habitat PointNav / Gibson

PointNav 用于已知 navmesh 下的室内 PointGoal 导航。每个样本使用：

- agent 第一视角 RGB；
- navmesh shortest path；
- `pathfinder.is_navigable()` 产生的局部 traversability raster；
- VO 更新后的 PointGoal；
- 碰撞观测形成的 obstacle raster。

建议同时导出 oracle-pose 与 VO-pose 两套数据，用它们的差异评估定位误差对
token 选择和轨迹规划的影响。

### 4.3 统一 raw record

两个数据源最终都由 `save_raw_record(...)` 写成 NPZ：

| 字段 | 形状/类型 | 含义 |
|---|---|---|
| `image` | `[H,W,3] uint8` | 第一视角 RGB |
| `reference_path_ego` | `[N,2]` 或 `[N,3]` | ego-frame 参考路径，可含 yaw |
| `expert_trajectory_ego` | `[Ne,2]` 或 `[Ne,3]`，推荐 | nuPlan 未来专家轨迹或 PointNav geodesic |
| `traversable` | `[Hm,Wm] bool` | 已知地图可通行区域 |
| `dynamic_obstacle` | `[Hm,Wm] bool` | 当前动态障碍 |
| `map_bounds` | `[4]` | `[xmin,xmax,ymin,ymax]` |
| `goal_xy` | `[2]` | ego-frame 目标坐标 |
| `ego_state` | `[8]` | 速度、加速度、转向等状态 |
| `task_text` | string | 当前规划任务文本 |
| `source_id` | scalar | nuPlan=0，PointNav=1 |
| `sample_id` | string | 唯一样本标识 |

坐标约定：x 轴向前，y 轴向左；raster 第 0 行对应 `ymax`，第 0 列对应
`xmin`。

V8.4 处理后，每个样本还包含：

- `candidate_trajectories`: `[32,16,3]`，8–32 条真实候选并 padding 到 32；
- `candidate_features`: `[32,8]`，部署时可获得的轨迹几何特征；
- `candidate_metrics`: `[32,8]`，teacher 监督；
- `candidate_valid`: `[32]`，真实候选为 1、padding 为 0；
- `candidate_count`: scalar，每个场景实际生成的候选数量；
- `expert_trajectory`: `[16,2]`，独立于候选集合的连续监督轨迹；
- `expert_is_fallback`: scalar，未导出专家轨迹时是否回退到 reference path；
- `sdf`: `[128,128]`，自由空间为正、障碍内部为负；
- `goal_state`: `[dx,dy,distance,bearing]`。

8 个 lower-is-better 指标依次为：`collision`、`non_traversable`、
`safety_risk`、`route_deviation`、`lack_of_progress`、`path_length`、
`discomfort`、`goal_error`。

## 5. 下载数据与代码

在项目目录运行：

```bash
bash download_planning_data_v84.sh /data/planning_sources
```

脚本会自动完成：

1. clone `motional/nuplan-devkit`；
2. clone `Xiaoming-Zhao/PointNav-VO`；
3. 下载并解压公开的 PointNav-Gibson-v2 episode 描述。

脚本不能替你接受数据许可证，因此下面两部分需要人工完成。

### 5.1 nuPlan 授权数据

登录 nuPlan 官方数据页面，接受条款后至少下载：

- maps；
- mini DB split；
- 与 DB 对应的 mini camera sensor blobs。

整理为：

```text
/data/planning_sources/nuplan/
├── maps/
└── nuplan-v1.1/
    ├── splits/mini/*.db
    └── sensor_blobs/<log>/CAM_F0/*.jpg
```

在 nuPlan 环境中设置：

```bash
export NUPLAN_DATA_ROOT=/data/planning_sources/nuplan/nuplan-v1.1
export NUPLAN_MAPS_ROOT=/data/planning_sources/nuplan/maps
export NUPLAN_EXP_ROOT=/data/planning_sources/nuplan/exp
```

### 5.2 Gibson 场景

接受 Gibson 使用条款后，将 Habitat-compatible 的 `.glb` 与 `.navmesh`
场景文件放入：

```text
/data/planning_sources/PointNav-VO/dataset/Gibson/gibson/
```

下载脚本只下载 episode JSON，不包含受许可约束的 Gibson 3D 场景。

## 6. 从 simulator 导出 raw NPZ

下载完成不等于已经得到训练集。必须分别在 nuPlan 和 Habitat 环境的 rollout
循环中调用统一导出函数：

```python
from raw_record_v84 import save_raw_record

save_raw_record(
    output_path="/data/v84_raw_all/nuplan/log_name/frame_000001.npz",
    image=front_rgb,
    reference_path_ego=route_xy_or_xyyaw,
    expert_trajectory_ego=future_ego_xy_or_xyyaw,
    traversable=local_drivable_raster,
    dynamic_obstacle=local_dynamic_obstacle_raster,
    map_bounds=[-5.0, 35.0, -20.0, 20.0],
    goal_xy=[30.0, 0.0],
    ego_state=[vx, vy, ax, ay, yaw_rate, steering, 0.0, 0.0],
    task_text="Drive safely to the goal while following the mapped route.",
    source_id=0,
    sample_id="log_name/iteration_12",
)
```

PointNav 使用相同接口，但设置 `source_id=1`，输入 egocentric RGB、navmesh
traversability、VO-updated PointGoal、室内参考路径，并将 geodesic shortest path
作为 `expert_trajectory_ego`。该字段省略时只用于 smoke test，处理器会回退到
reference path 并设置 `expert_is_fallback=1`。

nuPlan 导出循环需要依次完成：

1. 读取当前 iteration 的 `CAM_F0`；
2. 将 drivable/lane map rasterize 到 ego frame；
3. rasterize 当前 tracked vehicles/pedestrians；
4. 将 route polyline 转换到 ego frame；
5. 选取局部规划 horizon 的目标点；
6. 导出相同 horizon 内的未来 ego expert trajectory；
7. 调用 `save_raw_record`。

PointNav 导出循环需要依次完成：

1. 读取 `obs["rgb"]` 和可选的 `obs["depth"]`；
2. 用 PointNav-VO 更新 agent pose 和局部 PointGoal；
3. 查询 navmesh shortest path；
4. 在 ego-frame raster 上调用 `pathfinder.is_navigable()`；
5. 加入碰撞观测并调用 `save_raw_record`。

## 7. 划分 train/val/test

必须先按照完整 nuPlan log 或 Gibson scene 划分，再生成候选监督：

```bash
python split_planning_data_v84.py \
  --input-root /data/v84_raw_all/nuplan \
  --output-root /data/v84_raw_split/nuplan \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 2026 \
  --group-depth 1

python split_planning_data_v84.py \
  --input-root /data/v84_raw_all/pointnav \
  --output-root /data/v84_raw_split/pointnav \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 2026 \
  --group-depth 1
```

`group-depth=1` 表示 raw root 下第一层目录是 log/scene。脚本会生成
`split_manifest.json`，其中记录 seed、分组分配和准确样本数，应随实验结果保存。

## 8. 生成 V8.4 训练数据

对两个数据源、三个 split 分别运行：

```bash
for split in train val test; do
  python process_planning_data_v84.py \
    --raw-root /data/v84_raw_split/nuplan/${split} \
    --output-root /data/v84/${split}/nuplan \
    --source nuplan

  python process_planning_data_v84.py \
    --raw-root /data/v84_raw_split/pointnav/${split} \
    --output-root /data/v84/${split}/pointnav \
    --source pointnav
done
```

最终目录应为：

```text
/data/v84/
├── train/
│   ├── nuplan/*.npz
│   └── pointnav/*.npz
├── val/
│   ├── nuplan/*.npz
│   └── pointnav/*.npz
└── test/
    ├── nuplan/*.npz
    └── pointnav/*.npz
```

处理程序会确定性地完成：128×128 raster、SDF、基于路线曲率和静态地图复杂度
动态生成 8–32 条候选、padding 到 32、每条 16 个轨迹点、8 个指标 target，
以及独立的连续专家轨迹。动态障碍不参与候选数量分配，避免将训练标签泄漏到
候选生成阶段。

该数据协议已更新；旧版只有 11 条候选且没有 `expert_trajectory` 的 processed
NPZ 必须重新运行 `process_planning_data_v84.py`，不能直接用于新版训练器。

## 9. 生成或检查任务优化 JSON

仓库提供两个可以直接运行的默认任务：

```text
tasks/nuplan_safe_progress.json
tasks/pointnav_safe_short.json
```

也可以让 Qwen3-VL-8B 将新任务文本映射为固定 schema 的优化参数：

```bash
python optimization_task_v83.py \
  --model /data/lyi/models/Qwen3-VL-8B-Instruct \
  --domain nuplan \
  --task-text "Follow the mapped route, avoid collisions, and make progress." \
  --output tasks/my_nuplan_task.json \
  --device cuda:1 \
  --load-4bit
```

Agent 只生成包含 8 个指标权重、upper limits、constraint penalty 和说明的
JSON，不生成或执行 Python 求解器代码。正式实验前应人工检查 JSON，并在一次
受控实验中固定该 JSON。

## 10. 模型训练

训练建议分成两个阶段。

### 10.1 Stage 1：全 token warm-up

先使用全部 81 个区域学习候选指标、任务决策和轨迹生成器，不启用 DGD：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_DATA=/data/v84/train \
VAL_DATA=/data/v84/val \
VLM_MODEL=/data/lyi/models/Qwen3-VL-8B-Instruct \
DEVICE=1 \
BUDGET=1.0 \
EPOCHS=10 \
BATCH_SIZE=1 \
WORKERS=2 \
LOAD_4BIT=1 \
LAMBDA_DGD=0 \
LAMBDA_LATENT=0 \
LAMBDA_POLICY_DISTILL=0 \
bash run_v84.sh "$PWD" \
  --output checkpoints/v84_warmup.pt \
  --metrics results/v84_warmup_metrics.json
```

### 10.2 Stage 2：固定预算决策梯度蒸馏

从 warm-up 权重继续，恢复目标预算并训练 selector：

```bash
TRAIN_DATA=/data/v84/train \
VAL_DATA=/data/v84/val \
VLM_MODEL=/data/lyi/models/Qwen3-VL-8B-Instruct \
DEVICE=1 \
BUDGET=0.15 \
EPOCHS=30 \
BATCH_SIZE=1 \
WORKERS=2 \
LOAD_4BIT=1 \
RESUME=checkpoints/v84_warmup.pt \
NUPLAN_TEXT="Follow the mapped route safely and make progress." \
POINTNAV_TEXT="Reach the goal safely using the known map." \
bash run_v84.sh "$PWD" \
  --output checkpoints/v84_decision_flow_nd.pt \
  --metrics results/v84_training_metrics.json
```

`DEVICE=1` 对应 `cuda:1`。如果不设置 `NUPLAN_TEXT`/`POINTNAV_TEXT`，训练器
使用每个 NPZ 中的 `task_text`。

默认关键参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--region-grid` | 9 | 81 个 Qwen merge regions |
| `--prune-layer` | 4 | 在第 4 个 early vision block 后选择 |
| `--budget` | 0.15 | 视觉区域保留比例 |
| `--flow-steps` | 8 | Flow Euler integration steps |
| `--nd-steps` | 6 | ND unrolled refinement steps |
| `--nd-step-size` | 0.15 | ND 更新步长 |
| `--lambda-flow` | 1.0 | Flow Matching loss 权重 |
| `--lambda-dgd` | 0.25 | 决策梯度蒸馏权重 |
| `--lambda-latent` | 0.25 | Student/Teacher decision latent 对齐 |
| `--lambda-value` | 0.25 | latent 对下游规划损失的预测监督 |
| `--lambda-policy-distill` | 0.25 | full/student 候选决策分布蒸馏 |
| `--lr` | 2e-4 | selector/metric head/flow 学习率 |

Qwen backbone 全部冻结，训练参数来自 selector、candidate metric head 和
Flow Matching 网络。训练期使用 straight-through fixed-mass mask 保留梯度；
验证和推理期使用真正的 hard Top-K，并在 Qwen 后续层物理删除未选视觉区域。

### 10.3 24 GB GPU 建议

Qwen3-VL-8B 即使冻结也会占用较多激活显存。先使用：

```text
LOAD_4BIT=1, BATCH_SIZE=1, WORKERS=2
```

如果仍然 OOM，可先做功能验证：

```bash
TRAIN_DATA=/data/v84/train VAL_DATA=/data/v84/val \
LOAD_4BIT=1 BATCH_SIZE=1 \
bash run_v84.sh "$PWD" --flow-steps 4 --nd-steps 2
```

然后再逐步恢复论文实验默认的 `flow_steps=8`、`nd_steps=6`。不要通过减小
`budget` 来解决 early ViT 的全部显存问题，因为 selector 之前的层仍需处理
完整图像；更小 budget 主要减少 pruning layer 之后的计算。

## 11. 验证结果

```bash
python evaluate_decision_flow_nd_v84.py \
  --data /data/v84/test \
  --checkpoint checkpoints/v84_decision_flow_nd.pt \
  --nuplan-task tasks/nuplan_safe_progress.json \
  --pointnav-task tasks/pointnav_safe_short.json \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct \
  --device cuda:1 \
  --load-4bit \
  --budget 0.15 \
  --output results/v84_test_metrics.json
```

同一 checkpoint 会比较：

- `full`：保留全部 81 个区域；
- `random`：随机保留相同 K 个区域；
- `learned`：selector Top-K。

输出指标包括：candidate regret、expert trajectory error、ND energy、collision
rate、goal error、每场景 candidate count、expert fallback rate、retained regions
和 token ratio。正式实验中 `expert_fallback_rate` 应为 0。论文实验还应额外记录
wall-clock
latency、peak GPU memory、Qwen FLOPs、Flow 时间、ND 时间以及 benchmark 官方
指标。

## 12. 单样本推理

```bash
python infer_decision_flow_nd_v84.py \
  --data /data/v84/test \
  --index 0 \
  --checkpoint checkpoints/v84_decision_flow_nd.pt \
  --nuplan-task tasks/nuplan_safe_progress.json \
  --pointnav-task tasks/pointnav_safe_short.json \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct \
  --device cuda:1 \
  --load-4bit \
  --budget 0.15 \
  --output results/v84_prediction.npz
```

输出 NPZ 包含：

- `selected_region_indices` 和 `token_logits`；
- `predicted_candidate_metrics`、costs 和最终 candidate index；
- candidate warm start；
- Flow trajectory；
- ND-refined final trajectory。

## 13. 消融实验

建议至少报告以下组合：

| 实验 | 设置 |
|---|---|
| Full-token upper reference | `budget=1.0` |
| Equal-budget random | evaluator 自动提供 `random` |
| Learned selector | `budget=0.15` |
| w/o DGD | `--lambda-dgd 0` |
| w/o Flow Matching | `--flow-steps 0 --lambda-flow 0` |
| w/o ND refinement | `--nd-steps 0` |
| 不同预算 | `budget∈{0.05,0.10,0.15,0.20,1.0}` |

应优先验证 learned K-token 是否同时优于 random K-token，并接近 full-token；
随后再分析 Flow 和 ND 对任务误差、碰撞、速度和稳定性的贡献。

## 14. 测试

不加载 Qwen 权重的数据与求解器测试：

```bash
python -m unittest -v test_v83_optimization.py test_v84_data.py
python -m py_compile *_v84.py
bash -n download_planning_data_v84.sh run_v84.sh
```

当前测试覆盖：

- MILP task schema 与最优候选选择；
- raw record 写入和读取；
- log/scene 无泄漏划分；
- 8–32 条自适应候选、padding mask 与 8 个 metric target；
- 独立 expert trajectory 及 reference fallback 标记；
- SDF 正负距离；
- raw-to-processed 目录转换。

## 15. 常见问题

### `ModuleNotFoundError`

请进入代码目录再运行，或加入 `PYTHONPATH`：

```bash
cd /path/to/optimization/dcv_v74_decision_value_bottleneck
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

### `CUDA out of memory`

首先设置 `LOAD_4BIT=1`、`BATCH_SIZE=1` 和
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。确认 GPU 上没有其他进程，
再临时减小 `flow_steps` 与 `nd_steps`。

### 数据中没有 `task_text`

V8.4 raw schema 要求写入 `task_text`。训练时也可以通过 `NUPLAN_TEXT` 和
`POINTNAV_TEXT` 覆盖数据中的文本，但这不会替代 `source_id` 对应的优化 JSON。

### 下载脚本执行后为什么没有训练 NPZ

下载脚本只获取公开代码、episode 和可自动下载的资源。nuPlan/Gibson 授权数据、
simulator rollout、地图 rasterization 和 raw NPZ 导出仍需在各自环境中完成。

## 16. 当前实现边界

- 当前 warm-start teacher 在每个场景的 8–32 条自适应候选上比较成本；它不是
  任意规模的 edge-flow 全局规划器。最终 Flow/ND 轨迹由独立专家轨迹监督，
  不再被有限候选集合锁死。
- `candidate_valid` 区分真实候选和 padding；碰撞、动态障碍与不可通行仍通过
  任务指标和软约束处理。
- 当前验证器提供统一离线指标；正式论文还需接入 nuPlan closed-loop 与 Habitat
  PointNav 官方指标。
- 浏览器上传到 GitHub 的 `.sh` 可能没有 executable bit，可直接使用
  `bash script_name.sh ...`，或 clone 后执行 `chmod +x *.sh`。

## 17. 版本说明

- V7.4：顺序 counterfactual-value bottleneck；
- V7.5–V7.7：one-shot selector、position-C、BLIP alignment；
- V7.8–V7.9：Qwen3-VL-8B ViT 内 token pruning；
- V8.1：第一视角 RGB 数据模式；
- V8.2：已知地图与连续候选规划；
- V8.3：语言任务到优化 JSON 与确定性求解器；
- **V8.4-r2：动态候选 + 全信息 DGD + decision latent 蒸馏 + Flow + ND。**
