# Decision-Related Visual Token Selection for VLM Planning

本仓库研究面向机器人导航与路径规划的视觉 token 压缩：在固定视觉预算下，
让 Qwen3-VL-8B 只保留对最终优化决策有用的 BEV 区域，并用任务损失的梯度知识
训练一个可在固定步数内执行的视觉 mask 动力系统。本分支的 V8.5 输入已经从
第一视角相机改为固定物理尺度的 ego-centric BEV。

当前主线版本是 **V8.5：Teacher-Gradient Flow Matching + Fixed-Step
Neurodynamics + Endpoint Improvement**。V8.5 的核心变化是：

- Student 学习 full-information Teacher 的任务梯度下降方向；
- Flow Matching 与神经动力学描述的是**同一个视觉 mask 向量场**；
- ND 直接执行该向量场，不再使用 V8.4 的 `trajectory Flow -> ND refiner`
  串联结构；
- 使用端点损失 `L_improve`，约束固定步更新后的任务损失优于初始状态；
- nuPlan 输入不再读取 camera frame，而是地图、路线、信号灯、actors 与 2 秒历史
  组成的单帧 ego-centric BEV；
- V8.4 及更早版本继续保留，用于复现和消融。

V8.5 BEV 使用 schema 85；候选轨迹、任务指标、SDF 和优化 JSON 继续复用 V8.4
规划组件，但不接受旧的第一视角 processed NPZ。

## 1. V8.5 方法概览

```text
任务文本 ────────────────> task context c
                              │
ego-centric BEV ─> Qwen early ViT ─> R 个完整视觉区域 z
                              │
                              ▼
                 Decision Token Selector
                              │
                   初始固定预算 mask w⁰
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
 Full-information Teacher              Student mask field
 ∇w J_T(w) + 可行域投影               vθ(w,t,z,c)
            │                                   │
            └──── 梯度方向/步更新匹配 ──────────┘
                                                │
                                  K 步 projected ND
                                                │
                                      最终 mask wᴷ
                                                │
                           Straight-through hard Top-K
                                                │
                           Qwen 后续 ViT + 候选规划
                                                │
             L_task + L_gfm + L_improve + auxiliary losses
```

这里的“fixed-step”指推理时使用固定的更新次数 `K`。每一步的正步长由模型学习，
初始化值由 `--mask-step-size` 指定。

### 1.1 关键边界

- Student **不输入 `graph_feat`、SDF 或 teacher candidate metrics**。
- Qwen3-VL 的视觉输入是固定尺度的真实 ego-centric BEV，语言输入是真实任务文本。
- 默认 BEV 为 `288×288`，`x∈[-16,64]m`、`y∈[-40,40]m`；每个样本禁止
  自适应缩放。
- 当前/历史场景要素可以绘入 BEV；未来 ego trajectory 仅作为监督保存，禁止
  绘入视觉输入。
- known-map 候选轨迹只进入独立的候选规划分支，不伪装成视觉 token。
- Teacher 在训练时使用完整 early-vision regions 和真实候选指标构造任务监督。
- Student 部署时只保留预算内的 hard Top-K regions。
- 默认 `region_grid=9`，共有 `9×9=81` 个视觉区域；`budget=0.15` 时保留
  约 12 个区域。
- V8.5 优化的是**视觉信息选择状态**，不再生成一条 Flow trajectory 后交给
  另一个 ND trajectory refiner。

## 2. 预算约束的视觉 mask 动力学

令 `w in [0,1]^R` 为 relaxed visual-region mask，固定保留质量为 `k`：

```text
C_k = {w | 0 <= w_r <= 1, sum_r w_r = k}.
```

`Pi_Ck` 是 capped-simplex 投影。它同时保证：

- 所有 mask 权重位于 `[0,1]`；
- 无效区域的权重为 0；
- 每个样本的总视觉预算严格等于 `k`。

Selector 首先产生 task-conditioned 初始状态：

```text
w⁰ = Pi_Ck(sigmoid(selector_logits / tau_mask)).
```

## 3. Full-information Teacher 梯度

在随机采样的 ND 中间状态 `w` 上，Teacher 对真实规划任务目标求梯度：

```text
g_T(w) = grad_w J_T(w).
```

无约束负梯度不一定满足 token budget，因此 Teacher target 使用投影后的可执行
下降方向：

```text
w_T+ = Pi_Ck(w - eta_T * g_T(w))
v_T  = (w_T+ - w) / eta_T.
```

这一步借鉴 gradient-direction matching 的思想，但匹配对象从网络参数空间转为
视觉 mask 状态空间。Teacher 能访问完整视觉表示并通过下游任务损失判断哪些区域
应该被增强或抑制；Student 学习在没有在线反向求导 Teacher 的情况下预测该方向。

## 4. Flow Matching 与 ND 的关系

V8.5 中二者不是前后两个模块：

- **Flow Matching**：训练视角，监督局部 Student velocity；
- **Neurodynamics**：执行视角，用固定次数迭代同一个 velocity field。

Student 向量场为：

```text
v_theta = v_theta(w, t, z, c),
```

其中 `z` 是 decision-related region latent，`c` 是任务上下文，`t=s/K` 是归一化
步索引。Student velocity 在有效区域上做零均值处理，再通过可行域投影保持预算。

固定步 ND 为：

```text
w^(s+1) = Pi_Ck(w^s + eta_s * v_theta(w^s, s/K, z, c)),
s = 0, ..., K-1.
```

### 4.1 Gradient Flow Matching loss

方向项是主要蒸馏信号：

```text
L_dir = 1 - cos(v_theta, v_T).
```

为避免只匹配方向却产生不可控步幅，代码同时使用可调的 magnitude 和 projected
one-step 辅助项：

```text
L_mag  = SmoothL1(log(1 + ||v_theta||), log(1 + ||v_T||))
L_step = SmoothL1(Pi_Ck(w + eta_s v_theta), w_T+)

L_gfm = L_dir
        + alpha_mag  * L_mag
        + alpha_step * L_step.
```

Teacher 梯度非常小时，该样本不参与 cosine direction 平均，避免数值不稳定。

## 5. Endpoint improvement loss

`L_improve` 只比较初始和最终状态：

```text
L_improve = mean ReLU(
    J_T(wᴷ) - stopgrad(J_T(w⁰)) + delta
).
```

其中：

- `J_T(w⁰)` 是初始 task-conditioned mask 的任务损失；
- `J_T(wᴷ)` 在 straight-through hard Top-K mask 上计算，forward 与实际部署一致；
- `stopgrad` 阻止模型通过故意恶化初始状态来降低 hinge loss；
- `delta >= 0` 是要求的最小改善 margin；
- `delta=0` 时，仅在最终任务损失高于初始任务损失时产生惩罚。

它不是逐步单调约束。V8.5 允许中间状态暂时上升，只要求固定 `K` 步后的端点
产生净改善。这比原先设想的 `L_prog` 更符合有限步优化器的训练目标。

## 6. V8.5 总损失

```text
L_total = L_task
          + lambda_metric  * L_metric
          + lambda_gfm     * L_gfm
          + lambda_latent  * L_latent
          + lambda_value   * L_value
          + lambda_improve * L_improve.
```

各项含义：

| 损失 | 作用 |
|---|---|
| `L_task` | 最终稀疏视觉信息下的 expected normalized candidate regret |
| `L_metric` | 有效候选的 8 维任务指标回归 |
| `L_gfm` | Student 对 Teacher projected descent velocity 的匹配 |
| `L_latent` | 稀疏信息集与 full-information decision latent 对齐 |
| `L_value` | 预测当前所选信息集对应的下游任务损失 |
| `L_improve` | 最终状态相对初始状态的端点改善约束 |

V8.5 不再包含 V8.4 的 trajectory `L_flow` 与 trajectory ND energy，也不引入
per-step `L_prog`。

## 7. 主要文件

实验代码位于 `dcv_v74_decision_value_bottleneck/`。

### 7.1 V8.5 主线

| 文件 | 功能 |
|---|---|
| `gradient_flow_selector_v85.py` | capped-simplex 投影、Student mask field、固定步 ND |
| `train_gradient_flow_selector_v85.py` | Teacher 梯度、`L_gfm`、`L_improve`、训练与验证 |
| `evaluate_gradient_flow_selector_v85.py` | full/random/learned 三种视觉预算模式评估 |
| `run_v85.sh` | V8.5 训练启动脚本 |
| `test_v85_gradient_flow.py` | 投影、rollout 和 detached baseline 测试 |
| `README_V85_GRADIENT_FLOW.md` | V8.5 数学与实现补充说明 |
| `bev_renderer_v85.py` | 固定物理尺度 BEV renderer 与区域坐标映射 |
| `export_nuplan_bev_v85.py` | 无 camera 的 nuPlan DB 到 raw BEV 导出器 |
| `raw_record_bev_v85.py` | camera-free schema 85 raw record |
| `planning_data_bev_v85.py` | BEV raw record 到候选规划样本 |
| `planning_dataset_bev_v85.py` | BEV-only PyTorch dataset 与尺度校验 |
| `test_v85_bev.py` | BEV 几何、语义层和无 camera schema 测试 |
| `README_V85_BEV_INPUT.md` | nuPlan BEV 数据处理完整说明 |

### 7.2 复用的规划组件

| 文件 | 功能 |
|---|---|
| `split_planning_data_v84.py` | 按完整 log/scene 划分 train/val/test |
| `planning_data_v84.py` | 候选轨迹、8 个指标与 SDF 构造 |
| `optimization_task_v83.py` | 任务文本到优化 JSON |
| `candidate_milp_solver_v83.py` | 有限候选 MILP teacher/audit solver |
| `qwen3vl_selector_v79.py` | 支持 ViT 内物理 token pruning 的 Qwen backbone |
| `mapped_vlm_optimizer_v83.py` | Candidate metric head |

### 7.3 历史 V8.4 基线

`flow_nd_planner_v84.py`、`train_decision_flow_nd_v84.py`、`run_v84.sh` 和
`README_V84_COMPLETE_PIPELINE.md` 保留原始 trajectory Flow + ND 串联设计，
仅用于历史复现和结构消融，不代表当前主框架。

## 8. 环境准备

建议分开使用 simulator 数据导出环境与 Qwen 训练环境，避免 nuPlan/Habitat 的
旧依赖覆盖训练所需的 PyTorch、Transformers 和 CUDA。

```bash
git clone https://github.com/ylic204/optimization.git
cd optimization/dcv_v74_decision_value_bottleneck

conda create -n dcv-v85 python=3.12 -y
conda activate dcv-v85
pip install -r requirements.txt
```

主要依赖：PyTorch、Transformers、Accelerate、bitsandbytes、NumPy、SciPy、
Pillow、tqdm 和 pytest。

Qwen3-VL-8B 权重建议存放在数据盘：

```text
/data/lyi/models/Qwen3-VL-8B-Instruct
```

代码默认 `local_files_only=True`，训练时不会自动下载模型权重。

## 9. BEV 数据协议

V8.5 只读取 `input_mode=ego_centric_bev_v85` 的 schema 85 processed NPZ。
推荐目录为：

```text
/data/v85_bev/processed/
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

每个 processed sample 的主要字段：

| 字段 | 形状 | 含义 |
|---|---|---|
| `bev_rgb` | `[288,288,3]` | ego-centric BEV，唯一视觉输入 |
| `bev_semantic` | `[15,288,288]` | audit-only 语义层，不输入 Student |
| `bev_config_json` | string | 固定范围、分辨率、grid 与 history 配置 |
| `region_world_bounds` | `[81,4]` | 每个视觉区域的物理边界 |
| `candidate_trajectories` | `[32,16,3]` | 8–32 条真实候选，padding 到 32 |
| `candidate_features` | `[32,8]` | 部署时可获得的候选几何特征 |
| `candidate_metrics` | `[32,8]` | 仅用于训练监督的真实指标 |
| `candidate_valid` | `[32]` | 有效候选 mask |
| `goal_state` | `[4]` | `dx,dy,distance,bearing` |
| `ego_state` | `[8]` | 车辆/机器人当前状态 |
| `task_text` | string | 当前规划任务文本 |
| `source_id` | scalar | nuPlan=0，PointNav=1 |

8 个 lower-is-better 指标为：`collision`、`non_traversable`、`safety_risk`、
`route_deviation`、`lack_of_progress`、`path_length`、`discomfort`、`goal_error`。

nuPlan 导出阶段以 `include_cameras=False` 构造 scenario。完整导出、raw-to-processed
命令、字段定义与 leakage boundary 见
[`README_V85_BEV_INPUT.md`](dcv_v74_decision_value_bottleneck/README_V85_BEV_INPUT.md)。
train/val/test 必须继续按完整 log 划分，不能随机拆散同一 log 的帧。

## 10. 任务优化 JSON

默认任务：

```text
tasks/nuplan_safe_progress.json
tasks/pointnav_safe_short.json
```

其中包含 8 个任务指标的权重、upper limits 与 constraint penalty。正式实验应
固定并保存相同 JSON，避免不同运行之间的任务目标发生变化。

## 11. V8.5 训练

```bash
cd /path/to/optimization/dcv_v74_decision_value_bottleneck
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_DATA=/data/v85_bev/processed/train \
VAL_DATA=/data/v85_bev/processed/val \
VLM_MODEL=/data/lyi/models/Qwen3-VL-8B-Instruct \
DEVICE=1 \
BUDGET=0.15 \
EPOCHS=30 \
BATCH_SIZE=1 \
WORKERS=2 \
LOAD_4BIT=1 \
MASK_STEPS=4 \
MASK_STEP_SIZE=0.25 \
TEACHER_STEP_SIZE=0.25 \
LAMBDA_GFM=1.0 \
LAMBDA_LATENT=0.25 \
LAMBDA_VALUE=0.25 \
LAMBDA_IMPROVE=0.10 \
IMPROVE_MARGIN=0.0 \
bash run_v85.sh "$PWD" \
  --output checkpoints/v85_bev_gradient_flow.pt \
  --metrics results/v85_bev_training_metrics.json
```

`DEVICE=1` 对应 `cuda:1`。可通过 `NUPLAN_TEXT` 和 `POINTNAV_TEXT` 覆盖 NPZ
中的任务文本。

### 11.1 默认关键参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--region-grid` | 9 | 81 个 Qwen merge regions |
| `--prune-layer` | 4 | early vision 第 4 层后选择 |
| `--budget` | 0.15 | 视觉区域保留比例 |
| `--mask-temperature` | 0.35 | 初始 relaxed mask 温度 |
| `--mask-steps` | 4 | Student ND 固定更新次数 |
| `--mask-step-size` | 0.25 | 可学习步长的初始化值 |
| `--teacher-step-size` | 0.25 | Teacher projected gradient step |
| `--gfm-magnitude-weight` | 0.1 | `L_mag` 在 `L_gfm` 内的权重 |
| `--gfm-step-weight` | 0.5 | `L_step` 在 `L_gfm` 内的权重 |
| `--lambda-gfm` | 1.0 | Teacher gradient-flow matching 权重 |
| `--lambda-latent` | 0.25 | decision latent 对齐权重 |
| `--lambda-value` | 0.25 | set-value prediction 权重 |
| `--lambda-improve` | 0.1 | endpoint improvement 权重 |
| `--improve-margin` | 0.0 | 要求的最小端点改善量 |
| `--lr` | `2e-4` | selector、metric head、mask field 学习率 |

### 11.2 从旧 checkpoint 初始化

checkpoint 会记录 `input_mode` 与精确的 `bev_config_json`。默认只允许恢复相同
BEV geometry 的 checkpoint。若必须从旧第一视角权重初始化，需要显式设置
`ALLOW_CROSS_VIEW_RESUME=1`；这只是权重迁移，不代表两个输入分布等价。

## 12. 独立评估

```bash
python evaluate_gradient_flow_selector_v85.py \
  --checkpoint checkpoints/v85_bev_gradient_flow.pt \
  --data /data/v85_bev/processed/test \
  --nuplan-task tasks/nuplan_safe_progress.json \
  --pointnav-task tasks/pointnav_safe_short.json \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct \
  --device cuda:1 \
  --batch 1 \
  --workers 2 \
  --load-4bit \
  --output results/v85_bev_test_metrics.json
```

同一 checkpoint 比较：

- `full`：保留全部视觉区域；
- `random`：随机保留相同数量的区域；
- `learned`：初始 mask 经固定步 Student dynamics 后执行 hard Top-K。

当前输出指标包括：

| 指标 | 含义 |
|---|---|
| `regret` | 最终离散候选选择的 normalized regret |
| `expected_regret` | soft candidate policy 的期望任务 regret |
| `metric_mae` | 有效候选的指标预测误差 |
| `mask_change` | ND 终态与初始 relaxed mask 的平均变化 |
| `tokens` | 实际保留区域数量 |
| `token_ratio` | 实际区域保留比例 |

训练日志还记录 `direction`、`magnitude`、`step`、`improve`、`task_gain` 和
`teacher_direction_norm`，用于判断 Teacher signal 是否可靠以及固定步动力学是否
真正改善任务。

## 13. 建议的核心实验

### 13.1 准确性—效率 trade-off

```text
budget in {0.05, 0.10, 0.15, 0.20, 1.0}
```

每个预算同时报告 regret、retained tokens、wall-clock latency、peak GPU memory
和可选 FLOPs。主要比较 learned Top-K、equal-budget random 与 full-token upper
reference。

### 13.2 损失消融

| 实验 | 设置 |
|---|---|
| w/o Teacher gradient matching | `--lambda-gfm 0` |
| direction only | `--gfm-magnitude-weight 0 --gfm-step-weight 0` |
| w/o endpoint improvement | `--lambda-improve 0` |
| w/o latent alignment | `--lambda-latent 0` |
| w/o set-value prediction | `--lambda-value 0` |
| positive improvement margin | `--improve-margin 0.01` 等 |

### 13.3 动力学消融

比较 `mask_steps in {1,2,4,8}`，报告性能、延迟和 `task_gain`。V8.5 关注的是
固定计算预算下是否获得更好的信息选择，而不是证明所有中间步骤单调下降。

### 13.4 与 V8.4 的结构对比

- V8.4：candidate warm start → trajectory Flow → trajectory ND；
- V8.5：Teacher gradient supervision → one mask field → fixed-step mask ND。

该实验用于验证收益来自 decision-driven information dynamics，而不是额外串联
一个轨迹生成/修正网络。

## 14. 测试

不加载 Qwen 权重的 V8.5 检查：

```bash
python -m py_compile \
  bev_renderer_v85.py \
  raw_record_bev_v85.py \
  planning_data_bev_v85.py \
  planning_dataset_bev_v85.py \
  export_nuplan_bev_v85.py \
  gradient_flow_selector_v85.py \
  train_gradient_flow_selector_v85.py \
  evaluate_gradient_flow_selector_v85.py \
  test_v85_gradient_flow.py

bash -n run_v85.sh
python -m unittest test_v85_bev.py -v
pytest -q test_v85_gradient_flow.py
```

单元测试覆盖：

- capped-simplex 投影的预算、上下界和无效区域；
- mask ND 每一步后的可行性；
- `L_improve` 对初始 baseline 的梯度阻断。
- BEV metric-pixel 往返映射、81 个区域的世界坐标边界；
- semantic raster、dynamic obstacle 与 camera-free raw schema。

## 15. 24 GB GPU 建议

Qwen3-VL-8B 即使冻结也会占用较多激活显存，建议从以下配置开始：

```text
LOAD_4BIT=1
BATCH_SIZE=1
WORKERS=2
MASK_STEPS=4
```

如果仍然 OOM：

1. 确认同一 GPU 上没有其他进程；
2. 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；
3. 用较少 `MASK_STEPS` 做 smoke test；
4. 再考虑 activation checkpointing 或把不同 Teacher forward 分阶段计算。

减小 `budget` 主要减少 pruning layer 之后的计算，无法消除 early ViT 处理完整
BEV raster 时的显存开销。

## 16. 当前实现边界

- Teacher 是训练期由下游任务损失反向得到的 projected mask-gradient oracle，
  不是额外训练的独立大模型。
- 当前候选规划仍在每个场景的 8–32 条自适应候选上进行，不是任意规模的
  edge-flow 全局规划器。
- V8.5 当前重点是 token selection 的准确性—效率 trade-off；尚未把同一框架
  扩展到通信资源、传感器频率或多机器人带宽分配。
- 当前 evaluator 提供统一离线指标；正式论文还需接入 nuPlan closed-loop、
  Habitat PointNav 官方指标以及硬件延迟统计。
- `L_improve` 保证的是训练目标中的端点 hinge 约束，不等价于对任意未见样本的
  数学下降保证。

## 17. 版本说明

- V7.4：顺序 counterfactual-value bottleneck；
- V7.5–V7.7：one-shot selector、position-C、BLIP alignment；
- V7.8–V7.9：Qwen3-VL-8B ViT 内 token pruning；
- V8.1：第一视角 RGB 数据模式；
- V8.2：已知地图与连续候选规划；
- V8.3：任务文本到优化 JSON 与确定性求解器；
- V8.4-r2：动态候选、full-information DGD、trajectory Flow + ND；
- **V8.5 BEV：Teacher projected gradient matching、统一 mask flow/ND、端点
  `L_improve`，输入改为 camera-free ego-centric BEV。**

V8.5 的详细数学说明见
[`README_V85_GRADIENT_FLOW.md`](dcv_v74_decision_value_bottleneck/README_V85_GRADIENT_FLOW.md)，
nuPlan BEV 数据流程见
[`README_V85_BEV_INPUT.md`](dcv_v74_decision_value_bottleneck/README_V85_BEV_INPUT.md)。
