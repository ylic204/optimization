# V8.5 Tilebased：冻结 VLM 建模与最小地形成本路径

本分支 `v85-tilebased-llm-solver` **基于用户指定的 `v85-bev-selector-llm-solver`**
（基线提交 `fbe42f1`），接入 Tilebased navigation datasets。
主任务只有“从给定 S 到 G，最小化累计地形成本”，**不增加步数/能源约束，不做探索、记忆或车辆控制**。
视觉 token 预算属于选择器，不属于路径规划问题。

## 1. 与旧接口的对应

| 原 V8.5 route/GT 接口 | Tilebased 接口 |
|---|---|
| `PlanningDatasetBEVV85` | `TilebasedDatasetV85` |
| nuPlan BEV 图 | 原始 tile map RGB，保持原分辨率 |
| task + ego/goal + 米制 BEV bounds | 自动任务文本 + 原始 S/G 的 `[row,column]` + 网格空间区域 |
| 16 个带时间戳的车辆轨迹点 | 变长的格子路径，不伪造时间、航向或 ADE/FDE |
| 冻结 Qwen 生成连续轨迹 QP | 同一冻结 Qwen 生成最小成本路径的节点成本 JSON 实例 |
| SLSQP 连续求解 | Dijkstra 精确求解八邻接节点成本问题 |
| ADE/FDE、安全代理反馈 | 真实成本、相对最优成本差距、最优率、建模/求解失败率 |
| `metric → distill → rl` | 对齐三阶段流程；预训练代理改为地形成本和路径成本代理 |

复用 `FrozenQwen3VLPrunableBackbone`、`DecisionRelatedQwenSelector`、
`MaskGradientFlowND`、固定质量投影和固定预算 RL。旧 BEV 脚本仍可用；
新任务走独立入口 `run_tilebased_v85.py`，不把新样本塞进旧候选轨迹/车辆 QP 字段。
**旧 BEV checkpoint 不能作为 Tilebased checkpoint 直接加载。**

## 2. 原始数据与必须保持的规则

来源：<https://github.com/archettialberto/tilebased_navigation_datasets>
固定版本：`384900fd82d8404a6eb0deb326412755620f3747`。

- Pokémon：3,000/500/500 张 train/val/test 地图；每张 4 个原始 S/G 任务，合计 16,000。
- 图像 320×320，地形网格 20×20。Qwen3-VL patch=16、merge=2 时为 10×10=100 个视觉区域；
  **一个视觉区域覆盖 2×2 地形格，地形格不等于视觉 token**。`budget=.15` 保留 15 个区域。
- 读取原始 `maps/costs/sources/targets/paths`，不用 `exp_nodes` 作为输入。
- 坐标为零基 `[row,column]`；行向下、列向右。不是 nuPlan 的 x-forward/y-left 米制坐标。
- 八邻接，包括斜向；斜向不乘 √2。原上游求解器不删除任何有限成本节点，
  **Pokémon 的 wall=25 是高成本格，不在这里擅自改成不可通行障碍**。
- 成本统一报告 `sum(costs * path_mask)`，包含起点和终点。
  上游 A* 累积从起点之后开始；加入固定的起点成本不改变最优路径。
- `paths` 是路径掩码。预处理恢复有序格子序列，并验证端点、连通性、无多余格子和最优成本。
  同成本不同路径都视为最优，不使用逐格完全一致率作为主指标。
- Pokémon 公开地形成本规则来自上游 `pkmn_dataset_generation.ipynb`：
  sand/stairs=1, grass=1.5, snow=1.8, dirt=2, leaves=2.5,
  plants=3.5, lake=6, mud=8, ocean=10, wall=25。

求解规则核对来源：
<https://github.com/archettialberto/neural_weighted_a_star/blob/main/a_star/ray_a_star.py>
和同仓库 `a_star/neighbor_utils.py`。

Warcraft 的下载、网格/成本协议也支持，但预处理要求 `--legend-file` 提供核对过的公开地形—成本说明，
不会凭颜色虚构其语义。它原图只有 96×96，对应 Qwen 的 3×3 个视觉区域；
不要为了制造 token 节省而默认放大。**本轮完整实验入口推荐 Pokémon。**

## 3. 下载与预处理

切换后在 `optimization/dcv_v74_decision_value_bottleneck` 下运行。
下载与预处理只需 CPU、Python >=3.9、NumPy、Pillow，不需要 Habitat、nuPlan 或 CUDA。

```bash
python -m pip install numpy pillow

python download_tilebased_v85.py \
  --output-root /data/planning_sources/tilebased \
  --dataset pkmn

python preprocess_tilebased_v85.py \
  --raw-root /data/planning_sources/tilebased \
  --output-root /data/v85_tilebased/pkmn \
  --dataset pkmn
```

下载器按固定提交下载三个 `.npz`，逐文件验证固定 SHA256，断网重跑会复用已校验文件。
不关闭 TLS 校验。服务器不能联网时：在本地电脑用同一 Python 下载器，
把整个 `tilebased` 文件夹上传到服务器，再执行：

```bash
python download_tilebased_v85.py \
  --output-root /data/planning_sources/tilebased --dataset pkmn --offline
```

也可以把已下载的官方文件直接放在 `tilebased/pkmn/{train,val,test}.npz` 后离线校验。
Windows 下载命令示例：`python download_tilebased_v85.py --output-root D:/datasets/tilebased --dataset pkmn`。

小规模检查可在**另一个输出目录**运行预处理，加 `--max-maps-per-split 20`。
完整转换必须去掉该选项。导出目录非空时拒绝混写；中断导出标记 `complete=false`，读取器拒绝使用。

输出：

- `dataset.json`：图像/网格/坐标/成本契约、公开图例、上游版本、来源 SHA256、完成状态。
- `train/val/test/tasks.jsonl`：每行一个原始 S/G 任务及自动文本，4 个任务共享同一张地图。
- `train/val/test/000000.npz` 等：每张图存一次，含 RGB、真实成本和原始及有序 GT。
- `preprocess_report.json`：逐 split 数量、GT 校验、跨 split 完全相同图像的检查。
- `previews/{split}/*_input.png`：原始模型输入；`*_GT_ONLY.png`：参考路径可视化，**不进入模型**。

保留官方地图划分，不按 task 随机重新划分。原数据由若干大地图裁剪得到，
没有相同图像不等于没有相邻/重叠裁剪；该划分不能宣称是“未见真实世界”的泛化验证。

## 4. 数据读取与任务封装

```python
from torch.utils.data import DataLoader
from planning_dataset_tilebased_v85 import TilebasedDatasetV85

dataset = TilebasedDatasetV85('/data/v85_tilebased/pkmn', split='train')
batch = next(iter(DataLoader(dataset, batch_size=1)))
public = batch['public_inputs']
labels = batch['supervision']
print(public['image'].shape)                 # [1,3,320,320]
print(public['region_world_bounds'].shape)   # [1,100,4]，这里单位为格子
print(public['task_text'][0])
print(labels['optimal_cost'])
```

`public_inputs`：RGB、任务文本、S/G、规则/图例、区域几何、数据集名和样本 ID。
`supervision`：真实成本、GT mask、有序 GT、有效长度和最优成本。
可变长度 GT 只在存储/批处理时 padding 到 `grid_rows * grid_cols`，`-1` 不表示真实轨迹点。

模型侧只接收公开字段。真实成本/GT 只进入训练代理监督或求解后的评价。
不会把完整 cost matrix、GT 轨迹、A* expanded nodes 或训练候选路径写进 prompt。
图中不用画 GT；S/G 已经在任务文本中精确给出。

## 5. 冻结模型与求解器

生成 JSON 的 schema 是 `tilebased_node_cost_path_v1`，包含：
`schema, source_rc, target_rc, node_costs`。
节点成本矩阵必须与原网格一致，成本属于公开词表，起终点必须与任务一致。
解析器拒绝额外字段、重复键、非有限值、负值和不合法的坐标。
不会执行模型生成的 Python，也不把 GT 成本补回预测矩阵。

这里 VLM 在**固定优化问题族内实例化视觉参数**；图的邻接规则、流守恒和目标形式由模板固定。
它不训练新的优化结构推理能力，也不直接输出轨迹。缺失视觉区域保持未知语义，
允许 VLM 根据保留证据推断成本；推断错误由真实成本评价承担，不默认为零成本/免费通行。

Dijkstra 从模型预测矩阵求解，随后才读取真实标签计算：

`relative_regret = max(0, (true_path_cost - optimal_cost) / optimal_cost)`。

成功 loss 为 `min(relative_regret, .999 * failure_cost)`；建模或求解失败记 `failure_cost`，
默认 10。日志始终保留未裁剪 regret；失败的成本/差距记 null。
当前原始图是所有节点有限成本的连通图，**成功率主要反映格式与求解流程是否成功**，
不是车辆安全性或真实避障率；最优率与成本差距是主要质量指标。

## 6. 先运行完整视觉基线

模型阶段使用现有 Qwen 环境，依赖见仓库 `requirements.txt`。CPU 预处理不需要安装完整模型依赖。
建议在已有能运行 Qwen3-VL-8B 的环境中执行：

```bash
python run_tilebased_v85.py \
  --mode evaluate --split val --modes full random --limit 40 \
  --data /data/v85_tilebased/pkmn \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct \
  --device cuda:0 --load-4bit --budget 0.15 \
  --output results/tilebased_full_baseline
```

不需要 checkpoint 就能运行 full/random，检查冻结 VLM 是否能识别地形成本并生成可解实例。
`--modeling-max-new-tokens 4096` 为 20×20 成本 JSON 留出空间；输出截断按建模失败处理。
预处理后的图像不放大；程序核对 Qwen 实际 patch/merge 几何，若会改变输入尺度则报错。

`summary.json` 汇总主指标；`outcomes.jsonl` 记录每个样本的选择区域、原始模型回答、
预测路径、GT、真实成本、状态和时间。`environment_seconds` 是建模+求解+评价耗时，
**不含共享的 early vision、选择器和 mask rollout，因此不是完整端到端加速比**。
100→15 tokens 是后续视觉层/merger/LLM 保留量；剪枝前视觉层仍处理完整图像。

## 7. 三阶段训练

共同参数可按已有机器设置，以下命令使用默认网络尺寸。各阶段只从指定 checkpoint 初始化权重，
不承诺恢复优化器或随机状态；输出目录必须是新的。

```bash
python run_tilebased_v85.py \
  --mode metric --epochs 10 --batch 1 \
  --data /data/v85_tilebased/pkmn \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct --device cuda:0 --load-4bit \
  --output checkpoints/tilebased_metric

python run_tilebased_v85.py \
  --mode distill --epochs 10 --batch 1 --budget 0.15 \
  --checkpoint checkpoints/tilebased_metric/best.pt \
  --data /data/v85_tilebased/pkmn \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct --device cuda:0 --load-4bit \
  --output checkpoints/tilebased_distill

python run_tilebased_v85.py \
  --mode rl --epochs 5 --batch 1 --budget 0.15 --rl-group-size 4 \
  --checkpoint checkpoints/tilebased_distill/best.pt \
  --data /data/v85_tilebased/pkmn \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct --device cuda:0 --load-4bit \
  --output checkpoints/tilebased_rl

python run_tilebased_v85.py \
  --mode evaluate --split test --budget 0.15 \
  --checkpoint checkpoints/tilebased_rl/best.pt \
  --data /data/v85_tilebased/pkmn \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct --device cuda:0 --load-4bit \
  --output results/tilebased_test
```

先检查小批运行时，可给训练命令加 `--limit 40 --val-limit 20 --epochs 1`；
全量训练去掉 limit。RL 每个样本会调用多次冻结模型生成，时间主要不在 CPU Dijkstra。

阶段含义与边界：

1. **metric**：冻结 Qwen 和 selector，只训练地形成本代理头；全视觉及随机稀疏视觉各做一次监督。
2. **distill**：冻结代理头和 Qwen；以训练专用候选路径集合上的期望真实成本差距作为可微代理。
   候选来自 GT、均匀成本及扰动成本的精确求解，仅用于损失，不进入模型输入。
   Teacher 是该固定代理任务损失对 mask 的梯度；FM 学习这个下降场，ND 执行固定步更新。
   `L = L_task_proxy + lambda_gfm*L_gfm + lambda_improve*L_improve + lambda_metric*L_cost_regression`。
   `L_gfm` 包含方向、幅度和投影步匹配；`L_improve` 比较终点与 detach 的初始代理损失，
   **是改善惩罚，不是单调下降保证**。
3. **rl**：只训练 selector + mask field。生成 JSON→真实求解→外部成本反馈；
   对固定预算集合使用既有 without-replacement policy gradient、组内 leave-one-out baseline、KL 和熵。
   reward 包含最终任务损失与相对初始选择的 `L_improve` 惩罚。
   JSON 解码/Dijkstra 没有被假装成可微；代理头不参与 RL 奖励或正式预测。
4. **evaluate**：full/random/initial/learned 统一使用冻结建模器和真实求解器。
   metric/distill 的验证日志显式标记 `not_solver_evaluation=true`；正式主结果来自 evaluate。

固定预算内的迭代会联合替换区域集合；不是每步追加一个区域。
DGD/FM 监督的对象是 mask 更新方向，不是 GT 路径的 flow matching。

## 8. 验证

```bash
python -m pytest -q test_v85_tilebased.py
```

包含：八邻接/起点成本、同成本不同路径、GT 连通性、非法模型拒绝、标签隔离、
失败计分、批读取、冻结参数及 distill/RL 梯度、下载校验、旧 checkpoint 拒绝。
CPU 测试使用小型 backbone 替身测试接口和梯度，真实图求解使用真实 Dijkstra。
Qwen3-VL-8B 的识别准确率、GPU 显存、训练收益及端到端加速需在你的模型环境实测，
不能由这些接口测试推断。

本次已执行：Pokémon 三个原始文件下载与 SHA256 校验、全量 4,000 张地图/16,000 个任务转换与 GT 校验、
三个 split 的 batch=4 读取；相关回归测试 55 passed、1 skipped（缺少可选 Transformers 依赖）。
详细记录见 `validation_tilebased_v85.json`。此外修复了共享 RL 的 masked categorical KL 在反向传播中
可能出现 `(-inf)-(-inf)` 的 NaN；新旧 RL 测试均通过。
