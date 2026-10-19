# V8.5 — BEV Selector with Frozen LLM Modeling and Solver Feedback

## Tilebased 最小地形成本路径适配

新增分支 `v85-tilebased-llm-solver`，基于 `v85-bev-selector-llm-solver`。
接入原始 Pokémon/Warcraft tile-map 数据协议；推荐先用 Pokémon 的现成
**地图＋起点＋终点＋最小成本路径 GT**，不添加步数约束或探索/记忆任务。

- `download_tilebased_v85.py`：固定版本下载、SHA256 校验及离线复用。
- `preprocess_tilebased_v85.py`：保留官方地图划分，恢复并校验全部原始 GT，自动生成任务文本。
- `planning_dataset_tilebased_v85.py`：将模型公开输入与真实成本/GT 分开读取。
- `tilebased_solver_environment_v85.py`：冻结 Qwen 生成节点成本实例，Dijkstra 求解，真实成本反馈。
- `vqa_tilebased_v85.py`：用英文环境提示词和公开 S/G 做导航 VQA，检查连接关系、路线代价与决策相关视觉区域，保存逐任务提示词和回答。
- `diagnose_tilebased_v85.py`：原生/自定义 Qwen 同输入对照、局部格子识别、完整矩阵诊断与可选灰图对照。
- `run_tilebased_v85.py`：复用 V8.5 selector/固定步 FM-ND/RL，正式建模默认接入英文导航环境提示词；`--compare-prompts` 在相同视觉选择下配对评估新旧提示词。

**完整下载、预处理、模型调用和训练命令：
[README_V85_TILEBASED.md](dcv_v74_decision_value_bottleneck/README_V85_TILEBASED.md)。**
新任务不用车辆 QP、ADE/FDE 或旧 BEV checkpoint。
下面保留原 BEV 实验说明，供旧版本复现。


本分支 `v85-bev-selector-llm-solver` 在 `v85-bev-rl-posttraining` 基础上补齐主链路：
**选择 BEV 区域 → 冻结 LLM 生成优化实例 → 连续轨迹求解器 → 与时间对齐 GT 比较 → 更新区域选择策略**。
GitHub `main` 仍是旧第一视角版本，请切换到本分支运行。

## 学习什么

动作只有固定预算的 BEV 区域集合。Qwen3-VL 和求解器始终冻结；没有建模能力的参数训练。
Selector 输入包含任务及建模证据需求：路线、参考点、可行走廊、actors 和停止边界。
FM/Teacher 梯度匹配训练同一个 mask 场，固定步 ND 执行它，没有 trajectory-FM 与 ND 串联。

| 阶段 | 实际执行 | 更新参数 |
|---|---|---|
| 数据 | 当前/历史 BEV；未来 expert 与真实时间戳单独保存 | 无 |
| `metric` warm-up | 全 token + 随机剪枝，监督候选指标代理头 | 代理 metric head |
| `distill` warm-up | 冻结代理 Teacher 的 mask 梯度，共享场匹配和端点改善 | Selector + mask/ND field |
| `rl` 主后训练 | 硬剪枝 → 冻结 Qwen 生成 QP → SciPy 求解 → trajectory/GT 奖励 | Selector + mask/ND field |
| 推理/评估 | 确定性 Top-K → 同一冻结建模器和求解器 → trajectory | 无 |

候选成本只用于前两阶段的可微初始化代理。主 RL 不使用候选 argmin 或预测头奖励，
不向建模器传递候选列表、GT 或栅格标签。默认 `--rl-backend llm_solver`；
旧 `candidate_proxy` 仅作为显式消融保留。

## 模型和损失

冻结 LLM 生成受限 JSON：16 个未来参考点、逐步可行走廊、tracking/smoothness/goal 权重、
可选 separating/stop 半空间。统一模板优化全部连续未来 xy，施加初始位置、实测初始速度、
逐轴速度/加速度界。JSON 经尺寸/有限值/凸性/范围验证，SciPy SLSQP 求解 QP 后检查成功与残差。
这是受限凸模型族，不生成任意可执行程序。

```math
J_{raw}(a)=\lambda_A ADE(\hat\tau(a),\tau^{GT})
+\lambda_F FDE(\hat\tau(a),\tau^{GT})+\lambda_S(C_{static}+O_{static}).
```

默认权重 1/0.25/1。成功损失上限为 `0.999 × failure_cost`，建模格式错误/不可行/求解失败
记 `failure_cost`（默认100），不输出伪成功轨迹。ADE/FDE 报告保留原始米单位；失败记 null，不记0。

```math
L_{improve}=ReLU(J(a)-stopgrad(J(a_0))+m),\quad
r=-J(a)-\lambda_iL_{improve},\quad
L_{RL}=-mean(A\log\pi_\theta)+\beta\bar D_{KL}-\alpha\bar H.
```

`a_0` 是同预算初始 Selector Top-K。`L_improve` 只约束端点，margin=0允许持平，不保证每步下降。
联合动作 log-probability 与 leave-one-out advantage 更新选择策略，不对 JSON 或求解器求导。
这是离线 contextual-bandit REINFORCE，不是 PPO/GRPO 或闭环驾驶 RL。

## BEV 与 GT

BEV 由 nuPlan 地图、路线、当前 actors、信号灯与过去历史直接渲染，无 camera 输入。
默认288×288、x∈[-16,64]m、y∈[-40,40]m、81区域，15%预算约12区域。
所有 early-ViT 区域先编码；删除的是后续 ViT/merger/DeepStack/LLM tokens，不能理解为只观察15%环境。

nuPlan 有 logged expert trajectory GT，没有优化建模 JSON GT。GT 与求解器均采用当前 rear axle 原点，
x前向/y左向。默认未来8秒16点；新导出保存实际 `expert_time_s` 与协议 `planning_time_s`，
按时间插值并保留停车状态。GT 不进入图像或建模 prompt。
processed 数据升级 **schema85 revision3**。旧 raw 缺时间戳时必须重新导出，不能凭路径长度补造时序 GT。

## 运行

```bash
cd dcv_v74_decision_value_bottleneck
pip install -r requirements.txt
# nuPlan devkit 单独安装；按完整 log 划分 train/val/test 后分别导出。
python export_nuplan_bev_v85.py \
  --data-root /data/nuplan/dataset --map-root /data/nuplan/maps \
  --db-files /data/nuplan/splits/train/*.db \
  --output-root /data/v85_bev/raw_r3/train --future-horizon 8 --future-samples 16
python process_planning_bev_v85.py \
  --raw-root /data/v85_bev/raw_r3/train --output-root /data/v85_bev/r3/train
# val/test 使用相同参数处理；相邻帧不能跨 split。
export TRAIN_DATA=/data/v85_bev/r3/train VAL_DATA=/data/v85_bev/r3/val
export VLM_MODEL=/data/lyi/models/Qwen3-VL-8B-Instruct LOAD_4BIT=1 DEVICE=0
STAGE=metric EPOCHS=10 bash run_v85.sh "$PWD" --output checkpoints/v85_metric.pt
STAGE=distill EPOCHS=20 RESUME=checkpoints/v85_metric.pt \
  bash run_v85.sh "$PWD" --output checkpoints/v85_distill.pt
RESUME=checkpoints/v85_distill.pt EPOCHS=10 \
  bash run_v85_rl.sh "$PWD" --lr 5e-5 --rl-backend llm_solver \
  --output checkpoints/v85_solver_rl.pt
python evaluate_gradient_flow_selector_v85.py \
  --checkpoint checkpoints/v85_solver_rl.pt --data /data/v85_bev/r3/test \
  --nuplan-task tasks/nuplan_safe_progress.json --pointnav-task tasks/pointnav_safe_short.json \
  --backend llm_solver --device cuda:0 --load-4bit --output results/v85_solver_test.json
```

epoch/学习率只是起始设置。`--resume` 初始化权重，不恢复optimizer/RNG。
新数据/文本契约要求重训，不直接复用旧revision2 checkpoint。BEV导出目前仅nuPlan；
PointNav参数保留给旧代理头的双域接口。

## Baseline、报告与边界

`full/random/initial/learned` 共用冻结Qwen、prompt/schema、求解器、时间网格、ego/goal与GT。
后三者同预算；full是全信息参考。外部baseline可向 `FrozenBEVSolverEnvironment.evaluate` 传入indices。
报告所有样本 `task_loss`、成功样本ADE/FDE、建模/求解失败率、静态碰撞/越界代理、token ratio、环境耗时。
`*.outcomes.jsonl` 保存区域、生成模型、状态、trajectory、时间和GT。环境耗时不含early编码/Selector/ND，
不能冒充完整推理延迟。失败率必须和成功样本ADE/FDE一起报告。

QP使用点轨迹与逐轴动力学近似；静态栅格代理没有车辆扫掠包络或未来actors。
LLM也可能漏检走廊/交通约束，QP可行不等于道路安全。
本版rear-axle/16点ADE协议不能直接当nuPlan官方center/评测协议。
已做CPU和小型真实Qwen结构验证，未运行预训练8B权重、真实nuPlan训练或闭环测试。

## 文档

- [完整方法与训练损失](dcv_v74_decision_value_bottleneck/README_V85_POSTTRAINING.md)
- [BEV与时序GT](dcv_v74_decision_value_bottleneck/README_V85_BEV_INPUT.md)
- [共享场与ND](dcv_v74_decision_value_bottleneck/README_V85_GRADIENT_FLOW.md)
- [建模和求解器接口](dcv_v74_decision_value_bottleneck/README_V85_SOLVER_FEEDBACK.md)
- [验证记录](dcv_v74_decision_value_bottleneck/VALIDATION_V85_POSTTRAINING.md)

旧版本脚本保留复现；V8.3候选MILP入口不是本分支主RL链路。
