# V8.5 — BEV Decision-Related Token Selection with RL Post-Training

本分支在 `v85-bev-input` 上修复数据与剪枝接口，并实现真正的 outcome-based
RL 后训练。GitHub 默认 `main` 仍保留原 RGB V8.5；运行本版本前请切到包含本
提交的新分支，不要把不同分支的数据、训练入口和 checkpoint 混用。

## 结论与框架边界

- 视觉输入是固定物理尺度的 ego-centric **BEV RGB**，不是第一视角 camera，
  也不是相机经感知网络预测的 BEV。它由 nuPlan 地图、路线、当前 actors、
  交通灯与过去 2 秒历史直接渲染；未来 ego trajectory 不进入图像。
- 默认图像 `288×288`、`x∈[-16,64]m`、`y∈[-40,40]m`，
  `9×9=81` 个区域；约 15% 预算保留 12 个区域。禁止按样本 resize。
- Student 学习 **mask 空间**中的投影任务梯度方向；FM/梯度场匹配训练
  同一个向量场，固定步 ND 执行它。没有 `trajectory Flow -> ND` 串联。
- Qwen3-VL 始终冻结。RL 更新小型 Selector 与 mask field，**不是对 Qwen
  本体做 RL/LoRA，也不是闭环驾驶 RL**。
- 模型还接收候选轨迹几何、自车/目标状态；已知地图用于候选生成。
  专家未来轨迹和真实候选指标不能作为 Student 的视觉/策略输入。
- 原 V8.5 仅做任务监督与梯度蒸馏；本提交新增的第三阶段才是 RL。
  旧 `train_outcome_rl_v7.py` / `train_latent_outcome_rl_v73.py`
  不属于此 V8.5 训练链路。

## 完整步骤

| 阶段 | 实际实现 | 学习对象 |
|---|---|---|
| 数据导出 | nuPlan 当前/历史场景 → 固定尺度 BEV | 无 |
| 数据处理 | 坐标适配 → 候选轨迹、8 项指标、SDF，schema 85 revision 2 | 无 |
| metric 预训练 | 全 token + 随机预算剪枝，监督任务指标与候选排序 | metric head |
| distill 初始化 | 固定规划头产生局部 mask 梯度，训练共享场及端点改善 | Selector + ND |
| RL post-training | 同场景采样多个预算区域集合，实际删除 token 后规划，外部标签评分 | Selector + ND |
| 推理 | 全部 early-ViT 特征 → 初始 mask → 默认 4 步 ND → 确定性 Top-K → 候选决策 | 无 |

Teacher 是预训练后冻结的规划指标头与 Qwen 的组合，不是独立专家模型。
梯度目标经过可微软 mask 分支；训练端点与部署则使用实际删除后的 Top-K
序列。端点的 selected-only straight-through 梯度是有偏代理；RL 使用动作
联合 log-probability，绕开对删除操作求导的需求。

## 本次修复

1. **BEV 指标坐标错误**：渲染器用 x 行/y 列，旧指标器用 y 行/x 列。
   新的转置/翻转适配器只转换指标栅格，输出 SDF 再转回 BEV，图像不旋转。
2. **旧标签被静默复用**：processed 数据升级 revision 2，数据集拒绝旧标签。
   原 raw BEV 可直接重新处理；旧 BEV checkpoint 必须重新训练。
3. **置零与删除不一致**：训练端点、评估和 RL 都实际缩短视觉 token 序列；
   两个 `L_improve` 端点使用同一 hard-pruning 路径。
4. **Teacher 随预测头漂移**：metric → distill → RL 显式分阶段，后两阶段
   冻结规划头，不再从随机预测头同时蒸馏。
5. **投影反向梯度不正确**：使用自由坐标上的中心化切空间 Jacobian，
   饱和/无效坐标梯度为零，并加入有限差分测试。排名与 RL 采样统一使用
   最后一步投影前分数，避免二值 mask 的梯度消失和饱和并列。
6. **RL 未接入**：新增预算约束的 Plackett-Luce 不放回采样、
   联合 log-probability、外部结果 reward、leave-one-out advantage、
   reference anchor 与策略梯度更新。
7. **checkpoint 契约不完整**：记录并核对几何、标签 revision、训练阶段和
   架构；记录任务规格。RL 必须从蒸馏 checkpoint 初始化。

## 损失

metric 阶段：`J + λ_metric L_metric`，平均全 token 与随机预算观察。

distill 阶段：
`L_task + λ_gfm L_gfm + λ_latent L_latent + λ_value L_value
+ λ_improve L_improve`。固定头的 `L_metric` 仍作为常数监控项报告；
`L_value` 是任务损失回归，不是 RL critic。

`L_improve = ReLU(J_final - stopgrad(J_initial) + margin)`：
只约束**起点到终点**，不保证每一步下降，默认 margin=0 允许持平。

RL 阶段：
`-mean(A × logπ) + β conditional_KL - α conditional_entropy`。
reward 使用所选候选的真实标签 regret、标签限制违反和端点未改善惩罚。
同场景独立采样 G 个动作，`A_g = r_g - mean(other rewards)`。
这是 **REINFORCE，不是 PPO/GRPO**；完整定义见方法文档。

## 快速开始

```bash
cd dcv_v74_decision_value_bottleneck

# 先从 raw BEV 重建 revision-2 标签；使用新的输出目录。
python process_planning_bev_v85.py \
  --raw-root /data/v85_bev/raw/train \
  --output-root /data/v85_bev/r2/train

# val/test 也必须重新处理，并按完整 log/scenario 划分。
export TRAIN_DATA=/data/v85_bev/r2/train
export VAL_DATA=/data/v85_bev/r2/val
export VLM_MODEL=/data/lyi/models/Qwen3-VL-8B-Instruct
export DEVICE=0
export LOAD_4BIT=1

STAGE=metric EPOCHS=10 bash run_v85.sh "$PWD" \
  --output checkpoints/v85_bev_metric.pt

STAGE=distill EPOCHS=20 RESUME=checkpoints/v85_bev_metric.pt \
  bash run_v85.sh "$PWD" --output checkpoints/v85_bev_distill.pt

EPOCHS=10 RESUME=checkpoints/v85_bev_distill.pt \
  bash run_v85_rl.sh "$PWD" --lr 5e-5 --output checkpoints/v85_bev_rl.pt
```

以上 epoch/学习率只是起始设置，不是已验证最优值。后两阶段须保持架构、
Qwen 权重、BEV 几何、预算与任务 JSON 一致。`--resume` 是权重初始化，
不恢复 optimizer/RNG 状态。默认阶段为 metric，不能默默从零开始 RL。

## 验证

```bash
python -m pytest -q test_v85_posttraining.py test_v85_bev.py \
  test_v85_gradient_flow.py test_v84_data.py test_v79_spatial_paths.py
bash -n run_v85.sh run_v85_rl.sh
```

测试不下载 Qwen 权重：覆盖坐标/碰撞标签、旧标签拒绝、投影 gradcheck、
真实剪枝 forward 对齐、采样预算/概率、奖励与梯度，以及小型组件的三个
训练循环。CPU 测试通过不等于真实 Qwen GPU 训练或 nuPlan 闭环验证通过。

## 合理性与尚未实现的能力

“监督规划头 → 梯度场蒸馏 warm-up → 硬剪枝 outcome RL”是一个一致的
研究框架：梯度知识提供初始化，RL 优化部署时离散区域选择的实际结果。
但当前 reward 仍是离线代理指标，不能声称完整自动驾驶安全：

- 碰撞基于中心线与当前障碍栅格，不包含车辆扫掠包络或未来 actors 时序；
- 交通灯虽渲染，但没有显式闯红灯标签；历史不是运动预测器；
- 舒适性主要是曲率代理，不是完整加速度/jerk；
- 约束罚项不保证轨迹严格安全可行；专家未来轨迹当前未进入主损失；
- 全部区域先经过 early ViT，节省的是后续视觉/LLM计算，不是只观察了
  15% 的环境；ND、Selector 和多个 RL rollout 的开销都应计入；
- 固定预算下 token 数惩罚是常数。应做预算扫描、held-out 消融与真实
  延迟/显存统计，再增加高质量时序指标和闭环 reward。

## 文档

- [三阶段训练、完整损失与 RL 定义](dcv_v74_decision_value_bottleneck/README_V85_POSTTRAINING.md)
- [nuPlan BEV 导出与数据契约](dcv_v74_decision_value_bottleneck/README_V85_BEV_INPUT.md)
- [共享梯度场与 ND](dcv_v74_decision_value_bottleneck/README_V85_GRADIENT_FLOW.md)
- [V8.4 串联基线](dcv_v74_decision_value_bottleneck/README_V84_COMPLETE_PIPELINE.md)
- [Qwen 环境说明](dcv_v74_decision_value_bottleneck/ENVIRONMENT_QWEN3VL_V81.md)

旧版本的模型、脚本与 README 保留用于复现，不应作为本版本训练协议。
