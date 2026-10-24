# V8.5：区域建模与陆地最小成本规划

当前分支：`v85-tilebased-llm-solver`。
研究目标仍为：学习固定预算视觉 token 选择器，冻结 Qwen，由 Qwen 根据任务及保留视觉信息生成优化实例，再调用求解器获得真实任务反馈。
不增加探索、记忆或步数约束；不要求预测路径逐格等于参考解。

## 本轮改变

2026-10-24：默认输出升级为多个不规则多边形（含凹多边形），通行与禁行区域均支持多个；保留旧矩形读取。解析、求解和可视化已同步更新。CPU 回归测试通过；尚未运行真实 Qwen 8B 多边形生成实验，不能据此声称 SR 或成本提升。

旧 `segments` 接口把搜索范围限制在线段上。如果模型只提出一条链，求解器无法向链外绕行。
新 `regions` 接口让 Qwen 描述**有面积的相关通道、成本变化区及水域边界**，程序在区域并集内建立邻接图。
求解器可以在区域内部转弯，也可以选择不同连接通道。它仍不能找出未被建模覆盖的路线；区域质量需要实际验证。

数据协议改为 `pkmn_land_no_water_no_corner_cut_v1`：

- 同时排除 S 或 G 位于 lake/ocean 的任务，不移动原始端点。
- lake=6、ocean=10 的标签用于离线建立禁止通行掩码；其他地形（包括 wall=25）仍按公开有限成本规则处理。
- 八邻接、斜步长度为 1；斜向移动要求两个正交侧邻格均可通行，禁止穿水域拐角。
- 在原始 train/val/test 划分内，按上述规则排除陆地不连通任务，不按 VLM 成败或路径差异筛选。
- 重新计算最小地形成本参考路径、最优成本，以及独立的最短步数参考。保留原始 RGB、S/G 和任务 ID。
- 原始 GT 存在 `original_gt_path_rc/original_gt_path_mask/original_gt_path_length/original_optimal_cost`；新评价使用新参考解。

这是一个**派生陆地导航协议**，不能把结果当作原始完整基准指标，也不能与旧结果直接做配对提升比较。
`land_report.json` 记录每个划分的保留数与排除原因；排除按 water_start → water_goal → land_disconnected 顺序归类，每个任务只计一次。

## 1. 更新与生成新数据

在 `optimization/dcv_v74_decision_value_bottleneck` 中执行：

```bash
git pull --ff-only origin v85-tilebased-llm-solver
python prepare_tilebased_land_v85.py \
  --data /data/v85_tilebased/pkmn \
  --output /data/v85_tilebased/pkmn_land_v3
```

**输入必须是原始预处理目录 `pkmn`，不要使用 `pkmn_no_water_goal`。**
不用重新下载；CPU + NumPy 即可。输出须新建或为空，不能覆写旧数据。
新数据不再沿用旧的 GT 最优值，也不沿用 Chebyshev 距离计算 SPL。
筛选规则、通行图与 protocol 都进入 checkpoint 的数据签名，旧数据训练的 checkpoint 会被拒绝；请在新数据上重新训练。

## 2. 先运行 Full 并看路线

```bash
python run_tilebased_v85.py \
  --mode evaluate --split val --modes full --limit 16 \
  --data /data/v85_tilebased/pkmn_land_v3 \
  --model-representation regions --modeling-prompt navigation \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct \
  --device cuda:0 --load-4bit --batch 1 \
  --modeling-max-new-tokens 4096 --seed 1234 \
  --output results/tilebased_land_full_r4

python visualize_tilebased_v85.py \
  --data /data/v85_tilebased/pkmn_land_v3 \
  --outcomes results/tilebased_land_full_r4/outcomes.jsonl \
  --split val --mode full \
  --output results/tilebased_land_full_r4/path_visualizations
```

打开输出 `index.html` 或 PNG。四个面板分别是原图、实际求解路径、新陆地参考路径、预测区域及实际路径。
第四面板的橙色边框是模型区域，蓝色遮罩是**模型预测的**水域，未使用真实水域替模型纠错。
失败路线照样显示，标题标明 `constraint_failure`，不会事后补一条成功路线。
Qwen 仍可能误判区域或水域，图中出现穿水路径就会计失败，不能因已生成路线而算 SR 成功。

## 3. 新模型输出与求解

Qwen 输出五个字段，不生成整幅地图的成本矩阵，也不输出最终路径或节点 ID：

```json
{
  "schema": "tilebased_land_polygons_v4",
  "source_rc": [2, 0],
  "target_rc": [2, 4],
  "regions": [
    {"polygon_rc": [[0,0],[0,5],[2,5],[2,2],[5,2],[5,0]], "cost": 1.5},
    {"polygon_rc": [[2,2],[2,5],[5,5],[5,2]], "cost": 8.0}
  ],
  "blocked_regions": [
    {"polygon_rc": [[1,2],[1,3],[2,3],[2,4],[3,4],[3,2]]},
    {"polygon_rc": [[4,0],[4,1],[5,1],[5,0]]}
  ]
}
```

以上只是格式示例，不是实际 Qwen 输出或数据集 GT。现在默认使用多个简单多边形，支持凹多边形、不连通区域和多个障碍；内部孔洞通过禁行多边形扣除。不同成本区域分别描述，不要求覆盖全图。

`polygon_rc` 按顺时针或逆时针列出边界顶点，不重复首点、不交叉、不折返。每个列表最多 64 个区域，每个多边形 3–64 个顶点。
**多边形坐标是网格角点**：20×20 地图允许 0–20；S/G 仍是 0–19 的格子索引。格子 `[r,c]` 的中心为 `[r+0.5,c+0.5]`；中心在多边形内部或边界上才计入区域。禁行区域优先。因此这是明确的离散化规则，不是像素级障碍保证。

旧 `tilebased_land_regions_v3` / `bounds_rc` 矩形输出与历史可视化仍可读取，但新提示词只请求 `tilebased_land_polygons_v4`。数据与参考解未改变：已生成的 `pkmn_land_v3` 可继续使用，无须重新预处理。CLI 仍为 `--model-representation regions`；请使用新的结果目录重新评价，旧输出不会自动变成多边形。训练比较应在相同新提示词下重新运行所有方法。

- 区域并集内的格子可参与搜索；区域外为未建模空间，不自动补全。
- 每个格子使用所属区域的预测成本；多个区域重叠取最大成本，与列表顺序无关。
- `blocked_regions` 覆盖所有可通行区域；模型漏掉水域时，不会被真实标签提前修复。
- 只连接区域并集内部满足禁止斜穿规则的相邻格子，边代价为进入格子的预测成本，起点成本计一次。
- S/G 未覆盖或被预测为阻塞时求解失败，不自动插入端点或直连。
- Dijkstra 在模型图上求解；**正式建模和求解不接收真实成本、真实水域掩码或 GT**。
- 完整 JSON 出现时停止解码；最终严格检查结构、坐标、成本及重复区域覆盖、自交和退化多边形。长度上限仍为所有方法共用的硬上限，不保证每次模型生成都正确。

程序内部会用小型数组建立区域邻接图，这是编译模型的实现，不是要求 VLM 逐格输出，也不是偷偷读取真值。

## 4. 评价与训练反馈

实际路径生成后，才检查真实通行标签。穿过水域或斜切水角记 `constraint_failure`，失败损失为 10（可配置）；
原始路径保存到 `path_rc` 供可视化。模型格式失败与不连通仍分别记录为 `model_failure/solver_failure`。

| 指标 | 定义 |
|---|---|
| `sr` / `success_rate` | 合法到达 G 且全程满足陆地通行规则的比例 |
| `spl` | 所有样本上的成功加权最短步数效率；最短步数在真实陆地图离线求得 |
| `success_weighted_cost_efficiency` | 所有样本上平均的 `S * C*/max(C,C*)`，其中 C 使用真实地形成本 |
| `constraint_failure_rate` | 已生成路径但违反真实水域/拐角规则的比例 |
| `true_cost_successful` | 合法成功路径的平均真实成本，只作为成功条件下的补充量 |

同成本不同形状的路线同等评价。失败在 SPL 和成本效率中贡献 0；成功 loss 仍为裁剪后的真实成本 regret。
仅有 SR 高不表示成本最优，SPL 高也不等价于地形成本低。

辅助诊断在实际求解**之后**进行：`region_oracle_cost` 在已生成的图内排除真实违规连接并使用真实成本重算，
只用于区分 `region_restriction_regret` 与 `region_cost_model_regret`，不会替换模型输出的路径。
前者高表示已建模区域/连接仍遗漏更好的路线；后者高表示该范围内还有更好的路线但成本估计影响了选择。

## 5. 学习选择器

```bash
python run_tilebased_v85.py \
  --mode experiment \
  --data /data/v85_tilebased/pkmn_land_v3 \
  --model-representation regions --modeling-prompt navigation \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct \
  --device cuda:0 --load-4bit --batch 1 \
  --limit 128 --rl-limit 32 --val-limit 16 \
  --metric-epochs 3 --distill-epochs 3 --rl-epochs 2 \
  --rl-group-size 2 --budget 0.15 --seed 1234 \
  --modeling-max-new-tokens 4096 \
  --output results/tilebased_land_selector_r4
```

仍执行代理预热 → 梯度蒸馏/FM/ND → 真实求解器反馈 RL。Qwen 冻结。
代理候选路径也遵守同一陆地规则，地形成本回归仅在可通行格计算；代理监督仍只用于训练，不供正式求解器使用。
该代理不等于 VLM 区域建模误差的精确梯度，障碍误判由真实 RL 结果反馈；教师监督范围的局限依然存在。
Full、未训练/已训练选择器及可选 Random 使用同一环境提示词、区域模型、求解器和成功规则。
主结果仍是同任务固定预算下的 SR、成本效率和计算开销；不能仅凭 token 比例宣称端到端加速。

## 兼容与验证边界

`--model-representation auto` 为新的 CLI 默认值：陆地数据选 `regions`，旧数据选 `segments`。
显式 `regions` 与旧数据混用，或陆地数据与旧模型表示混用，会提前报错。
旧 `segments/corridor/matrix` 仍能用旧数据复现。可选 VQA 根据数据协议自动选择陆地提示词；旧逐格/矩阵诊断工具仅支持旧数据。

本轮 73 项 CPU 回归测试覆盖区域内绕行、不同成本通道选择、禁止斜穿、端点覆盖、非法结构、GT 不进入建模/求解、
真实约束失败计分、参考解重算与旧 GT 归档、陆地 SPL、失败路线可视化，以及原协议/陆地协议的一键三阶段训练。
训练和生成测试使用小型冻结模型替身，图求解使用真实 Dijkstra。
**尚未运行预训练 Qwen3-VL-8B；没有真实准确率、训练提升或加速结论。**
