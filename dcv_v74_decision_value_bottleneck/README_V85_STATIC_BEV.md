# V8.5 静态、多任务 BEV 导出（revision 4）

本版用于验证：同一环境在不同任务需求下，视觉区域选择能否保持最终优化结果质量。
`export_nuplan_bev_v85.py` 默认 `--planning-mode static_multitask`。
这是一种利用 nuPlan 地图与当前对象构造的静态路径规划实验，不是 nuPlan 闭环驾驶评测。

## 图像与任务

输入只包含当前地图、当前对象、起点 S 和目标 G。没有紫色参考线、路线高亮、
绿色历史轨迹，也不读取或绘制未来 GT。灰色车道中心线是所有附近车道的地图几何，
不是选定的路径。静态模式不绘制信号灯：本任务没有交通灯时序和让行决策。

| 属性 | 默认值 |
|---|---|
| 图像 | 288×288 RGB；不将网格或预览坐标轴写入模型输入 |
| 物理范围 | x、y 均为 [-40,40] 米 |
| 坐标 | 当前自车 rear axle 为 S=(0,0)，x 向前、y 向左 |
| 图像方向 | 上方 +x、左侧 +y；起点在图像中央 |
| 视觉区域 | 9×9=81 个区域，每个 32×32 像素 |
| 目标 | 当前地图上可达、图内距边界至少 3 米的点；默认距 S 为 18–32 米 |
| 目标随机性 | 固定 seed + log/scenario/iteration 的 SHA256；可复现 |
| 对象 | 当前车辆、自行车、行人及静态障碍，全部冻结 |
| 规划代理 | 半径 1 米的圆形代理，不采用完整车辆运动学 |
| 危险区域 | 当前施工标志、路障、交通锥的外接框向各侧扩展 3 米 |

蓝色填充是 LANE 与 LANE_CONNECTOR 多边形并集，白色背景不可通行；
车辆蓝色、行人棕色、自行车粉色、施工对象黑色、危险区域桃色。
自车为橙色，S 和白色方向箭头标识起点；金色星形 G 是任务终点。
自车车身按真实中心绘制，坐标原点仍在 rear axle。

危险区域是**实验操作性标注**，不是 nuPlan 风险 GT。当前版本不随机制造危险标签。
没有施工对象的场景仍可导出，但对应任务的 `task_applicable=false`。
没有行人的场景同理；统计时要分别报告有相关对象和无相关对象的结果。
对象存在不保证不同任务的最优路径不同，任务冲突与有效性还需下游求解分析。

同一帧导出以下三个任务。RGB、S/G 和评价地图完全一致，变化的是任务文本与目标/约束定义。

| task_id | 优化定义 |
|---|---|
| `shortest` | 恒定名义速度 v=1 m/s，最小化 L/v；等价于最短路，不是动态最快到达 |
| `pedestrian_clearance` | 最小化沿路径的积分 ∫[1+4 exp(-d/4)]ds；d 是到最近行人轮廓的米制距离 |
| `avoid_hazards` | 在共同碰撞/道路约束上增加危险区域及代理半径缓冲的禁止进入约束，最小化 L |

数值目标在 `task_spec_json` 中给出，任务文本同时解释目标与图例。
任务说明包含目标坐标、代理半径和公开权重；不包含对象坐标、占据图或参考路径。
不存在行人时，距离图为 +inf，行人附加代价自然为 0。

## 目标与可行性

先用当前完整地图构造评价栅格。把所有对象轮廓与非道路区域视为障碍，通过距离变换
施加圆形代理半径间距；视野外也是障碍。4 邻接连通检查避免对角穿越障碍角点。
目标采样只使用从 S 可达的栅格；默认导出危险规避任务，因此还要求存在不进入危险区域
缓冲的连通路径。没有合适目标、起点被占据或道路断开时跳过，不绘制不存在的终点。
目标采样并不输出选定的路线，连通性图只属于评价信息。

`--goal-xy X Y` 可以指定米制目标。超出图内边距、过近、位于障碍或不可达时跳过场景，
不会偷偷替换目标。它不需要等于原驾驶日志的目的地。
静态道路并集按无向可通行区域处理，不施加单行规则或自行车模型；不能当作车辆驾驶安全验证。

## 在现有服务器上重新导出

先按完整 log 划分 train/val/test，三个任务版本与相邻帧必须处于同一 split。
下面是 mini 数据的预览导出，不是训练/测试划分：

```bash
conda activate nuplan
cd /home/lyi/neurodynamic/optimization/dcv_v74_decision_value_bottleneck
python export_nuplan_bev_v85.py \
  --planning-mode static_multitask \
  --data-root /data/planning_sources/nuplan/data/cache \
  --map-root /data/planning_sources/nuplan/maps \
  --db-files /data/planning_sources/nuplan/data/cache/mini/*.db \
  --output-root /data/v85_static_bev/mini_preview \
  --limit-scenarios 20 --frame-stride 20 \
  --goal-seed 7
```

不要传入原来的 `--history-seconds 2 --history-samples 5`，静态模式会拒绝它们。
`--future-horizon/--future-samples` 仅用于旧 route_tracking 模式，在静态模式中不参与导出。
旧 raw/processed 数据不能靠改名变成静态数据，必须重新导出并放在新的目录。

每个有效帧输出：

```text
mini_preview/<log>/<scenario>/<iteration>/
  shortest.npz
  pedestrian_clearance.npz
  avoid_hazards.npz
  bev_preview.png
```

`bev_preview.png` 可直接打开，带米制坐标、方向说明和图例。
NPZ 内 `bev_rgb` 是不带外围注释的真正模型输入。
`--no-preview` 可以关闭预览；`--tasks shortest pedestrian_clearance` 可以选择导出的任务。

## 数据契约与边界

| 字段 | 用途 |
|---|---|
| `input_mode` | `ego_centric_static_bev_v85` |
| `raw_schema_version / schema_revision` | 85 / 4，与旧时序 revision3 分开 |
| `bev_rgb` | 视觉输入 |
| `task_text / task_spec_json` | 任务输入与可计算的公开目标/约束 |
| `start_xy / goal_xy / map_bounds / region_world_bounds` | 固定几何及任务条件 |
| `scene_id / sample_id / log_name / scenario_token` | 分组、划分与追踪来源 |
| `bev_semantic` | 16 层：保留旧 15 层编号并追加 hazard；route/reference/history/信号层恒为零 |
| `region_semantic_counts` | [81,16]，仅评价/审计标签 |
| `traversable / static_obstacle / planning_free` | 道路、当前冻结对象及代理半径膨胀后的可行域；仅评价 |
| `hazard_mask / hazard_exclusion` | 施工缓冲区及代理间距禁止区域；仅评价 |
| `pedestrian_mask / pedestrian_distance_m` | 行人轮廓和距离代价；仅评价 |
| `reachable / hazard_free_reachable` | 目标生成诊断；仅评价 |
| `trajectory_gt_available` | false；不写入任何伪造的专家轨迹 |

没有 `reference_path_ego`、`expert_trajectory_ego` 或未来时间网格。
评价目标为到达成功、约束违反、任务成本以及计算效率，不把与原驾驶轨迹的距离当作主奖励。
模型输入要采用明确 allowlist；完整标签用于评价器和全信息数值参考，不能直接给被评估的建模器。
保留 token 方案与全图方案必须共用同一冻结 VLM、问题族、求解器和评价地图。

**本次更新完成的是静态导出与数据契约。旧 `process_planning_bev_v85.py`、
`run_v85.sh`、`run_v85_rl.sh`、旧评估脚本仍是参考路线/候选/时序 ADE-FDE 流程，
不支持这些静态记录。** 旧处理器会给出明确错误，避免训练目标被静默替换。
下游接入必须按 `task_spec_json` 实现静态问题、完整路径段的碰撞检查和任务成本评价，
再接入选择器后训练；不能复用旧 GT 奖励宣称完成新实验。

旧实验复现时明确使用 `--planning-mode route_tracking`，见
[旧 BEV 与 GT 文档](README_V85_BEV_INPUT.md)。

## 验证

```bash
python -m unittest test_v85_static_bev test_v85_bev -v
```

覆盖无路线/历史/未来 API 调用、三任务共享图像、图内可达目标、不可达目标跳过、
障碍间距、禁止对角穿障、静态/旧模式隔离及旧导出回归。
验证使用 nuPlan API 形状的模拟对象；未在本地运行真实 nuPlan mini 数据导出。
