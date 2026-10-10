# V8.5 静态车辆多任务 BEV（revision 6）

分支：`v85-bev-gt-boundary`。本版扩大视野，以日志 GT 在视野内边缘的首次交点为终点，保存对应 GT 前缀。
研究流程仍是 **同一 BEV/起终点 + 不同任务文本 → 区域选择 → 冻结 VLM 建模 → 求解器**。
本次实现数据导出、数值参考求解和独立几何评价，不训练 VLM；旧 route/GT 的 RL 入口尚未接入此契约。

## 1. 图像与坐标

| 项目 | 默认配置 |
|---|---|
| 图像 | 原生 1440×1440 RGB；无文字、坐标轴、边框、图例、历史或参考路径 |
| 范围 | x∈[-24,72] m，y∈[-48,48] m；统一 96×96 m，不按单图拉伸 |
| 坐标 | 当前自车后轴中心为 (0,0,0)，x 前向、y 左向；图上方是 +x |
| 起点 | 橙色真实车身 + 白色朝向箭头；位于图像下方约 3/4 处，前方留 72 m |
| 终点 | 金色星形；目标位置和 GT 朝向另存在任务文本/数值字段中 |
| 道路 | 淡蓝底色、细灰边界与方向箭头；显示所有附近车道，不显示选定路线 |
| 对象 | 蓝色车辆、棕色行人、粉色自行车、黑色障碍，全部冻结在当前帧 |
| 危险区 | 桃色；施工标志、路障、交通锥轮廓外扩 3 m 的操作性定义，不是原始风险 GT |
| 区域 | 9×9=81，每区域 160×160 像素；区域数不等于模型实际视觉 token 数 |

相较 revision5（720 像素、48 m），视野边长和像素边长均扩大为 2 倍，保持约 15 像素/米。
这是固定局部视野，不自动扩大到整段日志或城市任务的最终目的地。原始像素总量为旧版 4 倍；
模型实际视觉 token 和显存取决于处理器是否缩放，不能假定代价不变。
`bev_preview.png` 与 NPZ 的 `bev_rgb` 完全一致，不是带注释的另一张图。
背景空白仍表示非道路，不会为了填满图片而拉伸道路或制造对象。
如需其他视野，必须同时设置 x/y 等跨度范围，并在所有 baseline 上使用同一协议。
模型处理器可能再次缩放图片；接入训练时应记录实际像素和 token 数，不能只按 PNG 大小声称效率收益。

BEV 来自地图及当前对象标注，不读取相机；查看原始前视序列需要另下载对应 sensor blobs。
全球坐标在减去自车原点前保留 float64，避免 UTM 大坐标的 float32 舍入破坏车道细节。

## 2. 静态车辆约束与有向规划图

`static_road_planner_v85.py` 独立于 nuPlan，输入为车道、多边形、对象和真实车身尺寸。

- 按车道基线正向每 4 m 采样，横向偏移默认 {-0.5,0,0.5} m；同车道允许向前跨 1–2 个站点。
- 跨车道连接使用地图的 `outgoing_edges`，不因两个车道距离近就自动连接。
- 默认允许地图明确标注、相互确认的相邻车道间换道。候选换道向前延伸 8–16 m，
  朝向相近，且必须通过相同车身/曲率检查。`--lane-change-policy outgoing_only` 可关闭侧向换道。
- 每条运动边是带起终朝向的三次 Hermite 曲线；仅向前，不允许横向平移或倒车。
- 默认最小转弯半径 5.5 m；在曲线上检查解析曲率，并对输出采样轨迹再次检查。
- 自车采用 nuPlan 真实长、宽、后轴到车身中心的偏移。安全边距默认 0.15 m。
- 检查整个车身及相邻姿态间的扫掠包络，不能只检查后轴点或两个端点。
  默认检查间隔不大于 0.25 m，旋转包络附加弓高缓冲；道路外、对象内和视野外均不可行。
- 后轴必须始终位于方向兼容、拓扑连续的车道走廊内。车道孔洞保留为非道路。

这是**低速、静态、几何车辆规划**。地图邻接不等价于完整的交通法规许可：
本版不解析实线禁跨、让行、信号时序、动态车辆交互、速度/加速度及转向速率。
曲率与扫掠检查是记录了分辨率的数值验证，不宣称连续空间的形式化安全保证。
不能用本版结果替代 nuPlan 官方闭环驾驶成绩。

## 3. 从原始 GT 截取边缘目标

默认 `--goal-source logged_gt_boundary`：

1. 从当前后轴姿态读取未来 ego GT，默认请求 20 秒、约 0.1 秒采样间隔；使用实际返回时间戳。
2. 转成当前自车坐标系，补入真实的 t=0 起点，保留停车样本。
3. 在 BEV 外边界内留出车身空间：内缩量取 `--goal-margin-m` 与含安全边距的车身外接半径加 0.05 m 的较大者。
4. 沿 GT 按时间查找首次触及该内边界的位置；若交点位于相邻采样间，线性插值位置、时间，并按最短角差插值朝向。
5. 将交点作为精确终点，保存从 t=0 到交点的 GT 前缀。不会吸附到车道中心，也不会改变 GT 终点朝向。

因此金色目标位于地图边缘附近，完整车身仍在视野内；后续出界再入界不会延长截断轨迹。
模型输入只画目标，不画 GT。`audit/logged_gt.png` 另画绿色未来 GT，供人工核查。
同一帧各任务共享 BEV、目标、车辆和 logged GT；GT 不因任务变化而重新生成。

可用 GT 没到边缘时，记录 `logged_gt_does_not_reach_boundary` 并跳过，不造轨迹、不随机换目标。
读取失败也汇总原因。可提高 `--gt-horizon-s`（如 40），但日志末尾不能外推。
这一采样协议会偏向能在指定时段离开视野的场景，需报告跳过数量，不能声称覆盖全部停车场景。
GT 真实转弯时，目标可能在侧边甚至后方；本模式不使用旧版 18–28 m/前向 8 m 的目标筛选。
扩大视野增加可见上下文，但不保证各任务的最优路径一定不同。

保留旧的 `--goal-source road_graph`：从 18–28 m 内前向可达的下游车道站点确定性选择目标，
不调用未来 GT；`--goal-xy X Y` 自动启用该模式，仍拒绝不满足原有范围/可达性要求的显式目标。
若完整复现旧视野，再指定：

```bash
--goal-source road_graph --image-size 720 --x-min -12 --x-max 36 --y-min -24 --y-max 24
```

## 4. 任务与数值参考解

| task_id | 目标/额外约束 |
|---|---|
| `shortest` | 最小化 L/v，默认 v=1 m/s；恒速下等价于最短路，不是动态时间最优 |
| `pedestrian_clearance` | 最小化 ∫[1+4 exp(-d/4)]ds；d 是**加安全边距的车身**到最近行人轮廓的距离 |
| `avoid_hazards` | 整个加边距车身不进入危险区；在可行路径中最小化 L |

三者共享车辆、道路和对象碰撞约束。无行人时附加代价为 0；无相关危险区时危险约束不改变问题。
`task_applicable` 标记相关对象是否存在，不保证它们影响最优解。

在同一有向图上分别运行 Dijkstra，输出每个任务的参考轨迹、边序列、目标成本和几何诊断。
随后独立重查拼接路径的朝向、道路拓扑、车身扫掠、曲率和数值成本；验证不一致会报错。

- `optimal_on_graph`：对保存的**有限图及采样边成本**最优；不是连续自由空间全局最优，也不是人类 GT。
- `infeasible_on_graph`：该任务在此图上不可达；不能据此断言连续空间中一定无解。
- 任何任务在图上无解时保存空参考轨迹、成本 +inf 和无解状态，**不更换 GT 目标、不给零成本伪答案**。
- `--skip-numeric-reference` 仅适用于 GT 模式：跳过建图，保存 `not_computed` 状态及空参考，不表示无解。
- 同时保存各任务参考路径在其他任务下的成本矩阵，用于区分任务冲突和无冲突场景。
  行是参考路径，列是评价任务；不同列单位/目标不同，只在同一列内比较。
- 参考路径只画在 `audit/reference_*.png`，绝不覆盖模型输入。

GT 模式保存 `trajectory_gt_available=true` 和 `logged_gt_is_task_optimum=false`。
日志 GT 是真实车辆在动态环境下的行为记录；冻结当前车辆后，它可能撞上原本会开走的对象，
也不保证满足新增危险约束或达到每个任务的最优成本。逐任务保存 GT 的静态可行性和成本，
即使不可行也保留 GT 和诊断；不可行的 logged GT 不能作为成功规划标签。
`road_graph` 模式没有 logged GT，仍保存 `trajectory_gt_available=false`。
优先报告成功率、碰撞/越界/危险约束违反、同任务成本和计算效率。
对连续求解器仅报告 `cost_gap_to_graph_reference`，它可能为负，不能强称非负最优 regret。
先按参考可行性和任务适用性分层，再统计任务结果；不要将缺失参考记为成功或零误差。

## 5. 在你的服务器上重新导出

本版需要 Shapely，nuPlan 环境通常已有；缺少时安装：

```bash
conda activate nuplan
python -m pip install 'shapely>=1.8,<3'
cd /home/lyi/neurodynamic/optimization/dcv_v74_decision_value_bottleneck
python export_nuplan_bev_v85.py \
  --planning-mode static_multitask \
  --data-root /data/planning_sources/nuplan/data/cache \
  --map-root /data/planning_sources/nuplan/maps \
  --db-files /data/planning_sources/nuplan/data/cache/mini/*.db \
  --output-root /data/v85_static_bev/mini_gt_boundary_r6 \
  --goal-source logged_gt_boundary \
  --gt-horizon-s 20 --gt-sample-interval-s 0.1 \
  --skip-numeric-reference \
  --limit-scenarios 20 --frame-stride 20
```

上述命令先导出扩大后的图片与 GT，便于检查；如需各任务数值参考，删除 `--skip-numeric-reference`，
并使用另一新目录重新导出。扩大后的道路图建图和碰撞检查会更耗时。

使用**新的输出目录**；若帧目录已存在，导出器拒绝覆盖，避免混入旧数据或残缺结果。
静态导出不需要 actor 过去 2 秒轨迹或相机文件，不读取 route、交通灯。
GT 模式只为确定目标和保存对照读取未来 ego trajectory；模型可见目标，不可见中间 GT 路径。
`--no-preview` 关闭所有 PNG；不改变是否计算数值参考。`--tasks shortest pedestrian_clearance` 可指定任务。
`--agent-radius-m` 已移除，改用真实车身 + `--safety-margin-m`。
改变图分辨率不会改变几何参考规划；改变站点间隔、偏移或转弯半径则定义了不同的参考图，须记录并统一。
按完整 log 划分 train/val/test，不能让同一帧的任务变体或相邻帧跨 split。

每个有效帧输出：

```text
mini_gt_boundary_r6/<log>/<scenario>/<iteration>/
  bev_preview.png                  # 唯一模型输入图片，无文字/路径
  shortest.npz
  pedestrian_clearance.npz
  avoid_hazards.npz
  audit/
    scene_geometry.json            # 完整环境，评价器专用
    logged_gt.png                  # 绿色未来 GT，仅供核查；输入图不包含这条线
    logged_gt_summary.json         # 截断参数、可用时长和静态几何诊断
    planning_graph.npz             # 未跳过数值参考时才生成，评价器专用
    reference_summary.json         # 状态、成本和跨任务成本矩阵
    reference_shortest.png         # 紫色数值参考；跳过或无解时不生成
    reference_pedestrian_clearance.png
    reference_avoid_hazards.png
```

输出根目录还有 `export_summary.json`，记录参数及跳过原因。

## 6. 查看参考解或评价自己的求解结果

设置下面 `RECORD` 为实际导出的一条 NPZ 的完整路径：

```bash
RECORD=/data/v85_static_bev/mini_gt_boundary_r6/<log>/<scenario>/<iteration>/shortest.npz
python evaluate_static_bev_v85.py --record "$RECORD" --logged-gt
# 计算过数值参考的记录，可复核其参考路径
python evaluate_static_bev_v85.py --record "$RECORD"
# 自己的输出：float array [N,3]，每行为后轴 x/y/heading（米/米/弧度）
python evaluate_static_bev_v85.py --record "$RECORD" --trajectory /data/predicted_path.npy
```

既不传 `--trajectory` 也不传 `--logged-gt` 时，复核保存的数值参考；无解或未计算参考时不能报告成功。
评价器检查起点、目标位置/朝向、车身碰撞、道路/方向/拓扑、曲率、危险区，并报告任务成本。
终点位置容差 0.75 m、朝向容差 0.15 rad；参考图自身到达精确目标节点。
数值参考和外部待评路径是无时间参数的几何路径，不接受横移、倒车或重复零长度点。
`--logged-gt` 仅在静态检查前去掉完全相同的连续停车姿态，保存的 GT 和时间戳不会被去重。
本版评价器不计算动态舒适性或 ADE/FDE；以后计算时间对齐 ADE/FDE 应使用 `logged_gt_time_s`，
不得按点数或弧长把它当作旧 8 秒/16 点标签，亦不得把 GT 对自身的零误差作为模型结果。

## 7. 输入边界与训练兼容性

数据为 `raw_schema_version=85, schema_revision=6, input_mode=ego_centric_static_bev_v85`。
`evaluate_static_bev_v85.model_inputs(record)` 使用显式白名单返回：
RGB、任务文本与公开目标/约束、坐标范围、区域边界、起终姿态和车身尺寸。

以下均是评价/教师专用，**不能直接传给被评估的建模器**：
`bev_semantic`、区域语义计数、占据/危险/行人掩码、距离图、完整车道拓扑、图边、
数值参考轨迹、参考成本、目标车道编号、参考解状态、完整 logged GT 及其时间戳/可行性。所有 baseline 共用同一输入协议和求解协议。

两类路径分开保存：

| 字段 | 含义 |
|---|---|
| `logged_gt_trajectory_ego` | [N,3] 原始未来 GT 的截断前缀，包含 t=0 和边界终点；float64 |
| `logged_gt_time_s` | [N] 相对当前帧的真实秒数，最后交点可能为插值时间 |
| `logged_gt_source` / `logged_gt_end_interpolated` | 来源与末点是否插值 |
| `logged_gt_boundary_inset_m` | 为完整车身留出的边界距离 |
| `logged_gt_static_task_feasible` / `logged_gt_static_task_cost` | 在当前任务静态约束下的可行性/成本，不可行成本为 +inf |
| `logged_gt_is_task_optimum` | 固定 false；不把日志行为当作所有任务最优解 |
| `reference_trajectory_ego` / `reference_status` / `reference_cost` | 独立求出的有限图数值参考及状态/成本 |

模型输入白名单支持 revision5/6，但生成上述 GT 必须从原始日志重新导出。
两类路径都是变长 [N,3]，不能混用旧 16 点时序协议。
旧 `planning_free`/圆形代理连通掩码已移除，它们不能表达车身朝向相关的可行域。
`pedestrian_distance_m` 仅是栅格点到行人的诊断距离，真正成本使用车身距离。

旧 `process_planning_bev_v85.py`、`run_v85.sh`、`run_v85_rl.sh` 仍采用 route/GT 时序协议，
会明确拒绝静态记录。**本次没有将新静态任务接入 Selector 的 RL 训练入口**。
后续接入应使用这里的白名单与任务成本评价，不能直接复用旧 ADE/FDE 奖励。
旧实验继续显式传 `--planning-mode route_tracking`，参见 [旧文档](README_V85_BEV_INPUT.md)。

## 8. 验证范围

```bash
python -m unittest test_v85_logged_gt test_v85_static_bev test_v85_bev -v
```

24 项测试通过，覆盖 GT 首次边界交点、侧向出界、朝向跨 ±π、停车和真实时间戳保留、
短 GT 跳过、精确目标不吸附、超出旧目标范围的完整导出、有/无数值参考与防止 GT 输入泄露；
亦覆盖旧道路目标模式无 future 调用、无 route/history 调用、无文字渲染、三任务共享输入、前向目标、无效目标不替换、
大坐标精度、车身边角与段间碰撞、逆行/横移/曲率拒绝、明确相邻换道、断开车道拒绝、
Dijkstra 直线成本、行人任务改变参考解、危险任务无解状态、完整导出后重算成本和旧模式回归。
验证采用模拟 nuPlan API 和受控几何；本地未安装 nuPlan/下载真实 mini 数据，
未运行真实 mini 导出、Qwen 权重或 RL 训练。真实场景数量与渲染效果需用上述服务器命令确认。

详细验证记录见 [VALIDATION_V85_STATIC_BEV.md](VALIDATION_V85_STATIC_BEV.md)。
