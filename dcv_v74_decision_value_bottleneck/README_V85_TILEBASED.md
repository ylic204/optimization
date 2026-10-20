# V8.5 Tilebased：冻结 VLM 建模与最小地形成本路径

本分支 `v85-tilebased-llm-solver` **基于用户指定的 `v85-bev-selector-llm-solver`**
（基线提交 `fbe42f1`），接入 Tilebased navigation datasets。
主任务只有“从给定 S 到 G，最小化累计地形成本”，**不增加步数/能源约束，不做探索、记忆或车辆控制**。
视觉 token 预算属于选择器，不属于路径规划问题。

## 当前默认：围绕 S→G 建立通路图，不重建完整地图

`run_tilebased_v85.py` 默认 `--model-representation corridor --modeling-prompt navigation`。
Full、Random、initial、learned 共用同一英文环境 prompt、建模 schema、解码上限、几何转换和求解器。
本轮直接验证最终路径质量，不需要再先运行 VQA 或重新预处理数据。

```bash
python run_tilebased_v85.py \
  --mode evaluate --model-representation corridor --modeling-prompt navigation \
  --data /data/v85_tilebased/pkmn \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct \
  --device cuda:0 --load-4bit \
  --split val --limit 40 --batch 1 \
  --modes full random --budget 0.15 --seed 1234 \
  --modeling-max-new-tokens 4096 \
  --output results/tilebased_corridor_r1
```

保持此前量化设置；若此前未量化则去掉 `--load-4bit`。40 个任务、两个方案共 80 次建模。
这是新表示的首轮实验命令，不是已经测得的性能提升。

**VLM 输出什么？** `tilebased_sg_corridor_v1` 包含六个字段：

| 字段 | 含义 |
|---|---|
| `schema` | `tilebased_sg_corridor_v1` |
| `source_rc`, `target_rc` | 公开 S/G，零起始行列坐标 |
| `source_cost` | 对起点成本的估计，只计一次 |
| `nodes_rc` | 少量路标点，0 为 S，1 为 G，其余可放在转折、连接点或成本变化处 |
| `edges` | `[起点编号, 终点编号, 平均到达成本]` 的有向连接列表 |

图中可以包含多条备选通路及共享连接，求解器决定最终路径；不要求 VLM 输出完整路线或 400 个成本格子。
最多 32 个路标和 64 条连接是输出 schema 的大小限制，不是导航任务的新资源约束。
可只输出一个有证据支持的通路，不强制凑出多条；遗漏有价值备选路线会体现在最终 regret 中。

**如何得到实际路径？** 两个路标间的连接是直线段，沿较长轴用最近整数插值展开成八邻接步；
正反方向使用同一规范端点次序，半整数向上取整。转弯和成本变化用额外路标表达。
每一步的代价是该段预测的平均到达成本，排除出发格、包括到达格；起点单独计入，斜步不乘距离系数。
段平均成本可为公开最小/最大地形成本之间的连续数值，不要求它等于单一地形类别。
同一采样格处的交叉连接共享节点；重叠的同向步对成本估计取平均。不在这些连接上的邻接边不会自动添加。
稀疏 Dijkstra 在展开后的通路图上求解。这个过程中**不读取 GT 成本、不调用 GT 绕路，也不从完整地图补全连接**。
生成图不连通时记 `solver_failure`，不会用 GT 或直线兜底。

这仍保留了栅格坐标作为统一空间坐标及评价工具，但没有把全图逐格识别作为 VLM 的任务。
当前数据集所有地形均为有限成本，未建模连接表示搜索范围没有覆盖，不表示那里物理不可通行。
新表示限制了可搜索的路径集合，不能宣称与完整栅格优化等价；需要靠结果评估这一近似的影响。

**评价与训练反馈：** 生成路线后才读取完整 GT 成本，计算真实路径成本、最优率、相对 regret 和原有失败惩罚。
主要结果仍是最终任务质量与视觉预算；不再要求完整矩阵准确率。日志另记录：

- `compiled_graph`：实际送入求解器的空间节点、单位连接和预测成本。
- `corridor_restriction_regret`：在生成通路图内使用真实成本所得最优值，相对完整 GT 最优值的差距。
- `corridor_cost_model_regret`：实际所选路径相对上述通路图内最优值的差距。
- `path_cost_relative_error`：预测路径成本与真实路径成本的相对误差。
- `waypoint_count/segment_count/modeled_cell_count/modeled_arc_count`：建模规模；与视觉 token 数分开统计。

前两项均除以完整 GT 最优成本，理论上相加等于该有效路径的未裁剪 regret。
通路图内的 GT 求解**仅在评价阶段诊断，不替换实际预测路径，也不进入 prompt**。
成功率是建模并求解成功率，不能当作避障安全率。

RL 已使用此接口：选择区域 → 冻结 Qwen 生成通路图 → 稀疏求解 → 真实路径成本反馈。
当前 `metric/distill` 仍保留训练专用的地形成本代理头及候选路径代理损失，用于获得 mask 梯度；
这个代理头内部使用完整训练标签，**不向正式 VLM 提供矩阵、不决定 RL/evaluate 的路径**。
它不是通路图求解器的精确可微替代。旧 metric/distill checkpoint 可以初始化，新 RL 应按通路图环境训练。
旧 RL checkpoint 也可做迁移评估，但日志会标记原表示；它不等于已完成通路图奖励下训练。

复现旧的矩阵实验需显式添加 `--model-representation matrix`，包括下面保留的新旧提示词对照命令。
`--compare-prompts` 始终在同一种表示内比较提示词，不能用于比较矩阵与通路图。
`diagnose_tilebased_v85.py --probes matrix` 仍是旧矩阵接口诊断，不是当前默认规划任务。

## 先检查导航 VQA：起终点、连接关系和路线代价

`vqa_tilebased_v85.py` 使用英文 system prompt 定义俯视二维像素地图环境，
明确没有天空、对象贴图的立体细节不改变俯视含义，结合纹理、形状和邻接区域理解地图。
提示词在 `prompts/tilebased_vqa_system_v85.txt`，可用 `--system-prompt-file` 替换。

VQA 现在读取 `tasks.jsonl` 中的公开 S/G 查询，询问五类导航问题：
起终点附近布局、可能的低成本连接区域、可能促使绕行的高成本边界、值得重点检查的视觉区域、
会改变路线选择的剩余不确定性。不再要求逐项盘点水域、建筑、树木，也不要求精确命名与决策无关的对象。
**若类别歧义会改变公开图例中的成本，仍须保留该不确定性。**

当前基准中所有格子均为有限正成本，连 wall 也可通行；因此只判断“能否到达”不能验证视觉理解。
VQA 强调低成本通道与代价差异，不能凭常识把树木、水域或建筑设为禁止通行。
不增加障碍规则、步数约束或记忆，也不修改原图、GT、数据划分和正式任务损失。

```bash
python vqa_tilebased_v85.py \
  --data /data/v85_tilebased/pkmn \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct \
  --device cuda:0 --load-4bit \
  --split val --map-indices 0 3 6 9 --task-index 0 \
  --output results/tilebased_navigation_vqa_r4
```

与之前实验保持相同精度；原来不用量化时去掉 `--load-4bit`。
`--task-index` 是每张图在任务清单中的查询序号，从 0 开始，默认选择第一个；可改为 1、2、3 检查不同 S/G。
缺少查询或坐标越界会报错，不会猜测端点。无需重新预处理已有数据。
只准备输入、不加载模型时，使用 `--prepare-only`，并可省略 `--vlm-model`。
输出目录必须不存在或为空。

输入包含原始 RGB、公开 S/G、坐标与移动规则及共用成本图例。
NPZ 只读取 `image`；任务清单只提取查询标识和公开端点，任务文本按统一规则封装，不向模型提供 GT 成本、路径或最优值。
原图不额外画端点标记；提示词给出 20×20 网格和每格 16×16 像素的对应关系。

输出包含 `system_prompt.txt`、共用 `questions.txt`、逐图实际 `map_*_question.txt`、
原图 `map_*_input.png`、回答 `map_*_answer.txt`、`answers.jsonl` 和配置。
JSONL 保存公开查询、实际提示词、chat 文本、视觉 token 数及停止原因。
人工检查端点定位、区域连接和成本判断是否有可见依据；**生成结束不等于回答正确，自述可行不等于求解器验证**。
VQA 不输出完整路径、成本矩阵或模型 JSON，不能替代正式建模和规划成本评价。

脚本使用官方 processor 和 `model.generate` 调用冻结 Qwen，不经过选择器或求解器；
保留 `mm_token_type_ids` 并检查原始 320×320 图像网格。
接口参考 [Qwen3-VL 官方文档](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)。
同一英文环境提示词已接入 `run_tilebased_v85.py` 的正式建模与 RL 求解器反馈，默认使用 `navigation`。
正式调用只输出成本模型 JSON，不先生成 VQA 回答，也不把 VQA 回答当成 GT。

## 历史矩阵接口：新旧提示词的配对规划实验

新增 `--modeling-prompt navigation|legacy`（默认 `navigation`）和 `--compare-prompts`。
`navigation` 使用独立 system message 加导航建模指令；`legacy` 保留原来的 user prompt 且没有 system message。
共享逻辑位于 `tilebased_prompts_v85.py`，原生与裁剪 Qwen 接口均支持 system message。
选择器仍接收原有任务文本；新环境提示词用于选定区域后的冻结建模器，RL 的奖励因此来自新建模提示词下的真实求解结果。
metric/distill 的代理损失不变。本次没有微调 Qwen，也没有更改成本、图结构或 GT。

在脚本目录运行以下命令，直接复用已经预处理的数据：

```bash
python run_tilebased_v85.py \
  --mode evaluate --model-representation matrix --compare-prompts \
  --data /data/v85_tilebased/pkmn \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct \
  --device cuda:0 --load-4bit \
  --split val --limit 40 --batch 1 \
  --modes full random --budget 0.15 --seed 1234 \
  --modeling-max-new-tokens 4096 \
  --output results/tilebased_prompt_ab_r1
```

保持原实验的模型精度；原来未量化则去掉 `--load-4bit`。目录必须为空或不存在。
这轮是 40 个任务 × 2 个视觉预算方案 × 2 种提示词，共 160 次建模调用。
Full 保留 100 个视觉 token，Random 保留 15 个。同一任务在两种提示词下共享图像编码、S/G 和完全相同的选择区域，
只加载一次 Qwen，并交替提示词执行顺序。两组均采用贪心解码和相同输出 token 上限，不依赖旧日志的运行配置。
新提示词增加的文本开销单独记录，不将视觉 token 比例当成总计算量加速比。

输出：

| 文件 | 内容 |
|---|---|
| `summary.json` | `legacy`、`navigation` 各自的 Full/Random 结果，以及 `paired` 配对变化 |
| `outcomes.jsonl` | 每个任务、预算方案和提示词版本的回答、选择区域、路径、GT 评价和错误 |
| `prompt_config.json` | 实际 system prompt、版本和 SHA256 |
| `args.json` | 模型路径、预算、种子、数据签名、checkpoint 原提示词版本等运行配置 |

优先查看 `paired.full.mean_task_loss_improvement`，计算为旧损失减新损失，正数表示改善，**失败样本也保留在计算中**。
同时检查 `success_rate_change`、`mean_regret_improvement_both_success`、两组的 `optimal_rate_all` 和 `cost_cell_accuracy_successful`。
共同成功样本上的 regret 单独配对，避免因成功样本集合变化产生误读；常数矩阵率和 uniform-cost 路径重合率辅助解释模型是否读取了图像。
`mean_input_tokens`、`mean_nonvisual_input_tokens`、生成 token 数和 `environment_seconds` 用于观察开销；后者不含共享的早期编码和选择器计算，不是端到端延迟。

40 个任务仅用于初步诊断，同图多个 S/G 查询相关，不应据此宣称统计显著提升。
确认 Full 的成本建模改善后，再扩大验证集、固定提示词并在 test 上报告结果；随后用该提示词重新进行 RL。
已有 checkpoint 可以做配对评估，选择器在两组中固定；旧 RL checkpoint 的奖励是在旧提示词下学习的，迁移评估不等于已经完成新提示词下的 RL 训练。
单独恢复旧基线使用 `--modeling-prompt legacy`；单独运行新方法使用默认值或 `--modeling-prompt navigation`。
当前环境提示词面向 Pokémon；Warcraft 请显式使用 `legacy`。

验证状态：集成与配对流程已用 CPU 测试，包含小型随机初始化真实 Qwen 的 system-prefix 位置编码和原生/裁剪输出一致性。
这些测试不代表 Qwen3-VL-8B 的任务能力；当前工作环境没有服务器模型权重和 GPU，真实 8B 新旧提示词实验尚待运行，暂无提升数值。

## 当前 Full 基线较弱时，先运行接口与识别诊断

新增 `diagnose_tilebased_v85.py`，直接复用已经预处理的数据，无需重下载、重做 GT 或先训练。
当 Full 成功率仅 65%、成功样本相对 regret 约 3.86 时，格式成功不能证明视觉建模有效；
先检查冻结模型能否正确读取地形成本，再解释选择器的收益。

在本目录运行，`--vlm-model` 替换成服务器上 **Qwen3-VL-8B-Instruct** 的实际路径：

```bash
python diagnose_tilebased_v85.py \
  --data /data/v85_tilebased/pkmn \
  --vlm-model /path/to/Qwen3-VL-8B-Instruct \
  --device cuda:0 \
  --split val --map-indices 0 3 6 9 \
  --backends native custom --probes cells matrix \
  --max-new-tokens 4096 --cell-max-new-tokens 768 \
  --output results/tilebased_diagnostics_r1
```

如原实验使用量化，添加 `--load-4bit` 并保持两条接口相同精度。脚本只从本地加载模型。
输出目录须不存在或为空。默认选择四张地图，每张只取第一个公开 S/G 查询；
不会按 GT 成本或模型结果挑选地图。两条接口使用相同 RGB、预处理像素、chat token ID、提示词和解码预算：

- `native`：官方 `model.generate`，由 Qwen 原生计算视觉特征、DeepStack 和 MRoPE。
- `custom`：当前选择器使用的自定义生成接口，诊断时保留全部视觉区域。
- `cells`：只询问 S/G 和预先固定位置的少量格子，减少完整 20×20 矩阵的序列化负担；允许回答 unknown。
- `matrix`：沿用 `legacy` 完整成本矩阵提示词与求解规则；该接口诊断默认不添加新环境提示词。新旧提示词任务效果比较使用上面的 `--compare-prompts`。

可先用 `--probes cells` 做短检查，再换一个输出目录运行完整命令。
可选 `--blank-control` 增加同尺寸灰图对照，用于检查答案是否依赖视觉；灰图结果不混入正式规划评估。
原始图不加 GT 路线、网格标签或裁剪重绘；公开地形图例与 S/G 继续保留。

输出文件：

| 文件 | 内容与用途 |
|---|---|
| `config.json` | 运行参数、数据签名、PyTorch/Transformers 版本 |
| `prefill.jsonl` | 两条接口首个生成 token 的 logits 最大/平均绝对差与 argmax 是否相同 |
| `outcomes.jsonl` | 每次原始输出、提示词及哈希、停止原因、生成 token 数、错误、逐格/路径指标 |
| `summary.json` | 按 probe、图像条件、backend 分组的结果与成对答案一致数 |

解读时按以下顺序：

1. 两条接口首 token logits 有明显差异时，先检查服务器版本、精度及视觉接入；小浮点差异或仅首 token 一致都不能单独证明整条链路正确。
2. 两条接口一致但局部格子成本识别都很差，优先检查地形图例、视觉可辨性及冻结模型能力。
3. 局部识别较好但完整矩阵常数化、行列错误或截断明显，说明完整矩阵建模/序列化仍是瓶颈。
4. Full 的成本识别和路径质量达到可用水平后，再比较训练后的 selector 与同预算 random。

诊断 v2 区分外层格式合规与内容可评分：返回裸数组 `[...]` 时，仍严格验证每个格子的
字段、顺序、坐标、数量及成本，然后计算识别准确率；`strict_format_valid=false` 和
`format_normalization=bare_list_to_cells` 明确记录缺少 `{"cells":...}` 包装。
`success_rate` 在 cells 中表示内容可评分率；`strict_format_success_rate` 才是严格格式合规率。
这只影响局部诊断，正式矩阵建模的校验和奖励完全不变。

已有诊断日志可离线重评分，无需 VLM 权重、Transformers 或 CUDA（读取器仍需 PyTorch）：

```bash
python diagnose_tilebased_v85.py \
  --data /data/v85_tilebased/pkmn --split val \
  --rescore-from results/tilebased_diagnostics_r1/outcomes.jsonl \
  --output results/tilebased_diagnostics_rescored_r2
```

原始日志不覆盖；新日志保留原回答、`previous_scoring` 及旧矩阵结果。
查询坐标会与数据集中相应任务核对。v2 的新生成提示词也显式展示完整 JSON 包装。

`cells` 的 `cell_accuracy_successful` 是内容可评分输出中、查询格子的成本准确率；unknown 计错。
`known_fraction_successful` 是非空回答比例，不等于正确率。这里不将地形名称当作人工语义 GT。
矩阵 `cost_cell_accuracy_successful` 在全部地形格上计算；`cost_mae_successful` 是平均成本绝对误差。
`same_path_as_uniform_cost` 表示与统一成本 Dijkstra 返回的路径完全相同，仅作诊断，不是最优性的判据。

`run_tilebased_v85.py --model-representation matrix` 的日志也保存这些矩阵诊断及 `generation` 字段；
`finish_reason=length` 表示耗尽输出预算，`eos` 表示遇到结束 token；`generated_tokens` 不包含 EOS。
继续严格拒绝错误形状和非法成本，不自动补齐矩阵、不从 GT 修复输出、不改变 reward。
旧 checkpoint 和已生成数据的契约保持兼容。诊断脚本不会训练或更新任何模型参数。

验证边界：用随机初始化的小型真实 Qwen 实现检查完整输入的 prefill/连续解码一致性，
同时检查稀疏输入、GT 隔离和原有训练流程。小模型测试不代表 8B 的实际识别准确率；
仍须在服务器运行上述对照，不能将本次更新报告为准确率已提升。

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

默认生成 `tilebased_sg_corridor_v1`，字段和几何规则见本文开头。VLM 根据当前任务和保留视觉信息，
提出空间连接及成本估计；稀疏 Dijkstra 在生成的通路子图上求解，随后才读取完整标签评价。
`--model-representation matrix` 保留旧 `tilebased_node_cost_path_v1`，仅用于复现旧实验。
解析器均拒绝额外字段、重复键、非有限值及非法坐标，不执行模型生成的代码。

`relative_regret = max(0, (true_path_cost - optimal_cost) / optimal_cost)`。

成功 loss 为 `min(relative_regret, .999 * failure_cost)`；建模或求解失败记 `failure_cost`，默认 10。
日志保留未裁剪 regret；失败成本为 null。完整 GT 图连通，但 VLM 提出的通路图可能不连通。
成功率反映模型及连接是否可求解；最优率与真实成本差距评价路线质量。

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
`--modeling-max-new-tokens 4096` 保留统一的解码上限；实际生成长度另行记录。当前输出为少量通路连接，输出截断按建模失败处理。
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
python -m pytest -q test_v85_tilebased_corridor.py test_v85_tilebased_prompts.py test_v85_tilebased.py test_v85_tilebased_diagnostics.py test_v85_solver_environment.py test_v85_gradient_flow.py test_v85_posttraining.py
```

包含：八邻接/起点成本、同成本不同路径、GT 连通性、非法模型拒绝、标签隔离、
失败计分、批读取、冻结参数及 distill/RL 梯度、下载校验、旧 checkpoint 拒绝。
CPU 测试使用小型 backbone 替身测试接口和梯度，真实图求解使用真实 Dijkstra。
Qwen3-VL-8B 的识别准确率、GPU 显存、训练收益及端到端加速需在你的模型环境实测，
不能由这些接口测试推断。

初次适配已执行：Pokémon 三个原始文件下载与 SHA256 校验、全量 4,000 张地图/16,000 个任务转换与 GT 校验、
三个 split 的 batch=4 读取。
本次诊断更新在 Transformers 4.57.6 和 5.17.0 上分别通过 63 项回归测试，无跳过；
小型真实 Qwen 包括剪枝前运行 0/1 个视觉 block 的完整输入一致性与稀疏解码检查。
详细记录见 `validation_tilebased_v85.json`。此外修复了共享 RL 的 masked categorical KL 在反向传播中
可能出现 `(-inf)-(-inf)` 的 NaN；新旧 RL 测试均通过。
