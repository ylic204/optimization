# 路径规划模板指导更新

参考 MM-Agent 固定版本 `74d84f393a0557d41ae45a01625d15e0d61df983`：
- [方法检索](https://github.com/usail-hkust/LLM-MM-Agent/blob/74d84f393a0557d41ae45a01625d15e0d61df983/MMAgent/agent/retrieve_method.py)：层次方法库，embedding 或 LLM 打分。
- [建模流程](https://github.com/usail-hkust/LLM-MM-Agent/blob/74d84f393a0557d41ae45a01625d15e0d61df983/MMAgent/utils/mathematical_modeling.py)：分析、检索、公式与建模。

本实现借鉴工作流，未复制上游实现。当前任务类别由公开 benchmark 任务规则确定为 path_planning/minimum_terrain_cost，检索唯一适用的 land_flow_v1 模板；这不是通用 LLM 分类器，不扩展到其他竞赛任务。未知任务明确拒绝。

新提示词提供完整路径模型范例：成本是参数，binary 连接选择变量、成本目标、逐节点流量守恒、度数上界与禁行约束；要求 Qwen 用当前图像实例化全部内容。模板仅进入提示词，不送进编译器作为默认约束。原始生成结果仍独立校验、编译、求解；遗漏约束不会自动补上。所有 baseline 共用模板，新增文本 token 计入输入统计。

兼容数值表达式与字符串键参数字典，避免把正常 JSON 数字误判为数学错误。非线性乘积、未定义符号和缺少输出绑定仍拒绝。每条日志保存 modeling_template，验证签名包含模板模块；旧 Full 报告须重跑。当前没有增加自动重试或声称复现 MM-Agent 全部多轮流程。

命令不变，请将结果目录改为 `results/tilebased_algebra_template_full_r2`。数据无需重建，真实 Qwen 效果仍须运行验证。

---

# V8.5：Qwen 生成数学模型，HiGHS 求解

当前分支 `v85-tilebased-llm-solver`。陆地数据的 `auto` 默认改为 `algebra`。
**正式流程：视觉 token → 冻结 Qwen → 变量、目标、约束 → 表达式编译 → HiGHS → 真实地图独立评价。**

不再把区域或连接成本送入固定的 Dijkstra 模板作为正式求解。历史 `regions/segments/matrix` 显式选项仅用于复现旧实验。
新模式支持线性规划和混合整数线性规划，不宣称支持任意非线性数学模型。求解接口通过 `scipy.optimize.milp` 调用 HiGHS（依赖已包含在 requirements.txt）。

## 生成模型与执行边界

Qwen 输出 `tilebased_algebra_v1` JSON：

| 字段 | 内容 |
|---|---|
| `sets` | 命名的有限整数索引集合 |
| `parameters` | 由视觉及任务得到的数值、数组，例如成本与连接关系 |
| `variables` | 自主声明的标量／索引变量、类型及有限上下界 |
| `objective` | `min/max` 和实际执行的线性表达式 |
| `constraints` | 自主声明的等式／不等式及 `forall` 索引 |
| `output` | 把某一变量族的选中项绑定到空间连接，用于还原轨迹 |

连同 `schema` 一共七个字段。变量名称和约束数量不固定。编译器不会插入流量守恒、起终点、连通性、禁行或其他路径约束。**漏写约束时，求解器照常求解漏写约束的模型，解码失败就计为失败。**

表达式例子（仅为语法示例，不是预填的导航模型）：

```json
{
  "variables": {
    "z": {"index":"ITEMS", "type":"binary", "lb":"0", "ub":"1"}
  },
  "objective": {"sense":"min", "expression":"sum(price[j]*z[j] for j in ITEMS)"},
  "constraints": [
    {"name":"capacity", "forall":{}, "lhs":"sum(weight[j]*z[j] for j in ITEMS)", "sense":"<=", "rhs":"capacity"}
  ]
}
```

完整 JSON 还需要集合、参数、schema 与输出绑定；上述片段不可以直接用于导航。
表达式支持数值、参数数组索引、决策变量、加减、参数乘除、有限求和、参数条件与条件表达式。
支持参数字典的字符串键索引、JSON 数值常量。仍不支持变量乘变量、变量条件、任意函数调用、属性、导入或执行 Python。使用 AST 解释器，不使用 `eval/exec`。
每条约束可有一个 `forall` 索引，每个求和有一个生成器；可以使用多个求和。索引不能遮蔽已有名称。

### 路径输出接口不是预制优化模型

`output={"variable":"x", "arcs":[{"index":0,"polyline_rc":[[0,0],[0,2],[1,2]]}, ...]}`。

- 每个索引绑定一条有方向、可拐弯的空间连接。Qwen 可声明多个候选连接和辅助变量。
- polyline 是连接的空间几何，不是指定最终路线；HiGHS 根据**生成的目标和约束**选择变量值。
- 线段仅进行固定八邻域几何展开，不用真实地图绕障或补路。相交连接不会自动添加转接边。
- 解码器要求绑定变量的求解值为 0/1，只跟随已经选中的连接；遇到缺口、分叉、重复格子或多余选中连接，记录 `invalid_solution`。
- 不从错误解中抽取一个较好子路径，不再运行 Dijkstra 修复它。
- 这是用于当前导航任务的输出约定，数学结构仍由 Qwen 生成。空间建模仍可能遗漏候选路线；改为完整数学模型并不能消除视觉误识别。

最多 1024 个展开变量、4096 条展开约束、256 个输出连接；每个连接最多 64 个点。求解时间上限 30 秒。超时不算成功，不自动接受尚未证明最优的 incumbent。

## 先做 Full

继续使用 `pkmn_land_v3`，无需重新预处理。不要复用区域版本结果或检查点。
所有 Full/random/learned 方法统一使用英文环境提示词及同一建模语法，只有任务和保留的视觉位置不同。

```bash
python run_tilebased_v85.py \
  --mode evaluate --split val --modes full --limit 32 \
  --data /data/v85_tilebased/pkmn_land_v3 \
  --model-representation algebra --modeling-prompt navigation \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct \
  --device cuda:0 --load-4bit --batch 1 \
  --modeling-max-new-tokens 8192 --seed 1234 \
  --output results/tilebased_algebra_full_r1

python validate_tilebased_full_v85.py --run results/tilebased_algebra_full_r1

python visualize_tilebased_v85.py \
  --data /data/v85_tilebased/pkmn_land_v3 \
  --outcomes results/tilebased_algebra_full_r1/outcomes.jsonl \
  --split val --mode full \
  --output results/tilebased_algebra_full_r1/path_visualizations
```

`outcomes.jsonl` 保存原始 `model_answer`、`generated_model`、实际编译的 `compiled_model`（变量、目标系数、约束系数和上下界）、`solver_result`（变量解、目标值、状态）、路径和具体 `error`。
`full_validation.json` 是训练前检查报告；`failure_examples.json` 汇总每类失败的前两个完整例子。

当前工程检查门槛：至少 32 个不重复的 val 任务，HiGHS 最优返回率 ≥95%，真实任务 SR ≥90%。这是开始选择器实验的最低检查门槛，**不是统计上证明稳定的标准**；应扩展验证集并检查 SPL/成本质量。成本效率差时，即使通过，也应先改善 Full。
检查失败脚本退出码为 2，仍保存报告和失败例子。不能用 test 调参或筛选成功样本后声称通过。

### 失败分类

| category | 含义 |
|---|---|
| `format_error` | 非法／重复键 JSON |
| `length_limit` | 生成截断 |
| `model_error` | 缺字段、未定义符号、非法索引、非线性或错误输出绑定 |
| `infeasible` | 生成的数学模型不可行 |
| `solver_limit` / `solver_error` | 求解超时或异常 |
| `invalid_solution` | 数学模型有解，但变量赋值不能解码为唯一完整 S→G 轨迹 |
| `water_or_corner_violation` | 解码轨迹在真实地图中涉水或斜穿水角 |

`status` 保留 model_failure / solver_failure / constraint_failure 以兼容已有汇总，具体原因看 category 和 error。SR/SPL/成本效率的失败项仍计零，task_loss 失败惩罚仍为 10；不会把失败样本移出分母。
生成模型的目标值 `generated_objective_value` 与真实地形成本分别保存，二者不混用。
真实成本／水域掩码／GT 在生成和正式求解后才读取。Dijkstra 仅保留在离线参考、训练代理和事后诊断，不进入 algebra 正式求解链。

## Full 通过后训练

`--full-validation` 必须引用上面的报告。程序会重查原始日志和配置，拒绝失败／修改／过期报告或不同数据、模型、解码预算与提示词。旧区域环境检查点不能迁移进来继续训练。

```bash
python run_tilebased_v85.py \
  --mode experiment --data /data/v85_tilebased/pkmn_land_v3 \
  --model-representation algebra --modeling-prompt navigation \
  --full-validation results/tilebased_algebra_full_r1/full_validation.json \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct \
  --device cuda:0 --load-4bit --batch 1 \
  --modeling-max-new-tokens 8192 --seed 1234 \
  --limit 128 --rl-limit 32 --val-limit 32 \
  --metric-epochs 3 --distill-epochs 3 --rl-epochs 2 \
  --rl-group-size 2 --budget 0.15 \
  --output results/tilebased_algebra_selector_r1
```

保留 proxy→梯度蒸馏/FM/ND→求解反馈 RL。proxy 仍是训练用的地形决策代理，**不是生成式 MILP 的精确梯度**。正式 RL 的成功和成本来自本页的新建模／求解／评价流程。
本次实现已进行 CPU 编译器、HiGHS、故障分类及环境测试；当前工作环境无用户服务器上的 Qwen 8B 权重和真实数据，未运行真实 Full、没有启动选择器训练，也不声称 SR 提升。
