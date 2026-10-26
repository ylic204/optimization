# V8.5 — Visual evidence, symbolic optimization, solver-generated trajectory

Current branch: `v85-tilebased-llm-solver`. Current generated schema:
`tilebased_symbolic_v1`; environment contract: `frozen_tilebased_symbolic_scene_milp_v3`.
The CLI name `--model-representation algebra` is retained; land data `auto` selects it.

## What changed

Frozen Qwen receives selected visual features plus the task. In ONE generation it
outputs (1) compact scene evidence and (2) a complete indexed mathematical model.
The compiler expands geometric sets and generated expressions; HiGHS chooses the
trajectory. Qwen does not enumerate nodes/arcs, output parameter matrices, or draw
candidate routes. Visual patches are observation units, not planning nodes.
The previous candidate-graph/patch-port approach is retired from the formal path.
Its documentation is archived in `README_V85_ALGEBRA_HISTORY.md`.

This adopts a neural-symbolic separation of visual facts and mathematical reasoning,
not a reproduction of a particular neuro-symbolic planner. The modeling template
provides mathematical guidance (inspired by the previously examined MM-Agent
workflow); no retrieval agent, extra LLM call, or memory system is introduced.

## Generated output

Exactly three top-level fields: `schema`, `scene`, `model`.

- `scene.regions`: up to 64 visual evidence regions, each with `geometry`, per-cell
  `cost` in [1,25], and boolean `blocked`. Geometry is an inclusive cell box
  `bounds_rc` or a simple, possibly concave, `polygon_rc` with grid-corner vertices.
  Multiple regions express terrain differences and water boundaries. These are
  perception facts, not prescribed routes or a replacement for the mathematical model.
- `model.sets`: one `grid_cells` and one `neighbor_pairs` operator (4 or 8 neighbors).
  These are declarative set constructors, never lists of IDs.
- `model.parameters`: aliases bound to estimated scene fields and public geometry/task
  data, or finite scalar constants. Explicit arrays and GT bindings are rejected.
- `model.variables`, `objective`, `constraints`: generated indexed linear/integer
  expressions, evaluated by the bounded AST interpreter (no Python eval/exec).
- `model.output`: the indexed movement variable to decode after solving, e.g.
  `{"variable":"x","index":"A"}`. No geometry list is generated.

Example declarations:
```json
{"sets":{"V":{"op":"grid_cells"},"A":{"op":"neighbor_pairs","of":"V","connectivity":8}},
 "parameters":{"c":"scene.cost","s":"task.source","g":"task.goal",
 "tail":"space.tail","head":"space.head","outgoing":"space.outgoing","incoming":"space.incoming"},
 "variables":{"x":{"index":"A","type":"binary","lb":0,"ub":1}},
 "objective":{"sense":"min","expression":"c[s] + sum(c[head[e]]*x[e] for e in A)"}}
```
This excerpt is NOT a complete model. The full executable prompt example is
`MODEL_EXAMPLE` in `tilebased_symbolic_v85.py`, including flow, degree, blocked-cell
and diagonal-corner constraints. Qwen must output these expressions itself.

Supported parameter sources: `scene.cost`, `scene.blocked`, `scene.unknown`,
`scene.corner_blocked`, `task.source`, `task.goal`, `space.tail`, `space.head`,
`space.outgoing`, `space.incoming`, `space.step_length`. A move has unit length,
including diagonals, matching this dataset's evaluation protocol. Source-cell
cost must be included by the generated objective. S=G uses the same flow balance
expression and a singleton decoded route when no movement is selected.

## What the compiler does and does not do

Geometry creates all in-bounds neighboring moves, INCLUDING moves through blocked
cells. It neither filters water nor installs flow constraints. Generated equations
must exclude blocked endpoints and diagonal corner cuts. Changing or deleting
constraints changes the actual solved model; missing constraints are never repaired.
The compiler supports up to 8192 expanded variables and 16384 rows on bounded grids;
this is distinct from the short generated JSON. Sparse incoming/outgoing incidence
lists avoid expensive all-edge filtering for every node.

The decoder follows the solver's selected moves. Branches, gaps, cycles, fractional
route values and unused selected components fail. It does not run Dijkstra to fix
the result. Dijkstra remains only for independent evaluation diagnostics/references.
No teacher labels, GT cost grid or reference trajectory enters scene compilation.
All methods use the same environment interpretation and symbolic modeling prompt.

## Unknown evidence and partial observations

Uncovered cells are UNKNOWN: fixed cost 25, blocked=false. This is an explicit,
shared pessimistic-cost assumption, not a claim that they are land. Missing visual
tokens never delete nodes. Predicted evidence in unretained regions is logged as
inferred evidence. Overlapping evidence uses maximum declared cost and blocked OR.
No GT repairs evidence. Unknown policy, coverage, direct visibility and inferred
coverage are exported. This policy can bias routes and cannot guarantee true-world
feasibility; independent land-map evaluation remains mandatory. Full may also
leave unknown regions if Qwen omits evidence.

## Run Full first

No dataset preprocessing changes are needed for existing land data. Use a NEW output
folder and no old checkpoint. Old Full reports/checkpoints have incompatible signatures.

```bash
python run_tilebased_v85.py \
  --mode evaluate --split val --modes full --limit 2 \
  --data /data/v85_tilebased/pkmn_land_v3 \
  --model-representation algebra --modeling-prompt navigation \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct \
  --device cuda:0 --load-4bit --batch 1 \
  --modeling-max-new-tokens 8192 --seed 1234 \
  --output results/tilebased_symbolic_full_smoke
```

Inspect this small run, then repeat with `--limit 32` and a new folder such as
`results/tilebased_symbolic_full_r1`.

```bash
python validate_tilebased_full_v85.py --run results/tilebased_symbolic_full_r1
```

The existing training gate requires >=32 unique Full cases, >=95% optimal solver
status and >=90% true-map SR. These are engineering readiness checks, not proof of
statistical stability. Supply the resulting `full_validation.json` to subsequent
training with `--full-validation`. Selector/proxy/distillation/RL mechanisms are
unchanged; no claim is made that proxy gradients are exact derivatives of MILP.

## Live output and saved interfaces

Live Qwen text is printed in 16-token chunks; `--quiet-generation` hides printing
but preserves files. Each attempt uses a unique filename under
`<output>/model_answers/navigation/`:

| Suffix | Contents |
|---|---|
| `.txt` | Raw LLM text, including incomplete output |
| `.generated_model.json` | Parsed scene + model response, potentially invalid |
| `.symbolic_model.json` | The generated model before expansion |
| `.scene_estimate.json` | Estimated costs/blocked/unknown/visibility provenance |
| `.expanded_model.json` | Instantiated sets/parameters plus generated expressions |
| `.model.txt` | Readable generated variables, objective and constraint families |
| `.compiled_model.json` | Numerical objective, bounds, integrality and sparse rows |
| `.solver_result.json` | Solver termination and assignment |
| `.path_rc.json` | Decoded trajectory, including routes that fail true-map checks |
| `.status.json` | Sample ID, failure information and artifact paths |

`outcomes.jsonl` retains the raw answer and stage records. Only reached stages are
exported; failures do not fabricate downstream results. The terminal prints model
expressions and expanded dimensions after each call. A streaming array limit
stops invalid runaway output; it does not fix modeling or perception errors.

Failure categories distinguish format/model errors, length stops, infeasibility,
solver limits, invalid assignments and independent map violations. SR, SPL and
success-weighted cost efficiency remain primary evaluation summaries.

## Verification and limitations

CPU tests verify objective-dependent route changes, obstacle and corner constraints,
missing-constraint behavior, aliases, unknown areas, polygon evidence, S=G,
20x20 expansion vs an independent reference, artifact output, and end-to-end
independence from changed GT labels using a frozen-response mock backend.
Real Qwen generation, custom/native backend equivalence and improved navigation
metrics have NOT been verified here. Keep full-vision validation before selector
training. This implementation supports indexed linear/MILP expressions, not
arbitrary nonlinear optimization or general PDDL task planning.
