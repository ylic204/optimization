# V8.5 frozen modeling and continuous solver interface

## Frozen downstream environment

`FrozenBEVSolverEnvironment.evaluate(encoded, indices, batch, texts)` accepts one fixed-budget
subset per scene. It sorts indices, physically gathers K groups, and calls frozen Qwen's
`generate_selected_json`. Greedy autoregressive decoding uses a real chat prefix, sparse original
row/column RoPE positions and DeepStack only for retained groups; subsequent tokens use a KV cache.
No full-image generation fallback or candidate-head prediction substitutes for this call.

The prompt allowlist is task/model requirements, fixed map bounds, public planning times,
current ego velocity and goal xy. The visual stream contains selected RGB features. GT, candidate
trajectories/metrics, map reference coordinates, semantic masks/counts and occupancy grids are
excluded. GT and occupancy are read only after solving for external reward. Public time grids are
protocol metadata, not future pose information.

## JSON model family

The LLM supplies a validated instance of `bev_trajectory_qp_v1`:

```json
{
  "schema": "bev_trajectory_qp_v1",
  "reference_xy": [[0.5,0.0], [1.0,0.0]],
  "corridor": [[-16.0,64.0,-40.0,40.0], [-16.0,64.0,-40.0,40.0]],
  "weights": {"tracking":1.0, "smoothness":1.0, "goal":0.1},
  "halfspaces": [{"step":2, "normal":[1.0,0.0], "upper":1.2}]
}
```

This two-step example illustrates the grammar; V8.5 production arrays have 16 rows.
Halfspaces mean `normal dot p_step <= upper`; one-based steps, at most64. Empty is allowed.
Corridors have positive widths and remain within BEV support. Reference xy stays within support.
Tracking weight is [0.001,100]; smoothness/goal [0,100]. Unknown fields, nonfinite values and wrong
lengths are model failures. Python/code strings are never executed. The map/route, obstacles,
traffic stopping decisions must be grounded in the retained visual information by the frozen LLM.

## Deterministic optimization template

For p_i=(x_i,y_i), p_0=(0,0), times t_i and dt_i=t_i-t_{i-1}, set
v_i=(p_i-p_{i-1})/dt_i and a_i=(v_i-v_{i-1})/dt_i, with measured v_0.

```math
\min_{p_1,...,p_N}\ \frac{w_t}{N}\sum_i\|p_i-r_i\|^2
+\frac{w_s}{N}\sum_i\|a_i\|^2+w_g\|p_N-g\|^2
```

Subject to corridor coordinate bounds, generated halfspaces, and fixed componentwise
`|v_i| <= solver_max_speed`, `|a_i| <= solver_max_acceleration`. Defaults15m/s and4m/s² are
**per axis**, not Euclidean norms. This is a strictly convex quadratic in continuous xy with
linear constraints. It does not select existing candidates. Derived yaw is each segment angle,
not a bicycle-dynamics state; heading/nonholonomic/jerk constraints are not modeled.

`solve_trajectory_qp` builds analytic Hessian/gradient and calls SciPy SLSQP, then checks solver
success, finite solution and maximum feasibility residual (default1e-4). A failed solve has no
successful trajectory. Invalid/infeasible model yields a penalty, without a stop-trajectory fallback.
This template restriction gives all baselines the same modeling capacity/solver interface.

## Reward, reports and baseline reuse

At common planning times, compute rear-axle ADE/FDE in meters. Current centerline occupancy gives
static collision/offroad proxies. Bounded success loss and failure penalty are defined in the
[training protocol](README_V85_POSTTRAINING.md). The environment has no trainable parameters.
Only policy-gradient logp leads back to Selector/field. The proxy head is not called in primary RL.

Full, random, initial and learned use the same frozen backbone, JSON prompt/validator, solver and
external targets. Random/initial/learned share K; full is an R-token reference. External selection
baselines can reuse this environment directly with their own indices; do not feed their own labels
or separate solver settings if claiming controlled comparisons.

Validation writes `*.outcomes.jsonl`: sample_id, mode, selected_indices, model_answer,
optimization_model (if valid), solver status/residual, time_s, trajectory (if successful),
gt_trajectory, ADE/FDE/proxies/loss and environment_seconds. Failure rows preserve model/error data.
Aggregate JSON includes all-scene task loss and failure rates; ADE/FDE are successful-only and null
if all fail. Environment seconds excludes early encoding/Selector/ND and is not end-to-end latency.

Limitations: convex corridors/separating planes depend on LLM perception and may miss hazards.
No future actor prediction or swept-vehicle collision reward. No official nuPlan red-light/comfort/
closed-loop scoring. GT is logged human behavior, not a uniquely optimal plan or model annotation.
A corridor is restricted to visible BEV support. If an 8-second expert future exceeds x=64m,
the default instance cannot match that part; choose a suitable fixed extent/horizon at export
and report that protocol rather than comparing incompatible scenarios or dropping them silently.
A low ADE alone does not establish safety. Inspect actual 8B model outputs and failure rates on
nuPlan before treating this implementation as a validated research result.

API references: [Qwen3-VL](https://huggingface.co/docs/transformers/model_doc/qwen3_vl),
[SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html).
