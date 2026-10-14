# V8.5 staged Selector training with frozen modeling and solver outcomes

## Scope and audit

`main` uses first-person RGB. `v85-bev-input` added BEV; `v85-bev-rl-posttraining`
added policy gradients but still rewarded a candidate-cost argmin. This branch adds:
**BEV subset -> frozen Qwen optimization JSON -> continuous trajectory solver -> GT feedback**.
The only learned action is a region subset. Modeling weights and solver are frozen.
All baselines use the same downstream environment.

## Stages and total losses

1. Export current/past BEV; separately preserve future logged rear-axle poses and actual timestamps.
   Process revision3. Split by full log. Primary RL rejects absent/fallback GT before loading Qwen.
2. Metric warm-up: freeze Qwen/Selector/field, train the candidate metric proxy head on full and random
   physically pruned observations. Loss `J_proxy + λ_metric L_metric`; select by random-mask metric MAE.
3. Distill warm-up: freeze Qwen and proxy head; train Selector/field using the projected mask gradient
   of label-based expected normalized candidate regret. This Teacher is a differentiable proxy,
   not a derivative through discrete JSON generation or SLSQP.
4. Primary RL: initialize from distillation. Freeze Qwen/head. Sample G>=2 independent fixed-budget
   region actions; generate and solve each problem, score trajectories with GT, and update only Selector/field.
   Reuse early vision features, execute each downstream outcome serially.
5. Evaluate deterministic Top-K using the same modeling/solver chain; checkpoint by learned task loss
   over all scenes including failures. Save episode models, solver statuses, trajectories and GT.

For valid regions C_k={0<=w<=1, sum(w)=k}:

```math
w_T^+=\Pi_{C_k}(w-\eta_T\nabla_w J_{proxy}(w)),\quad v_T=(w_T^+-w)/\eta_T,
\qquad w^{s+1}=\Pi_{C_k}(w^s+\eta_s v_\theta(w^s,z,c,s/S)).
```

Gradient/FM matching trains this same field; ND executes it. No trajectory-FM/ND cascade.
`L_gfm` matches direction, log-norm and projected next state. The projection uses the active-set
Jacobian. Deployment and RL use final pre-projection scores to avoid saturated ties/zero gradients.

```math
L_{distill}=L_{task,proxy}+\lambda_m L_{metric}+\lambda_g L_{gfm}+\lambda_z L_{latent}
+\lambda_v L_{value}+\lambda_i L_{improve,proxy}.
```

Proxy improvement is `ReLU(J_final-stopgrad(J_initial)+margin)`. Both endpoints actually gather K
visual groups; unit-forward selected-only gates give a biased hard-selection derivative. Teacher
gradients instead use the separate soft-mask path. Value loss is proxy set-loss regression, not
an RL critic. Freezing the metric-head parameters does not detach its inputs: `L_metric`
still backpropagates through retained tokens to Selector/field and is part of the optimized loss.
Proxy warm-up does not certify
primary trajectory loss.

## Primary reward and RL loss

[The environment contract](README_V85_SOLVER_FEEDBACK.md) excludes candidate geometry/labels, GT,
semantic counts and audit rasters from modeling. Selector text describes both task and model evidence needs.

For success `J_raw = λ_A ADE + λ_F FDE + λ_S(collision_static + offroad_static)`,
`J=min(J_raw,0.999 C_fail)`. Invalid models/infeasible/failed solves have `J=C_fail`, no successful
trajectory. Defaults: 1/0.25/1, C_fail=100. ADE/FDE use identical times/rear-axle frame in meters;
reports stay unclipped, failed errors are null. Safety terms are current-raster centerline proxies.

```math
L_{improve}=ReLU(J(a)-stopgrad(J(a_0))+m),\quad r_g=-J(a_g)-\lambda_i L_{improve}(a_g).
```

Initial Selector Top-K a_0 uses identical K and environment. This is endpoint improvement;
margin0 allows ties and does not ensure stepwise descent. Rewards/model/solver are detached.

The final scores h define a Plackett-Luce ordered draw of K distinct regions:

```math
\log\pi_\theta(a|o)=\sum_{j=1}^K\log
\frac{\exp(h_{a_j}/\tau)}{\sum_{i\notin\{a_1,...,a_{j-1}\}}\exp(h_i/\tau)}.
```

Order is an auxiliary action; sort spatially before Qwen. We do not claim unordered-set logp.
No new observation, ND update or modeling call occurs between draws. With detached leave-one-out
advantage `A_g=r_g-mean(other rewards)`:

```math
L_{RL}=-mean(A_g\log\pi_\theta(a_g|o))+\beta\bar D_{conditional}-\alpha\bar H_{conditional}.
```

KL/entropy average over K conditional draws; KL is an anchor surrogate, not exact subset KL.
The reference copies only small modules, sharing frozen 8B backbone/head. This is offline
contextual-bandit REINFORCE, not PPO/GRPO/closed-loop driving. Fixed K gives a constant token penalty;
use budget sweeps and complete latency measurements to assess efficiency.

## Commands, migration and evaluation

Use [root README commands](../README.md). Training/evaluation default to `llm_solver`.
`candidate_proxy` is an explicit historical ablation; its metrics are candidate regret/MAE,
not trajectory ADE/FDE. A new-contract distilled checkpoint can be evaluated in solver mode before RL.
Old timestamp-free raw requires re-export; old revision2 data/checkpoints or task-text-only checkpoints
are rejected. Retrain metric -> distill -> RL. Resume initializes weights, not optimizer/RNG.
Keep backbone weights, architecture, task, budget and geometry fixed across stages.

Full/random/initial/learned use identical prompts, schema, decoder, solver, times, ego/goal and GT.
Full uses R tokens; the other three use K. Each validation emits aggregate JSON and episode JSONL.
Task loss includes failures. ADE/FDE are success-only, so always report success/failure rates alongside them.
Environment time excludes early encoding/Selector/ND and is not complete inference latency.

The frozen model can return invalid JSON; failure rate/decode budget require real-data measurement.
The point QP has per-axis speed/acceleration, not bicycle dynamics. Generated corridors/halfspaces may
omit hazards. Swept footprints, time-indexed actors, official red-light/comfort checks and nuPlan
closed-loop evaluation remain extensions. No claim of measured improvement from CPU tests.

Sources: [Qwen3-VL API](https://huggingface.co/docs/transformers/model_doc/qwen3_vl),
[SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html),
[nuPlan metrics](https://github.com/motional/nuplan-devkit/blob/master/docs/metrics_description.md).
