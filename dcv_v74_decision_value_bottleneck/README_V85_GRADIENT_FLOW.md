# V8.5: Teacher-gradient flow matching with fixed-step neurodynamics

V8.5 replaces the V8.4 serial `trajectory Flow -> ND refiner` design.  Flow
matching and neurodynamics now refer to the same visual-mask vector field:

- the full-information Teacher differentiates the true planning task objective;
- the Student matches the Teacher's projected descent direction;
- fixed-step neurodynamics executes the learned Student field;
- `L_improve` checks only whether the final state improves over the initial
  state.  It does **not** impose per-step monotonicity.

V8.4 remains in the repository as the reproducible serial baseline.

## State and Teacher direction

Let `w in [0, 1]^R` be the relaxed visual-region mask, with a fixed token
budget `sum(w) = k`.  The feasible set is

```text
C_k = {w | 0 <= w_r <= 1, sum_r w_r = k}.
```

For the full-information task loss `J_T(w)`, the Teacher produces a feasible
descent velocity

```text
w_T+ = Pi_Ck(w - eta_T * grad_w J_T(w))
v_T  = (w_T+ - w) / eta_T.
```

This adapts the gradient-direction matching idea to the visual-mask state:
the Teacher can see all early Qwen visual regions, while the deployed Student
must decide which top-k regions survive.

The Student field `v_theta(w, t, z, c)` is trained primarily by cosine
direction matching.  Magnitude and one-step state matching stabilize the
finite-step rollout:

```text
L_gfm = (1 - cos(v_theta, v_T))
        + alpha * SmoothL1(log(1 + ||v_theta||), log(1 + ||v_T||))
        + beta * SmoothL1(Pi_Ck(w + eta_theta v_theta), w_T+).
```

The same field is executed for `K` projected neurodynamic steps:

```text
w^(s+1) = Pi_Ck(w^s + eta_s * v_theta(w^s, s/K, z, c)).
```

There is no separately trained flow trajectory followed by a second ND model.

## Endpoint improvement loss

The requested endpoint loss is

```text
L_improve = mean ReLU(J_T(w^K) - stopgrad(J_T(w^0)) + delta).
```

`stopgrad` is important: without it, the model could reduce the hinge by
making the initial baseline worse.  `delta >= 0` is the requested minimum
improvement margin.  With `delta = 0`, the term is active only when the final
selection is worse than the initial selection.

The total training objective is

```text
L_total = L_task
          + lambda_metric  * L_metric
          + lambda_gfm     * L_gfm
          + lambda_latent  * L_latent
          + lambda_value   * L_value
          + lambda_improve * L_improve.
```

The implementation evaluates `J_T(w^K)` on the straight-through hard top-k
mask, so the forward pass matches the actually retained token set while the
backward pass still reaches the relaxed ND state.

## Files

- `gradient_flow_selector_v85.py`: constrained mask projection, Student
  vector field, and fixed-step ND rollout.
- `train_gradient_flow_selector_v85.py`: Teacher-gradient construction,
  direction matching, `L_improve`, training, and validation.
- `evaluate_gradient_flow_selector_v85.py`: standalone full/random/learned
  comparison for a saved checkpoint.
- `run_v85.sh`: environment-variable-driven training launcher.
- `test_v85_gradient_flow.py`: projection, rollout, and detached-baseline
  unit tests.

## Training

```bash
TRAIN_DATA=/path/to/train.jsonl \
VAL_DATA=/path/to/val.jsonl \
VLM_MODEL=/path/to/Qwen3-VL-8B-Instruct \
bash run_v85.sh /path/to/dcv_v74_decision_value_bottleneck
```

Important controls:

```bash
MASK_STEPS=4
MASK_STEP_SIZE=0.25
TEACHER_STEP_SIZE=0.25
LAMBDA_GFM=1.0
LAMBDA_IMPROVE=0.10
IMPROVE_MARGIN=0.0
```

## Evaluation

```bash
python evaluate_gradient_flow_selector_v85.py \
  --checkpoint checkpoints/v85_gradient_flow.pt \
  --data /path/to/test.jsonl \
  --nuplan-task tasks/nuplan_safe_progress.json \
  --pointnav-task tasks/pointnav_safe_short.json
```

Validation reports physical token counts and task regret for `full`, `random`,
and `learned` masks.  `mask_change` additionally measures how much the learned
fixed-step dynamics changes the initialized relaxed mask.
