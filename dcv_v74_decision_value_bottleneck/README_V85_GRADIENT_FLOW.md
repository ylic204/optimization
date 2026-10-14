# V8.5: Teacher-gradient matching and fixed-step neurodynamics

Flow/gradient-field matching and ND are two views of the **same mask field**:
gradient matching trains its local velocity, and projected ND executes it.
There is no separately generated trajectory followed by another ND model.

This revision uses explicit BEV input and three stages:
`metric -> distill -> rl`. See
[the full staged protocol and losses](README_V85_POSTTRAINING.md) and
[the BEV data contract](README_V85_BEV_INPUT.md).

## Teacher direction

On valid regions, `C_k = {w: 0 <= w <= 1, sum(w) = k}`.

```math
w_T^+ = \Pi_{C_k}(w-\eta_T\nabla_w J_{\mathrm{relaxed}}(w)),
\qquad v_T=(w_T^+-w)/\eta_T.
```

A supervised pretrained metric head is frozen during distillation and RL.
Together with frozen Qwen and label-based expected-regret supervision it
provides the mask-space gradient oracle. It is not an independent expert
planner, nor a direct gradient of a physical trajectory cost. Full early
visual features are available before selection; this is not sensor acquisition.

Direction, log-norm and one-step projected-state matching train the field:

```text
L_gfm = (1 - cos(v_theta, v_T))
        + alpha * SmoothL1(log(1 + ||v_theta||), log(1 + ||v_T||))
        + beta * SmoothL1(Pi_Ck(w + eta_theta v_theta), w_T+)
```

The same field runs for S fixed iterations:

```math
w^{s+1}=\Pi_{C_k}(w^s+\eta_s v_\theta(w^s,z,c,s/S)).
```

Positive step sizes are learned. The projection backward centers gradients on
free active-set coordinates, with zero gradients on saturated/invalid entries.

Top-K and the RL sampler both use the last pre-projection proposal as their
decision scores, resolving clipped-mask ties and preserving policy gradients
at binary final masks. Initial endpoint selection uses the selector logits;
all intermediate ND states still satisfy the fixed budget.

## Deployment-matched endpoint improvement

```math
L_{\mathrm{improve}}=\operatorname{ReLU}
(J_{\mathrm{hard}}(w^S)-\operatorname{stopgrad}J_{\mathrm{hard}}(w^0)+m).
```

Both endpoints physically delete unselected tokens. Training uses unit-forward
selected-only gates to provide a biased surrogate gradient through the retained
groups; it no longer leaves 81 zero-filled slots in the deployment objective.
Teacher differentiation still uses a separately identified relaxed-mask proxy.

This is not per-step monotonicity or a mathematical convergence guarantee.
Zero margin permits ties. Detaching the baseline removes its direct gradient
incentive, but shared-parameter updates may still change future baselines.

The distillation objective combines task, metric prediction, gradient matching,
latent alignment, set-value regression and endpoint improvement. Metric-head
parameters are frozen, but the metric loss still backpropagates through its
inputs to Selector/field. Set-value regression is not an RL critic.

## Outcome RL follows distillation

The new RL stage samples budget-valid region subsets with exact ordered-draw
log-probabilities and evaluates actual pruning under no-grad Qwen inference.
The primary environment now generates a validated continuous QP with frozen
Qwen, solves it, and compares the resulting trajectory with time-aligned GT.
Independent labels determine reward. REINFORCE updates the small Selector/ND
modules with a leave-one-out baseline and reference/entropy regularization.

The RL stage is an offline contextual bandit, not PPO/GRPO or closed-loop
driving. Its reward includes the analogous hard-decision endpoint penalty;
that penalty is learned through policy gradients, not fake differentiation
through discrete model generation or the solver. The earlier candidate-regret
reward remains only as the explicit `candidate_proxy` ablation. Teacher gradients
still come from the differentiable proxy during warm-up, not the actual solver.

## Entry points

- `train_gradient_flow_selector_v85.py --stage metric|distill|rl`
- `run_v85.sh` (default metric stage) and `run_v85_rl.sh`
- `evaluate_gradient_flow_selector_v85.py`
- `test_v85_gradient_flow.py` and `test_v85_posttraining.py`

The full commands, checkpoint requirements, reward definitions and remaining
planning/safety limitations are in `README_V85_POSTTRAINING.md`.
