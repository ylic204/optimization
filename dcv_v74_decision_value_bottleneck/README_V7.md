# V7 — Decision-Gradient Distilled Flow + Fixed-Step Neurodynamic Acceleration + Outcome Visual RL

## Core idea

V7 removes DPO. The pipeline is:

**downstream outcome → decision gradient → gradient distillation → Flow Matching → fixed-step neurodynamic acceleration → outcome-based visual RL**.

The roles are deliberately separated:

- **Decision gradient** answers *which direction in visual-information space reduces downstream optimization loss?*
- **Gradient distillation** transfers that expensive privileged direction to the lightweight student.
- **Flow Matching** learns how the useful gradient direction changes when newly acquired high-resolution information changes the optimization problem.
- **Neurodynamics** learns fixed-step step sizes and inertia to reach the new gradient direction in fewer inner iterations.
- **Outcome RL** uses only the final shortest-path result to correct the visual policy.

There is no STOP action, no DPO, no critic, and no pairwise preference loss.

## Teacher gradient

At binary visual state `w_t`, the privileged differentiable teacher computes

`g_t = normalize(ReLU(-∂L_dec/∂w))`.

This is a dense `M`-dimensional direction, not a winner/rejected pair.

## Gradient-distilled Flow Matching

For consecutive states, use `g_{t-1}` as warm start and `g_t` as the new target after the optimization problem has changed:

`z_tau = (1-tau) g_{t-1} + tau g_t`

`u* = g_t - g_{t-1}`

`L_FM = ||v_phi(z_tau, state_t, tau) - u*||^2`.

The student fixed-step output `z_K` is directly distilled toward the teacher gradient:

`L_GD = 1 - cos(z_K, g_t) + 0.25 ||z_K-g_t||^2`.

## Fixed-step neurodynamic acceleration

The learned integrator is

`z^{k+1} = z^k + alpha_k v_phi^k + beta_k (z^k-z^{k-1})`.

`alpha_k` and `beta_k` are trained jointly with the Flow field. The default is only `K_ND=4` inner steps.

`benchmark_nd_v7.py` compares the learned Flow-ND integrator against plain Flow-Euler integration with `1,2,4,8,16,32` steps. It reports how many Euler steps are required to match the gradient-direction accuracy of the fixed-step learned ND. This is the intended TPAMI-style acceleration test.

## Stage-I objective

`L = lambda_FM L_FM + lambda_GD L_GD + lambda_task L_task`.

`L_task` trains the cheap Preview representation only through downstream decision regret. There is no preview classification auxiliary loss.

## Outcome-based visual RL

Stage II uses only the final path outcome:

`R = -Regret(final shortest path)`.

A simple terminal policy-gradient update is used. There is no step reward, DPO pair, STOP action, or critic.

## No visual leakage

Policy-visible visual content is always causal:

- acquired patch: high-resolution feature;
- unacquired patch: preview feature only.

The continuous variable used to differentiate the privileged teacher is an abstract information-fidelity variable only; it never mixes unseen high-resolution image features into the student input.

## Final metrics

`evaluate_v7.py` prints only:

1. **SelectionAcc@1** — did the model choose a region whose exact one-step downstream gain is maximal?
2. **PathAcc** — after the fixed visual budget, is the final route truly optimal?
3. **Regret** — true cost gap from the optimum.
4. **Saving** — fraction of high-resolution visual tokens not processed.

The ND acceleration result is printed separately by `benchmark_nd_v7.py`.

## Run

```bash
bash run_v7_from_scratch.sh 0.15 20 40 20 32
```

Quick smoke test:

```bash
N_TRAIN=64 N_VAL=32 N_TEST=32 bash run_v7_from_scratch.sh 0.15 1 2 2 16
```

## Scaling beyond 36 visual regions

`36` is only `GRID=6 -> 6x6`. The V7 code supports larger candidate sets:

```bash
GRID=8 bash run_v7_from_scratch.sh 0.10 20 40 20 32   # M=64
GRID=10 bash run_v7_from_scratch.sh 0.10 20 40 20 32  # M=100
```

The 33 downstream graph edges are randomly embedded into the `M` visual regions; remaining regions are distractors. Thus `M` can be increased independently for controlled scalability studies.
