# V7.5 - Joint Decision-Related Visual Token Selection

V7.5 replaces V7.4's monotone, one-patch-at-a-time acquisition policy with a
single fixed-budget token-set decision.

## Problem

For preview visual tokens `H={h_1,...,h_M}`, learn one complete mask

```text
m* = arg min_m L_task(m),  subject to ||m||_0 = K.
```

`L_task` is the downstream path-decision regret.  It is not an optimization
model reconstruction loss.

## Decision-related latent

The selector produces contextual token latents and one joint set of scores:

```text
Z_dec = Transformer(H_preview, graph/task prior, budget)
scores = ScoreHead(Z_dec, GlobalPool(Z_dec))
m_soft = BudgetProjection(scores, K)
m_hard = TopK(scores, K)
```

The latent is trained by four task-facing signals:

1. `L_task`: differentiable downstream path regret under `m_soft`;
2. `L_task_KL`: full-information vs selected-token path-decision distribution;
3. `L_DGD`: task-loss gradient with respect to the complete current mask;
4. `L_set`: predict and rank the downstream task loss of complete K-token sets.

No edge-cost, optimization-state, or model-structure reconstruction target is
used.

## Why Flow Matching and ND are excluded now

V7.5 is deliberately a one-shot joint selector.  Flow Matching and fixed-step
ND should be added only after this baseline satisfies all of the following:

1. validation `hard_regret` is clearly below `random_regret`;
2. the advantage is stable over multiple seeds and budgets;
3. `grad_cos` is positive and set-value ranking is better than chance;
4. remaining errors can be attributed to set interactions that one-shot Top-K
   cannot resolve.

If those conditions hold, the next ablation should add projected fixed-step ND
first.  Flow Matching is justified only when there are useful oracle/teacher
mask trajectories to distill.  Adding both now would confound the latent
learning failure with optimizer dynamics.

## Run

First train the frozen perception model as in V7.4, then run:

```bash
cd dcv_v74_decision_value_bottleneck
bash run_v75.sh /absolute/path/to/dcv_v74_decision_value_bottleneck
```

The primary validation comparison printed every epoch is:

```text
hardR (joint selector)  vs  randomR (same exact K-token budget)
```

The best checkpoint is selected by validation hard task regret, not by latent
reconstruction, gradient cosine, or an intermediate value-prediction metric.

