# V7.3 — Decision Information Bottleneck + Outcome Visual RL

This revision replaces the hand-crafted "minimal sufficient representation" ablation
with a learned optimization-aware latent representation.

The key idea is:

\[
\text{causal visual state}
\rightarrow
Z_t^{opt}
\rightarrow
\{\hat c_t,\hat g_t,\hat L_{\rm task}\}.
\]

`Z_t^{opt}` is not manually specified as edge cost / path cost / q.
Instead it is learned under four signals:

\[
L =
\lambda_c L_{\rm cost}
+
\lambda_g L_{\rm GD}
+
\lambda_t L_{\rm task}
+
\beta L_{\rm IB}.
\]

where

- `L_cost`: optimization-state supervision through current edge task cost;
- `L_GD`: decision-gradient distillation;
- `L_task`: differentiable soft downstream decision regret;
- `L_IB`: variational information-bottleneck compression.

The information term is NOT raw image entropy.
Decision information is measured through the entropy of the candidate-path decision
distribution

\[
H(q_t)=-\sum_p q_t(p)\log q_t(p).
\]

The visual-RL stage then uses

\[
r_t =
\lambda_R(R_t-R_{t+1})
+
\lambda_I(H_t-H_{t+1})
-
c_{patch},
\]

with a terminal outcome term

\[
-\lambda_T R_K.
\]

---

## Stage 2B — Learn the Decision Information Bottleneck

Run this first.

```bash
BASE=/absolute/path/to/your/original/V7/project

python train_dib_v73.py \
  --data "$BASE/data_v7/train" \
  --val "$BASE/data_v7/val" \
  --teacher "$BASE/teacher_v7/train_15" \
  --teacher-val "$BASE/teacher_v7/val_15" \
  --perception "$BASE/checkpoints/perception_v7.pt" \
  --out checkpoints/v73_dib.pt \
  --budget 0.15 \
  --epochs 40 \
  --batch 32 \
  --hidden 256 \
  --latent-dim 32 \
  --layers 4 \
  --heads 8 \
  --lambda-cost 1.0 \
  --lambda-grad 1.0 \
  --lambda-task 0.5 \
  --beta-ib 0.001
```

### Main outputs

Offline:

- normalized EdgeCost MAE
- normalized PathCost MAE
- gradient cosine
- hard/soft regret
- Optimal-Path Rate
- VIB KL
- decision entropy

Closed-loop:

- on-policy gradient cosine
- final hard regret
- final Optimal-Path Rate
- decision entropy trace
- decision-information gain per acquisition step
- true regret gain per acquisition step

### Recommended gate before RL

Do NOT run RL merely because training converges.

Prefer:

- learned-gradient closed-loop regret clearly better than the old direct visual Student;
- optimization representation has meaningful edge/path cost accuracy;
- entropy does not collapse trivially at step 0;
- selected patches produce positive average true regret gain.

The final downstream metric remains `Hard Regret@K`.

---

## What the bottleneck is doing

For every visual patch:

\[
q_\theta(z_j|o_t)
=
\mathcal N(\mu_j,\operatorname{diag}(\sigma_j^2)).
\]

Training samples

\[
z_j=\mu_j+\sigma_j\odot\epsilon.
\]

Compression is

\[
L_{\rm IB}
=
D_{KL}
\left[
q_\theta(Z|O)
\|
\mathcal N(0,I)
\right].
\]

This discourages the latent from retaining arbitrary visual detail.
The downstream losses determine which information must survive the bottleneck.

Important:
`beta_ib` must be swept rather than assumed.
Recommended first sweep:

```text
0
1e-4
3e-4
1e-3
3e-3
1e-2
```

This gives the actual information–decision tradeoff.

A good bottleneck is NOT the one with the smallest KL.
It is the smallest-information representation that preserves low decision regret.

---

## Stage 2C — Outcome-based sequential visual RL

Only after Stage 2B is satisfactory:

```bash
python train_latent_outcome_rl_v73.py \
  --data "$BASE/data_v7/train" \
  --val "$BASE/data_v7/val" \
  --dib-checkpoint checkpoints/v73_dib.pt \
  --perception "$BASE/checkpoints/perception_v7.pt" \
  --out checkpoints/v73_latent_outcome_rl.pt \
  --budget 0.15 \
  --epochs 20 \
  --batch 32 \
  --lambda-regret-reward 1.0 \
  --lambda-info-reward 0.10 \
  --lambda-terminal 1.0 \
  --patch-cost 0.0
```

The DIB is frozen in this first RL sanity test.

State:

\[
s_t=(Z_t^{opt},w_t,\mathcal G).
\]

Action:

\[
a_t\in\{j:w_{t,j}=0\}.
\]

Dense reward:

\[
r_t
=
\lambda_R\Delta R_t
+
\lambda_I\Delta H_t.
\]

Terminal reward:

\[
-\lambda_T R_K.
\]

There is no critic and no DPO in this sanity version.

---

## Why entropy is auxiliary rather than the final objective

A high-entropy patch is not automatically useful.
Random texture may contain high visual entropy but no decision value.

Likewise, lowering path entropy can be harmful if the model becomes confidently wrong.

Therefore:

- `Delta H` = decision-information shaping;
- `Delta Regret` = actual decision improvement;
- terminal `-Regret_K` = final outcome criterion.

This makes the RL objective decision-conditioned rather than saliency-conditioned.

---

## Recommended experiment sequence

1. `beta_ib = 0`: verify the new visual -> Zopt -> cost/gradient/task architecture can learn.
2. Sweep `beta_ib`: find the information–decision Pareto region.
3. Compare:
   - Direct Visual Student
   - DIB without IB (`beta=0`)
   - DIB with IB
   - Teacher Gradient
4. Run closed-loop K=5.
5. Only then train Outcome-RL.
6. Flow Matching and fixed-step ND remain disabled until the learned representation + RL path is established.

The immediate scientific question is:

\[
\boxed{
\text{Can a compressed visual latent preserve the information required for downstream optimization decisions?}
}
\]
