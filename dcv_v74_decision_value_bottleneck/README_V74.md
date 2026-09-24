# V7.4 — Counterfactual Downstream Decision-Value Bottleneck

V7.4 changes the supervision target of the information bottleneck.

V7.3 asked the latent to reconstruct an optimization state and then indirectly
produce a decision. The result showed that self-entropy reduction did not imply
good downstream decisions.

V7.4 therefore defines what information is useful directly through the
counterfactual downstream decision value.

## 1. Core target

At acquisition state \(w_t\), for every legal patch \(j\),

\[
V_{t,j}^{soft}
=
R_{soft}(w_t)-R_{soft}(w_t+e_j)
\]

and

\[
V_{t,j}^{hard}
=
R_{hard}(w_t)-R_{hard}(w_t+e_j).
\]

The default privileged training target is

\[
V_{t,j}^{T}
=
V_{t,j}^{soft}
+
0.5V_{t,j}^{hard}.
\]

This is training-only privileged information. It is NOT an inference input.

The learned causal pipeline is

\[
o_t
\rightarrow
Z_t^{DV}
\rightarrow
\hat V_{t,1:M}.
\]

The selected patch is

\[
a_t=\arg\max_j \hat V_{t,j}.
\]

## 2. Why this is an information bottleneck

The latent is stochastic:

\[
q_\theta(z|o_t)
=
\mathcal N(\mu_\theta,\operatorname{diag}(\sigma_\theta^2)).
\]

Compression:

\[
L_{IB}
=
D_{KL}\left[q_\theta(Z|O)\Vert\mathcal N(0,I)\right].
\]

But compression is NOT the primary objective.

The downstream decision-value supervision determines what information must
survive the bottleneck. Thus the operational definition is:

> preserve information that predicts how much observing each visual region will
> improve the downstream optimization decision.

## 3. Training loss

\[
L
=
\lambda_{reg}L_{value-reg}
+
\lambda_{KL}L_{value-policy}
+
\lambda_{rank}L_{rank}
+
\lambda_D L_{decision}
+
\lambda_G L_{gradient-aux}
+
\beta L_{IB}.
\]

### Value regression

Predict the normalized full vector of counterfactual gains.

### Value-policy KL

The Teacher action distribution is

\[
p^T(j|o_t)
\propto
\exp(V^T_{t,j}/T_V).
\]

The Student policy is induced by the predicted values.

### Pairwise ranking

Preserve the ordering of decision value across candidate patches.

### Direct downstream decision loss

For the Student acquisition distribution \(\pi_\theta(j|o_t)\),

\[
L_{decision}
=
\mathbb E_{j\sim\pi_\theta}
\left[
R_{soft}(w_t+e_j)
\right].
\]

This makes downstream decision quality directly differentiable with respect to
the acquisition scores.

### Gradient auxiliary

The old decision gradient is retained only as an auxiliary dense signal, not
the primary Teacher.

## 4. DAgger-style mixed state training

V7.4 no longer trains only on Teacher trajectory states.

At every outer step, training states are sampled from a mixture of:

- stored Teacher states;
- states visited by the current Student policy.

`--onpolicy-start 0.25 --onpolicy-end 0.80`

gradually increases Student-state supervision.

The Student rollout also uses epsilon exploration.

This directly addresses the large offline/on-policy gap seen in earlier
versions.

## 5. Global value head

The value of a patch is not a local visual property.

V7.4 predicts each action value from

\[
[z_j,\ z_{global},\ graph_j,\ w_j,\ t,\ budget].
\]

So the model can represent path competition and relations among candidate
regions without receiving privileged optimization variables as input.

---

# Run

Use the original V7 data and Teacher trajectories:

```bash
BASE=/absolute/path/to/original/V7/project

python train_decision_value_bottleneck_v74.py \
  --data "$BASE/data_v7/train" \
  --val "$BASE/data_v7/val" \
  --teacher "$BASE/teacher_v7/train_15" \
  --teacher-val "$BASE/teacher_v7/val_15" \
  --perception "$BASE/checkpoints/perception_v7.pt" \
  --out checkpoints/v74_decision_value_bottleneck.pt \
  --budget 0.15 \
  --epochs 40 \
  --batch 32 \
  --hidden 256 \
  --latent-dim 32 \
  --layers 4 \
  --heads 8 \
  --hard-gain-weight 0.5 \
  --lambda-value-reg 1.0 \
  --lambda-value-kl 0.5 \
  --lambda-value-rank 0.25 \
  --lambda-decision 1.0 \
  --lambda-grad-aux 0.10 \
  --beta-ib 0.0001 \
  --onpolicy-start 0.25 \
  --onpolicy-end 0.80 \
  --rollout-epsilon 0.10
```

Or:

```bash
bash run_v74.sh "$BASE"
```

## 6. Metrics

### Offline decision-value prediction

- `Value Top1`
- `Value GainRatio`
- `Value Spearman`
- `Actionable-state fraction`
- `Gradient cosine (auxiliary only)`
- `Value-policy entropy`
- per-step Top1 and GainRatio

`GainRatio` is especially important:

\[
\text{GainRatio}
=
\frac{
V^T(a_{student})
}{
\max_j V^T(j)
}
\]

on actionable states.

### Closed-loop

Four policies are reported under exactly the same K-step budget:

- `student`
- `random`
- `gradient_teacher`
- `exact_value`

For each:

- `Hard Regret@K`
- `OptimalPathRate@K`
- full regret trace from \(t=0\) to \(K\)

The primary metric remains

\[
\boxed{HardRegret@K}.
\]

## 7. What counts as success

Do not use gradient cosine as the main gate anymore.

A useful first target is:

1. Student GainRatio clearly above random;
2. Student Regret@K substantially below Random-K;
3. Student approaches Exact-Value / Gradient-Teacher performance;
4. increasing `beta_ib` compresses the latent without materially damaging
   Regret@K.

Only after these are satisfied should Outcome-RL fine-tuning be added.

## 8. First ablation

Run `beta_ib=0` first.

If the architecture cannot learn downstream decision value without compression,
do not tune the information bottleneck.

Then sweep:

```text
0
1e-5
3e-5
1e-4
3e-4
1e-3
```

The desired operating point is the strongest compression that preserves
closed-loop decision performance.

## 9. Interpretation

This version implements the intended research question more directly:

\[
\boxed{
\text{Which visual region contains information that is valuable for the
downstream optimization decision?}
}
\]

It does not equate:

- high raw visual entropy,
- low model entropy,
- accurate edge reconstruction,

with decision usefulness.
