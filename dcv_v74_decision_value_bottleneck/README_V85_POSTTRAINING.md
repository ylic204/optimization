# V8.5: corrected BEV + gradient distillation + outcome RL post-training

## Audit and scope

The upstream `main` used `PlanningDatasetV84` and a generic RGB entry point.
The upstream `v85-bev-input` branch added explicit ego-centric BEV but did
not wire V8.5 into the older V7/V7.3 RL trainers. This revision builds on the
BEV branch and adds a genuine policy-gradient post-training stage.

This is **RL post-training of the Selector and ND field on a frozen VLM**,
not RL fine-tuning of Qwen weights or closed-loop driving RL. Given one
offline scene, the action is a fixed-budget region subset and the outcome
is a chosen candidate trajectory: an offline contextual-bandit formulation.

## Full workflow

1. Export current/past nuPlan map, actors, route and traffic controls into a
   fixed-metric BEV; do not render future ego trajectories or load cameras.
2. Convert raw records to **schema 85 revision 2**, generating candidate
   trajectories and eight independent metric labels. Metric raster adapters
   align x/y conventions without modifying the visual image. Old revision-1
   processed records and BEV checkpoints must not be used for post-training.
3. Split by entire log/scenario; adjacent frames cannot cross train/val/test.
4. `--stage metric`: supervise the metric head on full-token and random
   budgeted observations. Freeze Qwen, Selector and mask field. Use actual
   token deletion, not zero-filled slots. Select checkpoints by random-mask
   validation metric MAE.
5. `--stage distill`: load the metric checkpoint, freeze the metric head and
   Qwen, and train Selector/mask field. Differentiate a relaxed mask task
   objective to obtain a local projected Teacher direction. Distill the same
   field that fixed-step ND executes. Evaluate task loss and `L_improve` on
   physically pruned initial/final Top-K sets. Selected-only unit-forward
   gates provide a **biased surrogate**, not an exact discrete derivative.
6. `--stage rl`: load the distilled checkpoint, keep Qwen/head frozen, and
   sample G >= 2 budget-valid region subsets per scene. Run each sampled
   subset through the actual shortened ViT/merger/LLM stream with no Qwen
   autograd graph; reuse cached early visual features. Compute rewards using
   metric labels, not predicted metrics as self-assigned rewards. Update
   Selector/ND via REINFORCE with a leave-one-out baseline and a frozen
   reference-policy anchor. Sampled outcomes are executed serially.
7. Evaluate deterministic final Top-K against full/random baselines on a
   held-out split. Report selected-trajectory regret, predicted metric MAE,
   safety proxies, and actual end-to-end latency/peak memory. GPU throughput
   and closed-loop safety require separate experiments, not unit tests.

## Shared field and losses

Let `C_k = {w: 0 <= w_i <= 1, sum_i w_i = k}` on valid regions.

```math
w_T^+ = \Pi_{C_k}(w - \eta_T \nabla_w J_{\mathrm{relaxed}}(w)),
\qquad v_T = (w_T^+ - w)/\eta_T.
```

The Teacher is a **fixed pretrained metric head plus frozen Qwen**, evaluated
at a relaxed mask with access to full early visual features and training
labels. It is not a separate all-token expert planner or an expert-trajectory
flow. `J` is label-based expected normalized candidate regret, with candidate
probabilities derived from predicted penalized costs. The Teacher direction
is in mask space, not vehicle-control or Qwen-parameter space.

```math
w^{s+1}=\Pi_{C_k}(w^s+\eta_s v_\theta(w^s,z,c,s/S)).
```

Gradient matching trains this field; ND executes this same field. There is
no `trajectory Flow -> separate ND` cascade. The projection backward now
uses its active-set tangent Jacobian rather than a detached-threshold identity.
At active-set boundaries, the derivative is inherently nonsmooth.

Metric pretraining:

```math
L_{\mathrm{metric-stage}}=J+\lambda_{\mathrm{metric}}L_{\mathrm{metric}},
```

averaged over full and random budgeted observations.

Distillation:

```math
L_{\mathrm{distill}}=L_{\mathrm{task}}
+\lambda_{\mathrm{gfm}}L_{\mathrm{gfm}}
+\lambda_{\mathrm{latent}}L_{\mathrm{latent}}
+\lambda_{\mathrm{value}}L_{\mathrm{value}}
+\lambda_{\mathrm{improve}}L_{\mathrm{improve}}.
```

The metric head is fixed here; `L_metric` is still logged/added as a constant
monitoring term, not a metric-head training signal. `L_gfm` matches direction,
norm and projected next state. `L_value` regresses task loss; it is not a critic
used by the RL stage. Both endpoint costs use hard Top-K deletion:

```math
L_{\mathrm{improve}}=\operatorname{ReLU}
(J_{\mathrm{hard}}(w^S)-\operatorname{stopgrad}J_{\mathrm{hard}}(w^0)+m).
```

This penalizes insufficient **endpoint** improvement. `m=0` allows ties;
neither this loss nor a fixed positive step size guarantees every step descends.
Stop-gradient removes the direct incentive to worsen the baseline, not all
effects of future shared-parameter updates.

## Exact sampled-action probability

The last **pre-projection proposal** supplies scores `h` for a Plackett-Luce
draw of K distinct regions; all ND states themselves remain projected and
budget-feasible. The same proposal supplies deterministic deployment Top-K.
This avoids arbitrary saturated-mask ties and a zero policy Jacobian when
the final projected mask is entirely binary. With zero ND steps, the initial
selector logits are used. For an ordered action `a=(a_1,...,a_k)`:

```math
\log\pi_\theta(a\mid o)=\sum_{j=1}^k
\log\frac{\exp(h_{a_j}/\tau)}
{\sum_{i\notin\{a_1,...,a_{j-1}\}}\exp(h_i/\tau)}.
```

Order is an auxiliary action variable: Qwen receives the selected indices in
spatial order. The code records the joint probability of the **ordered draw**,
not a falsely claimed unordered-set probability. A subset's outcome is
order-invariant, so score-function learning over draws is valid. Sampling
without replacement is not sequential patch acquisition: there is no new
observation, Qwen call or ND state update between individual draws.

Let `j(a)` be the candidate chosen by the frozen predicted-cost head. Reward
uses independent labels for that chosen candidate:

```math
r_g=-\mathrm{regret}_{\mathrm{true}}(j(a_g))
-\lambda_{\mathrm{violation}}\mathbf1[\text{labelled limit violation}]
-\lambda_{\mathrm{improve}}\operatorname{ReLU}
(\mathrm{regret}_g-\operatorname{stopgrad}\mathrm{regret}_{\mathrm{initial}}+m).
```

The same hard decision regret is used at both RL endpoints; this differs from
distillation's differentiable expected-regret objective. Reward limits are
still dataset proxies, not a safety certificate. At fixed K a token-count
penalty is constant, so none is advertised as learning efficiency.

For independently sampled outcomes, `A_g = r_g - mean_{h != g}(r_h)`.
Rewards and advantages are detached; no learned critic is required.

```math
L_{\mathrm{RL}}=-\operatorname{mean}_g[A_g\log\pi_\theta(a_g\mid o)]
+\beta\overline{D}_{\mathrm{conditional}}
-\alpha\overline{H}_{\mathrm{conditional}}.
```

KL and entropy are averaged over the K conditional categorical decisions
along sampled draws; the KL is a reference-anchor surrogate, not asserted to
be the exact unordered-set KL. The reference freezes the small modules from
the starting distilled checkpoint; it does not duplicate the 8B backbone.
This implementation is **REINFORCE, not PPO/GRPO**: no old-policy ratio,
clipped multi-epoch objective or claim of GRPO reproduction is made.

## Commands

From `dcv_v74_decision_value_bottleneck`, after regenerating processed data:

```bash
export TRAIN_DATA=/data/v85_bev/processed/train
export VAL_DATA=/data/v85_bev/processed/val
export VLM_MODEL=/data/lyi/models/Qwen3-VL-8B-Instruct
export DEVICE=0
export LOAD_4BIT=1

STAGE=metric EPOCHS=10 bash run_v85.sh "$PWD" \
  --output checkpoints/v85_bev_metric.pt

STAGE=distill EPOCHS=20 RESUME=checkpoints/v85_bev_metric.pt \
  bash run_v85.sh "$PWD" --output checkpoints/v85_bev_distill.pt

EPOCHS=10 RESUME=checkpoints/v85_bev_distill.pt RL_GROUP_SIZE=4 \
  bash run_v85_rl.sh "$PWD" --lr 5e-5 --output checkpoints/v85_bev_rl.pt

python evaluate_gradient_flow_selector_v85.py \
  --checkpoint checkpoints/v85_bev_rl.pt \
  --data /data/v85_bev/processed/test \
  --nuplan-task tasks/nuplan_safe_progress.json \
  --pointnav-task tasks/pointnav_safe_short.json \
  --device cuda:0 --load-4bit
```

Hyperparameters/epochs above are starting points, not measured optimal values.
Keep model architecture, model weights, BEV geometry, budget and task JSON
consistent between stages. `--resume` initializes weights; it does **not**
restore optimizer/RNG/data-loader state for an exact interrupted-run resume.
Cross-view initialization is only an explicit option for metric pretraining.
Stage, geometry, architecture and label revision are checked before loading
resumed modules. CLI default stage is `metric`; RL cannot silently start fresh.

## Tests and remaining limits

```bash
python -m pytest -q test_v85_posttraining.py test_v85_bev.py \
  test_v85_gradient_flow.py test_v84_data.py test_v79_spatial_paths.py
bash -n run_v85.sh run_v85_rl.sh
```

CPU tests cover raster conversion and actor-center detection, processed-label
alignment, stale-data/checkpoint rejection, projection finite-difference
gradients, binary-mask policy-gradient continuity, hard-pruning forward parity,
ordered-action probabilities and
budget validity, external rewards, a rewarded-action policy-gradient update,
and all three training loops with small substitute components. They do not
load actual Qwen weights or exercise the nuPlan simulator.

Remaining research limitations:

- Collision labels use centerline samples and present obstacles, not swept
  vehicle footprints or time-indexed future actor motion.
- Traffic-light semantics are rendered, but there is no explicit red-light
  violation label; history is not itself a dynamics predictor.
- Discomfort is a curvature proxy, not measured acceleration/jerk.
- Exported expert trajectories/SDF are not fed into the current main losses.
- Soft penalty objectives do not guarantee safe feasible trajectories.
- All regions are processed by early ViT before selection; only later ViT,
  merger/DeepStack and LLM work can be reduced. ND and multiple RL rollouts
  also cost compute. Memory/latency claims require actual GPU measurement.
- Geometry repair invalidates old labels; RL cannot compensate for incorrect
  ground truth. Better labels and closed-loop rewards are a separate extension.

API background: [PyTorch score-function / distributions documentation](https://docs.pytorch.org/docs/stable/distributions.html).
For the distinct PPO/GRPO formulation, see [DeepSeekMath](https://arxiv.org/abs/2402.03300).
