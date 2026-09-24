# V7.2 — Decision-Regret + Gradient-Distillation Capacity Suite

This patch makes two changes before continuing to DAgger / Flow / ND.

## 1. Task objective is defined by task cost, not path ID

For each candidate path P:

C_true(P) = sum_e c_true(e)

The full-information optimum is

C* = min_P C_true(P).

A limited-vision model estimates path costs and selects

P_hat = argmin_P C_hat(P | w_K).

Final evaluation uses the TRUE cost of that selected path:

R_hard = [C_true(P_hat) - C*] / C*.

This `Hard Decision Regret` is the primary downstream metric.

`Optimal-Path Rate` is cost based. A path is counted as optimal whenever
its TRUE task cost equals C* within numerical tolerance. Its path ID does
not need to equal one particular stored `optimal_path_idx`.

### Differentiable training surrogate

Hard argmin is not differentiable, so training uses

q(P|w) = softmax(-C_hat(P|w)/tau)

and

L_soft = [sum_P q(P|w) C_true(P) - C*] / C*.

Teacher gradient:

g_T = Normalize(ReLU(- d L_soft / d w)).

Files:
- `decision.py`
- `evaluate_v72.py`

## 2. Student capacity is no longer fixed to dim=128

`student_v72.py` contains a configurable Transformer Student.

The capacity sweep tests:

- 128 hidden, 2 layers, 4 heads
- 256 hidden, 4 layers, 8 heads
- 512 hidden, 4 layers, 8 heads
- 512 hidden, 6 layers, 8 heads

and two output heads:

- `relu_l2`: sparse nonnegative direction + L2 normalization
- `softmax`: ranking/probability-style output

The improved gradient-distillation objective is

L_GD = lambda_cos (1-cos) + lambda_KL KL(p_T || p_S) + lambda_rank L_rank.

This tests whether the previous ~0.72 same-subset cosine was a capacity /
output-parameterization bottleneck before changing the scientific framework.

## Run the capacity sweep

Use the same 128 fixed training scenes as the previous 1.5-A test:

```bash
python train_student_capacity_sweep_v72.py \
  --data /ABS/PATH/data_v7/train \
  --teacher /ABS/PATH/teacher_v7/train_15 \
  --perception /ABS/PATH/checkpoints/perception_v7.pt \
  --outdir capacity_sweep_v72 \
  --num-samples 128 \
  --epochs 200 \
  --batch 32 \
  --budget 0.15 \
  --lr 1e-3
```

For a faster first pass:

```bash
python train_student_capacity_sweep_v72.py \
  --data /ABS/PATH/data_v7/train \
  --teacher /ABS/PATH/teacher_v7/train_15 \
  --perception /ABS/PATH/checkpoints/perception_v7.pt \
  --outdir capacity_sweep_v72 \
  --num-samples 128 \
  --epochs 120 \
  --batch 32 \
  --budget 0.15 \
  --archs 128x2x4,256x4x8,512x4x8 \
  --heads relu_l2,softmax
```

Primary diagnostic:

- `same_cos >= 0.95`: capacity sanity passes.
- `0.85 <= same_cos < 0.95`: capacity helps but representation still limits fitting.
- `< 0.85` even at 512: do not keep enlarging the model; move to optimization-state representation sanity.

Do not choose the architecture only by cosine. Also inspect:

- `onpol_cos`
- `Optimal-Path Rate`
- `Hard Decision Regret`

The downstream decision metric is more important than exact gradient reconstruction.

## Evaluate a trained V7 checkpoint with revised metrics

```bash
python evaluate_v72.py \
  --data data_v7/test \
  --ckpt checkpoints/v7_outcome_rl_15.pt \
  --budget 0.15
```

It reports:

- Hard Decision Regret (primary)
- Median Hard Decision Regret
- Optimal-Path Rate (cost based)
- Mean selected true task cost
- Mean full-information optimal task cost
- Visual saving

## Visualize the selected patches and final path

For a known selection sequence:

```bash
python visualize_path_decision_v72.py \
  --data data_v7/test \
  --index 0 \
  --selected 2,7,18,23,31 \
  --out sample_decision.png
```

The left panel marks the K selected visual patches. The right panel shows the
controlled graph and compares the limited-information selected path with one
full-information minimum-task-cost path.

If multiple different paths have the same minimum TRUE task cost, all of them
are treated as optimal by the metric.

## Recommended next decision

Run the capacity sweep first.

- If 256/512 raises same-subset cosine to >=0.95 and hard regret approaches the Teacher, keep gradient distillation and use the stronger Student.
- If scaling saturates below ~0.85, the problem is not just model width. Then test explicit optimization-state / path-competition representation before DAgger, Flow, or ND.
