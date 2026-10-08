# V8.5 BEV/RL repair validation

Reviewed upstream: `v85-bev-input` at
`571d9feca3a63e86f40044207981763016afdcc3`.
The isolated local snapshot tree matched upstream
`4cc6312d2bc225510530142729c851e53c24a54d` before editing.

Validation date: 2026-10-08. CPU environment: Python 3.12,
PyTorch 2.14.1+cpu, NumPy 2.3.5, SciPy 1.17.0; CUDA unavailable.

## Executed checks

From `dcv_v74_decision_value_bottleneck`:

```bash
python -m pytest -q
python train_gradient_flow_selector_v85.py --help
python evaluate_gradient_flow_selector_v85.py --help
```

From the repository root:

```bash
python -m compileall -q dcv_v74_decision_value_bottleneck
bash -n dcv_v74_decision_value_bottleneck/run_v85.sh \
  dcv_v74_decision_value_bottleneck/run_v85_rl.sh
git diff --check
```

Result: **39 pytest tests passed**. Eight warnings report disabled nested-tensor
optimization for pre-norm Transformer encoders; they are not failed assertions.
Python compilation, launcher syntax and whitespace checks passed.

New regressions cover:

- Actor at ego `(15, 1.5)` is detected by corrected BEV sampling; the old
  direct V8.4 mapping misses the actor center.
- BEV/legacy raster conversion, corrected processed collision labels and
  rejection of stale revision-1 data.
- Projection finite-difference Jacobian, zero saturated/invalid-coordinate
  gradients and policy gradients at entirely binary projected masks.
- Physically shortened hard-pruning streams and equality of unit-forward
  selected gates versus deployment forward output.
- Exact ordered-draw probability normalization, fixed budget, no replacement,
  invalid-token exclusion, leave-one-out advantages and external-label rewards.
- Policy-gradient update raises the rewarded action's score.
- Small-component metric/distillation/RL loops run with finite losses;
  the metric head stays fixed in later stages and reference parameters stay frozen.
- CLI checkpoint chain parses arguments, trains, validates, saves and reloads
  all three stages with revision/geometry/stage checks and safety-proxy logs.

## Not validated here

No real Qwen3-VL checkpoint, 4-bit CUDA execution, full nuPlan export,
production dataset training, end-to-end GPU memory/latency benchmark or
closed-loop driving simulation was executed. Tiny substitute components test
training and checkpoint contracts, not real-model planning quality.

The stage chain is implemented, but **no trained model or measured planning
improvement is delivered with this code commit**. Existing centerline/current-
obstacle labels remain proxies; red-light, swept-footprint and temporal actor
safety require further labels/simulation. See `README_V85_POSTTRAINING.md`.
