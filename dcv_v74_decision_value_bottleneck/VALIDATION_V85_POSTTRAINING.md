# V8.5 frozen BEV modeling/solver feedback validation

Validation date: 2026-10-09. Branch: `v85-bev-selector-llm-solver`.
Base: published `v85-bev-rl-posttraining` commit `c3172386097807fae9d5a353411609d22bd62893`.

## Executed checks

- `python -m pytest -q`: **54 passed**, 14 non-fatal pre-norm Transformer nested-tensor warnings.
- `python -m compileall -q .`: passed.
- `bash -n run_v85.sh run_v85_rl.sh`: passed.
- Training/evaluation `--help`: passed.
- `git diff --check`: passed.

Environment: Python3.12, torch2.14.1+cpu, numpy2.3.5, scipy1.17.0,
transformers4.57.1. No pretrained model or nuPlan data was downloaded.

## Meaningful coverage

Existing BEV geometry, actor-center labeling, old-data rejection, capped-simplex gradcheck,
hard-pruning/selected-gate parity, ordered no-replacement policy probabilities and stage freezing
remain covered. New regression coverage includes:

- Actual recorded timestamps, no future extrapolation, nonuniform motion and stationary targets.
- Timestamp-free raw cannot serve as trajectory-GT RL data; no reference-path substitute.
- Real SciPy optimization of continuous xy; a halfspace produces a coordinate outside the candidate pool.
- Infeasible solves expose failure and no successful trajectory; invalid/nonfinite/nonconvex JSON rejected.
- GT/candidate-label changes leave frozen modeling output and trajectory identical, changing reward only.
- Model failures incur the specified loss and null ADE/FDE; aggregate failure rates are explicit.
- Actual solver-reward policy gradients update Selector/field with frozen head/reference.
- Full/random/initial/learned use one environment and record problems/trajectories/GT JSONL.
- Metric -> distill -> primary solver-RL CLI checkpoint chain runs with a frozen visual generation stub
  plus the actual SciPy solver (not an argmin substitute).
- A **real randomly initialized miniature Qwen3-VL** in Transformers4.57.1 exercises vision stem,
  physical region gathering, DeepStack, original sparse-grid RoPE chat prefill, and cached greedy decode.
  This tests architecture/API flow, not pretrained visual reasoning or valid optimization generation.

## Not executed and result limits

No pretrained Qwen3-VL-8B/4-bit CUDA run, production nuPlan export, real-data training,
closed-loop simulator evaluation, end-to-end GPU latency/peak-memory measurement or measured
accuracy improvement was performed. Transformers5.x is not validated by this run.

The convex QP is a bounded modeling family using point rear-axle positions and per-axis dynamics.
Occupancy rewards are current-raster centerline proxies, not time-indexed vehicle-footprint safety.
ADE/FDE use this repository's time-aligned rear-axle/16-point protocol, not official nuPlan scoring.
A real-data smoke run must inspect generated JSON, failure rates and solved trajectories before
starting expensive RL training. Old raw without timestamps requires re-export; revision2
processed data/checkpoints require revision3 rebuilding and new warm-up training.
