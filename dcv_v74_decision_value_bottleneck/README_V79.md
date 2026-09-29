# V7.9: spatially continuous path-aligned patches

V7.9 uses `qwen3vl_selector_v79.py` for Qwen3-VL-8B token pruning and replaces
the abstract 33-edge layered graph with a graph that is visible in the image.

## Geometry

The controlled scene uses a 9x9 Qwen merge-region grid. Five hub patches lie
from west to east. Between every two hubs, the upper, middle and lower patch
represent three parallel candidate edges:

```text
                 upper edge patch
                /                \
hub s ---------- middle edge patch ---------- hub s+1
                \                /
                 lower edge patch
```

There are four consecutive stages. A route chooses one of three edge patches
in each stage, so the dataset still contains `3^4 = 81` candidate paths.
With the default 32-pixel tile and Qwen's 2x2 visual merger, one logical scene
patch is exactly one Qwen merge region; no edge is split across selector units.

For every candidate path `p`:

- `path_mask[p]` selects exactly four graph edges;
- `edge_patch[e]` is the unique image patch containing edge `e`;
- `path_patch_sequence[p]` is
  `[hub0, edge0_patch, hub1, ..., edge3_patch, hub4]`;
- every consecutive pair in that sequence is 8-neighbor adjacent.

Thus graph selection and image geometry cannot disagree silently. The
generator calls `validate_spatial_topology` and the test suite verifies all 81
paths.

## Important tensors

| Field | Shape | Meaning |
|---|---:|---|
| `edge_patch` | `[12]` | one-to-one edge to image-patch mapping |
| `path_mask` | `[81, 12]` | four selected edges for each route |
| `path_edge_indices` | `[81, 4]` | ordered edge IDs along each route |
| `path_patch_sequence` | `[81, 9]` | ordered physical patch route including hubs |
| `optimal_edge_patches` | `[4]` | edge patches on the full-information optimum |
| `optimal_path_patches` | `[9]` | continuous optimum including five hubs |

`patch_graph_feat` remains in saved files only for old teacher compatibility.
It is not passed to the Qwen selector.

## Token budget

At `GRID=9`, Qwen receives 81 merge groups (324 raw ViT patches before its 2x2
merger). The default V7.9 budget is `0.075`, which rounds to six groups:

- early vision encoder: 81 groups / 324 raw tokens;
- after hard pruning: 6 groups / 24 raw tokens;
- language-model visual prefix: 6 tokens.

Six is deliberately smaller than the 12 decision-edge patches; otherwise the
selector could keep every edge and the selection problem would be trivial.

## Run

```bash
cd dcv_v74_decision_value_bottleneck
VLM_MODEL=/absolute/path/to/Qwen3-VL-8B-Instruct \
  bash run_v79.sh "$(pwd)"
```

Topology-only tests do not load Qwen weights:

```bash
GRID=9 pytest -q test_v79_spatial_paths.py
```

The dataset entry point is `generate_dataset_v79_standalone.py`. It defines
its task-cost and coarse-oracle functions locally and does not import any
`generate_dataset_v7*` module.

The training entry point is `train_qwen3vl_selector_v79_multigpu.py`. Its
fixed-budget masks, task-gradient teacher and DGD loss are defined locally; it
does not import the V7.2, V7.5 or V7.6 training modules.

## Task instruction from command-line arguments

The trainer does not require `task_text` in each `.npz` sample. The frozen
Qwen language stream receives `--task-text` directly. The corresponding
numerical edge costs are reconstructed with `--risk-normal`, `--risk-rough`,
`--risk-hazard` and `--risk-blocked`, so the text and downstream task loss stay
consistent.

For example:

```bash
TASK_TEXT="Find the minimum-cost path from S to G." \
RISK_NORMAL=0.0 RISK_ROUGH=0.35 RISK_HAZARD=1.0 RISK_BLOCKED=20.0 \
DEVICE=1 \
VLM_MODEL=/data/lyi/models/Qwen3-VL-8B-Instruct \
  bash run_v79_multigpu.sh "$(pwd)"
```

`DEVICE` accepts an integer GPU ID (`0`, `1`, ...), a full device string such
as `cuda:1`, `cpu`, or `auto`. The selected CUDA device is fixed before the
Qwen3-VL checkpoint is loaded.

The learning objective is unchanged: downstream task regret is primary, with
decision-gradient and set-value auxiliary supervision. Flow Matching and ND
are still intentionally deferred.
