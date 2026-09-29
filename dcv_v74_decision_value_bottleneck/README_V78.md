# V7.8: Qwen3-VL-8B decision-related vision-token pruning

V7.8 replaces the BLIP prototype with a frozen `Qwen3VLForConditionalGeneration`
backbone. It is designed for a local Qwen3-VL-8B or Qwen3-VL-8B-Instruct
checkpoint and keeps the student free of privileged graph features.

## Actual model path

```text
image -> Qwen patch embedding -> first L vision blocks
                                      |
task text -> Qwen token embeddings ---+-> decision-related selector
                                      |
                         soft fixed-budget gate (training)
                         hard group Top-K (deployment)
                                      |
             remaining Qwen vision blocks + visual merger + DeepStack
                                      |
                            frozen Qwen language model
                                      |
                              PathDecisionHead
                                      |
                     expected true path regret / decision KL
```

Qwen3-VL-8B uses a 1152-dimensional vision stream, a 2x2 spatial merger and a
4096-dimensional language model. The controlled input is resized to 192x192,
which gives a 12x12 raw ViT patch grid and exactly 6x6=36 merge groups. A Top-K
decision therefore keeps `4K` raw ViT tokens after the pruning layer and `K`
visual tokens after the merger.

With the default budget 0.15, `K=5`:

- before pruning: 36 groups / 144 raw ViT tokens;
- after hard pruning: 5 groups / 20 raw ViT tokens;
- LLM visual prefix: 5 visual tokens.

Unlike V7.7, pruning happens after the first `--prune-layer` vision blocks and
before all remaining vision blocks. It therefore reduces both later ViT
computation and LLM visual-token computation. The default is 4; it must remain
at or before Qwen's first DeepStack tap so every DeepStack feature follows the
same selected token set.

## Student inputs and loss

The selector receives only:

- early Qwen visual group token `v_i`;
- Qwen task-token embedding `q_j`;
- observable 2-D group position `p_i`;
- budget `b`.

It does **not** receive `graph_feat`, true edge state, future path or true path
cost. Those fields are privileged training supervision used to calculate the
downstream path regret and the fixed-budget DGD target.

The implemented objective is:

```text
L = lambda_task * expected_path_regret
  + lambda_task_kl * path_policy_KL
  + lambda_hard_task * hard_TopK_path_regret
  + lambda_dgd * fixed_budget_DGD
  + lambda_set_value * set_value_regression
  + lambda_set_rank * set_ranking
```

The task loss is primary. There is no model-structure reconstruction loss,
edge-state classifier, Flow Matching or neurodynamic update in V7.8.

## Why there are soft and hard paths

Hard Top-K is discrete and cannot send the path-regret gradient into the
selector. During training, the soft path preserves all 36 groups but repeatedly
gates them through the remaining frozen vision blocks. This supplies a task
gradient to the selector. The hard path physically gathers K spatial groups,
rebuilds the packed ViT sequence and trains/evaluates the deployment path.

The reported `soft_hard_gap` measures the relaxation gap. `hard_regret` must be
compared with `random_regret` under the same exact K-token budget.

## Run

Install dependencies in the environment that already contains the local
Qwen3-VL checkpoint:

```bash
pip install -r requirements.txt
```

BF16 run:

```bash
cd dcv_v74_decision_value_bottleneck
VLM_MODEL=/absolute/path/to/Qwen3-VL-8B-Instruct \
  bash run_v78.sh "$(pwd)"
```

Four-bit frozen backbone (requires CUDA and bitsandbytes):

```bash
VLM_MODEL=/absolute/path/to/Qwen3-VL-8B-Instruct \
LOAD_4BIT=1 BATCH_SIZE=1 \
  bash run_v78.sh "$(pwd)"
```

The saved checkpoint contains only the learned selector and path head, plus
the training arguments and local base-model identifier. The frozen 8B weights
are not duplicated.
