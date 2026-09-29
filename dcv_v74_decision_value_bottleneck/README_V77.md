# V7.7: aligned-VLM decision-aware position-C selection

V7.7 corrects the modality mismatch in V7.6. The vision and text features now
come from the same frozen BLIP model and use its pretrained visual/text
projection layers.

```text
image -> BLIP vision encoder -> raw region tokens -------+-> gather Top-K
                              -> aligned region tokens --|
task  -> BLIP text encoder   -> aligned text tokens -----+-> cross-modal selector
```

The student inputs are only deployable observations:

- aligned visual region tokens;
- aligned task-text tokens;
- observable 2-D region coordinates;
- token budget.

There is no graph feature in the student. Graph/path/true-cost fields are used
only to calculate training task regret and the privileged DGD target.

## Why ITC/ITM are not retrained here

BLIP already learned image-text alignment during pretraining. The three task
prompts are optimization instructions, not one-to-one image captions, and they
repeat across a batch. Standard diagonal ITC would create false negatives;
standard ITM would incorrectly call another valid objective a mismatch.

The frozen VLM therefore supplies the aligned space, while V7.7 trains only:

```text
task regret + task decision KL + fixed-budget DGD + token-set value/ranking
```

After selection, the visual tokens enter BLIP's pretrained text-vision
cross-attention. A small `PathDecisionHead` converts the fused task state into
candidate-path logits and is trained only by expected true path regret and
decision-distribution KL. There is no edge-state classification or
optimization-structure reconstruction loss.

Training uses a soft fixed-budget mask to pass task gradients into the
selector, plus a hard Top-K forward pass to train the downstream path head on
the same K-token sequence used at evaluation. `soft_hard_gap` records their
distribution mismatch.

## Spatial region pooling

The controlled dataset supervises 6x6 regions. BLIP produces a finer visual
patch grid, so V7.7 pools the frozen BLIP patch tokens into 6x6 raw/aligned
region tokens. The selector keeps K of these 36 tokens. A later real-image VLM
experiment can select native VLM tokens directly.

V7.7 also makes the controlled edge-to-region layout observable through fixed
2-D positions. The older random hidden permutation is impossible to infer once
`graph_feat` is removed. This fixed topology is only a proof-of-concept; a real
robot should provide observable local-map coordinates and pose, never a GT
future graph feature.

## Flow Matching and ND

They are intentionally excluded. First establish that this aligned selector
beats equal-budget random, visual-only, and shuffled-text baselines. The next
isolated extension should be projected fixed-step ND. Flow Matching should be
added only after defining a reliable teacher trajectory over fixed-budget
masks or decision latents.

## Run

Use a locally downloaded BLIP checkpoint, for example an image-text aligned
BLIP base checkpoint compatible with `transformers.BlipModel`:

```bash
cd dcv_v74_decision_value_bottleneck
VLM_MODEL=/absolute/path/to/blip-model bash run_v77.sh "$(pwd)"
```

The main comparison is validation `hard_regret` against `random_regret`, both
at the same exact K-region budget.
