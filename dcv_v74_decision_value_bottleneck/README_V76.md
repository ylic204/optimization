# V7.6: Position-C, task-conditioned visual-token selection

V7.6 implements the requested deployment interface:

```text
full vision encoder -> M visual tokens -> decision selector -> K tokens -> LLM
```

The student selector receives only information available at inference:

```text
visual token + 2-D position + frozen-LLM task embedding q + budget
```

It never receives `patch_graph_feat`, candidate paths, true edge costs, or the
optimal solution. Those variables are training-only supervision for task
regret and decision-gradient distillation.

## Why an LLM task encoder is included

Generative VLMs usually reuse the hidden states of a decoder-only LLM rather
than a separate language encoder. `FrozenLLMTaskEncoder` loads any compatible
Hugging Face language backbone, mean-pools its last hidden states, and freezes
it. A small trainable projection maps this task embedding into the selector.

The new dataset has three task profiles (`balanced`, `safety_first`, and
`efficiency_first`) with different risk costs and prompts. Without task
variation, q would be constant and could not teach task-conditioned selection.

## Important scope

`gather_selected_tokens` returns the actual `[B,K,D]` sequence intended for the
downstream LLM. The current controlled experiment still evaluates those tokens
with the repository's path-decision surrogate. It does not yet splice them
into Qwen3-VL internals; that final adapter is model-specific.

For Qwen3-VL specifically, one pruning index must be applied consistently to
the final image embeddings and every DeepStack visual feature. The visual
position mask and MRoPE position IDs must then be rebuilt for the shorter
sequence. Deleting only the final image-embedding rows is not a valid Qwen3-VL
position-C implementation.

## Run

Use a small frozen Qwen language backbone for this controlled test rather than
an 8B model. For an offline local model:

```bash
cd dcv_v74_decision_value_bottleneck
TEXT_MODEL=/absolute/path/to/Qwen3-0.6B bash run_v76.sh "$(pwd)"
```

The LLM is evaluated only once per unique prompt and cached, so the three task
profiles do not repeatedly invoke it during training.

## Ablations required

Compare at least:

1. random K tokens;
2. visual-only selector (zero or remove q);
3. visual + q selector;
4. visual + q with shuffled prompts;
5. V7.5 privileged graph-input selector as an upper-bound diagnostic only.

The principal metric remains hard downstream task regret at exactly K tokens.
