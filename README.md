             Visual + Task + Robot State
                         │
                         ▼
        ① Optimization-aware Latent Learning
                         │
                    Z_t^opt
                         │
                         ▼
        ② Decision Information Bottleneck
        保留 decision-relevant information
                         │
                         ▼
        ③ Budgeted Visual Subset Selection
        decision value + embodied priors
                 S_t , |S_t| ≤ K
                         │
                         ▼
             Downstream Optimization
                         │
                         ▼
               Robot Action a_t
                         │
                         ▼
                Environment
                         │
                         ▼
                 O_{t+1}
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
     ④ Outcome-based RL       ⑤ Flow + Fixed-Step ND
     学长期任务策略             学 latent 演化并加速更新

## Current implementation status

- `dcv_v74_decision_value_bottleneck/`: historical V7.4 sequential
  counterfactual-value bottleneck.
- `joint_decision_bottleneck_v75.py` +
  `train_joint_token_selector_v75.py`: current one-shot, fixed-budget visual
  token selector trained by downstream task loss and decision-gradient/set
  supervision.
- Flow Matching and fixed-step ND remain later ablations; they are not enabled
  in V7.5 until the joint latent selector reliably outperforms equal-budget
  random selection.
- `position_c_selector_v76.py` + `train_position_c_selector_v76.py`: position-C
  selection after the full vision encoder. The student has no graph input; it
  uses visual tokens, observable 2-D positions, a frozen-LLM task embedding,
  and the budget, then emits the actual K-token sequence for the downstream
  language model.
- `aligned_vlm_selector_v77.py` + `train_aligned_vlm_selector_v77.py`: frozen
  BLIP alignment baseline. It selects after the complete BLIP vision encoder
  and is retained only as a controlled ablation.
- `qwen3vl_selector_v78.py` + `train_qwen3vl_selector_v78.py`: current mainline
  Qwen3-VL-8B implementation. It learns task-conditioned
  decision latents from early Qwen visual groups and Qwen task-token
  embeddings, then performs group Top-K inside the vision encoder. Remaining
  ViT blocks, the visual merger, DeepStack and the Qwen language model all use
  the same selected visual set. The student still has no graph feature.
- `generate_dataset_v79_standalone.py` + `run_v79.sh`: current experiment entry point.
  V7.9 replaces the abstract graph layout with 81 spatially continuous routes
  on a visible 9x9 corridor. Every one of the 12 graph edges maps one-to-one to
  a real image patch, and every route's ordered patch sequence is validated.
  Training uses `train_qwen3vl_selector_v79_multigpu.py`, which has no
  dependency on older versioned training modules.
  Its Qwen backbone and selector are implemented in `qwen3vl_selector_v79.py`.
