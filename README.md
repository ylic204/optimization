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
