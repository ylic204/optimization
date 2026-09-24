import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from dataset import DCVDataset
from models import DualResolutionPerception
from decision import (
    move_batch,
    causal_state_features,
    legal_mask,
    semantic_hard_decision,
)
from dib_model_v73 import (
    DecisionInformationBottleneck,
    categorical_entropy,
)
from train_dib_v73 import (
    load_perception,
    freeze_high_resolution,
    predicted_decision_terms,
)


def safe_mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


class LatentAcquisitionPolicy(nn.Module):
    """
    Policy over visual regions based only on the learned optimization-aware
    bottleneck Z_opt, graph features, and acquisition mask.
    """

    def __init__(
        self,
        latent_dim,
        graph_dim=4,
        hidden=128,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                latent_dim
                + graph_dim
                + 1,
                hidden,
            ),
            nn.GELU(),
            nn.Linear(
                hidden,
                hidden,
            ),
            nn.GELU(),
            nn.Linear(
                hidden,
                1,
            ),
        )

    def forward(
        self,
        z,
        graph_feat,
        w,
        eligible,
    ):
        x = torch.cat(
            [
                z,
                graph_feat,
                w[
                    ...,
                    None,
                ],
            ],
            dim=-1,
        )

        logits = self.net(
            x
        ).squeeze(
            -1
        )

        return logits.masked_fill(
            ~eligible,
            -1e9,
        )


def load_dib_checkpoint(
    checkpoint,
    perception_path,
    device,
):
    ckpt = torch.load(
        checkpoint,
        map_location=device,
    )

    args = ckpt[
        "args"
    ]

    perception = load_perception(
        perception_path,
        device,
    )

    freeze_high_resolution(
        perception
    )

    if "preview_encoder" in ckpt:
        perception.preview_encoder.load_state_dict(
            ckpt[
                "preview_encoder"
            ]
        )

    dib = DecisionInformationBottleneck(
        feat_dim=CFG.feat_dim,
        graph_dim=4,
        hidden=int(
            args[
                "hidden"
            ]
        ),
        latent_dim=int(
            args[
                "latent_dim"
            ]
        ),
        layers=int(
            args[
                "layers"
            ]
        ),
        heads=int(
            args[
                "heads"
            ]
        ),
    ).to(
        device
    )

    dib.load_state_dict(
        ckpt[
            "dib"
        ]
    )

    dib.eval()
    perception.eval()

    for p in dib.parameters():
        p.requires_grad_(
            False
        )

    for p in perception.parameters():
        p.requires_grad_(
            False
        )

    return (
        dib,
        perception,
        int(
            args[
                "latent_dim"
            ]
        ),
    )


@torch.no_grad()
def latent_and_decision(
    dib,
    perception,
    batch,
    preview_feat,
    high_feat,
    w,
    outer,
    budget,
):
    state_feat = causal_state_features(
        preview_feat,
        high_feat,
        w,
    )

    eligible = legal_mask(
        batch,
        w,
    )

    out = dib(
        state_feat=state_feat,
        graph_feat=batch[
            "patch_graph_feat"
        ],
        w_binary=w,
        outer_t=outer,
        budget_frac=budget,
        eligible=eligible,
        base_cost=batch[
            "base_cost"
        ],
        edge_patch=batch[
            "edge_patch"
        ],
        sample=False,
    )

    dec = predicted_decision_terms(
        out[
            "pred_edge_cost"
        ],
        batch,
    )

    return (
        out[
            "mu"
        ],
        eligible,
        dec,
    )


def train_epoch(
    policy,
    dib,
    perception,
    loader,
    optimizer,
    args,
    device,
):
    policy.train()

    K = CFG.visual_budget_k(
        args.budget
    )

    losses = []
    rewards_all = []
    final_regrets = []
    entropy_gains = []
    regret_gains = []

    for batch in tqdm(
        loader,
        desc="Outcome-RL train",
    ):
        batch = move_batch(
            batch,
            device,
        )

        B = batch[
            "image"
        ].shape[
            0
        ]

        M = CFG.n_patches

        rows = torch.arange(
            B,
            device=device,
        )

        with torch.no_grad():
            high_feat = perception.encode_all_high(
                batch[
                    "image"
                ]
            )

            preview_feat = perception.encode_preview(
                batch[
                    "image"
                ]
            )

        budget = torch.full(
            (
                B,
            ),
            float(
                args.budget
            ),
            device=device,
        )

        w = torch.zeros(
            B,
            M,
            device=device,
        )

        log_probs = []
        policy_entropies = []
        rewards = []

        with torch.no_grad():
            prev_true_regret = semantic_hard_decision(
                w,
                batch,
            )[
                "hard_regret"
            ]

            outer0 = torch.zeros(
                B,
                device=device,
            )

            _, _, dec0 = latent_and_decision(
                dib,
                perception,
                batch,
                preview_feat,
                high_feat,
                w,
                outer0,
                budget,
            )

            prev_dec_entropy = dec0[
                "decision_entropy"
            ]

        for t in range(
            K
        ):
            outer = torch.full(
                (
                    B,
                ),
                t
                / max(
                    K - 1,
                    1,
                ),
                device=device,
            )

            with torch.no_grad():
                z, eligible, _ = latent_and_decision(
                    dib,
                    perception,
                    batch,
                    preview_feat,
                    high_feat,
                    w,
                    outer,
                    budget,
                )

            logits = policy(
                z.detach(),
                batch[
                    "patch_graph_feat"
                ],
                w,
                eligible,
            )

            dist = torch.distributions.Categorical(
                logits=logits
            )

            action = dist.sample()

            log_probs.append(
                dist.log_prob(
                    action
                )
            )

            policy_entropies.append(
                dist.entropy()
            )

            w_next = w.clone()

            w_next[
                rows,
                action,
            ] = 1.0

            with torch.no_grad():
                next_true_regret = semantic_hard_decision(
                    w_next,
                    batch,
                )[
                    "hard_regret"
                ]

                next_outer = torch.full(
                    (
                        B,
                    ),
                    min(
                        t + 1,
                        K - 1,
                    )
                    / max(
                        K - 1,
                        1,
                    ),
                    device=device,
                )

                _, _, next_dec = latent_and_decision(
                    dib,
                    perception,
                    batch,
                    preview_feat,
                    high_feat,
                    w_next,
                    next_outer,
                    budget,
                )

                next_dec_entropy = next_dec[
                    "decision_entropy"
                ]

                d_regret = (
                    prev_true_regret
                    - next_true_regret
                )

                d_info = (
                    prev_dec_entropy
                    - next_dec_entropy
                )

                reward = (
                    args.lambda_regret_reward
                    * d_regret
                    + args.lambda_info_reward
                    * d_info
                    - args.patch_cost
                )

            rewards.append(
                reward
            )

            rewards_all.extend(
                reward.cpu()
                .tolist()
            )

            entropy_gains.extend(
                d_info.cpu()
                .tolist()
            )

            regret_gains.extend(
                d_regret.cpu()
                .tolist()
            )

            prev_true_regret = next_true_regret
            prev_dec_entropy = next_dec_entropy
            w = w_next

        # Terminal outcome reward.
        terminal_regret = semantic_hard_decision(
            w,
            batch,
        )[
            "hard_regret"
        ].detach()

        rewards[
            -1
        ] = (
            rewards[
                -1
            ]
            - args.lambda_terminal
            * terminal_regret
        )

        final_regrets.extend(
            terminal_regret.cpu()
            .tolist()
        )

        # Reward-to-go.
        returns = []

        running = torch.zeros(
            B,
            device=device,
        )

        for r in reversed(
            rewards
        ):
            running = (
                r
                + args.gamma
                * running
            )

            returns.append(
                running
            )

        returns.reverse()

        returns = torch.stack(
            returns,
            dim=1,
        )

        logp = torch.stack(
            log_probs,
            dim=1,
        )

        ent = torch.stack(
            policy_entropies,
            dim=1,
        )

        # Batch/time standardized advantage. No critic.
        advantage = (
            returns
            - returns.mean()
        ) / returns.std().clamp_min(
            1e-6
        )

        loss_pg = -(
            advantage.detach()
            * logp
        ).mean()

        loss_entropy = -ent.mean()

        loss = (
            loss_pg
            + args.entropy_coef
            * loss_entropy
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            5.0,
        )

        optimizer.step()

        losses.append(
            float(
                loss.detach()
                .cpu()
            )
        )

    return {
        "loss":
            safe_mean(
                losses
            ),

        "reward":
            safe_mean(
                rewards_all
            ),

        "final_semantic_regret":
            safe_mean(
                final_regrets
            ),

        "information_gain":
            safe_mean(
                entropy_gains
            ),

        "true_regret_gain":
            safe_mean(
                regret_gains
            ),
    }


@torch.no_grad()
def evaluate(
    policy,
    dib,
    perception,
    loader,
    args,
    device,
):
    policy.eval()

    K = CFG.visual_budget_k(
        args.budget
    )

    semantic_regrets = []
    semantic_optimal = []

    dib_regrets = []
    dib_optimal = []

    info_gain = []
    regret_gain = []

    entropy_trace = [
        []
        for _ in range(
            K + 1
        )
    ]

    for batch in tqdm(
        loader,
        desc="Outcome-RL val",
    ):
        batch = move_batch(
            batch,
            device,
        )

        B = batch[
            "image"
        ].shape[
            0
        ]

        M = CFG.n_patches

        rows = torch.arange(
            B,
            device=device,
        )

        high_feat = perception.encode_all_high(
            batch[
                "image"
            ]
        )

        preview_feat = perception.encode_preview(
            batch[
                "image"
            ]
        )

        budget = torch.full(
            (
                B,
            ),
            float(
                args.budget
            ),
            device=device,
        )

        w = torch.zeros(
            B,
            M,
            device=device,
        )

        outer0 = torch.zeros(
            B,
            device=device,
        )

        _, _, dec = latent_and_decision(
            dib,
            perception,
            batch,
            preview_feat,
            high_feat,
            w,
            outer0,
            budget,
        )

        prev_entropy = dec[
            "decision_entropy"
        ]

        prev_regret = semantic_hard_decision(
            w,
            batch,
        )[
            "hard_regret"
        ]

        entropy_trace[
            0
        ].extend(
            prev_entropy.cpu()
            .tolist()
        )

        for t in range(
            K
        ):
            outer = torch.full(
                (
                    B,
                ),
                t
                / max(
                    K - 1,
                    1,
                ),
                device=device,
            )

            z, eligible, _ = latent_and_decision(
                dib,
                perception,
                batch,
                preview_feat,
                high_feat,
                w,
                outer,
                budget,
            )

            logits = policy(
                z,
                batch[
                    "patch_graph_feat"
                ],
                w,
                eligible,
            )

            action = logits.argmax(
                dim=-1
            )

            w = w.clone()

            w[
                rows,
                action,
            ] = 1.0

            next_outer = torch.full(
                (
                    B,
                ),
                min(
                    t + 1,
                    K - 1,
                )
                / max(
                    K - 1,
                    1,
                ),
                device=device,
            )

            _, _, next_dec = latent_and_decision(
                dib,
                perception,
                batch,
                preview_feat,
                high_feat,
                w,
                next_outer,
                budget,
            )

            next_entropy = next_dec[
                "decision_entropy"
            ]

            next_regret = semantic_hard_decision(
                w,
                batch,
            )[
                "hard_regret"
            ]

            info_gain.extend(
                (
                    prev_entropy
                    - next_entropy
                ).cpu()
                .tolist()
            )

            regret_gain.extend(
                (
                    prev_regret
                    - next_regret
                ).cpu()
                .tolist()
            )

            entropy_trace[
                t + 1
            ].extend(
                next_entropy.cpu()
                .tolist()
            )

            prev_entropy = next_entropy
            prev_regret = next_regret

        sem = semantic_hard_decision(
            w,
            batch,
        )

        semantic_regrets.extend(
            sem[
                "hard_regret"
            ].cpu()
            .tolist()
        )

        semantic_optimal.extend(
            sem[
                "optimal"
            ].float()
            .cpu()
            .tolist()
        )

        outerf = torch.ones(
            B,
            device=device,
        )

        _, _, final_dec = latent_and_decision(
            dib,
            perception,
            batch,
            preview_feat,
            high_feat,
            w,
            outerf,
            budget,
        )

        dib_regrets.extend(
            final_dec[
                "hard_regret"
            ].cpu()
            .tolist()
        )

        dib_optimal.extend(
            final_dec[
                "optimal"
            ].float()
            .cpu()
            .tolist()
        )

    return {
        "semantic_selected_set_regret":
            safe_mean(
                semantic_regrets
            ),

        "semantic_selected_set_optimal_rate":
            safe_mean(
                semantic_optimal
            ),

        "dib_final_decision_regret":
            safe_mean(
                dib_regrets
            ),

        "dib_final_optimal_rate":
            safe_mean(
                dib_optimal
            ),

        "mean_information_gain":
            safe_mean(
                info_gain
            ),

        "mean_true_regret_gain":
            safe_mean(
                regret_gain
            ),

        "decision_entropy_trace":
            [
                safe_mean(
                    x
                )
                for x in entropy_trace
            ],
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data",
        required=True,
    )

    ap.add_argument(
        "--val",
        required=True,
    )

    ap.add_argument(
        "--dib-checkpoint",
        required=True,
    )

    ap.add_argument(
        "--perception",
        required=True,
    )

    ap.add_argument(
        "--out",
        default="checkpoints/v73_latent_outcome_rl.pt",
    )

    ap.add_argument(
        "--budget",
        type=float,
        default=0.15,
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=32,
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    ap.add_argument(
        "--lambda-regret-reward",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--lambda-info-reward",
        type=float,
        default=0.10,
    )

    ap.add_argument(
        "--lambda-terminal",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--patch-cost",
        type=float,
        default=0.0,
        help=(
            "With fixed K this can stay 0. "
            "Use >0 only when later adding variable-budget/STOP."
        ),
    )

    ap.add_argument(
        "--gamma",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--entropy-coef",
        type=float,
        default=1e-3,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=4321,
    )

    args = ap.parse_args()

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_loader = DataLoader(
        DCVDataset(
            args.data
        ),
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        DCVDataset(
            args.val
        ),
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
    )

    dib, perception, latent_dim = load_dib_checkpoint(
        args.dib_checkpoint,
        args.perception,
        device,
    )

    policy = LatentAcquisitionPolicy(
        latent_dim=latent_dim,
        graph_dim=4,
        hidden=128,
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=CFG.wd,
    )

    Path(
        args.out
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best = None
    best_regret = float(
        "inf"
    )

    for ep in range(
        args.epochs
    ):
        tr = train_epoch(
            policy,
            dib,
            perception,
            train_loader,
            optimizer,
            args,
            device,
        )

        va = evaluate(
            policy,
            dib,
            perception,
            val_loader,
            args,
            device,
        )

        print(
            f"epoch={ep:02d} "
            f"loss={tr['loss']:.4f} "
            f"reward={tr['reward']:.4f} "
            f"train_sem_regret={tr['final_semantic_regret']:.4f} "
            f"IG={tr['information_gain']:.4f} "
            f"dR={tr['true_regret_gain']:.4f} | "
            f"val_sem_regret={va['semantic_selected_set_regret']:.4f} "
            f"val_dib_regret={va['dib_final_decision_regret']:.4f} "
            f"val_opt={100*va['dib_final_optimal_rate']:.2f}%"
        )

        if va[
            "dib_final_decision_regret"
        ] < best_regret:
            best_regret = va[
                "dib_final_decision_regret"
            ]

            best = va

            torch.save(
                {
                    "policy":
                        policy.state_dict(),

                    "metrics":
                        va,

                    "args":
                        vars(
                            args
                        ),

                    "version":
                        "v73_latent_outcome_visual_rl",
                },
                args.out,
            )

    Path(
        args.out
    ).with_suffix(
        ".json"
    ).write_text(
        json.dumps(
            best,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "V7.3 LATENT OUTCOME-RL — BEST RESULT"
    )

    print(
        "============================================================"
    )

    print(
        f"Selected-set true regret       = "
        f"{best['semantic_selected_set_regret']:.4f}"
    )

    print(
        f"Selected-set optimal rate      = "
        f"{100*best['semantic_selected_set_optimal_rate']:.2f}%"
    )

    print(
        f"DIB final decision regret      = "
        f"{best['dib_final_decision_regret']:.4f}"
    )

    print(
        f"DIB final optimal rate         = "
        f"{100*best['dib_final_optimal_rate']:.2f}%"
    )

    print(
        f"Mean decision information gain = "
        f"{best['mean_information_gain']:.4f}"
    )

    print(
        f"Mean true regret gain          = "
        f"{best['mean_true_regret_gain']:.4f}"
    )

    print(
        f"Decision entropy trace         = "
        f"{[round(x,4) for x in best['decision_entropy_trace']]}"
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()
