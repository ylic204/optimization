import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
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
    semantic_current_edge_cost,
    semantic_hard_decision,
    teacher_gradient_direction,
    true_path_costs,
)
from student_v72 import gradient_distillation_loss
from dib_model_v73 import (
    DecisionInformationBottleneck,
    vib_kl,
    categorical_entropy,
)


def safe_mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def load_perception(path, device):
    model = DualResolutionPerception(
        CFG.feat_dim
    ).to(device)

    ckpt = torch.load(
        path,
        map_location=device,
    )

    if "model" in ckpt:
        state = ckpt["model"]
    elif "perception" in ckpt:
        state = ckpt["perception"]
    else:
        state = ckpt

    model.load_state_dict(
        state,
        strict=False,
    )

    return model


def freeze_high_resolution(perception):
    """
    High-resolution semantics were already validated in Stage 0.
    Freeze them so this experiment isolates representation learning.

    Preview encoder stays trainable because the purpose is to make its
    representation decision-aware.
    """
    for p in perception.high_encoder.parameters():
        p.requires_grad_(False)

    for p in perception.high_head.parameters():
        p.requires_grad_(False)

    # Old preview cost head is not used by V7.3 DIB.
    for p in perception.preview_cost_head.parameters():
        p.requires_grad_(False)


def teacher_distribution(target, eligible, eps=1e-8):
    x = torch.relu(
        target
    ) * eligible.float()

    return (
        x
        / x.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(eps)
    )


def predicted_decision_terms(
    pred_edge_cost,
    batch,
    tau=None,
):
    tau = (
        CFG.teacher_tau
        if tau is None
        else tau
    )

    pred_path_cost = torch.einsum(
        "bpe,be->bp",
        batch["path_mask"],
        pred_edge_cost,
    )

    q = torch.softmax(
        -pred_path_cost / tau,
        dim=-1,
    )

    true_pc = true_path_costs(
        batch
    )

    expected_true_cost = (
        q * true_pc
    ).sum(
        dim=-1
    )

    optimal_true_cost = true_pc.min(
        dim=-1
    ).values

    soft_regret = (
        (
            expected_true_cost
            - optimal_true_cost
        )
        / optimal_true_cost.clamp_min(
            CFG.grad_eps
        )
    ).clamp_min(
        0.0
    )

    chosen_idx = pred_path_cost.argmin(
        dim=-1
    )

    chosen_true_cost = true_pc.gather(
        1,
        chosen_idx[
            :,
            None,
        ],
    ).squeeze(
        1
    )

    hard_regret = (
        (
            chosen_true_cost
            - optimal_true_cost
        )
        / optimal_true_cost.clamp_min(
            CFG.grad_eps
        )
    ).clamp_min(
        0.0
    )

    gap = (
        chosen_true_cost
        - optimal_true_cost
    )

    tol = (
        CFG.optimal_cost_atol
        + CFG.optimal_cost_rtol
        * optimal_true_cost.abs()
    )

    optimal = (
        gap <= tol
    )

    return {
        "pred_path_cost":
            pred_path_cost,

        "q":
            q,

        "soft_regret":
            soft_regret,

        "chosen_idx":
            chosen_idx,

        "hard_regret":
            hard_regret,

        "optimal":
            optimal,

        "decision_entropy":
            categorical_entropy(
                q
            ),
    }


def action_from_gradient(
    grad_prob,
    eligible,
):
    return grad_prob.masked_fill(
        ~eligible,
        -1e9,
    ).argmax(
        dim=-1
    )


def forward_state(
    dib,
    perception,
    batch,
    preview_feat,
    high_feat,
    w,
    outer_t,
    budget,
    sample,
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
        graph_feat=batch["patch_graph_feat"],
        w_binary=w,
        outer_t=outer_t,
        budget_frac=budget,
        eligible=eligible,
        base_cost=batch["base_cost"],
        edge_patch=batch["edge_patch"],
        sample=sample,
    )

    return out, eligible


def train_epoch(
    dib,
    perception,
    loader,
    optimizer,
    args,
    device,
):
    dib.train()

    perception.preview_encoder.train()
    perception.high_encoder.eval()
    perception.high_head.eval()

    losses = []
    cost_losses = []
    grad_cosines = []
    task_losses = []
    ib_losses = []
    entropies = []

    for batch in tqdm(
        loader,
        desc="DIB train",
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

        T = batch[
            "traj_grad"
        ].shape[
            1
        ]

        # Frozen high-res feature.
        with torch.no_grad():
            high_feat = perception.encode_all_high(
                batch["image"]
            )

        # Preview encoder is intentionally trainable.
        preview_feat = perception.encode_preview(
            batch["image"]
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

        total_loss = (
            next(
                dib.parameters()
            ).sum()
            * 0.0
        )

        batch_cost = []
        batch_cos = []
        batch_task = []
        batch_ib = []
        batch_entropy = []

        for t in range(
            T
        ):
            w = batch[
                "traj_w"
            ][
                :,
                t,
            ]

            target_grad = batch[
                "traj_grad"
            ][
                :,
                t,
            ]

            outer = torch.full(
                (
                    B,
                ),
                t
                / max(
                    T - 1,
                    1,
                ),
                device=device,
            )

            out, eligible = forward_state(
                dib,
                perception,
                batch,
                preview_feat,
                high_feat,
                w,
                outer,
                budget,
                sample=True,
            )

            # --------------------------------------------------
            # 1) Optimization-state supervision:
            #    visual latent -> current edge task cost
            # --------------------------------------------------
            target_edge_cost = semantic_current_edge_cost(
                w,
                batch,
            )

            scale = true_path_costs(
                batch
            ).min(
                dim=-1
            ).values[
                :,
                None,
            ].clamp_min(
                1.0
            )

            loss_cost = F.smooth_l1_loss(
                out[
                    "pred_edge_cost"
                ]
                / scale,
                target_edge_cost
                / scale,
            )

            # --------------------------------------------------
            # 2) Decision-gradient distillation
            # --------------------------------------------------
            gd = gradient_distillation_loss(
                pred=out[
                    "grad_prob"
                ],
                logits=out[
                    "grad_logits"
                ],
                target=target_grad,
                eligible=eligible,
                lambda_cos=args.lambda_cos,
                lambda_kl=args.lambda_grad_kl,
                lambda_rank=args.lambda_rank,
                rank_margin=args.rank_margin,
            )

            # --------------------------------------------------
            # 3) Differentiable downstream decision regret
            # --------------------------------------------------
            decision = predicted_decision_terms(
                out[
                    "pred_edge_cost"
                ],
                batch,
            )

            loss_task = decision[
                "soft_regret"
            ].mean()

            # --------------------------------------------------
            # 4) Variational information bottleneck
            # --------------------------------------------------
            graph_patch_mask = (
                batch[
                    "patch_graph_feat"
                ][
                    ...,
                    0,
                ]
                > 0.5
            )

            loss_ib = vib_kl(
                out[
                    "mu"
                ],
                out[
                    "logvar"
                ],
                valid_patch_mask=graph_patch_mask,
            )

            loss_t = (
                args.lambda_cost
                * loss_cost
                + args.lambda_grad
                * gd[
                    "loss"
                ]
                + args.lambda_task
                * loss_task
                + args.beta_ib
                * loss_ib
            )

            total_loss = (
                total_loss
                + loss_t
            )

            batch_cost.append(
                loss_cost
            )

            batch_cos.append(
                gd[
                    "cosine"
                ]
            )

            batch_task.append(
                loss_task
            )

            batch_ib.append(
                loss_ib
            )

            batch_entropy.append(
                decision[
                    "decision_entropy"
                ].mean()
            )

        total_loss = (
            total_loss
            / max(
                T,
                1,
            )
        )

        optimizer.zero_grad()

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(
                dib.parameters()
            )
            + list(
                perception.preview_encoder.parameters()
            ),
            5.0,
        )

        optimizer.step()

        losses.append(
            float(
                total_loss.detach()
                .cpu()
            )
        )

        cost_losses.append(
            float(
                torch.stack(
                    batch_cost
                ).mean()
                .detach()
                .cpu()
            )
        )

        grad_cosines.append(
            float(
                torch.stack(
                    batch_cos
                ).mean()
                .detach()
                .cpu()
            )
        )

        task_losses.append(
            float(
                torch.stack(
                    batch_task
                ).mean()
                .detach()
                .cpu()
            )
        )

        ib_losses.append(
            float(
                torch.stack(
                    batch_ib
                ).mean()
                .detach()
                .cpu()
            )
        )

        entropies.append(
            float(
                torch.stack(
                    batch_entropy
                ).mean()
                .detach()
                .cpu()
            )
        )

    return {
        "loss":
            safe_mean(
                losses
            ),

        "cost":
            safe_mean(
                cost_losses
            ),

        "grad_cos":
            safe_mean(
                grad_cosines
            ),

        "task":
            safe_mean(
                task_losses
            ),

        "ib":
            safe_mean(
                ib_losses
            ),

        "entropy":
            safe_mean(
                entropies
            ),
    }


@torch.no_grad()
def evaluate_teacher_states(
    dib,
    perception,
    loader,
    args,
    device,
):
    dib.eval()
    perception.eval()

    edge_mae = []
    path_mae = []
    grad_cos = []
    grad_top1 = []
    soft_regret = []
    hard_regret = []
    optimal_rate = []
    ib_values = []
    entropy_values = []
    per_step_cos = None

    for batch in tqdm(
        loader,
        desc="DIB offline val",
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

        T = batch[
            "traj_grad"
        ].shape[
            1
        ]

        if per_step_cos is None:
            per_step_cos = [
                []
                for _ in range(
                    T
                )
            ]

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

        for t in range(
            T
        ):
            w = batch[
                "traj_w"
            ][
                :,
                t,
            ]

            target_grad = batch[
                "traj_grad"
            ][
                :,
                t,
            ]

            outer = torch.full(
                (
                    B,
                ),
                t
                / max(
                    T - 1,
                    1,
                ),
                device=device,
            )

            out, eligible = forward_state(
                dib,
                perception,
                batch,
                preview_feat,
                high_feat,
                w,
                outer,
                budget,
                sample=False,
            )

            target_edge = semantic_current_edge_cost(
                w,
                batch,
            )

            pred_pc = torch.einsum(
                "bpe,be->bp",
                batch[
                    "path_mask"
                ],
                out[
                    "pred_edge_cost"
                ],
            )

            target_pc = torch.einsum(
                "bpe,be->bp",
                batch[
                    "path_mask"
                ],
                target_edge,
            )

            scale = true_path_costs(
                batch
            ).min(
                dim=-1
            ).values.clamp_min(
                1.0
            )

            edge_mae.extend(
                (
                    (
                        out[
                            "pred_edge_cost"
                        ]
                        - target_edge
                    ).abs()
                    / scale[
                        :,
                        None,
                    ]
                ).mean(
                    dim=-1
                ).cpu()
                .tolist()
            )

            path_mae.extend(
                (
                    (
                        pred_pc
                        - target_pc
                    ).abs()
                    / scale[
                        :,
                        None,
                    ]
                ).mean(
                    dim=-1
                ).cpu()
                .tolist()
            )

            p = (
                out[
                    "grad_prob"
                ]
                * eligible.float()
            )

            p_l2 = (
                p
                / torch.linalg.vector_norm(
                    p,
                    dim=-1,
                    keepdim=True,
                ).clamp_min(
                    1e-8
                )
            )

            g = (
                target_grad
                * eligible.float()
            )

            g_l2 = (
                g
                / torch.linalg.vector_norm(
                    g,
                    dim=-1,
                    keepdim=True,
                ).clamp_min(
                    1e-8
                )
            )

            cos = F.cosine_similarity(
                p_l2,
                g_l2,
                dim=-1,
                eps=1e-8,
            )

            grad_cos.extend(
                cos.cpu()
                .tolist()
            )

            per_step_cos[
                t
            ].extend(
                cos.cpu()
                .tolist()
            )

            pa = action_from_gradient(
                p,
                eligible,
            )

            ta = action_from_gradient(
                target_grad,
                eligible,
            )

            grad_top1.extend(
                (
                    pa
                    == ta
                ).float()
                .cpu()
                .tolist()
            )

            dec = predicted_decision_terms(
                out[
                    "pred_edge_cost"
                ],
                batch,
            )

            soft_regret.extend(
                dec[
                    "soft_regret"
                ].cpu()
                .tolist()
            )

            hard_regret.extend(
                dec[
                    "hard_regret"
                ].cpu()
                .tolist()
            )

            optimal_rate.extend(
                dec[
                    "optimal"
                ].float()
                .cpu()
                .tolist()
            )

            graph_patch_mask = (
                batch[
                    "patch_graph_feat"
                ][
                    ...,
                    0,
                ]
                > 0.5
            )

            ib_values.append(
                float(
                    vib_kl(
                        out[
                            "mu"
                        ],
                        out[
                            "logvar"
                        ],
                        graph_patch_mask,
                    ).cpu()
                )
            )

            entropy_values.extend(
                dec[
                    "decision_entropy"
                ].cpu()
                .tolist()
            )

    return {
        "edge_cost_mae":
            safe_mean(
                edge_mae
            ),

        "path_cost_mae":
            safe_mean(
                path_mae
            ),

        "gradient_cosine":
            safe_mean(
                grad_cos
            ),

        "gradient_top1":
            safe_mean(
                grad_top1
            ),

        "soft_regret":
            safe_mean(
                soft_regret
            ),

        "hard_regret":
            safe_mean(
                hard_regret
            ),

        "optimal_path_rate":
            safe_mean(
                optimal_rate
            ),

        "ib_kl":
            safe_mean(
                ib_values
            ),

        "decision_entropy":
            safe_mean(
                entropy_values
            ),

        "per_step_cosine":
            [
                safe_mean(
                    x
                )
                for x in per_step_cos
            ],
    }


@torch.no_grad()
def evaluate_closed_loop(
    dib,
    perception,
    loader,
    args,
    device,
):
    """
    Closed-loop K-step visual acquisition using the learned gradient head.

    Also records:
        decision-information gain:
            IG_t = H(q_t) - H(q_{t+1})

        true downstream gain:
            DG_t = R_t - R_{t+1}

    These are diagnostics now. They become RL reward components in Step 2C.
    """
    dib.eval()
    perception.eval()

    K = CFG.visual_budget_k(
        args.budget
    )

    final_regrets = []
    final_optimal = []
    teacher_regrets = []
    teacher_optimal = []

    onpolicy_cos = []
    onpolicy_top1 = []

    entropy_gain = [
        []
        for _ in range(
            K
        )
    ]

    regret_gain = [
        []
        for _ in range(
            K
        )
    ]

    entropy_trace = [
        []
        for _ in range(
            K + 1
        )
    ]

    semantic_regret_trace = [
        []
        for _ in range(
            K + 1
        )
    ]

    for batch in tqdm(
        loader,
        desc="DIB closed-loop val",
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

        ws = torch.zeros(
            B,
            M,
            device=device,
        )

        wt = torch.zeros_like(
            ws
        )

        # Initial predicted decision entropy.
        outer0 = torch.zeros(
            B,
            device=device,
        )

        out0, _ = forward_state(
            dib,
            perception,
            batch,
            preview_feat,
            high_feat,
            ws,
            outer0,
            budget,
            sample=False,
        )

        dec0 = predicted_decision_terms(
            out0[
                "pred_edge_cost"
            ],
            batch,
        )

        prev_entropy = dec0[
            "decision_entropy"
        ]

        prev_sem_regret = semantic_hard_decision(
            ws,
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

        semantic_regret_trace[
            0
        ].extend(
            prev_sem_regret.cpu()
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

            out, eligible = forward_state(
                dib,
                perception,
                batch,
                preview_feat,
                high_feat,
                ws,
                outer,
                budget,
                sample=False,
            )

            gt = teacher_gradient_direction(
                ws,
                batch,
            )

            pred = out[
                "grad_prob"
            ]

            p_l2 = (
                pred
                / torch.linalg.vector_norm(
                    pred,
                    dim=-1,
                    keepdim=True,
                ).clamp_min(
                    1e-8
                )
            )

            gt_l2 = (
                gt
                / torch.linalg.vector_norm(
                    gt,
                    dim=-1,
                    keepdim=True,
                ).clamp_min(
                    1e-8
                )
            )

            cos = F.cosine_similarity(
                p_l2,
                gt_l2,
                dim=-1,
                eps=1e-8,
            )

            onpolicy_cos.extend(
                cos.cpu()
                .tolist()
            )

            action = action_from_gradient(
                pred,
                eligible,
            )

            ta = action_from_gradient(
                gt,
                eligible,
            )

            onpolicy_top1.extend(
                (
                    action
                    == ta
                ).float()
                .cpu()
                .tolist()
            )

            ws = ws.clone()

            ws[
                rows,
                action,
            ] = 1.0

            # New decision belief after observing the selected patch.
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

            out_next, _ = forward_state(
                dib,
                perception,
                batch,
                preview_feat,
                high_feat,
                ws,
                next_outer,
                budget,
                sample=False,
            )

            dec_next = predicted_decision_terms(
                out_next[
                    "pred_edge_cost"
                ],
                batch,
            )

            next_entropy = dec_next[
                "decision_entropy"
            ]

            next_sem_regret = semantic_hard_decision(
                ws,
                batch,
            )[
                "hard_regret"
            ]

            entropy_gain[
                t
            ].extend(
                (
                    prev_entropy
                    - next_entropy
                ).cpu()
                .tolist()
            )

            regret_gain[
                t
            ].extend(
                (
                    prev_sem_regret
                    - next_sem_regret
                ).cpu()
                .tolist()
            )

            entropy_trace[
                t + 1
            ].extend(
                next_entropy.cpu()
                .tolist()
            )

            semantic_regret_trace[
                t + 1
            ].extend(
                next_sem_regret.cpu()
                .tolist()
            )

            prev_entropy = next_entropy
            prev_sem_regret = next_sem_regret

            # Privileged Teacher reference trajectory.
            et = legal_mask(
                batch,
                wt,
            )

            gt_ref = teacher_gradient_direction(
                wt,
                batch,
            )

            at = action_from_gradient(
                gt_ref,
                et,
            )

            wt = wt.clone()

            wt[
                rows,
                at,
            ] = 1.0

        # Final DIB hard decision from its own predicted edge costs.
        outerf = torch.ones(
            B,
            device=device,
        )

        outf, _ = forward_state(
            dib,
            perception,
            batch,
            preview_feat,
            high_feat,
            ws,
            outerf,
            budget,
            sample=False,
        )

        decf = predicted_decision_terms(
            outf[
                "pred_edge_cost"
            ],
            batch,
        )

        final_regrets.extend(
            decf[
                "hard_regret"
            ].cpu()
            .tolist()
        )

        final_optimal.extend(
            decf[
                "optimal"
            ].float()
            .cpu()
            .tolist()
        )

        td = semantic_hard_decision(
            wt,
            batch,
        )

        teacher_regrets.extend(
            td[
                "hard_regret"
            ].cpu()
            .tolist()
        )

        teacher_optimal.extend(
            td[
                "optimal"
            ].float()
            .cpu()
            .tolist()
        )

    return {
        "onpolicy_gradient_cosine":
            safe_mean(
                onpolicy_cos
            ),

        "onpolicy_teacher_top1":
            safe_mean(
                onpolicy_top1
            ),

        "final_hard_regret":
            safe_mean(
                final_regrets
            ),

        "final_optimal_path_rate":
            safe_mean(
                final_optimal
            ),

        "teacher_hard_regret":
            safe_mean(
                teacher_regrets
            ),

        "teacher_optimal_path_rate":
            safe_mean(
                teacher_optimal
            ),

        "mean_information_gain_per_step":
            [
                safe_mean(
                    x
                )
                for x in entropy_gain
            ],

        "mean_true_regret_gain_per_step":
            [
                safe_mean(
                    x
                )
                for x in regret_gain
            ],

        "decision_entropy_trace":
            [
                safe_mean(
                    x
                )
                for x in entropy_trace
            ],

        "semantic_regret_trace":
            [
                safe_mean(
                    x
                )
                for x in semantic_regret_trace
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
        "--teacher",
        required=True,
    )

    ap.add_argument(
        "--teacher-val",
        required=True,
    )

    ap.add_argument(
        "--perception",
        required=True,
    )

    ap.add_argument(
        "--out",
        default="checkpoints/v73_dib.pt",
    )

    ap.add_argument(
        "--budget",
        type=float,
        default=0.15,
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=32,
    )

    ap.add_argument(
        "--hidden",
        type=int,
        default=256,
    )

    ap.add_argument(
        "--latent-dim",
        type=int,
        default=32,
    )

    ap.add_argument(
        "--layers",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--heads",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )

    ap.add_argument(
        "--preview-lr",
        type=float,
        default=5e-5,
    )

    ap.add_argument(
        "--lambda-cost",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--lambda-grad",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--lambda-task",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--beta-ib",
        type=float,
        default=1e-3,
    )

    ap.add_argument(
        "--lambda-cos",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--lambda-grad-kl",
        type=float,
        default=0.25,
    )

    ap.add_argument(
        "--lambda-rank",
        type=float,
        default=0.10,
    )

    ap.add_argument(
        "--rank-margin",
        type=float,
        default=0.02,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=1234,
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
            args.data,
            args.teacher,
        ),
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        DCVDataset(
            args.val,
            args.teacher_val,
        ),
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
    )

    if len(
        train_loader.dataset
    ) == 0:
        raise RuntimeError(
            "Empty training dataset."
        )

    if len(
        val_loader.dataset
    ) == 0:
        raise RuntimeError(
            "Empty validation dataset."
        )

    perception = load_perception(
        args.perception,
        device,
    )

    freeze_high_resolution(
        perception
    )

    dib = DecisionInformationBottleneck(
        feat_dim=CFG.feat_dim,
        graph_dim=4,
        hidden=args.hidden,
        latent_dim=args.latent_dim,
        layers=args.layers,
        heads=args.heads,
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    dib.parameters(),

                "lr":
                    args.lr,
            },
            {
                "params":
                    perception.preview_encoder.parameters(),

                "lr":
                    args.preview_lr,
            },
        ],
        weight_decay=CFG.wd,
    )

    Path(
        args.out
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_regret = float(
        "inf"
    )

    best_metrics = None

    for ep in range(
        args.epochs
    ):
        tr = train_epoch(
            dib,
            perception,
            train_loader,
            optimizer,
            args,
            device,
        )

        va = evaluate_teacher_states(
            dib,
            perception,
            val_loader,
            args,
            device,
        )

        cl = evaluate_closed_loop(
            dib,
            perception,
            val_loader,
            args,
            device,
        )

        print(
            f"epoch={ep:02d} "
            f"loss={tr['loss']:.4f} "
            f"cost={tr['cost']:.4f} "
            f"grad_cos={tr['grad_cos']:.4f} "
            f"task={tr['task']:.4f} "
            f"ib={tr['ib']:.4f} | "
            f"val_edge_mae={va['edge_cost_mae']:.4f} "
            f"val_grad_cos={va['gradient_cosine']:.4f} "
            f"val_soft_regret={va['soft_regret']:.4f} | "
            f"closed_regret={cl['final_hard_regret']:.4f} "
            f"closed_opt={100*cl['final_optimal_path_rate']:.2f}%"
        )

        if cl[
            "final_hard_regret"
        ] < best_regret:
            best_regret = cl[
                "final_hard_regret"
            ]

            best_metrics = {
                "offline":
                    va,

                "closed_loop":
                    cl,
            }

            torch.save(
                {
                    "dib":
                        dib.state_dict(),

                    "preview_encoder":
                        perception.preview_encoder.state_dict(),

                    "args":
                        vars(
                            args
                        ),

                    "metrics":
                        best_metrics,

                    "version":
                        "v73_decision_information_bottleneck",
                },
                args.out,
            )

    metrics_path = Path(
        args.out
    ).with_suffix(
        ".json"
    )

    metrics_path.write_text(
        json.dumps(
            best_metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "V7.3 DECISION INFORMATION BOTTLENECK — BEST RESULT"
    )

    print(
        "============================================================"
    )

    off = best_metrics[
        "offline"
    ]

    cl = best_metrics[
        "closed_loop"
    ]

    print(
        "[Offline optimization representation]"
    )

    print(
        f"EdgeCost normalized MAE       = {off['edge_cost_mae']:.4f}"
    )

    print(
        f"PathCost normalized MAE       = {off['path_cost_mae']:.4f}"
    )

    print(
        f"Gradient cosine               = {off['gradient_cosine']:.4f}"
    )

    print(
        f"Gradient Top1                 = {100*off['gradient_top1']:.2f}%"
    )

    print(
        f"Soft decision regret          = {off['soft_regret']:.4f}"
    )

    print(
        f"Hard decision regret          = {off['hard_regret']:.4f}"
    )

    print(
        f"Optimal-Path Rate             = {100*off['optimal_path_rate']:.2f}%"
    )

    print(
        f"VIB KL                        = {off['ib_kl']:.4f}"
    )

    print(
        f"Decision entropy              = {off['decision_entropy']:.4f}"
    )

    print(
        f"Per-step gradient cosine      = "
        f"{[round(x,4) for x in off['per_step_cosine']]}"
    )

    print(
        "\n[Closed-loop K-step visual acquisition]"
    )

    print(
        f"On-policy gradient cosine     = "
        f"{cl['onpolicy_gradient_cosine']:.4f}"
    )

    print(
        f"On-policy Teacher-Top1        = "
        f"{100*cl['onpolicy_teacher_top1']:.2f}%"
    )

    print(
        f"Final Hard Regret             = "
        f"{cl['final_hard_regret']:.4f}"
    )

    print(
        f"Final Optimal-Path Rate       = "
        f"{100*cl['final_optimal_path_rate']:.2f}%"
    )

    print(
        f"Teacher Hard Regret           = "
        f"{cl['teacher_hard_regret']:.4f}"
    )

    print(
        f"Teacher Optimal-Path Rate     = "
        f"{100*cl['teacher_optimal_path_rate']:.2f}%"
    )

    print(
        f"Decision entropy trace        = "
        f"{[round(x,4) for x in cl['decision_entropy_trace']]}"
    )

    print(
        f"Information gain / step       = "
        f"{[round(x,4) for x in cl['mean_information_gain_per_step']]}"
    )

    print(
        f"True regret gain / step       = "
        f"{[round(x,4) for x in cl['mean_true_regret_gain_per_step']]}"
    )

    print(
        "============================================================"
    )

    print(
        f"Checkpoint: {args.out}"
    )

    print(
        f"Metrics:    {metrics_path}"
    )


if __name__ == "__main__":
    main()
