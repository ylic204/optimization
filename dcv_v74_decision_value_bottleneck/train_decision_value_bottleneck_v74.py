import argparse
import copy
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
    semantic_hard_decision,
    semantic_soft_decision_regret,
    teacher_gradient_direction,
)
from student_v72 import gradient_distillation_loss
from decision_value_bottleneck_v74 import (
    DecisionValueBottleneck,
    vib_kl,
    masked_policy_entropy,
)


# ============================================================
# V7.4
# Counterfactual Downstream Decision-Value Bottleneck
# ============================================================


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


def prepare_perception(perception):
    """
    High-resolution perception was validated in Stage 0.
    Keep it fixed. Let preview representation adapt slowly because the
    research question is whether the causal visual state can learn
    decision-relevant information.
    """
    for p in perception.high_encoder.parameters():
        p.requires_grad_(False)

    for p in perception.high_head.parameters():
        p.requires_grad_(False)

    for p in perception.preview_cost_head.parameters():
        p.requires_grad_(False)


def exact_counterfactual_decision_values(
    w,
    batch,
    hard_weight=0.5,
):
    """
    Privileged TRAINING-ONLY downstream decision value for every legal patch.

    For patch j:
        V_soft(j) = R_soft(w) - R_soft(w + e_j)
        V_hard(j) = R_hard(w) - R_hard(w + e_j)

    Default target:
        V(j) = V_soft(j) + hard_weight * V_hard(j)

    This is a counterfactual outcome target, not image entropy and not an
    optimization-state input.

    Returns:
        target_gain: [B,M]
        soft_gain:   [B,M]
        hard_gain:   [B,M]
        post_soft:   [B,M]  soft regret after each candidate observation
        post_hard:   [B,M]  hard regret after each candidate observation
        r0_soft:     [B]
        r0_hard:     [B]
        eligible:    [B,M]
    """
    B, M = w.shape
    device = w.device

    eligible = legal_mask(
        batch,
        w,
    )

    with torch.no_grad():
        r0_soft = semantic_soft_decision_regret(
            w.float(),
            batch,
        )

        r0_hard = semantic_hard_decision(
            w,
            batch,
        )[
            "hard_regret"
        ]

        post_soft = torch.full(
            (
                B,
                M,
            ),
            float("inf"),
            device=device,
        )

        post_hard = torch.full_like(
            post_soft,
            float("inf"),
        )

        for j in range(
            M
        ):
            rows = eligible[
                :,
                j,
            ]

            if not rows.any():
                continue

            wc = w.clone()

            wc[
                rows,
                j,
            ] = 1.0

            rs = semantic_soft_decision_regret(
                wc.float(),
                batch,
            )

            rh = semantic_hard_decision(
                wc,
                batch,
            )[
                "hard_regret"
            ]

            post_soft[
                rows,
                j,
            ] = rs[
                rows
            ]

            post_hard[
                rows,
                j,
            ] = rh[
                rows
            ]

        soft_gain = (
            r0_soft[
                :,
                None,
            ]
            - post_soft
        )

        hard_gain = (
            r0_hard[
                :,
                None,
            ]
            - post_hard
        )

        # Ineligible locations never enter any supervised term.
        soft_gain = torch.where(
            eligible,
            soft_gain,
            torch.zeros_like(
                soft_gain
            ),
        )

        hard_gain = torch.where(
            eligible,
            hard_gain,
            torch.zeros_like(
                hard_gain
            ),
        )

        target_gain = (
            soft_gain
            + float(
                hard_weight
            )
            * hard_gain
        )

    return {
        "target_gain":
            target_gain,

        "soft_gain":
            soft_gain,

        "hard_gain":
            hard_gain,

        "post_soft":
            post_soft,

        "post_hard":
            post_hard,

        "r0_soft":
            r0_soft,

        "r0_hard":
            r0_hard,

        "eligible":
            eligible,
    }


def normalize_value_target(
    gain,
    eligible,
    eps=1e-6,
):
    """
    Per-state scale normalization.

    It preserves:
      - sign,
      - ordering,
      - relative magnitude inside a state,

    while avoiding a few very large-regret scenes dominating training.
    """
    masked_abs = (
        gain.abs()
        * eligible.float()
    )

    scale = masked_abs.max(
        dim=-1,
        keepdim=True,
    ).values.clamp_min(
        eps
    )

    return (
        gain / scale,
        scale,
    )


def value_teacher_distribution(
    gain,
    eligible,
    temperature,
):
    logits = (
        gain
        / max(
            float(
                temperature
            ),
            1e-6,
        )
    )

    logits = logits.masked_fill(
        ~eligible,
        -1e9,
    )

    return torch.softmax(
        logits,
        dim=-1,
    )


def value_losses(
    pred_value,
    pred_logits,
    target_gain,
    post_soft,
    eligible,
    args,
):
    """
    Primary supervision for the bottleneck.

    1) Value regression:
       predict normalized downstream decision gain.

    2) Value-policy KL:
       match the full teacher ranking/distribution, not just argmax.

    3) Pairwise ranking:
       preserve which observation has larger decision value.

    4) Expected downstream regret:
       pi_student(j|o) directly minimizes the counterfactual post-observation
       decision regret.
    """
    target_norm, _ = normalize_value_target(
        target_gain,
        eligible,
    )

    m = eligible.float()

    # Regression only on legal actions.
    if eligible.any():
        loss_reg = F.smooth_l1_loss(
            pred_value[
                eligible
            ],
            target_norm[
                eligible
            ],
        )
    else:
        loss_reg = (
            pred_value.sum()
            * 0.0
        )

    p_teacher = value_teacher_distribution(
        target_norm,
        eligible,
        args.value_teacher_temp,
    )

    log_p_student = F.log_softmax(
        pred_logits.masked_fill(
            ~eligible,
            -1e9,
        )
        / args.policy_temp,
        dim=-1,
    )

    loss_kl = F.kl_div(
        log_p_student,
        p_teacher,
        reduction="batchmean",
    )

    # Pairwise ranking.
    ti = target_norm[
        :,
        :,
        None,
    ]

    tj = target_norm[
        :,
        None,
        :,
    ]

    pi = pred_value[
        :,
        :,
        None,
    ]

    pj = pred_value[
        :,
        None,
        :,
    ]

    valid = (
        eligible[
            :,
            :,
            None,
        ]
        & eligible[
            :,
            None,
            :,
        ]
    )

    teacher_better = (
        ti - tj
    ) > args.value_rank_margin

    pair_mask = (
        valid
        & teacher_better
    )

    if pair_mask.any():
        pair_loss = F.softplus(
            -(
                pi - pj
            )
        )

        loss_rank = pair_loss[
            pair_mask
        ].mean()

    else:
        loss_rank = (
            pred_value.sum()
            * 0.0
        )

    # Direct downstream-decision loss.
    p_student = torch.softmax(
        pred_logits.masked_fill(
            ~eligible,
            -1e9,
        )
        / args.policy_temp,
        dim=-1,
    )

    safe_post_soft = torch.where(
        eligible,
        post_soft,
        torch.zeros_like(
            post_soft
        ),
    )

    expected_post_regret = (
        p_student
        * safe_post_soft
    ).sum(
        dim=-1
    )

    loss_decision = expected_post_regret.mean()

    return {
        "loss_reg":
            loss_reg,

        "loss_kl":
            loss_kl,

        "loss_rank":
            loss_rank,

        "loss_decision":
            loss_decision,

        "p_teacher":
            p_teacher,

        "p_student":
            p_student,
    }


def action_from_logits(
    logits,
    eligible,
):
    return logits.masked_fill(
        ~eligible,
        -1e9,
    ).argmax(
        dim=-1
    )


def random_legal_action(
    eligible,
):
    probs = eligible.float()

    probs = (
        probs
        / probs.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(
            1.0
        )
    )

    return torch.multinomial(
        probs,
        num_samples=1,
    ).squeeze(
        -1
    )


def spearman_one(
    pred,
    target,
    mask,
):
    idx = torch.where(
        mask
    )[
        0
    ]

    n = int(
        idx.numel()
    )

    if n < 2:
        return float(
            "nan"
        )

    p = pred[
        idx
    ]

    t = target[
        idx
    ]

    # Ranking without special tie handling is sufficient for this diagnostic.
    pr = torch.empty_like(
        p,
        dtype=torch.float32,
    )

    tr = torch.empty_like(
        t,
        dtype=torch.float32,
    )

    pr[
        torch.argsort(
            p
        )
    ] = torch.arange(
        n,
        device=p.device,
        dtype=torch.float32,
    )

    tr[
        torch.argsort(
            t
        )
    ] = torch.arange(
        n,
        device=t.device,
        dtype=torch.float32,
    )

    pr = (
        pr - pr.mean()
    )

    tr = (
        tr - tr.mean()
    )

    denom = (
        torch.linalg.vector_norm(
            pr
        )
        * torch.linalg.vector_norm(
            tr
        )
    )

    if float(
        denom
    ) <= 1e-8:
        return float(
            "nan"
        )

    return float(
        (
            pr * tr
        ).sum()
        / denom
    )


def value_metrics(
    pred_value,
    target_gain,
    eligible,
):
    top1s = []
    gain_ratios = []
    spearmans = []
    actionable = []

    B = pred_value.shape[
        0
    ]

    for b in range(
        B
    ):
        mask = eligible[
            b
        ]

        if int(
            mask.sum()
        ) == 0:
            continue

        pred_action = pred_value[
            b
        ].masked_fill(
            ~mask,
            -1e9,
        ).argmax()

        target_action = target_gain[
            b
        ].masked_fill(
            ~mask,
            -1e9,
        ).argmax()

        top1s.append(
            float(
                pred_action
                == target_action
            )
        )

        best_gain = target_gain[
            b
        ][
            mask
        ].max()

        chosen_gain = target_gain[
            b,
            pred_action,
        ]

        is_actionable = float(
            best_gain > 1e-8
        )

        actionable.append(
            is_actionable
        )

        if is_actionable:
            gain_ratios.append(
                float(
                    chosen_gain
                    / best_gain.clamp_min(
                        1e-8
                    )
                )
            )

        rho = spearman_one(
            pred_value[
                b
            ],
            target_gain[
                b
            ],
            mask,
        )

        if np.isfinite(
            rho
        ):
            spearmans.append(
                rho
            )

    return {
        "top1":
            safe_mean(
                top1s
            ),

        "gain_ratio":
            safe_mean(
                gain_ratios
            ),

        "spearman":
            safe_mean(
                spearmans
            ),

        "actionable_fraction":
            safe_mean(
                actionable
            ),
    }


def forward_value_model(
    model,
    batch,
    preview_feat,
    high_feat,
    w,
    outer,
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

    out = model(
        state_feat=state_feat,
        graph_feat=batch[
            "patch_graph_feat"
        ],
        w=w,
        outer_t=outer,
        budget_frac=budget,
        eligible=eligible,
        sample=sample,
    )

    return (
        out,
        eligible,
    )


def choose_student_action(
    model,
    batch,
    preview_feat,
    high_feat,
    w,
    outer,
    budget,
    epsilon=0.0,
):
    with torch.no_grad():
        out, eligible = forward_value_model(
            model,
            batch,
            preview_feat,
            high_feat,
            w,
            outer,
            budget,
            sample=False,
        )

        greedy = action_from_logits(
            out[
                "value_logits"
            ],
            eligible,
        )

        if epsilon <= 0:
            return greedy

        rnd = random_legal_action(
            eligible
        )

        use_random = (
            torch.rand(
                w.shape[
                    0
                ],
                device=w.device,
            )
            < epsilon
        )

        return torch.where(
            use_random,
            rnd,
            greedy,
        )


def train_epoch(
    model,
    perception,
    loader,
    optimizer,
    args,
    device,
    epoch,
):
    model.train()

    perception.preview_encoder.train()
    perception.high_encoder.eval()
    perception.high_head.eval()

    K = CFG.visual_budget_k(
        args.budget
    )

    # DAgger-style schedule.
    if args.epochs <= 1:
        onpolicy_prob = args.onpolicy_end
    else:
        frac = epoch / (
            args.epochs - 1
        )

        onpolicy_prob = (
            args.onpolicy_start
            + frac
            * (
                args.onpolicy_end
                - args.onpolicy_start
            )
        )

    logs = {
        "loss": [],
        "value_reg": [],
        "value_kl": [],
        "value_rank": [],
        "decision": [],
        "grad_cos": [],
        "ib": [],
        "top1": [],
        "gain_ratio": [],
        "spearman": [],
    }

    for batch in tqdm(
        loader,
        desc=(
            f"V7.4 train p_on={onpolicy_prob:.2f}"
        ),
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

        w_student = torch.zeros(
            B,
            CFG.n_patches,
            device=device,
        )

        total = (
            next(
                model.parameters()
            ).sum()
            * 0.0
        )

        local = {
            k: []
            for k in logs
            if k != "loss"
        }

        for t in range(
            K
        ):
            teacher_w = batch[
                "traj_w"
            ][
                :,
                t,
            ]

            # Mixed Teacher/on-policy state per sample.
            use_on = (
                torch.rand(
                    B,
                    device=device,
                )
                < onpolicy_prob
            )

            w_train = torch.where(
                use_on[
                    :,
                    None,
                ],
                w_student,
                teacher_w,
            )

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

            targets = exact_counterfactual_decision_values(
                w_train,
                batch,
                hard_weight=args.hard_gain_weight,
            )

            out, eligible = forward_value_model(
                model,
                batch,
                preview_feat,
                high_feat,
                w_train,
                outer,
                budget,
                sample=True,
            )

            vl = value_losses(
                pred_value=out[
                    "value"
                ],
                pred_logits=out[
                    "value_logits"
                ],
                target_gain=targets[
                    "target_gain"
                ],
                post_soft=targets[
                    "post_soft"
                ],
                eligible=eligible,
                args=args,
            )

            # Gradient is now AUXILIARY, not the main teacher.
            target_grad = teacher_gradient_direction(
                w_train,
                batch,
            )

            grad_prob = torch.softmax(
                out[
                    "grad_logits_masked"
                ]
                / args.grad_temp,
                dim=-1,
            )

            gd = gradient_distillation_loss(
                pred=grad_prob,
                logits=out[
                    "grad_logits"
                ],
                target=target_grad,
                eligible=eligible,
                lambda_cos=1.0,
                lambda_kl=0.25,
                lambda_rank=0.05,
                rank_margin=0.02,
            )

            patch_mask = (
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
                patch_mask,
            )

            loss_t = (
                args.lambda_value_reg
                * vl[
                    "loss_reg"
                ]
                + args.lambda_value_kl
                * vl[
                    "loss_kl"
                ]
                + args.lambda_value_rank
                * vl[
                    "loss_rank"
                ]
                + args.lambda_decision
                * vl[
                    "loss_decision"
                ]
                + args.lambda_grad_aux
                * gd[
                    "loss"
                ]
                + args.beta_ib
                * loss_ib
            )

            total = (
                total
                + loss_t
            )

            vm = value_metrics(
                out[
                    "value"
                ].detach(),
                targets[
                    "target_gain"
                ],
                eligible,
            )

            local[
                "value_reg"
            ].append(
                vl[
                    "loss_reg"
                ]
            )

            local[
                "value_kl"
            ].append(
                vl[
                    "loss_kl"
                ]
            )

            local[
                "value_rank"
            ].append(
                vl[
                    "loss_rank"
                ]
            )

            local[
                "decision"
            ].append(
                vl[
                    "loss_decision"
                ]
            )

            local[
                "grad_cos"
            ].append(
                gd[
                    "cosine"
                ]
            )

            local[
                "ib"
            ].append(
                loss_ib
            )

            local[
                "top1"
            ].append(
                torch.tensor(
                    vm[
                        "top1"
                    ],
                    device=device,
                )
            )

            local[
                "gain_ratio"
            ].append(
                torch.tensor(
                    vm[
                        "gain_ratio"
                    ],
                    device=device,
                )
            )

            local[
                "spearman"
            ].append(
                torch.tensor(
                    vm[
                        "spearman"
                    ],
                    device=device,
                )
            )

            # DAgger rollout state: model chooses from its OWN current state.
            action = choose_student_action(
                model,
                batch,
                preview_feat.detach(),
                high_feat,
                w_student,
                outer,
                budget,
                epsilon=args.rollout_epsilon,
            )

            w_student = w_student.clone()

            w_student[
                rows,
                action,
            ] = 1.0

        total = (
            total / K
        )

        optimizer.zero_grad()

        total.backward()

        torch.nn.utils.clip_grad_norm_(
            list(
                model.parameters()
            )
            + list(
                perception.preview_encoder.parameters()
            ),
            5.0,
        )

        optimizer.step()

        logs[
            "loss"
        ].append(
            float(
                total.detach()
                .cpu()
            )
        )

        for k, vals in local.items():
            if vals:
                x = torch.stack(
                    vals
                ).mean()

                if torch.isfinite(
                    x
                ):
                    logs[
                        k
                    ].append(
                        float(
                            x.detach()
                            .cpu()
                        )
                    )

    return {
        k:
            safe_mean(
                v
            )
        for k, v in logs.items()
    } | {
        "onpolicy_prob":
            float(
                onpolicy_prob
            )
    }


@torch.no_grad()
def evaluate_value_states(
    model,
    perception,
    loader,
    args,
    device,
):
    model.eval()
    perception.eval()

    K = CFG.visual_budget_k(
        args.budget
    )

    all_top1 = []
    all_gain_ratio = []
    all_spearman = []
    all_grad_cos = []
    all_actionable = []
    all_policy_entropy = []

    per_step_top1 = [
        []
        for _ in range(
            K
        )
    ]

    per_step_gain_ratio = [
        []
        for _ in range(
            K
        )
    ]

    for batch in tqdm(
        loader,
        desc="V7.4 offline val",
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
            K
        ):
            w = batch[
                "traj_w"
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
                    K - 1,
                    1,
                ),
                device=device,
            )

            targets = exact_counterfactual_decision_values(
                w,
                batch,
                hard_weight=args.hard_gain_weight,
            )

            out, eligible = forward_value_model(
                model,
                batch,
                preview_feat,
                high_feat,
                w,
                outer,
                budget,
                sample=False,
            )

            vm = value_metrics(
                out[
                    "value"
                ],
                targets[
                    "target_gain"
                ],
                eligible,
            )

            all_top1.append(
                vm[
                    "top1"
                ]
            )

            if np.isfinite(
                vm[
                    "gain_ratio"
                ]
            ):
                all_gain_ratio.append(
                    vm[
                        "gain_ratio"
                    ]
                )

                per_step_gain_ratio[
                    t
                ].append(
                    vm[
                        "gain_ratio"
                    ]
                )

            if np.isfinite(
                vm[
                    "spearman"
                ]
            ):
                all_spearman.append(
                    vm[
                        "spearman"
                    ]
                )

            all_actionable.append(
                vm[
                    "actionable_fraction"
                ]
            )

            per_step_top1[
                t
            ].append(
                vm[
                    "top1"
                ]
            )

            target_grad = teacher_gradient_direction(
                w,
                batch,
            )

            grad_prob = torch.softmax(
                out[
                    "grad_logits_masked"
                ]
                / args.grad_temp,
                dim=-1,
            )

            g1 = (
                grad_prob
                / torch.linalg.vector_norm(
                    grad_prob,
                    dim=-1,
                    keepdim=True,
                ).clamp_min(
                    1e-8
                )
            )

            g2 = (
                target_grad
                / torch.linalg.vector_norm(
                    target_grad,
                    dim=-1,
                    keepdim=True,
                ).clamp_min(
                    1e-8
                )
            )

            all_grad_cos.extend(
                F.cosine_similarity(
                    g1,
                    g2,
                    dim=-1,
                ).cpu()
                .tolist()
            )

            all_policy_entropy.extend(
                masked_policy_entropy(
                    out[
                        "value_logits"
                    ]
                    / args.policy_temp,
                    eligible,
                ).cpu()
                .tolist()
            )

    return {
        "value_top1":
            safe_mean(
                all_top1
            ),

        "value_gain_ratio":
            safe_mean(
                all_gain_ratio
            ),

        "value_spearman":
            safe_mean(
                all_spearman
            ),

        "actionable_fraction":
            safe_mean(
                all_actionable
            ),

        "gradient_cosine_aux":
            safe_mean(
                all_grad_cos
            ),

        "value_policy_entropy":
            safe_mean(
                all_policy_entropy
            ),

        "per_step_top1":
            [
                safe_mean(
                    x
                )
                for x in per_step_top1
            ],

        "per_step_gain_ratio":
            [
                safe_mean(
                    x
                )
                for x in per_step_gain_ratio
            ],
    }


@torch.no_grad()
def rollout_policy(
    mode,
    model,
    perception,
    batch,
    preview_feat,
    high_feat,
    args,
):
    """
    mode:
      student
      random
      gradient_teacher
      exact_value
    """
    B = batch[
        "image"
    ].shape[
        0
    ]

    K = CFG.visual_budget_k(
        args.budget
    )

    device = batch[
        "image"
    ].device

    rows = torch.arange(
        B,
        device=device,
    )

    w = torch.zeros(
        B,
        CFG.n_patches,
        device=device,
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

    regret_trace = [
        semantic_hard_decision(
            w,
            batch,
        )[
            "hard_regret"
        ]
    ]

    action_entropy_trace = []

    for t in range(
        K
    ):
        eligible = legal_mask(
            batch,
            w,
        )

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

        if mode == "student":
            out, eligible = forward_value_model(
                model,
                batch,
                preview_feat,
                high_feat,
                w,
                outer,
                budget,
                sample=False,
            )

            action = action_from_logits(
                out[
                    "value_logits"
                ],
                eligible,
            )

            action_entropy_trace.append(
                masked_policy_entropy(
                    out[
                        "value_logits"
                    ]
                    / args.policy_temp,
                    eligible,
                )
            )

        elif mode == "random":
            action = random_legal_action(
                eligible
            )

        elif mode == "gradient_teacher":
            g = teacher_gradient_direction(
                w,
                batch,
            )

            action = action_from_logits(
                g,
                eligible,
            )

        elif mode == "exact_value":
            targets = exact_counterfactual_decision_values(
                w,
                batch,
                hard_weight=args.hard_gain_weight,
            )

            action = action_from_logits(
                targets[
                    "target_gain"
                ],
                eligible,
            )

        else:
            raise ValueError(
                mode
            )

        w = w.clone()

        w[
            rows,
            action,
        ] = 1.0

        regret_trace.append(
            semantic_hard_decision(
                w,
                batch,
            )[
                "hard_regret"
            ]
        )

    final = semantic_hard_decision(
        w,
        batch,
    )

    return {
        "w":
            w,

        "regret":
            final[
                "hard_regret"
            ],

        "optimal":
            final[
                "optimal"
            ],

        "regret_trace":
            regret_trace,

        "action_entropy_trace":
            action_entropy_trace,
    }


@torch.no_grad()
def evaluate_closed_loop(
    model,
    perception,
    loader,
    args,
    device,
):
    model.eval()
    perception.eval()

    modes = [
        "student",
        "random",
        "gradient_teacher",
        "exact_value",
    ]

    agg = {
        m: {
            "regret": [],
            "optimal": [],
            "trace": None,
        }
        for m in modes
    }

    student_entropy = None

    for batch in tqdm(
        loader,
        desc="V7.4 closed-loop val",
    ):
        batch = move_batch(
            batch,
            device,
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

        for mode in modes:
            out = rollout_policy(
                mode,
                model,
                perception,
                batch,
                preview_feat,
                high_feat,
                args,
            )

            agg[
                mode
            ][
                "regret"
            ].extend(
                out[
                    "regret"
                ].cpu()
                .tolist()
            )

            agg[
                mode
            ][
                "optimal"
            ].extend(
                out[
                    "optimal"
                ].float()
                .cpu()
                .tolist()
            )

            if agg[
                mode
            ][
                "trace"
            ] is None:
                agg[
                    mode
                ][
                    "trace"
                ] = [
                    []
                    for _ in out[
                        "regret_trace"
                    ]
                ]

            for i, x in enumerate(
                out[
                    "regret_trace"
                ]
            ):
                agg[
                    mode
                ][
                    "trace"
                ][
                    i
                ].extend(
                    x.cpu()
                    .tolist()
                )

            if mode == "student":
                if student_entropy is None:
                    student_entropy = [
                        []
                        for _ in out[
                            "action_entropy_trace"
                        ]
                    ]

                for i, x in enumerate(
                    out[
                        "action_entropy_trace"
                    ]
                ):
                    student_entropy[
                        i
                    ].extend(
                        x.cpu()
                        .tolist()
                    )

    result = {}

    for mode in modes:
        result[
            mode
        ] = {
            "hard_regret":
                safe_mean(
                    agg[
                        mode
                    ][
                        "regret"
                    ]
                ),

            "optimal_path_rate":
                safe_mean(
                    agg[
                        mode
                    ][
                        "optimal"
                    ]
                ),

            "regret_trace":
                [
                    safe_mean(
                        x
                    )
                    for x in agg[
                        mode
                    ][
                        "trace"
                    ]
                ],
        }

    result[
        "student"
    ][
        "action_entropy_trace"
    ] = [
        safe_mean(
            x
        )
        for x in (
            student_entropy
            or []
        )
    ]

    return result


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
        default="checkpoints/v74_decision_value_bottleneck.pt",
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

    # Primary downstream-value objective.
    ap.add_argument(
        "--hard-gain-weight",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--lambda-value-reg",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--lambda-value-kl",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--lambda-value-rank",
        type=float,
        default=0.25,
    )

    ap.add_argument(
        "--lambda-decision",
        type=float,
        default=1.0,
    )

    # Gradient is only auxiliary now.
    ap.add_argument(
        "--lambda-grad-aux",
        type=float,
        default=0.10,
    )

    # Information bottleneck.
    ap.add_argument(
        "--beta-ib",
        type=float,
        default=1e-4,
    )

    ap.add_argument(
        "--value-teacher-temp",
        type=float,
        default=0.20,
    )

    ap.add_argument(
        "--policy-temp",
        type=float,
        default=0.30,
    )

    ap.add_argument(
        "--grad-temp",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--value-rank-margin",
        type=float,
        default=0.05,
    )

    # DAgger-style state coverage.
    ap.add_argument(
        "--onpolicy-start",
        type=float,
        default=0.25,
    )

    ap.add_argument(
        "--onpolicy-end",
        type=float,
        default=0.80,
    )

    ap.add_argument(
        "--rollout-epsilon",
        type=float,
        default=0.10,
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

    train_ds = DCVDataset(
        args.data,
        args.teacher,
    )

    val_ds = DCVDataset(
        args.val,
        args.teacher_val,
    )

    if len(
        train_ds
    ) == 0:
        raise RuntimeError(
            f"Empty training dataset: {args.data}"
        )

    if len(
        val_ds
    ) == 0:
        raise RuntimeError(
            f"Empty validation dataset: {args.val}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
    )

    perception = load_perception(
        args.perception,
        device,
    )

    prepare_perception(
        perception
    )

    model = DecisionValueBottleneck(
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
                    model.parameters(),

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

    best_score = float(
        "inf"
    )

    best_metrics = None
    best_epoch = -1

    for ep in range(
        args.epochs
    ):
        tr = train_epoch(
            model,
            perception,
            train_loader,
            optimizer,
            args,
            device,
            ep,
        )

        off = evaluate_value_states(
            model,
            perception,
            val_loader,
            args,
            device,
        )

        cl = evaluate_closed_loop(
            model,
            perception,
            val_loader,
            args,
            device,
        )

        student_regret = cl[
            "student"
        ][
            "hard_regret"
        ]

        print(
            f"epoch={ep:02d} "
            f"loss={tr['loss']:.4f} "
            f"Vreg={tr['value_reg']:.4f} "
            f"Vkl={tr['value_kl']:.4f} "
            f"Vrank={tr['value_rank']:.4f} "
            f"Dec={tr['decision']:.4f} "
            f"Gcos={tr['grad_cos']:.4f} "
            f"IB={tr['ib']:.4f} "
            f"p_on={tr['onpolicy_prob']:.2f} | "
            f"val_top1={100*off['value_top1']:.2f}% "
            f"val_gain={off['value_gain_ratio']:.3f} "
            f"val_rho={off['value_spearman']:.3f} | "
            f"student_R={student_regret:.4f} "
            f"student_opt={100*cl['student']['optimal_path_rate']:.2f}% "
            f"exact_R={cl['exact_value']['hard_regret']:.4f}"
        )

        # Primary checkpoint criterion: actual downstream closed-loop regret.
        if student_regret < best_score:
            best_score = student_regret
            best_epoch = ep

            best_metrics = {
                "epoch":
                    ep,

                "train":
                    tr,

                "offline":
                    off,

                "closed_loop":
                    cl,
            }

            torch.save(
                {
                    "model":
                        model.state_dict(),

                    "preview_encoder":
                        perception.preview_encoder.state_dict(),

                    "args":
                        vars(
                            args
                        ),

                    "metrics":
                        best_metrics,

                    "version":
                        "v74_counterfactual_decision_value_bottleneck",
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
        "V7.4 DECISION-VALUE BOTTLENECK — BEST RESULT"
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
        f"Best epoch                     = {best_epoch}"
    )

    print(
        "\n[Offline counterfactual decision-value prediction]"
    )

    print(
        f"Value Top1                     = {100*off['value_top1']:.2f}%"
    )

    print(
        f"Value GainRatio                = {off['value_gain_ratio']:.4f}"
    )

    print(
        f"Value Spearman                 = {off['value_spearman']:.4f}"
    )

    print(
        f"Actionable-state fraction      = {100*off['actionable_fraction']:.2f}%"
    )

    print(
        f"Gradient cosine (auxiliary)    = {off['gradient_cosine_aux']:.4f}"
    )

    print(
        f"Value-policy entropy           = {off['value_policy_entropy']:.4f}"
    )

    print(
        f"Per-step Value Top1            = "
        f"{[round(100*x,2) for x in off['per_step_top1']]}"
    )

    print(
        f"Per-step GainRatio             = "
        f"{[round(x,4) for x in off['per_step_gain_ratio']]}"
    )

    print(
        "\n[Closed-loop K-step downstream decision]"
    )

    for mode in [
        "student",
        "random",
        "gradient_teacher",
        "exact_value",
    ]:
        x = cl[
            mode
        ]

        print(
            f"{mode:18s} "
            f"Regret={x['hard_regret']:.4f} "
            f"OptimalPath={100*x['optimal_path_rate']:.2f}% "
            f"Trace={[round(v,4) for v in x['regret_trace']]}"
        )

    print(
        f"\nStudent action-entropy trace   = "
        f"{[round(v,4) for v in cl['student']['action_entropy_trace']]}"
    )

    print(
        "\nPrimary interpretation:"
    )

    print(
        "The bottleneck is successful only if low-information Z preserves "
        "counterfactual downstream decision value and Student Regret@K moves "
        "substantially toward Exact-Value / Gradient-Teacher baselines."
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
