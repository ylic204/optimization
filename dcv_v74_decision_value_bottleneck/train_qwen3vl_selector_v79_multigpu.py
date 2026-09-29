"""V7.9 Qwen3-VL trainer with CLI task text and explicit GPU selection.

The fixed-budget masks, task teacher and gradient-distillation loss are defined
in this file; no V7.2/V7.5/V7.6 training module is imported.
"""

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
from qwen3vl_selector_v79 import (
    DecisionRelatedQwenSelector,
    FrozenQwen3VLPrunableBackbone,
    PathDecisionHead,
    mask_to_region_indices,
    normalized_region_positions,
    topk_region_indices,
)


def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def true_path_costs(batch):
    return torch.einsum(
        "bpe,be->bp", batch["path_mask"], batch["true_edge_cost"]
    )


def gather_edge_values(values, edge_patch):
    return torch.gather(values, 1, edge_patch)


def apply_cli_task_costs(batch, args):
    """Make the numerical task loss agree with the CLI language objective.

    The image and topology still come from the dataset.  Edge costs are
    reconstructed from base geometry and the four state-risk coefficients so
    that changing ``--task-text`` can be accompanied by an explicit change in
    the downstream optimization objective.
    """
    risks = torch.tensor(
        [
            args.risk_normal,
            args.risk_rough,
            args.risk_hazard,
            args.risk_blocked,
        ],
        device=batch["base_cost"].device,
        dtype=batch["base_cost"].dtype,
    )
    batch["true_edge_cost"] = (
        batch["base_cost"] + risks[batch["edge_state"]]
    )
    batch["task_risks"] = risks[None].expand(
        batch["base_cost"].shape[0], -1
    )
    return batch


def budgeted_soft_mask(logits, valid, k, temperature=0.35, iterations=40):
    """Differentiable fixed-mass approximation of a K-token mask."""
    valid_count = valid.sum(-1).clamp_min(1)
    if torch.is_tensor(k):
        target = k.to(logits).clamp(min=1.0, max=float(logits.shape[-1]))
        target = torch.minimum(target, valid_count.to(logits))
    else:
        target = torch.full_like(valid_count, float(k), dtype=logits.dtype)
        target = torch.minimum(target, valid_count.to(logits))

    with torch.no_grad():
        safe = logits.masked_fill(~valid, 0.0)
        lower = safe.min(-1).values - 30.0
        upper = safe.max(-1).values + 30.0
        for _ in range(iterations):
            threshold = 0.5 * (lower + upper)
            mass = (
                torch.sigmoid(
                    (logits - threshold[:, None]) / float(temperature)
                )
                * valid.float()
            ).sum(-1)
            too_many = mass > target
            lower = torch.where(too_many, threshold, lower)
            upper = torch.where(too_many, upper, threshold)
        threshold = 0.5 * (lower + upper)

    return (
        torch.sigmoid((logits - threshold[:, None]) / float(temperature))
        * valid.float()
    )


def hard_topk_mask(logits, valid, k):
    mask = torch.zeros_like(logits)
    for row in range(logits.shape[0]):
        candidates = torch.where(valid[row])[0]
        if candidates.numel() == 0:
            continue
        keep = min(int(k), int(candidates.numel()))
        chosen = candidates[
            torch.topk(logits[row, candidates], k=keep).indices
        ]
        mask[row, chosen] = 1.0
    return mask


def random_topk_mask(valid, k, generator=None):
    mask = torch.zeros_like(valid, dtype=torch.float32)
    for row in range(valid.shape[0]):
        candidates = torch.where(valid[row])[0]
        if candidates.numel() == 0:
            continue
        keep = min(int(k), int(candidates.numel()))
        permutation = torch.randperm(
            int(candidates.numel()), generator=generator, device="cpu"
        )[:keep].to(candidates.device)
        mask[row, candidates[permutation]] = 1.0
    return mask


def task_soft_regret(mask, batch, temperature):
    """Privileged differentiable teacher used only during training."""
    edge_mask = gather_edge_values(
        mask, batch["edge_patch"]
    ).clamp(0.0, 1.0)
    abnormal = (batch["edge_state"] != 0).float()
    coarse_edge_cost = (
        batch["base_cost"] + abnormal * CFG.preview_abnormal_penalty
    )
    estimated_edge_cost = (
        (1.0 - edge_mask) * coarse_edge_cost
        + edge_mask * batch["true_edge_cost"]
    )
    estimated_path_cost = torch.einsum(
        "bpe,be->bp", batch["path_mask"], estimated_edge_cost
    )
    true_cost = true_path_costs(batch)
    optimal_cost = true_cost.min(-1).values
    tau = max(float(temperature), 1e-6)
    path_probability = torch.softmax(-estimated_path_cost / tau, dim=-1)
    expected_cost = (path_probability * true_cost).sum(-1)
    regret = (
        (expected_cost - optimal_cost)
        / optimal_cost.clamp_min(CFG.grad_eps)
    ).clamp_min(0.0)
    full_probability = torch.softmax(-true_cost / tau, dim=-1)
    policy_kl = F.kl_div(
        torch.log(path_probability.clamp_min(1e-8)),
        full_probability,
        reduction="batchmean",
    )
    return {
        "task_regret": regret,
        "task_policy_kl": policy_kl,
        "path_probability": path_probability,
    }


def fixed_budget_gradient_target(mask, batch, valid, temperature):
    """Project the task-loss descent direction onto the fixed-budget plane."""
    with torch.enable_grad():
        reference = mask.detach().clone().requires_grad_(True)
        loss = task_soft_regret(reference, batch, temperature)[
            "task_regret"
        ].mean()
        gradient = torch.autograd.grad(loss, reference)[0]

    descent = -gradient * valid.float()
    valid_count = valid.sum(-1, keepdim=True).clamp_min(1)
    tangent = descent - (
        descent.sum(-1, keepdim=True) / valid_count
    ) * valid.float()
    large = torch.finfo(tangent.dtype).max
    minimum = tangent.masked_fill(~valid, large).min(-1, keepdim=True).values
    utility = (tangent - minimum).clamp_min(0.0) * valid.float()
    norm = torch.linalg.vector_norm(utility, dim=-1, keepdim=True)
    active = norm.squeeze(-1) > CFG.grad_eps
    return {
        "direction": (utility / norm.clamp_min(CFG.grad_eps)).detach(),
        "active": active.detach(),
    }


def teacher_distribution(target, eligible, eps=1e-8):
    probability = torch.relu(target) * eligible.float()
    return probability / probability.sum(
        dim=-1, keepdim=True
    ).clamp_min(eps)


def gradient_distillation_loss(
    pred,
    logits,
    target,
    eligible,
    lambda_cos=1.0,
    lambda_kl=0.25,
    lambda_rank=0.10,
    rank_margin=0.02,
):
    mask = eligible.float()
    predicted_direction = pred * mask
    predicted_direction = predicted_direction / torch.linalg.vector_norm(
        predicted_direction, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    target_direction = target * mask
    target_direction = target_direction / torch.linalg.vector_norm(
        target_direction, dim=-1, keepdim=True
    ).clamp_min(1e-8)

    cosine = F.cosine_similarity(
        predicted_direction, target_direction, dim=-1, eps=1e-8
    )
    cosine_loss = (1.0 - cosine).mean()
    teacher_probability = teacher_distribution(target, eligible)
    masked_logits = logits.masked_fill(~eligible, -1e9)
    kl_loss = F.kl_div(
        F.log_softmax(masked_logits, dim=-1),
        teacher_probability,
        reduction="batchmean",
    )

    target_i = target[:, :, None]
    target_j = target[:, None, :]
    logit_i = logits[:, :, None]
    logit_j = logits[:, None, :]
    valid_pair = eligible[:, :, None] & eligible[:, None, :]
    ordered_pair = (target_i - target_j) > rank_margin
    pair_mask = valid_pair & ordered_pair
    if pair_mask.any():
        rank_loss = F.softplus(-(logit_i - logit_j))[pair_mask].mean()
    else:
        rank_loss = logits.sum() * 0.0

    total = (
        lambda_cos * cosine_loss
        + lambda_kl * kl_loss
        + lambda_rank * rank_loss
    )
    return {
        "loss": total,
        "cosine": cosine.mean(),
        "loss_cos": cosine_loss,
        "loss_kl": kl_loss,
        "loss_rank": rank_loss,
    }


def mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


def resolve_device(device_argument):
    """Resolve auto, integer GPU IDs, cuda:N strings, or cpu."""
    value = str(device_argument).strip().lower()
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif value.isdigit():
        value = f"cuda:{value}"

    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"--device {device_argument} requested CUDA, but CUDA is unavailable"
            )
        index = 0 if device.index is None else int(device.index)
        count = torch.cuda.device_count()
        if index < 0 or index >= count:
            raise ValueError(
                f"CUDA device {index} is invalid; visible GPU count is {count}"
            )
        torch.cuda.set_device(index)
        device = torch.device(f"cuda:{index}")
    return device


def downstream_task_outputs(path_logits, batch):
    """Expected path regret: the primary downstream task objective."""
    true_cost = true_path_costs(batch)
    optimal_cost = true_cost.min(-1).values
    path_probability = torch.softmax(path_logits, dim=-1)
    expected_cost = (path_probability * true_cost).sum(-1)
    task_regret = (
        (expected_cost - optimal_cost)
        / optimal_cost.clamp_min(CFG.grad_eps)
    ).clamp_min(0.0)
    full_probability = torch.softmax(-true_cost / CFG.teacher_tau, dim=-1)
    task_policy_kl = F.kl_div(
        torch.log(path_probability.clamp_min(1e-8)),
        full_probability,
        reduction="batchmean",
    )
    return {
        "task_regret": task_regret,
        "task_policy_kl": task_policy_kl,
        "path_probability": path_probability,
    }


@torch.no_grad()
def hard_task_metrics(path_logits, batch):
    chosen_index = path_logits.argmax(-1)
    true_cost = true_path_costs(batch)
    chosen_cost = true_cost.gather(1, chosen_index[:, None]).squeeze(1)
    optimal_cost = true_cost.min(-1).values
    regret = (
        (chosen_cost - optimal_cost)
        / optimal_cost.clamp_min(CFG.grad_eps)
    ).clamp_min(0.0)
    tolerance = CFG.optimal_cost_atol + CFG.optimal_cost_rtol * optimal_cost.abs()
    return {
        "hard_regret": regret,
        "optimal": (chosen_cost - optimal_cost) <= tolerance,
    }


def set_value_losses(
    selector,
    output,
    chosen_mask,
    random_mask,
    valid,
    budget,
    batch,
    temperature,
):
    chosen_prediction = selector.predict_set_value(
        output["z"],
        output["task_context"],
        chosen_mask,
        valid,
        budget,
    )
    random_prediction = selector.predict_set_value(
        output["z"],
        output["task_context"],
        random_mask,
        valid,
        budget,
    )
    with torch.no_grad():
        chosen_target = task_soft_regret(
            chosen_mask, batch, temperature
        )["task_regret"]
        random_target = task_soft_regret(
            random_mask, batch, temperature
        )["task_regret"]
    value_loss = 0.5 * (
        F.smooth_l1_loss(chosen_prediction, chosen_target)
        + F.smooth_l1_loss(random_prediction, random_target)
    )
    target_difference = random_target - chosen_target
    predicted_difference = random_prediction - chosen_prediction
    ordered = target_difference.abs() > 1e-5
    if ordered.any():
        rank_loss = F.softplus(
            -target_difference[ordered].sign()
            * predicted_difference[ordered]
        ).mean()
    else:
        rank_loss = chosen_prediction.sum() * 0.0
    return value_loss, rank_loss


def run_epoch(
    backbone,
    selector,
    path_head,
    loader,
    optimizer,
    args,
    device,
    training,
):
    backbone.eval()
    selector.train(training)
    path_head.train(training)
    k = CFG.visual_budget_k(args.budget)
    logs = {
        name: []
        for name in (
            "loss",
            "task",
            "task_kl",
            "hard_train_task",
            "dgd",
            "grad_cos",
            "set_value",
            "set_rank",
            "hard_regret",
            "random_regret",
            "optimal",
            "selected_tokens",
            "vision_tokens_before",
            "vision_tokens_after",
            "soft_hard_gap",
            "mapped_edge_fraction",
            "edge_coverage",
        )
    }
    grad_context = torch.enable_grad if training else torch.no_grad
    description = "V7.9 train" if training else "V7.9 val"

    for batch in tqdm(loader, desc=description):
        batch_size = int(batch["image"].shape[0])
        task_text = [args.task_text] * batch_size
        batch = move_batch(batch, device)
        batch = apply_cli_task_costs(batch, args)
        features = backbone.encode_inputs(batch["image"], task_text)
        visual_tokens = features["region_visual_tokens"]
        text_tokens = features["text_tokens"]
        text_valid = features["text_valid"]
        batch_size, regions, _ = visual_tokens.shape
        if regions != CFG.n_patches:
            raise RuntimeError(
                f"Qwen3-VL produced {regions} regions, expected {CFG.n_patches}"
            )
        visual_valid = torch.ones(
            batch_size, regions, dtype=torch.bool, device=device
        )
        position_xy = normalized_region_positions(
            batch_size, regions, device, visual_tokens.dtype
        )
        budget = torch.full(
            (batch_size,), float(args.budget), device=device
        )

        with grad_context():
            output = selector(
                visual_tokens=visual_tokens,
                text_tokens=text_tokens,
                text_valid=text_valid,
                position_xy=position_xy,
                visual_valid=visual_valid,
                budget=budget,
            )
            soft_mask = budgeted_soft_mask(
                output["logits"], visual_valid, k, args.mask_temperature
            )
            soft_visual = backbone.continue_vision(
                features["vision_state"], region_mask=soft_mask
            )
            soft_fused = backbone.fuse(
                features["input_ids"],
                features["attention_mask"],
                soft_visual,
            )
            task = downstream_task_outputs(path_head(soft_fused), batch)

            gradient_target = fixed_budget_gradient_target(
                soft_mask,
                batch,
                visual_valid,
                args.decision_temperature,
            )
            predicted_direction = torch.softmax(
                output["logits"] / args.gradient_temperature, dim=-1
            )
            active = gradient_target["active"]
            if active.any():
                dgd = gradient_distillation_loss(
                    pred=predicted_direction[active],
                    logits=output["logits"][active],
                    target=gradient_target["direction"][active],
                    eligible=visual_valid[active],
                    lambda_cos=1.0,
                    lambda_kl=0.25,
                    lambda_rank=0.10,
                    rank_margin=0.01,
                )
            else:
                zero = output["logits"].sum() * 0.0
                dgd = {"loss": zero, "cosine": zero}

            hard_mask = hard_topk_mask(output["logits"], visual_valid, k)
            random_mask = random_topk_mask(visual_valid, k)
            selected_indices = topk_region_indices(
                output["logits"], visual_valid, k
            )
            hard_visual = backbone.continue_vision(
                features["vision_state"], selected_indices=selected_indices
            )
            hard_fused = backbone.fuse(
                features["input_ids"],
                features["attention_mask"],
                hard_visual,
            )
            hard_path_logits = path_head(hard_fused)
            hard_train_task = downstream_task_outputs(
                hard_path_logits, batch
            )
            value_loss, rank_loss = set_value_losses(
                selector,
                output,
                hard_mask,
                random_mask,
                visual_valid,
                budget,
                batch,
                args.decision_temperature,
            )
            task_loss = task["task_regret"].mean()
            loss = (
                args.lambda_task * task_loss
                + args.lambda_task_kl * task["task_policy_kl"]
                + args.lambda_hard_task
                * hard_train_task["task_regret"].mean()
                + args.lambda_dgd * dgd["loss"]
                + args.lambda_set_value * value_loss
                + args.lambda_set_rank * rank_loss
            )
        with torch.no_grad():
            # Diagnostics only. ``edge_patch`` is privileged metadata and is
            # not passed to the selector or used in the training loss.
            edge_region_mask = torch.zeros_like(hard_mask)
            edge_region_mask.scatter_(1, batch["edge_patch"], 1.0)
            selected_edge_count = (hard_mask * edge_region_mask).sum(-1)
            random_indices = mask_to_region_indices(random_mask)
            random_visual = backbone.continue_vision(
                features["vision_state"], selected_indices=random_indices
            )
            random_fused = backbone.fuse(
                features["input_ids"],
                features["attention_mask"],
                random_visual,
            )
            random_path_logits = path_head(random_fused)
            chosen_result = hard_task_metrics(hard_path_logits, batch)
            random_result = hard_task_metrics(random_path_logits, batch)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            parameters = list(selector.parameters()) + list(
                path_head.parameters()
            )
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()

        logs["loss"].append(float(loss.detach()))
        logs["task"].append(float(task_loss.detach()))
        logs["task_kl"].append(float(task["task_policy_kl"].detach()))
        logs["hard_train_task"].append(
            float(hard_train_task["task_regret"].mean().detach())
        )
        logs["dgd"].append(float(dgd["loss"].detach()))
        logs["grad_cos"].append(float(dgd["cosine"].detach()))
        logs["set_value"].append(float(value_loss.detach()))
        logs["set_rank"].append(float(rank_loss.detach()))
        logs["hard_regret"].extend(
            chosen_result["hard_regret"].cpu().tolist()
        )
        logs["random_regret"].extend(
            random_result["hard_regret"].cpu().tolist()
        )
        logs["optimal"].extend(
            chosen_result["optimal"].float().cpu().tolist()
        )
        logs["selected_tokens"].append(float(k))
        logs["vision_tokens_before"].append(
            float(regions * backbone.merge_unit)
        )
        logs["vision_tokens_after"].append(
            float(k * backbone.merge_unit)
        )
        logs["soft_hard_gap"].append(
            float(
                chosen_result["hard_regret"].mean().detach()
                - task_loss.detach()
            )
        )
        logs["mapped_edge_fraction"].extend(
            (selected_edge_count / float(k)).cpu().tolist()
        )
        logs["edge_coverage"].extend(
            (
                selected_edge_count
                / edge_region_mask.sum(-1).clamp_min(1.0)
            ).cpu().tolist()
        )
    return {name: mean_or_nan(values) for name, values in logs.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--vlm-model", required=True)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cpu, an integer GPU ID such as 1, or cuda:1.",
    )
    parser.add_argument("--vlm-local-files-only", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--prune-layer", type=int, default=4)
    parser.add_argument("--max-text-length", type=int, default=64)
    parser.add_argument(
        "--task-text",
        default=(
            "Find a spatially continuous path from the green S hub to the "
            "red G hub that minimizes total traversal cost. Account for "
            "rough terrain, hazards, and blocked regions."
        ),
        help="Task instruction encoded by the frozen Qwen language stream.",
    )
    parser.add_argument("--risk-normal", type=float, default=0.0)
    parser.add_argument("--risk-rough", type=float, default=0.35)
    parser.add_argument("--risk-hazard", type=float, default=1.0)
    parser.add_argument("--risk-blocked", type=float, default=20.0)
    parser.add_argument(
        "--out", default="checkpoints/v79_qwen3vl_spatial_selector.pt"
    )
    parser.add_argument(
        "--metrics", default="results/v79_spatial_metrics.json"
    )
    parser.add_argument("--budget", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--aligned-dim", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--position-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--path-hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-temperature", type=float, default=0.35)
    parser.add_argument("--decision-temperature", type=float, default=0.15)
    parser.add_argument("--gradient-temperature", type=float, default=0.50)
    parser.add_argument("--lambda-task", type=float, default=1.0)
    parser.add_argument("--lambda-task-kl", type=float, default=0.25)
    parser.add_argument("--lambda-hard-task", type=float, default=0.5)
    parser.add_argument("--lambda-dgd", type=float, default=0.25)
    parser.add_argument("--lambda-set-value", type=float, default=0.25)
    parser.add_argument("--lambda-set-rank", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not args.task_text.strip():
        raise ValueError("--task-text must not be empty")
    if min(
        args.risk_normal,
        args.risk_rough,
        args.risk_hazard,
        args.risk_blocked,
    ) < 0.0:
        raise ValueError("task risk coefficients must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        print(
            f"device: {device} ({torch.cuda.get_device_name(device.index)})"
        )
    else:
        print(f"device: {device}")
    print(f"task_text: {args.task_text}")
    print(
        "task_risks: "
        f"normal={args.risk_normal}, rough={args.risk_rough}, "
        f"hazard={args.risk_hazard}, blocked={args.risk_blocked}"
    )
    train_dataset = DCVDataset(args.data)
    val_dataset = DCVDataset(args.val)
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("training and validation datasets must be non-empty")
    train_sample = train_dataset[0]
    val_sample = val_dataset[0]
    n_paths = int(train_sample["path_mask"].shape[0])
    if int(val_sample["path_mask"].shape[0]) != n_paths:
        raise ValueError("train and validation datasets use different path counts")
    expected_image_side = CFG.grid * CFG.tile
    actual_image_shape = tuple(train_sample["image"].shape[-2:])
    if actual_image_shape != (expected_image_side, expected_image_side):
        raise ValueError(
            f"dataset image/grid mismatch: GRID={CFG.grid} expects "
            f"{expected_image_side}x{expected_image_side}, got {actual_image_shape}"
        )
    if (
        "spatial_grid" in train_sample
        and int(train_sample["spatial_grid"]) != CFG.grid
    ):
        raise ValueError(
            f"dataset uses GRID={int(train_sample['spatial_grid'])}, "
            f"but the process uses GRID={CFG.grid}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    backbone = FrozenQwen3VLPrunableBackbone(
        args.vlm_model,
        region_grid=CFG.grid,
        prune_layer=args.prune_layer,
        max_text_length=args.max_text_length,
        local_files_only=args.vlm_local_files_only,
        load_4bit=args.load_4bit,
        attn_implementation=args.attn_implementation,
        device=device,
    )
    selector = DecisionRelatedQwenSelector(
        visual_dim=backbone.visual_hidden_dim,
        text_dim=backbone.text_hidden_dim,
        aligned_dim=args.aligned_dim,
        hidden=args.hidden,
        latent_dim=args.latent_dim,
        position_dim=args.position_dim,
        layers=args.layers,
        heads=args.heads,
    ).to(device)
    path_head = PathDecisionHead(
        backbone.text_hidden_dim,
        n_paths=n_paths,
        hidden=args.path_hidden,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(selector.parameters()) + list(path_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_regret = float("inf")
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            backbone,
            selector,
            path_head,
            train_loader,
            optimizer,
            args,
            device,
            True,
        )
        val_metrics = run_epoch(
            backbone,
            selector,
            path_head,
            val_loader,
            None,
            args,
            device,
            False,
        )
        history.append(
            {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        )
        print(
            f"[{epoch:03d}] train task={train_metrics['task']:.4f} "
            f"hardR={train_metrics['hard_regret']:.4f} "
            f"gradCos={train_metrics['grad_cos']:.3f} | "
            f"val task={val_metrics['task']:.4f} "
            f"hardR={val_metrics['hard_regret']:.4f} "
            f"randomR={val_metrics['random_regret']:.4f} "
            f"optimal={val_metrics['optimal']:.3f}"
        )
        if val_metrics["hard_regret"] < best_regret:
            best_regret = val_metrics["hard_regret"]
            torch.save(
                {
                    "selector": selector.state_dict(),
                    "path_head": path_head.state_dict(),
                    "args": vars(args),
                    "best_epoch": epoch,
                    "best_val_hard_regret": best_regret,
                    "vlm_model": args.vlm_model,
                    "base_model": "Qwen3-VL-8B",
                    "n_paths": n_paths,
                    "data_layout": train_sample.get(
                        "observable_layout", "legacy"
                    ),
                },
                output_path,
            )
        metrics_path.write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
    print(f"saved best checkpoint: {output_path}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
