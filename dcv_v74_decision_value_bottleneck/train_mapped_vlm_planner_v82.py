"""Train Qwen3-VL visual-token selection for known-map path planning.

V8.2 uses downstream candidate trajectory cost as the task objective.  It
contains decision-gradient distillation (DGD), but intentionally contains no
Flow Matching and no neurodynamic (ND) module.
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

from mapped_planning_dataset_v82 import (
    MappedPlanningDataset,
    assert_compatible_sample,
)
from mapped_vlm_planner_v82 import build_mapped_vlm_planner


def resolve_device(value):
    value = str(value).strip().lower()
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif value.isdigit():
        value = f"cuda:{value}"
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = 0 if device.index is None else int(device.index)
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f"invalid GPU {index}; count={torch.cuda.device_count()}")
        torch.cuda.set_device(index)
        device = torch.device(f"cuda:{index}")
    return device


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def budget_to_k(budget, regions):
    if budget <= 0:
        raise ValueError("budget must be positive")
    k = round(budget * regions) if budget <= 1.0 else round(budget)
    return max(1, min(regions, int(k)))


def budgeted_soft_mask(logits, valid, k, temperature=0.35, iterations=40):
    """Differentiable sigmoid mask whose mass is exactly approximately K."""
    target = torch.full(
        (logits.shape[0],), float(k), device=logits.device, dtype=logits.dtype
    )
    target = torch.minimum(target, valid.sum(-1).to(logits))
    with torch.no_grad():
        safe = logits.masked_fill(~valid, 0.0)
        lower = safe.min(-1).values - 30.0
        upper = safe.max(-1).values + 30.0
        for _ in range(iterations):
            threshold = 0.5 * (lower + upper)
            mass = (
                torch.sigmoid((logits - threshold[:, None]) / temperature)
                * valid.float()
            ).sum(-1)
            too_many = mass > target
            lower = torch.where(too_many, threshold, lower)
            upper = torch.where(too_many, upper, threshold)
        threshold = 0.5 * (lower + upper)
    return (
        torch.sigmoid((logits - threshold[:, None]) / temperature)
        * valid.float()
    )


def hard_topk_mask(logits, valid, k):
    masked = logits.masked_fill(~valid, -1e9)
    index = torch.topk(masked, k=int(k), dim=-1).indices
    return torch.zeros_like(logits).scatter(1, index, 1.0) * valid.float()


def candidate_task_outputs(logits, costs, valid, teacher_temperature=0.25):
    """Primary path-planning objective and auxiliary policy losses."""
    masked_logits = logits.masked_fill(~valid, -1e9)
    safe_costs = costs.masked_fill(~valid, torch.inf)
    optimal_cost, optimal_index = safe_costs.min(dim=-1)
    maximum = costs.masked_fill(~valid, -torch.inf).max(dim=-1).values
    scale = (maximum - optimal_cost).clamp_min(1e-4)
    normalized = (costs - optimal_cost[:, None]) / scale[:, None]
    normalized = normalized.masked_fill(~valid, 1e4)

    probability = torch.softmax(masked_logits, dim=-1)
    expected_regret = (probability * normalized).sum(-1)
    teacher = torch.softmax(
        -normalized / max(float(teacher_temperature), 1e-6), dim=-1
    )
    policy_kl = F.kl_div(
        F.log_softmax(masked_logits, dim=-1), teacher, reduction="batchmean"
    )
    imitation = F.cross_entropy(masked_logits, optimal_index)
    return {
        "regret": expected_regret,
        "policy_kl": policy_kl,
        "imitation": imitation,
        "probability": probability,
        "optimal_index": optimal_index,
        "optimal_cost": optimal_cost,
        "scale": scale,
    }


def decision_gradient_target(task_loss, soft_mask, valid, eps=1e-8):
    """Task-loss descent direction projected to the fixed-budget plane."""
    gradient = torch.autograd.grad(
        task_loss, soft_mask, retain_graph=True, create_graph=False
    )[0]
    descent = -gradient * valid.float()
    count = valid.sum(-1, keepdim=True).clamp_min(1)
    tangent = descent - descent.sum(-1, keepdim=True) / count * valid.float()
    large = torch.finfo(tangent.dtype).max
    minimum = tangent.masked_fill(~valid, large).min(-1, keepdim=True).values
    utility = (tangent - minimum).clamp_min(0.0) * valid.float()
    norm = torch.linalg.vector_norm(utility, dim=-1, keepdim=True)
    return (utility / norm.clamp_min(eps)).detach(), (norm[:, 0] > eps).detach()


def decision_gradient_distillation(logits, target, valid, active):
    if not active.any():
        zero = logits.sum() * 0.0
        return zero, zero
    logits = logits[active]
    target = target[active]
    valid = valid[active]
    prediction = torch.softmax(logits.masked_fill(~valid, -1e9), dim=-1)
    target_probability = target / target.sum(-1, keepdim=True).clamp_min(1e-8)
    cosine = F.cosine_similarity(prediction, target_probability, dim=-1).mean()
    kl = F.kl_div(
        torch.log(prediction.clamp_min(1e-8)),
        target_probability,
        reduction="batchmean",
    )
    return (1.0 - cosine) + 0.25 * kl, cosine


@torch.no_grad()
def hard_metrics(logits, costs, valid):
    chosen = logits.masked_fill(~valid, -1e9).argmax(-1)
    safe_costs = costs.masked_fill(~valid, torch.inf)
    optimal, optimal_index = safe_costs.min(-1)
    selected = costs.gather(1, chosen[:, None]).squeeze(1)
    maximum = costs.masked_fill(~valid, -torch.inf).max(-1).values
    scale = (maximum - optimal).clamp_min(1e-4)
    return {
        "regret": ((selected - optimal) / scale).clamp_min(0.0),
        "optimal": chosen.eq(optimal_index),
        "chosen": chosen,
    }


def mean(values):
    return float(np.mean(values)) if values else float("nan")


def run_epoch(model, loader, optimizer, args, device, training):
    model.backbone.eval()
    model.selector.train(training)
    model.planning_head.train(training)
    logs = {name: [] for name in (
        "loss", "task_regret", "policy_kl", "imitation", "dgd",
        "gradient_cosine", "hard_regret", "optimal_rate", "selected_regions",
    )}
    iterator = tqdm(loader, desc="V8.2 train" if training else "V8.2 val")
    for raw_batch in iterator:
        batch = move_batch(raw_batch, device)
        text = (
            [args.task_text] * batch["image"].shape[0]
            if args.task_text
            else list(raw_batch["task_text"])
        )
        budget = torch.full(
            (batch["image"].shape[0],), float(args.budget), device=device
        )
        encoded, selector_output, visual_valid = model.selector_forward(
            batch["image"], text, budget
        )
        regions = selector_output["logits"].shape[-1]
        k = budget_to_k(args.budget, regions)

        if training:
            soft_mask = budgeted_soft_mask(
                selector_output["logits"], visual_valid, k, args.mask_temperature
            )
            hard_mask = hard_topk_mask(selector_output["logits"], visual_valid, k)
            # Hard forward values, soft backward derivatives.  This uses one
            # rectangular training pass; inference physically removes tokens.
            straight_through = hard_mask + soft_mask - soft_mask.detach()
            logits = model.logits_from_mask(encoded, straight_through, batch)
            task = candidate_task_outputs(
                logits,
                batch["candidate_costs"],
                batch["candidate_valid"],
                args.teacher_temperature,
            )
            primary = (
                args.lambda_task * task["regret"].mean()
                + args.lambda_policy_kl * task["policy_kl"]
                + args.lambda_imitation * task["imitation"]
            )
            target, active = decision_gradient_target(
                primary, soft_mask, visual_valid
            )
            dgd, cosine = decision_gradient_distillation(
                selector_output["logits"], target, visual_valid, active
            )
            loss = primary + args.lambda_dgd * dgd
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.selector.parameters())
                + list(model.planning_head.parameters()),
                args.grad_clip,
            )
            optimizer.step()
        else:
            with torch.no_grad():
                logits, _ = model.logits_from_topk(
                    encoded,
                    selector_output["logits"],
                    visual_valid,
                    k,
                    batch,
                )
                task = candidate_task_outputs(
                    logits,
                    batch["candidate_costs"],
                    batch["candidate_valid"],
                    args.teacher_temperature,
                )
                primary = (
                    args.lambda_task * task["regret"].mean()
                    + args.lambda_policy_kl * task["policy_kl"]
                    + args.lambda_imitation * task["imitation"]
                )
                loss = primary
                dgd = logits.sum() * 0.0
                cosine = logits.sum() * 0.0

        metrics = hard_metrics(
            logits, batch["candidate_costs"], batch["candidate_valid"]
        )
        logs["loss"].append(float(loss.detach()))
        logs["task_regret"].append(float(task["regret"].mean().detach()))
        logs["policy_kl"].append(float(task["policy_kl"].detach()))
        logs["imitation"].append(float(task["imitation"].detach()))
        logs["dgd"].append(float(dgd.detach()))
        logs["gradient_cosine"].append(float(cosine.detach()))
        logs["hard_regret"].extend(metrics["regret"].cpu().tolist())
        logs["optimal_rate"].extend(metrics["optimal"].float().cpu().tolist())
        logs["selected_regions"].append(float(k))
        iterator.set_postfix(
            loss=f"{logs['loss'][-1]:.3f}", regret=f"{logs['hard_regret'][-1]:.3f}"
        )
    return {key: mean(value) for key, value in logs.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--vlm-model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--task-text", default="", help="Optional global override")
    parser.add_argument("--out", default="checkpoints/v82_mapped_vlm_planner.pt")
    parser.add_argument("--metrics", default="results/v82_mapped_metrics.json")
    parser.add_argument("--resume", default="")
    parser.add_argument("--vlm-local-files-only", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--region-grid", type=int, default=9)
    parser.add_argument("--prune-layer", type=int, default=4)
    parser.add_argument("--max-text-length", type=int, default=96)
    parser.add_argument("--budget", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--aligned-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--position-dim", type=int, default=32)
    parser.add_argument("--selector-layers", type=int, default=4)
    parser.add_argument("--selector-heads", type=int, default=8)
    parser.add_argument("--mask-temperature", type=float, default=0.35)
    parser.add_argument("--teacher-temperature", type=float, default=0.25)
    parser.add_argument("--lambda-task", type=float, default=1.0)
    parser.add_argument("--lambda-policy-kl", type=float, default=0.25)
    parser.add_argument("--lambda-imitation", type=float, default=0.25)
    parser.add_argument("--lambda-dgd", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)

    image_side = args.region_grid * 2 * 16
    train_dataset = MappedPlanningDataset(
        args.data, args.task_text or None, image_side=image_side
    )
    val_dataset = MappedPlanningDataset(
        args.val, args.task_text or None, image_side=image_side
    )
    assert_compatible_sample(train_dataset[0])
    assert_compatible_sample(val_dataset[0])
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
    model = build_mapped_vlm_planner(
        args.vlm_model,
        device,
        region_grid=args.region_grid,
        prune_layer=args.prune_layer,
        max_text_length=args.max_text_length,
        load_4bit=args.load_4bit,
        local_files_only=args.vlm_local_files_only,
        attn_implementation=args.attn_implementation,
        aligned_dim=args.aligned_dim,
        hidden=args.hidden,
        latent_dim=args.latent_dim,
        position_dim=args.position_dim,
        selector_layers=args.selector_layers,
        selector_heads=args.selector_heads,
    )
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.selector.load_state_dict(checkpoint["selector"])
        model.planning_head.load_state_dict(checkpoint["planning_head"])
        print(f"loaded trainable weights from {args.resume}")
    parameters = list(model.selector.parameters()) + list(
        model.planning_head.parameters()
    )
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    print(f"device={device} train={len(train_dataset)} val={len(val_dataset)}")
    print("Flow Matching=off, ND=off, DGD=on")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, optimizer, args, device, training=True
        )
        val_metrics = run_epoch(
            model, val_loader, None, args, device, training=False
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(record, indent=2))
        if val_metrics["hard_regret"] < best:
            best = val_metrics["hard_regret"]
            torch.save(
                {
                    "version": "v8.2",
                    "args": vars(args),
                    "selector": model.selector.state_dict(),
                    "planning_head": model.planning_head.state_dict(),
                    "epoch": epoch,
                    "val": val_metrics,
                },
                output,
            )


if __name__ == "__main__":
    main()
