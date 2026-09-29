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
from decision import (
    causal_state_features,
    gather_edge_features,
    gather_edge_values,
    hard_decision_from_state_feat,
    move_batch,
    semantic_soft_decision_regret,
    state_risks,
    true_path_costs,
)
from joint_decision_bottleneck_v75 import (
    JointDecisionBottleneck,
    budgeted_soft_mask,
    hard_topk_mask,
    random_topk_mask,
)
from models import DualResolutionPerception
from student_v72 import gradient_distillation_loss


def load_perception(path, device):
    model = DualResolutionPerception(CFG.feat_dim).to(device)
    checkpoint = torch.load(path, map_location=device)
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    elif "perception" in checkpoint:
        checkpoint = checkpoint["perception"]
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def valid_token_mask(batch):
    # All image patches are candidates.  patch_graph_feat[..., 0] is an
    # embodied prior supplied to the model, not a privileged validity label.
    return torch.ones_like(batch["patch_state"], dtype=torch.bool)


def differentiable_task_outputs(
    perception,
    preview_feat,
    high_feat,
    mask,
    batch,
    temperature,
):
    """Task outcome under a continuous token-fidelity mask.

    This loss is defined on the downstream path decision.  It does not
    reconstruct an optimization model or any intermediate edge state.
    """
    preview_edge = gather_edge_features(preview_feat, batch["edge_patch"])
    high_edge = gather_edge_features(high_feat, batch["edge_patch"])
    edge_mask = gather_edge_values(mask, batch["edge_patch"]).clamp(0.0, 1.0)

    preview_penalty = perception.predict_preview_penalty(preview_edge)
    high_probability = perception.classify_high_features(high_edge).softmax(-1)
    high_penalty = (high_probability * state_risks(mask.device)).sum(-1)

    estimated_edge_cost = batch["base_cost"] + (
        (1.0 - edge_mask) * preview_penalty + edge_mask * high_penalty
    )
    estimated_path_cost = torch.einsum(
        "bpe,be->bp", batch["path_mask"], estimated_edge_cost
    )
    true_path_cost = true_path_costs(batch)
    optimal_true_cost = true_path_cost.min(-1).values

    student_path_probability = torch.softmax(
        -estimated_path_cost / max(float(temperature), 1e-6), dim=-1
    )
    expected_true_cost = (student_path_probability * true_path_cost).sum(-1)
    task_regret = (
        (expected_true_cost - optimal_true_cost)
        / optimal_true_cost.clamp_min(CFG.grad_eps)
    ).clamp_min(0.0)

    full_information_probability = torch.softmax(
        -true_path_cost / max(float(temperature), 1e-6), dim=-1
    )
    task_policy_kl = F.kl_div(
        torch.log(student_path_probability.clamp_min(1e-8)),
        full_information_probability,
        reduction="batchmean",
    )

    return {
        "task_regret": task_regret,
        "task_policy_kl": task_policy_kl,
        "estimated_path_cost": estimated_path_cost,
        "student_path_probability": student_path_probability,
    }


def decision_gradient_target(mask, batch, valid):
    """Teacher descent direction at the current complete token set."""
    with torch.enable_grad():
        reference = mask.detach().clone().requires_grad_(True)
        task_loss = semantic_soft_decision_regret(reference, batch).mean()
        gradient = torch.autograd.grad(task_loss, reference)[0]

    raw_utility = torch.relu(-gradient) * valid.float()
    raw_norm = torch.linalg.vector_norm(raw_utility, dim=-1)
    active = raw_norm > CFG.grad_eps

    utility = raw_utility
    norm = torch.linalg.vector_norm(utility, dim=-1, keepdim=True)
    dead = norm.squeeze(-1) <= CFG.grad_eps
    if dead.any():
        fallback = gradient.abs() * valid.float()
        utility = torch.where(dead[:, None], fallback, utility)
        norm = torch.linalg.vector_norm(utility, dim=-1, keepdim=True)
    return {
        "direction": (utility / norm.clamp_min(CFG.grad_eps)).detach(),
        "active": active.detach(),
    }


def set_value_losses(model, z, chosen_mask, random_mask, valid, budget, batch):
    predicted_chosen = model.predict_set_value(z, chosen_mask, valid, budget)
    predicted_random = model.predict_set_value(z, random_mask, valid, budget)

    with torch.no_grad():
        target_chosen = semantic_soft_decision_regret(chosen_mask, batch)
        target_random = semantic_soft_decision_regret(random_mask, batch)

    value_loss = 0.5 * (
        F.smooth_l1_loss(predicted_chosen, target_chosen)
        + F.smooth_l1_loss(predicted_random, target_random)
    )

    target_difference = target_random - target_chosen
    predicted_difference = predicted_random - predicted_chosen
    ordered = target_difference.abs() > 1e-5
    if ordered.any():
        sign = target_difference[ordered].sign()
        rank_loss = F.softplus(
            -sign * predicted_difference[ordered]
        ).mean()
    else:
        rank_loss = predicted_chosen.sum() * 0.0

    return value_loss, rank_loss


def hard_metrics(perception, preview_feat, high_feat, mask, batch):
    state = causal_state_features(preview_feat, high_feat, mask)
    return hard_decision_from_state_feat(perception, state, mask, batch)


def mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


def run_epoch(model, perception, loader, optimizer, args, device, training):
    model.train(training)
    perception.eval()
    k = CFG.visual_budget_k(args.budget)
    logs = {
        name: []
        for name in (
            "loss",
            "task",
            "task_kl",
            "dgd",
            "grad_cos",
            "set_value",
            "set_rank",
            "hard_regret",
            "optimal",
            "random_regret",
        )
    }

    context = torch.enable_grad if training else torch.no_grad
    for batch in tqdm(loader, desc="V7.5 train" if training else "V7.5 val"):
        batch = move_batch(batch, device)
        with torch.no_grad():
            preview_feat = perception.encode_preview(batch["image"])
            high_feat = perception.encode_all_high(batch["image"])

        valid = valid_token_mask(batch)
        budget = torch.full(
            (batch["image"].shape[0],), float(args.budget), device=device
        )

        with context():
            output = model(
                preview_feat=preview_feat,
                graph_feat=batch["patch_graph_feat"],
                valid=valid,
                budget_frac=budget,
            )
            soft_mask = budgeted_soft_mask(
                output["logits"], valid, k, args.mask_temperature
            )
            task = differentiable_task_outputs(
                perception,
                preview_feat,
                high_feat,
                soft_mask,
                batch,
                args.decision_temperature,
            )

            gradient_target = decision_gradient_target(soft_mask, batch, valid)
            target_gradient = gradient_target["direction"]
            predicted_direction = torch.softmax(
                output["logits"].masked_fill(~valid, -1e9)
                / args.gradient_temperature,
                dim=-1,
            )
            active = gradient_target["active"]
            if active.any():
                dgd = gradient_distillation_loss(
                    pred=predicted_direction[active],
                    logits=output["logits"][active],
                    target=target_gradient[active],
                    eligible=valid[active],
                    lambda_cos=1.0,
                    lambda_kl=0.25,
                    lambda_rank=0.10,
                    rank_margin=0.01,
                )
            else:
                zero = output["logits"].sum() * 0.0
                dgd = {"loss": zero, "cosine": zero}

            hard_mask = hard_topk_mask(output["logits"], valid, k)
            random_mask = random_topk_mask(valid, k)
            set_value_loss, set_rank_loss = set_value_losses(
                model,
                output["z"],
                hard_mask,
                random_mask,
                valid,
                budget,
                batch,
            )

            loss_task = task["task_regret"].mean()
            loss = (
                args.lambda_task * loss_task
                + args.lambda_task_kl * task["task_policy_kl"]
                + args.lambda_dgd * dgd["loss"]
                + args.lambda_set_value * set_value_loss
                + args.lambda_set_rank * set_rank_loss
            )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        with torch.no_grad():
            selected = hard_metrics(
                perception, preview_feat, high_feat, hard_mask, batch
            )
            random_result = hard_metrics(
                perception, preview_feat, high_feat, random_mask, batch
            )

        batch_size = int(batch["image"].shape[0])
        logs["loss"].append(float(loss.detach()))
        logs["task"].append(float(loss_task.detach()))
        logs["task_kl"].append(float(task["task_policy_kl"].detach()))
        logs["dgd"].append(float(dgd["loss"].detach()))
        logs["grad_cos"].append(float(dgd["cosine"].detach()))
        logs["set_value"].append(float(set_value_loss.detach()))
        logs["set_rank"].append(float(set_rank_loss.detach()))
        logs["hard_regret"].extend(selected["hard_regret"].cpu().tolist())
        logs["optimal"].extend(selected["optimal"].float().cpu().tolist())
        logs["random_regret"].extend(
            random_result["hard_regret"].cpu().tolist()
        )

    return {name: mean_or_nan(values) for name, values in logs.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--perception", required=True)
    parser.add_argument(
        "--out", default="checkpoints/v75_joint_decision_bottleneck.pt"
    )
    parser.add_argument("--metrics", default="results/v75_metrics.json")
    parser.add_argument("--budget", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-temperature", type=float, default=0.35)
    parser.add_argument("--decision-temperature", type=float, default=0.15)
    parser.add_argument("--gradient-temperature", type=float, default=0.50)
    parser.add_argument("--lambda-task", type=float, default=1.0)
    parser.add_argument("--lambda-task-kl", type=float, default=0.25)
    parser.add_argument("--lambda-dgd", type=float, default=0.25)
    parser.add_argument("--lambda-set-value", type=float, default=0.25)
    parser.add_argument("--lambda-set-rank", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = DCVDataset(args.data)
    val_dataset = DCVDataset(args.val)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("training and validation datasets must be non-empty")

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

    perception = load_perception(args.perception, device)
    model = JointDecisionBottleneck(
        feat_dim=CFG.feat_dim,
        graph_dim=4,
        hidden=args.hidden,
        latent_dim=args.latent_dim,
        layers=args.layers,
        heads=args.heads,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    best_regret = float("inf")
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model, perception, train_loader, optimizer, args, device, True
        )
        val_metrics = run_epoch(
            model, perception, val_loader, None, args, device, False
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"[{epoch:03d}] "
            f"train task={train_metrics['task']:.4f} "
            f"hardR={train_metrics['hard_regret']:.4f} "
            f"gradCos={train_metrics['grad_cos']:.3f} | "
            f"val task={val_metrics['task']:.4f} "
            f"hardR={val_metrics['hard_regret']:.4f} "
            f"randomR={val_metrics['random_regret']:.4f} "
            f"optimal={val_metrics['optimal']:.3f} "
            f"gradCos={val_metrics['grad_cos']:.3f}"
        )

        if val_metrics["hard_regret"] < best_regret:
            best_regret = val_metrics["hard_regret"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "best_epoch": epoch,
                    "best_val_hard_regret": best_regret,
                    "config": {
                        "grid": CFG.grid,
                        "feat_dim": CFG.feat_dim,
                        "n_patches": CFG.n_patches,
                    },
                },
                output_path,
            )

        metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"saved best checkpoint: {output_path}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
