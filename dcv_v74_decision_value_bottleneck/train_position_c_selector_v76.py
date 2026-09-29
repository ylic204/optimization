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
    move_batch,
    true_path_costs,
)
from joint_decision_bottleneck_v75 import (
    budgeted_soft_mask,
    hard_topk_mask,
    random_topk_mask,
)
from models import DualResolutionPerception
from position_c_selector_v76 import (
    FrozenLLMTaskEncoder,
    PositionCDecisionBottleneck,
    gather_selected_tokens,
    normalized_patch_positions,
)
from student_v72 import gradient_distillation_loss


def load_perception(path, device):
    model = DualResolutionPerception(CFG.feat_dim).to(device)
    checkpoint = torch.load(path, map_location=device)
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    elif "perception" in checkpoint:
        checkpoint = checkpoint["perception"]
    model.load_state_dict(checkpoint, strict=False)
    model.eval().requires_grad_(False)
    return model


class TaskEmbeddingCache:
    """Avoid running the frozen LLM again for repeated task prompts."""

    def __init__(self, encoder, device):
        self.encoder = encoder
        self.device = device
        self.cache = {}

    def __call__(self, texts):
        texts = [str(text) for text in texts]
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        if missing:
            embeddings = self.encoder(missing, self.device).cpu()
            for text, embedding in zip(missing, embeddings):
                self.cache[text] = embedding
        return torch.stack([self.cache[text] for text in texts]).to(self.device)


def valid_token_mask(batch):
    # No graph-derived validity flag is visible to the student.
    return torch.ones_like(batch["patch_state"], dtype=torch.bool)


def task_soft_regret(mask, batch, temperature):
    """Privileged training teacher; never an input to the student selector."""
    edge_mask = gather_edge_values(mask, batch["edge_patch"]).clamp(0.0, 1.0)
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
    path_probability = torch.softmax(
        -estimated_path_cost / max(float(temperature), 1e-6), dim=-1
    )
    expected_cost = (path_probability * true_cost).sum(-1)
    regret = (
        (expected_cost - optimal_cost)
        / optimal_cost.clamp_min(CFG.grad_eps)
    ).clamp_min(0.0)
    full_probability = torch.softmax(
        -true_cost / max(float(temperature), 1e-6), dim=-1
    )
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
    """Swap-aware teacher direction on the fixed-sum mask constraint.

    A constant component of -dL/dw cannot change a fixed-budget set, so it is
    removed. Shifting the remaining direction to non-negative utilities keeps
    the full add/remove ordering for distillation to token scores.
    """
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


def set_value_losses(
    model,
    output,
    chosen_mask,
    random_mask,
    valid,
    budget,
    batch,
    temperature,
):
    chosen_prediction = model.predict_set_value(
        output["z"],
        output["task_context"],
        chosen_mask,
        valid,
        budget,
    )
    random_prediction = model.predict_set_value(
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


def hard_metrics(perception, preview_tokens, vision_tokens, mask, batch):
    state = causal_state_features(preview_tokens, vision_tokens, mask)
    edge_features = gather_edge_features(state, batch["edge_patch"])
    edge_mask = gather_edge_values(mask, batch["edge_patch"])
    preview_penalty = perception.predict_preview_penalty(edge_features)
    high_probability = perception.classify_high_features(edge_features).softmax(-1)
    high_penalty = (
        high_probability * batch["task_risks"][:, None, :]
    ).sum(-1)
    estimated_edge_cost = batch["base_cost"] + torch.where(
        edge_mask.bool(), high_penalty, preview_penalty
    )
    estimated_path_cost = torch.einsum(
        "bpe,be->bp", batch["path_mask"], estimated_edge_cost
    )
    chosen_index = estimated_path_cost.argmin(-1)
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


def mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


def run_epoch(
    model,
    perception,
    task_embeddings,
    loader,
    optimizer,
    args,
    device,
    training,
):
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
            "selected_tokens",
        )
    }
    context = torch.enable_grad if training else torch.no_grad
    description = "V7.6 train" if training else "V7.6 val"

    for batch in tqdm(loader, desc=description):
        task_text = batch["task_text"]
        batch = move_batch(batch, device)
        with torch.no_grad():
            # Position C: all vision tokens are produced before pruning.
            preview_tokens = perception.encode_preview(batch["image"])
            vision_tokens = perception.encode_all_high(batch["image"])
            q = task_embeddings(task_text)

        valid = valid_token_mask(batch)
        batch_size, patches, _ = vision_tokens.shape
        position_xy = normalized_patch_positions(
            batch_size,
            patches,
            device,
            vision_tokens.dtype,
        )
        budget = torch.full(
            (batch_size,), float(args.budget), device=device
        )

        with context():
            output = model(
                vision_tokens=vision_tokens,
                position_xy=position_xy,
                task_embedding=q,
                valid=valid,
                budget=budget,
            )
            soft_mask = budgeted_soft_mask(
                output["logits"], valid, k, args.mask_temperature
            )
            task = task_soft_regret(
                soft_mask, batch, args.decision_temperature
            )
            gradient_target = fixed_budget_gradient_target(
                soft_mask,
                batch,
                valid,
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
            selected_tokens, _ = gather_selected_tokens(
                vision_tokens, output["logits"], valid, k
            )
            value_loss, rank_loss = set_value_losses(
                model,
                output,
                hard_mask,
                random_mask,
                valid,
                budget,
                batch,
                args.decision_temperature,
            )
            task_loss = task["task_regret"].mean()
            loss = (
                args.lambda_task * task_loss
                + args.lambda_task_kl * task["task_policy_kl"]
                + args.lambda_dgd * dgd["loss"]
                + args.lambda_set_value * value_loss
                + args.lambda_set_rank * rank_loss
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                optimizer.step()

        with torch.no_grad():
            selected_result = hard_metrics(
                perception,
                preview_tokens,
                vision_tokens,
                hard_mask,
                batch,
            )
            random_result = hard_metrics(
                perception,
                preview_tokens,
                vision_tokens,
                random_mask,
                batch,
            )
        logs["loss"].append(float(loss.detach()))
        logs["task"].append(float(task_loss.detach()))
        logs["task_kl"].append(float(task["task_policy_kl"].detach()))
        logs["dgd"].append(float(dgd["loss"].detach()))
        logs["grad_cos"].append(float(dgd["cosine"].detach()))
        logs["set_value"].append(float(value_loss.detach()))
        logs["set_rank"].append(float(rank_loss.detach()))
        logs["hard_regret"].extend(
            selected_result["hard_regret"].cpu().tolist()
        )
        logs["optimal"].extend(
            selected_result["optimal"].float().cpu().tolist()
        )
        logs["random_regret"].extend(
            random_result["hard_regret"].cpu().tolist()
        )
        logs["selected_tokens"].append(float(selected_tokens.shape[1]))
    return {name: mean_or_nan(values) for name, values in logs.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--perception", required=True)
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--text-local-files-only", action="store_true")
    parser.add_argument("--task-max-length", type=int, default=96)
    parser.add_argument(
        "--out", default="checkpoints/v76_position_c_selector.pt"
    )
    parser.add_argument("--metrics", default="results/v76_metrics.json")
    parser.add_argument("--budget", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--task-hidden", type=int, default=128)
    parser.add_argument("--position-dim", type=int, default=32)
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

    train_loader = DataLoader(
        DCVDataset(args.data),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        DCVDataset(args.val),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise RuntimeError("training and validation datasets must be non-empty")

    perception = load_perception(args.perception, device)
    text_encoder = FrozenLLMTaskEncoder(
        args.text_model,
        local_files_only=args.text_local_files_only,
        max_length=args.task_max_length,
    ).to(device)
    task_embeddings = TaskEmbeddingCache(text_encoder, device)
    model = PositionCDecisionBottleneck(
        vision_dim=CFG.feat_dim,
        task_dim=text_encoder.output_dim,
        hidden=args.hidden,
        latent_dim=args.latent_dim,
        task_hidden=args.task_hidden,
        position_dim=args.position_dim,
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
            model,
            perception,
            task_embeddings,
            train_loader,
            optimizer,
            args,
            device,
            True,
        )
        validation_metrics = run_epoch(
            model,
            perception,
            task_embeddings,
            val_loader,
            None,
            args,
            device,
            False,
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": validation_metrics,
            }
        )
        print(
            f"[{epoch:03d}] train task={train_metrics['task']:.4f} "
            f"hardR={train_metrics['hard_regret']:.4f} "
            f"gradCos={train_metrics['grad_cos']:.3f} | "
            f"val task={validation_metrics['task']:.4f} "
            f"hardR={validation_metrics['hard_regret']:.4f} "
            f"randomR={validation_metrics['random_regret']:.4f} "
            f"optimal={validation_metrics['optimal']:.3f}"
        )
        if validation_metrics["hard_regret"] < best_regret:
            best_regret = validation_metrics["hard_regret"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "best_epoch": epoch,
                    "best_val_hard_regret": best_regret,
                    "text_model": args.text_model,
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
