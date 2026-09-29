import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from aligned_vlm_selector_v77 import (
    AlignedCrossModalSelector,
    FrozenBlipAlignedEncoder,
    PathDecisionHead,
    gather_selected_visual_tokens,
    normalized_region_positions,
)
from config import CFG
from dataset import DCVDataset
from decision import move_batch, true_path_costs
from joint_decision_bottleneck_v75 import (
    budgeted_soft_mask,
    hard_topk_mask,
    random_topk_mask,
)
from student_v72 import gradient_distillation_loss
from train_position_c_selector_v76 import (
    fixed_budget_gradient_target,
    mean_or_nan,
    task_soft_regret,
)


def downstream_task_outputs(
    path_logits,
    batch,
):
    """Actual downstream expected path regret from the fused VLM output."""
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
def hard_task_metrics(
    backbone,
    path_head,
    input_ids,
    attention_mask,
    selected_visual_tokens,
    batch,
):
    selected_valid = torch.ones(
        selected_visual_tokens.shape[:2],
        dtype=torch.bool,
        device=selected_visual_tokens.device,
    )
    fused = backbone.fuse(
        input_ids,
        attention_mask,
        selected_visual_tokens,
        selected_valid,
    )
    chosen_index = path_head(fused).argmax(-1)
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
            "soft_hard_gap",
        )
    }
    context = torch.enable_grad if training else torch.no_grad

    for batch in tqdm(loader, desc="V7.7 train" if training else "V7.7 val"):
        task_text = batch["task_text"]
        batch = move_batch(batch, device)
        with torch.no_grad():
            features = backbone(batch["image"], task_text)
        raw_visual = features["raw_visual_tokens"]
        aligned_visual = features["aligned_visual_tokens"]
        aligned_text = features["aligned_text_tokens"]
        text_valid = features["text_valid"]
        batch_size, regions, _ = raw_visual.shape
        if regions != CFG.n_patches:
            raise RuntimeError(
                f"VLM produced {regions} regions, expected {CFG.n_patches}"
            )
        visual_valid = torch.ones(
            batch_size, regions, dtype=torch.bool, device=device
        )
        position_xy = normalized_region_positions(
            batch_size, regions, device, aligned_visual.dtype
        )
        budget = torch.full(
            (batch_size,), float(args.budget), device=device
        )

        with context():
            output = selector(
                aligned_visual=aligned_visual,
                aligned_text=aligned_text,
                text_valid=text_valid,
                position_xy=position_xy,
                visual_valid=visual_valid,
                budget=budget,
            )
            soft_mask = budgeted_soft_mask(
                output["logits"], visual_valid, k, args.mask_temperature
            )
            task = downstream_task_outputs(
                path_head(
                    backbone.fuse(
                        features["input_ids"],
                        features["attention_mask"],
                        raw_visual * soft_mask[..., None],
                        visual_valid,
                    )
                ),
                batch,
            )
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
            selected_tokens, _ = gather_selected_visual_tokens(
                raw_visual, output["logits"], visual_valid, k
            )
            selected_valid = torch.ones(
                selected_tokens.shape[:2],
                dtype=torch.bool,
                device=device,
            )
            hard_train_task = downstream_task_outputs(
                path_head(
                    backbone.fuse(
                        features["input_ids"],
                        features["attention_mask"],
                        selected_tokens,
                        selected_valid,
                    )
                ),
                batch,
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
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                parameters = list(selector.parameters()) + list(path_head.parameters())
                torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
                optimizer.step()

        random_tokens, _ = gather_selected_visual_tokens(
            raw_visual,
            random_mask,
            visual_valid,
            k,
        )
        chosen_result = hard_task_metrics(
            backbone,
            path_head,
            features["input_ids"],
            features["attention_mask"],
            selected_tokens,
            batch,
        )
        random_result = hard_task_metrics(
            backbone,
            path_head,
            features["input_ids"],
            features["attention_mask"],
            random_tokens,
            batch,
        )
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
        logs["hard_regret"].extend(chosen_result["hard_regret"].cpu().tolist())
        logs["random_regret"].extend(random_result["hard_regret"].cpu().tolist())
        logs["optimal"].extend(chosen_result["optimal"].float().cpu().tolist())
        logs["selected_tokens"].append(float(selected_tokens.shape[1]))
        logs["soft_hard_gap"].append(
            float(chosen_result["hard_regret"].mean().detach() - task_loss.detach())
        )
    return {name: mean_or_nan(values) for name, values in logs.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--vlm-model", required=True)
    parser.add_argument("--vlm-local-files-only", action="store_true")
    parser.add_argument("--max-text-length", type=int, default=64)
    parser.add_argument("--out", default="checkpoints/v77_aligned_vlm_selector.pt")
    parser.add_argument("--metrics", default="results/v77_metrics.json")
    parser.add_argument("--budget", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
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
    parser.add_argument("--lambda-hard-task", type=float, default=1.0)
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
    backbone = FrozenBlipAlignedEncoder(
        args.vlm_model,
        region_grid=CFG.grid,
        max_text_length=args.max_text_length,
        local_files_only=args.vlm_local_files_only,
    ).to(device)
    selector = AlignedCrossModalSelector(
        aligned_dim=backbone.aligned_dim,
        hidden=args.hidden,
        latent_dim=args.latent_dim,
        position_dim=args.position_dim,
        layers=args.layers,
        heads=args.heads,
    ).to(device)
    path_head = PathDecisionHead(
        backbone.text_hidden_dim,
        n_paths=CFG.n_paths,
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
            backbone, selector, path_head, train_loader, optimizer, args, device, True
        )
        val_metrics = run_epoch(
            backbone, selector, path_head, val_loader, None, args, device, False
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
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
                },
                output_path,
            )
        metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"saved best checkpoint: {output_path}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
