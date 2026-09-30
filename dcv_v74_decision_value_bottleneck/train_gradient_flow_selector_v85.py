"""Train V8.5 reliable gradient-flow distillation for visual-token selection."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from gradient_flow_selector_v85 import (
    build_v85_model,
    endpoint_improvement_loss,
    fixed_mass_mask,
    hard_topk_mask,
    project_capped_simplex,
)
from optimization_spec_v83 import METRIC_NAMES, OptimizationTask
from planning_dataset_v84 import PlanningDatasetV84


def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def task_texts(raw, source_id, args):
    overrides = (args.nuplan_text, args.pointnav_text)
    return [
        overrides[int(domain)] or text
        for text, domain in zip(raw["task_text"], source_id.tolist())
    ]


def task_parameters(source_id, tasks, device):
    weights = torch.tensor(
        [task.weight_vector for task in tasks], device=device, dtype=torch.float32
    )
    limits = torch.ones(
        len(tasks), len(METRIC_NAMES), device=device, dtype=torch.float32
    )
    limit_mask = torch.zeros_like(limits)
    penalties = torch.tensor(
        [task.constraint_penalty for task in tasks],
        device=device,
        dtype=torch.float32,
    )
    for task_index, task in enumerate(tasks):
        for name, value in task.limits.items():
            metric_index = METRIC_NAMES.index(name)
            limits[task_index, metric_index] = float(value)
            limit_mask[task_index, metric_index] = 1.0
    return (
        weights[source_id],
        limits[source_id],
        limit_mask[source_id],
        penalties[source_id],
    )


def augmented_candidate_cost(metrics, weights, limits, limit_mask, penalty):
    violation = F.relu(metrics - limits[:, None]) * limit_mask[:, None]
    objective = (metrics * weights[:, None]).sum(-1)
    return objective + penalty[:, None] * violation.sum(-1)


def candidate_regret_values(probability, oracle_cost, valid):
    minimum = oracle_cost.min(-1).values
    maximum = oracle_cost.masked_fill(~valid, -torch.inf).max(-1).values
    scale = (maximum - minimum).clamp_min(1e-4)
    normalized = (oracle_cost - minimum[:, None]) / scale[:, None]
    normalized = normalized.masked_fill(~valid, 0.0)
    return (probability * normalized).sum(-1)


def planning_from_fused(
    model,
    fused,
    batch,
    weights,
    limits,
    limit_mask,
    penalty,
    temperature,
):
    predicted_metrics = model.predict_metrics(fused, batch)
    predicted_cost = augmented_candidate_cost(
        predicted_metrics, weights, limits, limit_mask, penalty
    ).masked_fill(~batch["candidate_valid"], torch.inf)
    probability = torch.softmax(-predicted_cost / temperature, dim=-1)
    oracle_cost = augmented_candidate_cost(
        batch["candidate_metrics"], weights, limits, limit_mask, penalty
    ).masked_fill(~batch["candidate_valid"], torch.inf)
    task_values = candidate_regret_values(
        probability, oracle_cost, batch["candidate_valid"]
    )
    return {
        "predicted_metrics": predicted_metrics,
        "predicted_cost": predicted_cost,
        "probability": probability,
        "oracle_cost": oracle_cost,
        "task_values": task_values,
    }


def planning_from_mask(model, encoded, mask, batch, task_data, temperature):
    fused = model.fuse_mask(encoded, mask)
    return planning_from_fused(model, fused, batch, *task_data, temperature)


def teacher_descent_target(
    model,
    encoded,
    state,
    valid,
    mass,
    batch,
    task_data,
    args,
):
    """Full-information projected task-loss descent at one mask state."""
    teacher_state = state.detach().requires_grad_(True)
    teacher = planning_from_mask(
        model,
        encoded,
        teacher_state,
        batch,
        task_data,
        args.decision_temperature,
    )
    gradient = torch.autograd.grad(
        teacher["task_values"].sum(), teacher_state, retain_graph=False
    )[0]
    with torch.no_grad():
        next_state = project_capped_simplex(
            teacher_state - args.teacher_step_size * gradient,
            valid,
            mass,
        )
        velocity = (next_state - teacher_state) / args.teacher_step_size
    return velocity.detach(), next_state.detach()


def gradient_flow_loss(
    student_velocity,
    teacher_velocity,
    state,
    teacher_next,
    valid,
    mass,
    step_size,
    args,
):
    student_norm = torch.linalg.vector_norm(student_velocity, dim=-1)
    teacher_norm = torch.linalg.vector_norm(teacher_velocity, dim=-1)
    reliable = teacher_norm > args.gradient_eps
    cosine = F.cosine_similarity(
        student_velocity, teacher_velocity, dim=-1, eps=args.gradient_eps
    )
    direction = torch.where(
        reliable, 1.0 - cosine, torch.zeros_like(cosine)
    )
    direction = direction.sum() / reliable.float().sum().clamp_min(1.0)
    magnitude = F.smooth_l1_loss(
        torch.log1p(student_norm), torch.log1p(teacher_norm)
    )
    student_next = project_capped_simplex(
        state + step_size * student_velocity,
        valid,
        mass,
    )
    step = F.smooth_l1_loss(student_next, teacher_next)
    total = (
        direction
        + args.gfm_magnitude_weight * magnitude
        + args.gfm_step_weight * step
    )
    return total, direction, magnitude, step, teacher_norm.mean()


def train_epoch(model, loader, optimizer, args, device, tasks):
    model.selector.train()
    model.metric_head.train()
    model.mask_flow.train()
    keys = (
        "loss", "task", "metric", "gfm", "direction", "magnitude",
        "step", "latent", "value", "improve", "task_gain",
        "teacher_direction_norm",
    )
    logs = {key: [] for key in keys}

    for raw in tqdm(loader, desc="V8.5 train"):
        batch = move_batch(raw, device)
        task_data = task_parameters(batch["source_id"], tasks, device)
        budget = torch.full(
            (batch["image"].shape[0],), args.budget, device=device
        )
        encoded, selection, positions, valid = model.encode_and_select(
            batch["image"],
            task_texts(raw, batch["source_id"], args),
            budget,
        )
        region_count = selection["logits"].shape[-1]
        k = max(1, min(region_count, round(args.budget * region_count)))
        mass = torch.full(
            (batch["image"].shape[0],),
            float(k),
            device=device,
            dtype=selection["logits"].dtype,
        )

        # The selector supplies a simultaneous, task-conditioned relaxed mask.
        initial_mask = fixed_mass_mask(
            selection["logits"], valid, mass, args.mask_temperature
        )
        final_mask, states, _ = model.mask_flow.rollout(
            initial_mask,
            selection["z"],
            selection["task_context"],
            positions,
            valid,
            mass,
        )
        hard_mask = hard_topk_mask(final_mask, valid, k)
        straight_through_mask = hard_mask + final_mask - final_mask.detach()

        # The final Student task value is evaluated on the actually retained set.
        student = planning_from_mask(
            model,
            encoded,
            straight_through_mask,
            batch,
            task_data,
            args.decision_temperature,
        )
        task_loss = student["task_values"].mean()
        predicted_metrics = student["predicted_metrics"]
        metric_loss = F.smooth_l1_loss(
            predicted_metrics[batch["candidate_valid"]],
            batch["candidate_metrics"][batch["candidate_valid"]],
        )

        # L_improve compares only the initial and final information states.
        # The detached baseline prevents the model from making w0 worse.
        with torch.no_grad():
            initial = planning_from_mask(
                model,
                encoded,
                initial_mask.detach(),
                batch,
                task_data,
                args.decision_temperature,
            )
        improve_values = endpoint_improvement_loss(
            initial["task_values"],
            student["task_values"],
            args.improve_margin,
        )
        improve_loss = improve_values.mean()
        task_gain = (
            initial["task_values"] - student["task_values"].detach()
        ).mean()

        # Sample one latent optimization state.  FM matches the local Teacher
        # descent velocity; fixed-step projected ND executes this same field.
        if args.lambda_gfm > 0.0:
            sampled_step = random.randrange(args.mask_steps)
            sampled_state = states[sampled_step].detach()
            teacher_velocity, teacher_next = teacher_descent_target(
                model,
                encoded,
                sampled_state,
                valid,
                mass,
                batch,
                task_data,
                args,
            )
            time = torch.full(
                (sampled_state.shape[0],),
                sampled_step / max(1, args.mask_steps),
                device=device,
                dtype=sampled_state.dtype,
            )
            student_velocity = model.mask_flow.velocity(
                sampled_state,
                time,
                selection["z"],
                selection["task_context"],
                positions,
                valid,
            )
            step_size = model.mask_flow.step_size(sampled_step)
            gfm, direction, magnitude, step, teacher_direction_norm = (
                gradient_flow_loss(
                    student_velocity,
                    teacher_velocity,
                    sampled_state,
                    teacher_next,
                    valid,
                    mass,
                    step_size,
                    args,
                )
            )
        else:
            zero = task_loss.new_zeros(())
            gfm = direction = magnitude = step = teacher_direction_norm = zero

        full_latent = model.selector.pool_decision_latent(
            selection["z"], valid.float(), valid
        ).detach()
        selected_latent = model.selector.pool_decision_latent(
            selection["z"], final_mask, valid
        )
        latent_loss = (
            1.0 - F.cosine_similarity(selected_latent, full_latent, dim=-1)
        ).mean()
        predicted_value = model.selector.predict_set_value(
            selection["z"],
            selection["task_context"],
            final_mask,
            valid,
            budget,
        )
        value_loss = F.smooth_l1_loss(
            predicted_value, student["task_values"].detach()
        )

        loss = (
            task_loss
            + args.lambda_metric * metric_loss
            + args.lambda_gfm * gfm
            + args.lambda_latent * latent_loss
            + args.lambda_value * value_loss
            + args.lambda_improve * improve_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainable = (
            list(model.selector.parameters())
            + list(model.metric_head.parameters())
            + list(model.mask_flow.parameters())
        )
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()

        values = {
            "loss": loss,
            "task": task_loss,
            "metric": metric_loss,
            "gfm": gfm,
            "direction": direction,
            "magnitude": magnitude,
            "step": step,
            "latent": latent_loss,
            "value": value_loss,
            "improve": improve_loss,
            "task_gain": task_gain,
            "teacher_direction_norm": teacher_direction_norm,
        }
        for key, value in values.items():
            logs[key].append(float(value.detach()))

    return {key: float(np.mean(value)) for key, value in logs.items()}


@torch.no_grad()
def evaluate_mode(model, encoded, selection, positions, valid, batch, args, task_data, mode):
    region_count = selection["logits"].shape[-1]
    k = region_count if mode == "full" else max(
        1, min(region_count, round(args.budget * region_count))
    )
    if mode == "full":
        scores = selection["logits"]
        mask_change = 0.0
    elif mode == "random":
        scores = torch.rand_like(selection["logits"])
        mask_change = 0.0
    else:
        mass = torch.full(
            (selection["logits"].shape[0],),
            float(k),
            device=selection["logits"].device,
            dtype=selection["logits"].dtype,
        )
        initial = fixed_mass_mask(
            selection["logits"], valid, mass, args.mask_temperature
        )
        scores, _, _ = model.mask_flow.rollout(
            initial,
            selection["z"],
            selection["task_context"],
            positions,
            valid,
            mass,
        )
        mask_change = float((scores - initial).abs().mean())
    fused, _ = model.fuse_topk(encoded, scores, valid, k)
    result = planning_from_fused(
        model,
        fused,
        batch,
        *task_data,
        args.decision_temperature,
    )
    predicted_index = result["predicted_cost"].argmin(-1)
    oracle_index = result["oracle_cost"].argmin(-1)
    row = torch.arange(predicted_index.shape[0], device=predicted_index.device)
    chosen = result["oracle_cost"][row, predicted_index]
    optimal = result["oracle_cost"][row, oracle_index]
    maximum = result["oracle_cost"].masked_fill(
        ~batch["candidate_valid"], -torch.inf
    ).max(-1).values
    regret = ((chosen - optimal) / (maximum - optimal).clamp_min(1e-4)).mean()
    metric_mae = (
        result["predicted_metrics"][batch["candidate_valid"]]
        - batch["candidate_metrics"][batch["candidate_valid"]]
    ).abs().mean()
    return {
        "regret": float(regret),
        "expected_regret": float(result["task_values"].mean()),
        "metric_mae": float(metric_mae),
        "mask_change": mask_change,
        "tokens": float(k),
        "token_ratio": float(k / region_count),
    }


def validate(model, loader, args, device, tasks):
    model.selector.eval()
    model.metric_head.eval()
    model.mask_flow.eval()
    metric_names = (
        "regret", "expected_regret", "metric_mae", "mask_change",
        "tokens", "token_ratio",
    )
    logs = {
        mode: {key: [] for key in metric_names}
        for mode in ("full", "random", "learned")
    }
    for raw in tqdm(loader, desc="V8.5 validation"):
        batch = move_batch(raw, device)
        task_data = task_parameters(batch["source_id"], tasks, device)
        budget = torch.full(
            (batch["image"].shape[0],), args.budget, device=device
        )
        encoded, selection, positions, valid = model.encode_and_select(
            batch["image"],
            task_texts(raw, batch["source_id"], args),
            budget,
        )
        for mode in logs:
            result = evaluate_mode(
                model,
                encoded,
                selection,
                positions,
                valid,
                batch,
                args,
                task_data,
                mode,
            )
            for key, value in result.items():
                logs[mode][key].append(value)
    return {
        mode: {key: float(np.mean(value)) for key, value in metrics.items()}
        for mode, metrics in logs.items()
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--nuplan-task", required=True)
    parser.add_argument("--pointnav-task", required=True)
    parser.add_argument("--nuplan-text", default="")
    parser.add_argument("--pointnav-text", default="")
    parser.add_argument("--vlm-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--region-grid", type=int, default=9)
    parser.add_argument("--prune-layer", type=int, default=4)
    parser.add_argument("--max-text-length", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--budget", type=float, default=0.15)
    parser.add_argument("--mask-temperature", type=float, default=0.35)
    parser.add_argument("--mask-steps", type=int, default=4)
    parser.add_argument("--mask-step-size", type=float, default=0.25)
    parser.add_argument("--teacher-step-size", type=float, default=0.25)
    parser.add_argument("--decision-temperature", type=float, default=0.25)
    parser.add_argument("--gradient-eps", type=float, default=1e-6)
    parser.add_argument("--gfm-magnitude-weight", type=float, default=0.1)
    parser.add_argument("--gfm-step-weight", type=float, default=0.5)
    parser.add_argument("--lambda-metric", type=float, default=1.0)
    parser.add_argument("--lambda-gfm", type=float, default=1.0)
    parser.add_argument("--lambda-latent", type=float, default=0.25)
    parser.add_argument("--lambda-value", type=float, default=0.25)
    parser.add_argument("--lambda-improve", type=float, default=0.1)
    parser.add_argument("--improve-margin", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--output", default="checkpoints/v85_gradient_flow.pt")
    parser.add_argument("--metrics", default="results/v85_metrics.json")
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.budget <= 1.0:
        raise ValueError("budget must be in (0, 1]")
    if args.lambda_gfm > 0.0 and args.mask_steps <= 0:
        raise ValueError("mask_steps must be positive when lambda_gfm > 0")
    if args.mask_steps < 0:
        raise ValueError("mask_steps must be non-negative")
    if args.mask_step_size <= 0.0:
        raise ValueError("mask_step_size must be positive")
    if args.teacher_step_size <= 0.0:
        raise ValueError("teacher_step_size must be positive")
    if args.improve_margin < 0.0:
        raise ValueError("improve_margin must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    tasks = [
        OptimizationTask.load(args.nuplan_task),
        OptimizationTask.load(args.pointnav_task),
    ]
    image_side = args.region_grid * 32
    train_loader = DataLoader(
        PlanningDatasetV84(args.data, image_side),
        args.batch,
        shuffle=True,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        PlanningDatasetV84(args.val, image_side),
        args.batch,
        shuffle=False,
        num_workers=args.workers,
    )
    model = build_v85_model(args, device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.selector.load_state_dict(checkpoint["selector"])
        model.metric_head.load_state_dict(checkpoint["metric_head"])
        if "mask_flow" in checkpoint:
            model.mask_flow.load_state_dict(checkpoint["mask_flow"])

    trainable = (
        list(model.selector.parameters())
        + list(model.metric_head.parameters())
        + list(model.mask_flow.parameters())
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    output = Path(args.output)
    metrics_path = Path(args.metrics)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, args, device, tasks
        )
        validation = validate(model, val_loader, args, device, tasks)
        record = {"epoch": epoch, "train": train_metrics, "validation": validation}
        history.append(record)
        metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(record, indent=2))

        learned_regret = validation["learned"]["regret"]
        if learned_regret < best:
            best = learned_regret
            torch.save(
                {
                    "args": vars(args),
                    "selector": model.selector.state_dict(),
                    "metric_head": model.metric_head.state_dict(),
                    "mask_flow": model.mask_flow.state_dict(),
                    "validation": validation,
                },
                output,
            )


if __name__ == "__main__":
    main()
