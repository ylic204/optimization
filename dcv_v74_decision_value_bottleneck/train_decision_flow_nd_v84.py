"""End-to-end V8.4 training for token distillation, Flow Matching and ND."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow_nd_planner_v84 import build_v84_model
from optimization_spec_v83 import METRIC_NAMES, OptimizationTask
from planning_dataset_v84 import PlanningDatasetV84


def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def task_texts(raw, source_id, args):
    """Use optional CLI instructions, otherwise keep the dataset text."""
    overrides = (args.nuplan_text, args.pointnav_text)
    return [
        overrides[int(domain)] or text
        for text, domain in zip(raw["task_text"], source_id.tolist())
    ]


def fixed_mass_mask(logits, valid, k, temperature):
    """Differentiable K-mass mask used by decision-gradient distillation."""
    with torch.no_grad():
        lower = logits.min(-1).values - 30.0
        upper = logits.max(-1).values + 30.0
        for _ in range(40):
            threshold = 0.5 * (lower + upper)
            mass = (
                torch.sigmoid((logits - threshold[:, None]) / temperature)
                * valid.float()
            ).sum(-1)
            lower = torch.where(mass > k, threshold, lower)
            upper = torch.where(mass > k, upper, threshold)
        threshold = 0.5 * (lower + upper)
    return torch.sigmoid((logits - threshold[:, None]) / temperature) * valid.float()


def hard_topk_mask(logits, k):
    indices = torch.topk(logits, k=k, dim=-1).indices
    return torch.zeros_like(logits).scatter(1, indices, 1.0)


def task_parameters(source_id, tasks, device):
    """Convert the agent-generated JSON tasks to tensors."""
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

    selected_weights = weights[source_id]
    selected_limits = limits[source_id]
    selected_mask = limit_mask[source_id]
    selected_penalty = penalties[source_id]
    task_vector = torch.cat(
        [
            selected_weights,
            selected_limits,
            selected_mask,
            selected_penalty[:, None] / 100.0,
        ],
        dim=-1,
    )
    return (
        selected_weights,
        selected_limits,
        selected_mask,
        selected_penalty,
        task_vector,
    )


def augmented_candidate_cost(metrics, weights, limits, limit_mask, penalty):
    violation = F.relu(metrics - limits[:, None]) * limit_mask[:, None]
    objective = (metrics * weights[:, None]).sum(-1)
    return objective + penalty[:, None] * violation.sum(-1)


def oracle_trajectory(batch, weights, limits, limit_mask, penalty):
    """Exact finite-candidate teacher generated from simulator metrics."""
    cost = augmented_candidate_cost(
        batch["candidate_metrics"], weights, limits, limit_mask, penalty
    )
    cost = cost.masked_fill(~batch["candidate_valid"], torch.inf)
    oracle_index = cost.argmin(-1)
    row = torch.arange(cost.shape[0], device=cost.device)
    target = batch["candidate_trajectories"][row, oracle_index, :, :2]
    return target, oracle_index, cost


def soft_candidate_warm_start(
    predicted_metrics,
    batch,
    weights,
    limits,
    limit_mask,
    penalty,
    temperature,
):
    predicted_cost = augmented_candidate_cost(
        predicted_metrics, weights, limits, limit_mask, penalty
    )
    predicted_cost = predicted_cost.masked_fill(~batch["candidate_valid"], 1e4)
    probability = torch.softmax(-predicted_cost / temperature, dim=-1)
    candidate_xy = batch["candidate_trajectories"][..., :2]
    base = (probability[:, :, None, None] * candidate_xy).sum(1)
    return base, probability, predicted_cost


def expected_candidate_regret(probability, oracle_cost, valid):
    minimum = oracle_cost.min(-1).values
    maximum = oracle_cost.masked_fill(~valid, -torch.inf).max(-1).values
    scale = (maximum - minimum).clamp_min(1e-4)
    normalized = (oracle_cost - minimum[:, None]) / scale[:, None]
    normalized = normalized.masked_fill(~valid, 0.0)
    return (probability * normalized).sum(-1).mean()


def decision_gradient_teacher(task_loss, soft_mask, valid):
    """Token utility is the fixed-budget descent direction of final task loss."""
    gradient = torch.autograd.grad(task_loss, soft_mask, retain_graph=True)[0]
    utility = -gradient * valid.float()
    utility = utility - utility.mean(-1, keepdim=True)
    utility = utility - utility.min(-1, keepdim=True).values
    utility = utility * valid.float()
    return (utility / utility.sum(-1, keepdim=True).clamp_min(1e-8)).detach()


def trajectory_metrics(model, trajectory, target, batch, weights):
    """Final downstream loss after Flow Matching and ND refinement."""
    imitation = F.smooth_l1_loss(trajectory, target)
    energy = model.refiner.energy(
        trajectory,
        trajectory.detach(),
        batch["goal_state"],
        batch["sdf"],
        batch["map_bounds"],
        weights,
    ).mean()
    clearance = model.refiner.sample_sdf(
        batch["sdf"], trajectory, batch["map_bounds"]
    )
    collision_rate = (clearance < 0.0).float().mean()
    goal_error = torch.linalg.vector_norm(
        trajectory[:, -1] - batch["goal_state"][:, :2], dim=-1
    ).mean()
    return imitation, energy, collision_rate, goal_error


def train_epoch(model, loader, optimizer, args, device, tasks):
    model.selector.train()
    model.metric_head.train()
    model.flow.train()
    logs = {key: [] for key in (
        "loss", "task", "flow", "metric", "dgd", "collision", "goal_error"
    )}

    for raw in tqdm(loader, desc="V8.4 train"):
        batch = move_batch(raw, device)
        texts = task_texts(raw, batch["source_id"], args)
        weights, limits, limit_mask, penalty, task_vector = task_parameters(
            batch["source_id"], tasks, device
        )

        # 1. Encode RGB/text and predict a task-conditioned importance per region.
        budget = torch.full(
            (batch["image"].shape[0],), args.budget, device=device
        )
        encoded, selection, region_valid = model.encode_and_select(
            batch["image"], texts, budget
        )
        region_count = selection["logits"].shape[-1]
        k = max(1, round(args.budget * region_count))

        # 2. Straight-through Top-K: hard forward values, soft mask derivatives.
        soft_mask = fixed_mass_mask(
            selection["logits"], region_valid, k, args.mask_temperature
        )
        hard_mask = hard_topk_mask(selection["logits"], k)
        mask = hard_mask + soft_mask - soft_mask.detach()
        fused = model.fuse_mask(encoded, mask)

        # 3. The exact candidate teacher uses simulator/map metrics, not model input.
        target, _, oracle_cost = oracle_trajectory(
            batch, weights, limits, limit_mask, penalty
        )

        # 4. The pruned VLM predicts optimization coefficients for all candidates.
        predicted_metrics = model.predict_metrics(fused, batch)
        metric_loss = F.smooth_l1_loss(
            predicted_metrics[batch["candidate_valid"]],
            batch["candidate_metrics"][batch["candidate_valid"]],
        )
        base, probability, _ = soft_candidate_warm_start(
            predicted_metrics,
            batch,
            weights,
            limits,
            limit_mask,
            penalty,
            args.decision_temperature,
        )
        candidate_regret = expected_candidate_regret(
            probability, oracle_cost, batch["candidate_valid"]
        )

        # 5. Flow Matching distills the solver trajectory distribution.
        if args.lambda_flow > 0.0:
            flow_loss = model.flow.matching_loss(
                base,
                target,
                fused,
                batch["goal_state"],
                batch["ego_state"],
                task_vector,
            )
        else:
            flow_loss = fused.new_zeros(())
        flow_trajectory = model.flow.integrate(
            base,
            fused,
            batch["goal_state"],
            batch["ego_state"],
            task_vector,
            args.flow_steps,
        )

        # 6. ND refines safety, smoothness, length and terminal-goal constraints.
        refined = model.refiner(
            flow_trajectory,
            batch["goal_state"],
            batch["sdf"],
            batch["map_bounds"],
            weights,
            differentiable=True,
        )
        imitation, nd_energy, collision, goal_error = trajectory_metrics(
            model, refined, target, batch, weights
        )

        # 7. This is the downstream planning loss used for token distillation.
        task_loss = (
            candidate_regret
            + args.lambda_trajectory * imitation
            + args.lambda_constraint * nd_energy
        )
        teacher = decision_gradient_teacher(task_loss, soft_mask, region_valid)
        dgd = F.kl_div(
            F.log_softmax(selection["logits"], dim=-1),
            teacher,
            reduction="batchmean",
        )
        loss = (
            task_loss
            + args.lambda_metric * metric_loss
            + args.lambda_flow * flow_loss
            + args.lambda_dgd * dgd
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainable = (
            list(model.selector.parameters())
            + list(model.metric_head.parameters())
            + list(model.flow.parameters())
        )
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()

        logs["loss"].append(float(loss.detach()))
        logs["task"].append(float(task_loss.detach()))
        logs["flow"].append(float(flow_loss.detach()))
        logs["metric"].append(float(metric_loss.detach()))
        logs["dgd"].append(float(dgd.detach()))
        logs["collision"].append(float(collision.detach()))
        logs["goal_error"].append(float(goal_error.detach()))

    return {key: float(np.mean(value)) for key, value in logs.items()}


def evaluate_mode(model, encoded, selection, region_valid, batch, args, task_data, mode):
    weights, limits, limit_mask, penalty, task_vector = task_data
    region_count = selection["logits"].shape[-1]
    k = region_count if mode == "full" else max(1, round(args.budget * region_count))
    logits = selection["logits"]
    if mode == "random":
        logits = torch.rand_like(logits)
    fused, _ = model.fuse_topk(encoded, logits, region_valid, k)

    target, oracle_index, oracle_cost = oracle_trajectory(
        batch, weights, limits, limit_mask, penalty
    )
    predicted_metrics = model.predict_metrics(fused, batch)
    _, _, predicted_cost = soft_candidate_warm_start(
        predicted_metrics,
        batch,
        weights,
        limits,
        limit_mask,
        penalty,
        args.decision_temperature,
    )
    predicted_index = predicted_cost.argmin(-1)
    row = torch.arange(predicted_index.shape[0], device=predicted_index.device)
    base = batch["candidate_trajectories"][row, predicted_index, :, :2]

    flow_trajectory = model.flow.integrate(
        base,
        fused,
        batch["goal_state"],
        batch["ego_state"],
        task_vector,
        args.flow_steps,
    )
    refined = model.refiner(
        flow_trajectory,
        batch["goal_state"],
        batch["sdf"],
        batch["map_bounds"],
        weights,
        differentiable=False,
    )
    imitation, energy, collision, goal_error = trajectory_metrics(
        model, refined, target, batch, weights
    )
    chosen_cost = oracle_cost.gather(1, predicted_index[:, None]).squeeze(1)
    optimal_cost = oracle_cost.gather(1, oracle_index[:, None]).squeeze(1)
    maximum = oracle_cost.masked_fill(
        ~batch["candidate_valid"], -torch.inf
    ).max(-1).values
    regret = ((chosen_cost - optimal_cost) / (maximum - optimal_cost).clamp_min(1e-4)).mean()
    return {
        "regret": float(regret),
        "trajectory_error": float(imitation),
        "nd_energy": float(energy),
        "collision_rate": float(collision),
        "goal_error": float(goal_error),
        "tokens": float(k),
        "token_ratio": float(k / region_count),
    }


def validate(model, loader, args, device, tasks):
    model.selector.eval()
    model.metric_head.eval()
    model.flow.eval()
    logs = {
        mode: {key: [] for key in (
            "regret", "trajectory_error", "nd_energy", "collision_rate",
            "goal_error", "tokens", "token_ratio"
        )}
        for mode in ("full", "random", "learned")
    }
    for raw in tqdm(loader, desc="V8.4 validation"):
        batch = move_batch(raw, device)
        task_data = task_parameters(batch["source_id"], tasks, device)
        with torch.no_grad():
            budget = torch.full(
                (batch["image"].shape[0],), args.budget, device=device
            )
            encoded, selection, region_valid = model.encode_and_select(
                batch["image"], task_texts(raw, batch["source_id"], args), budget
            )
        for mode in logs:
            with torch.no_grad():
                result = evaluate_mode(
                    model,
                    encoded,
                    selection,
                    region_valid,
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
    parser.add_argument("--decision-temperature", type=float, default=0.25)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--nd-steps", type=int, default=6)
    parser.add_argument("--nd-step-size", type=float, default=0.15)
    parser.add_argument("--lambda-trajectory", type=float, default=1.0)
    parser.add_argument("--lambda-constraint", type=float, default=1.0)
    parser.add_argument("--lambda-metric", type=float, default=1.0)
    parser.add_argument("--lambda-flow", type=float, default=1.0)
    parser.add_argument("--lambda-dgd", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output", default="checkpoints/v84_decision_flow_nd.pt")
    parser.add_argument("--metrics", default="results/v84_metrics.json")
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()
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
    train_set = PlanningDatasetV84(args.data, image_side)
    val_set = PlanningDatasetV84(args.val, image_side)
    train_loader = DataLoader(
        train_set, args.batch, shuffle=True, num_workers=args.workers
    )
    val_loader = DataLoader(
        val_set, args.batch, shuffle=False, num_workers=args.workers
    )

    model = build_v84_model(args, device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.selector.load_state_dict(checkpoint["selector"])
        model.metric_head.load_state_dict(checkpoint["metric_head"])
        model.flow.load_state_dict(checkpoint["flow"])
    trainable = (
        list(model.selector.parameters())
        + list(model.metric_head.parameters())
        + list(model.flow.parameters())
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
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation,
        }
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
                    "flow": model.flow.state_dict(),
                    "validation": validation,
                },
                output,
            )


if __name__ == "__main__":
    main()
