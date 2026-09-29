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
    """Return the continuous expert target and the best discrete candidate.

    The expert trajectory supervises Flow Matching and final trajectory loss.
    The finite-candidate optimum is retained only for warm-start regret.
    """
    cost = augmented_candidate_cost(
        batch["candidate_metrics"], weights, limits, limit_mask, penalty
    )
    cost = cost.masked_fill(~batch["candidate_valid"], torch.inf)
    oracle_index = cost.argmin(-1)
    target = batch["expert_trajectory"]
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
    predicted_cost = predicted_cost.masked_fill(
        ~batch["candidate_valid"], torch.inf
    )
    probability = torch.softmax(-predicted_cost / temperature, dim=-1)
    candidate_xy = batch["candidate_trajectories"][..., :2]
    base = (probability[:, :, None, None] * candidate_xy).sum(1)
    return base, probability, predicted_cost


def candidate_regret_values(probability, oracle_cost, valid):
    minimum = oracle_cost.min(-1).values
    maximum = oracle_cost.masked_fill(~valid, -torch.inf).max(-1).values
    scale = (maximum - minimum).clamp_min(1e-4)
    normalized = (oracle_cost - minimum[:, None]) / scale[:, None]
    normalized = normalized.masked_fill(~valid, 0.0)
    return (probability * normalized).sum(-1)


def expected_candidate_regret(probability, oracle_cost, valid):
    return candidate_regret_values(probability, oracle_cost, valid).mean()


def decision_gradient_teacher(task_loss, soft_mask, valid):
    """Token utility is the fixed-budget descent direction of final task loss."""
    gradient = torch.autograd.grad(task_loss, soft_mask, retain_graph=False)[0]
    utility = -gradient * valid.float()
    utility = utility - utility.mean(-1, keepdim=True)
    utility = utility - utility.min(-1, keepdim=True).values
    utility = utility * valid.float()
    total = utility.sum(-1, keepdim=True)
    uniform = valid.float() / valid.float().sum(-1, keepdim=True).clamp_min(1.0)
    normalized = torch.where(
        total > 1e-8, utility / total.clamp_min(1e-8), uniform
    )
    return normalized.detach()


def trajectory_metric_values(model, trajectory, target, batch, weights):
    """Per-sample downstream values after Flow Matching and ND refinement."""
    imitation = F.smooth_l1_loss(
        trajectory, target, reduction="none"
    ).mean(dim=(1, 2))
    energy = model.refiner.energy(
        trajectory,
        trajectory.detach(),
        batch["goal_state"],
        batch["sdf"],
        batch["map_bounds"],
        weights,
    )
    clearance = model.refiner.sample_sdf(
        batch["sdf"], trajectory, batch["map_bounds"]
    )
    collision_rate = (clearance < 0.0).float().mean(-1)
    goal_error = torch.linalg.vector_norm(
        trajectory[:, -1] - batch["goal_state"][:, :2], dim=-1
    )
    return imitation, energy, collision_rate, goal_error


def trajectory_metrics(model, trajectory, target, batch, weights):
    values = trajectory_metric_values(model, trajectory, target, batch, weights)
    return tuple(value.mean() for value in values)


def planning_rollout(
    model,
    fused,
    batch,
    weights,
    limits,
    limit_mask,
    penalty,
    task_vector,
    target,
    oracle_cost,
    args,
    differentiable,
):
    """Run coefficient prediction, warm start, Flow and fixed-step ND."""
    predicted_metrics = model.predict_metrics(fused, batch)
    base, probability, predicted_cost = soft_candidate_warm_start(
        predicted_metrics,
        batch,
        weights,
        limits,
        limit_mask,
        penalty,
        args.decision_temperature,
    )
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
        differentiable=differentiable,
    )
    imitation, energy, collision, goal_error = trajectory_metric_values(
        model, refined, target, batch, weights
    )
    regret = candidate_regret_values(
        probability, oracle_cost, batch["candidate_valid"]
    )
    task_values = (
        regret
        + args.lambda_trajectory * imitation
        + args.lambda_constraint * energy
    )
    return {
        "predicted_metrics": predicted_metrics,
        "predicted_cost": predicted_cost,
        "base": base,
        "probability": probability,
        "flow_trajectory": flow_trajectory,
        "refined": refined,
        "regret": regret,
        "imitation": imitation,
        "energy": energy,
        "collision": collision,
        "goal_error": goal_error,
        "task_values": task_values,
    }


def train_epoch(model, loader, optimizer, args, device, tasks):
    model.selector.train()
    model.metric_head.train()
    model.flow.train()
    logs = {key: [] for key in (
        "loss", "task", "flow", "metric", "dgd", "latent", "value",
        "policy_distill", "collision", "goal_error"
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

        # 2. The continuous expert is independent of the finite candidate bank.
        target, _, oracle_cost = oracle_trajectory(
            batch, weights, limits, limit_mask, penalty
        )

        # 3. Full-information online teacher: every visual region reaches the
        # later Qwen blocks.  DGD is evaluated at this all-one information state.
        needs_full_teacher = (
            args.lambda_dgd > 0.0 or args.lambda_policy_distill > 0.0
        )
        teacher_probability = None
        if needs_full_teacher:
            teacher_mask = (
                region_valid.float()
                .detach()
                .requires_grad_(args.lambda_dgd > 0.0)
            )
            teacher_fused = model.fuse_mask(encoded, teacher_mask)
            teacher_rollout = planning_rollout(
                model,
                teacher_fused,
                batch,
                weights,
                limits,
                limit_mask,
                penalty,
                task_vector,
                target,
                oracle_cost,
                args,
                differentiable=args.lambda_dgd > 0.0,
            )
            teacher_probability = teacher_rollout["probability"].detach()
            if args.lambda_dgd > 0.0:
                teacher_utility = decision_gradient_teacher(
                    teacher_rollout["task_values"].mean(),
                    teacher_mask,
                    region_valid,
                )
            else:
                teacher_utility = None
            del teacher_rollout, teacher_fused, teacher_mask
        else:
            teacher_utility = None

        # 4. Student straight-through Top-K: hard values in the forward pass,
        # differentiable fixed-mass mask in the backward pass.
        soft_mask = fixed_mass_mask(
            selection["logits"], region_valid, k, args.mask_temperature
        )
        hard_mask = hard_topk_mask(selection["logits"], k)
        mask = hard_mask + soft_mask - soft_mask.detach()
        fused = model.fuse_mask(encoded, mask)

        # 5. The pruned Student predicts coefficients and a continuous plan.
        student = planning_rollout(
            model,
            fused,
            batch,
            weights,
            limits,
            limit_mask,
            penalty,
            task_vector,
            target,
            oracle_cost,
            args,
            differentiable=True,
        )
        predicted_metrics = student["predicted_metrics"]
        metric_loss = F.smooth_l1_loss(
            predicted_metrics[batch["candidate_valid"]],
            batch["candidate_metrics"][batch["candidate_valid"]],
        )

        # 6. Flow Matching targets the continuous expert trajectory rather than
        # the best member of a fixed eleven-trajectory bank.
        if args.lambda_flow > 0.0:
            flow_loss = model.flow.matching_loss(
                student["base"],
                target,
                fused,
                batch["goal_state"],
                batch["ego_state"],
                task_vector,
            )
        else:
            flow_loss = fused.new_zeros(())

        # 7. Explicit decision-related latent learning.
        full_latent = model.selector.pool_decision_latent(
            selection["z"], region_valid.float(), region_valid
        ).detach()
        selected_latent = model.selector.pool_decision_latent(
            selection["z"], soft_mask, region_valid
        )
        latent_loss = (1.0 - F.cosine_similarity(
            selected_latent, full_latent, dim=-1
        )).mean()
        predicted_set_value = model.selector.predict_set_value(
            selection["z"],
            selection["task_context"],
            soft_mask,
            region_valid,
            budget,
        )
        value_loss = F.smooth_l1_loss(
            predicted_set_value, student["task_values"].detach()
        )

        if teacher_probability is not None:
            policy_distill = F.kl_div(
                student["probability"].clamp_min(1e-8).log(),
                teacher_probability,
                reduction="batchmean",
            )
        else:
            policy_distill = fused.new_zeros(())
        if teacher_utility is not None:
            dgd = F.kl_div(
                F.log_softmax(selection["logits"], dim=-1),
                teacher_utility,
                reduction="batchmean",
            )
        else:
            dgd = fused.new_zeros(())

        # 8. The final task loss supervises the actually retained information.
        task_loss = student["task_values"].mean()
        loss = (
            task_loss
            + args.lambda_metric * metric_loss
            + args.lambda_flow * flow_loss
            + args.lambda_dgd * dgd
            + args.lambda_latent * latent_loss
            + args.lambda_value * value_loss
            + args.lambda_policy_distill * policy_distill
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
        logs["latent"].append(float(latent_loss.detach()))
        logs["value"].append(float(value_loss.detach()))
        logs["policy_distill"].append(float(policy_distill.detach()))
        logs["collision"].append(float(student["collision"].mean().detach()))
        logs["goal_error"].append(float(student["goal_error"].mean().detach()))

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
        "candidate_count": float(batch["candidate_count"].float().mean()),
        "expert_fallback_rate": float(
            batch["expert_is_fallback"].float().mean()
        ),
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
            "goal_error", "candidate_count", "expert_fallback_rate",
            "tokens", "token_ratio"
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
    parser.add_argument("--lambda-latent", type=float, default=0.25)
    parser.add_argument("--lambda-value", type=float, default=0.25)
    parser.add_argument("--lambda-policy-distill", type=float, default=0.25)
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
