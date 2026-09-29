"""Train task-conditioned visual token pruning for solver-based planning.

The trainable model predicts the coefficients of an optimization problem.
The exact MILP solver is used for validation/inference; a differentiable soft
decision loss supplies gradients during training.  Flow Matching and ND are
not part of this file.
"""

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from candidate_milp_solver_v83 import solve_candidate_milp
from mapped_optimization_dataset_v83 import MappedOptimizationDataset
from mapped_vlm_optimizer_v83 import build_model
from optimization_spec_v83 import METRIC_NAMES, OptimizationTask


def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def budget_to_k(budget, region_count):
    return max(1, round(float(budget) * region_count))


def fixed_mass_mask(logits, valid, k, temperature=0.35):
    """Produce a differentiable region mask whose total mass is K."""
    with torch.no_grad():
        lower = logits.min(-1).values - 30.0
        upper = logits.max(-1).values + 30.0
        for _ in range(40):
            threshold = (lower + upper) / 2.0
            mass = (
                torch.sigmoid((logits - threshold[:, None]) / temperature)
                * valid.float()
            ).sum(-1)
            lower = torch.where(mass > k, threshold, lower)
            upper = torch.where(mass > k, upper, threshold)
        threshold = (lower + upper) / 2.0
    return (
        torch.sigmoid((logits - threshold[:, None]) / temperature)
        * valid.float()
    )


def hard_topk_mask(logits, k):
    indices = torch.topk(logits, k=k, dim=-1).indices
    return torch.zeros_like(logits).scatter(1, indices, 1.0)


def task_parameter_batch(source_id, nuplan_task, pointnav_task, device):
    """Choose language-generated weights, limits and penalties per sample."""
    tasks = (nuplan_task, pointnav_task)
    weight_table = torch.tensor(
        [nuplan_task.weight_vector, pointnav_task.weight_vector],
        dtype=torch.float32,
        device=device,
    )
    limit_table = torch.ones(
        2, len(METRIC_NAMES), dtype=torch.float32, device=device
    )
    limit_mask = torch.zeros(
        2, len(METRIC_NAMES), dtype=torch.float32, device=device
    )
    penalties = torch.tensor(
        [task.constraint_penalty for task in tasks],
        dtype=torch.float32,
        device=device,
    )
    for task_index, task in enumerate(tasks):
        for metric_name, upper_bound in task.limits.items():
            metric_index = METRIC_NAMES.index(metric_name)
            limit_table[task_index, metric_index] = float(upper_bound)
            limit_mask[task_index, metric_index] = 1.0
    return (
        weight_table[source_id],
        limit_table[source_id],
        limit_mask[source_id],
        penalties[source_id],
    )


def differentiable_task_loss(
    predicted_metrics,
    target_metrics,
    candidate_valid,
    weights,
    limits,
    limit_mask,
    constraint_penalty,
    temperature,
):
    """Approximate the discrete solver during back-propagation.

    The final solver takes argmin/MILP decisions.  During training we use a
    softmin over predicted costs and measure its expected ground-truth regret.
    """
    # For a one-hot path decision, the MILP slack penalty is exactly a hinge
    # penalty on each candidate's constraint violation.
    predicted_violation = F.relu(predicted_metrics - limits[:, None])
    target_violation = F.relu(target_metrics - limits[:, None])
    predicted_violation = (predicted_violation * limit_mask[:, None]).sum(-1)
    target_violation = (target_violation * limit_mask[:, None]).sum(-1)
    predicted_cost = (predicted_metrics * weights[:, None]).sum(-1)
    target_cost = (target_metrics * weights[:, None]).sum(-1)
    predicted_cost = predicted_cost + constraint_penalty[:, None] * predicted_violation
    target_cost = target_cost + constraint_penalty[:, None] * target_violation

    # Put different language tasks on a comparable temperature scale.
    normalizer = (
        weights.sum(-1) + constraint_penalty * limit_mask.sum(-1)
    ).clamp_min(1.0)
    predicted_cost = predicted_cost / normalizer[:, None]
    target_cost = target_cost / normalizer[:, None]

    large = torch.finfo(predicted_cost.dtype).max
    predicted_cost = predicted_cost.masked_fill(~candidate_valid, large)
    target_cost = target_cost.masked_fill(~candidate_valid, large)

    probability = torch.softmax(-predicted_cost / temperature, dim=-1)
    optimal_cost, optimal_index = target_cost.min(-1)
    maximum = target_cost.masked_fill(~candidate_valid, -torch.inf).max(-1).values
    scale = (maximum - optimal_cost).clamp_min(1e-4)
    normalized_target = (target_cost - optimal_cost[:, None]) / scale[:, None]
    normalized_target = normalized_target.masked_fill(~candidate_valid, 0.0)

    expected_regret = (probability * normalized_target).sum(-1).mean()
    optimal_ce = F.cross_entropy(-predicted_cost / temperature, optimal_index)
    metric_loss = F.smooth_l1_loss(
        predicted_metrics[candidate_valid], target_metrics[candidate_valid]
    )
    return expected_regret, optimal_ce, metric_loss


def decision_gradient_target(task_loss, soft_mask, valid):
    """Convert downstream planning gradients into a token-importance teacher."""
    gradient = torch.autograd.grad(task_loss, soft_mask, retain_graph=True)[0]
    descent = -gradient * valid.float()

    # Token budget is fixed, so useful directions must sum to zero.
    descent = descent - descent.mean(-1, keepdim=True)

    # Shift the tangent direction into a non-negative target distribution.
    importance = descent - descent.min(-1, keepdim=True).values
    importance = importance * valid.float()
    importance = importance / importance.sum(-1, keepdim=True).clamp_min(1e-8)
    return importance.detach()


def dgd_loss(region_logits, gradient_teacher):
    predicted = torch.log_softmax(region_logits, dim=-1)
    return F.kl_div(predicted, gradient_teacher, reduction="batchmean")


def solver_metrics(predicted, target, valid, sources, tasks):
    """Call the real SciPy MILP and score its selected path."""
    regrets = []
    violations = []
    for row in range(predicted.shape[0]):
        task = tasks[int(sources[row])]
        result = solve_candidate_milp(predicted[row], valid[row], task)
        weights = np.asarray(task.weight_vector)
        true_cost = target[row] @ weights
        feasible_cost = true_cost[valid[row]]
        chosen_cost = true_cost[result.selected_index]
        scale = max(float(feasible_cost.max() - feasible_cost.min()), 1e-4)
        regrets.append((chosen_cost - feasible_cost.min()) / scale)
        violations.append(sum(result.constraint_slacks.values()))
    return float(np.mean(regrets)), float(np.mean(violations))


def run_epoch(model, loader, optimizer, args, device, tasks, training):
    model.selector.train(training)
    model.metric_head.train(training)
    logs = {key: [] for key in (
        "loss", "task_regret", "optimal_ce", "metric", "dgd",
        "solver_regret", "solver_violation",
    )}

    for raw_batch in tqdm(loader, desc="train" if training else "validation"):
        batch = move_batch(raw_batch, device)
        texts = list(raw_batch["task_text"])
        budget = torch.full(
            (batch["image"].shape[0],), args.budget, device=device
        )

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            # Step 1: run frozen early Qwen layers and the trainable selector.
            encoded, selection, region_valid = model.encode_and_select(
                batch["image"], texts, budget
            )
            region_count = selection["logits"].shape[-1]
            k = budget_to_k(args.budget, region_count)

            if training:
                # Step 2: hard values in the forward pass, soft mask derivatives.
                soft_mask = fixed_mass_mask(
                    selection["logits"], region_valid, k, args.mask_temperature
                )
                hard_mask = hard_topk_mask(selection["logits"], k)
                straight_through = hard_mask + soft_mask - soft_mask.detach()
                predicted = model.predict_with_mask(
                    encoded, straight_through, batch
                )
            else:
                # Validation executes the actual compute-saving Top-K path.
                predicted, _ = model.predict_with_topk(
                    encoded, selection["logits"], region_valid, k, batch
                )

            # Step 3: construct the language-conditioned differentiable objective.
            weights, limits, limit_mask, penalty = task_parameter_batch(
                batch["source_id"], tasks[0], tasks[1], device
            )
            task_regret, optimal_ce, metric_loss = differentiable_task_loss(
                predicted,
                batch["candidate_metrics"],
                batch["candidate_valid"],
                weights,
                limits,
                limit_mask,
                penalty,
                args.decision_temperature,
            )
            # Task terms evaluate the planning decision. Metric regression is
            # an auxiliary calibration loss for the solver coefficients.
            task_objective = task_regret + args.lambda_ce * optimal_ce
            primary = task_objective + args.lambda_metric * metric_loss

            # Step 4: distill the task-loss gradient into the token selector.
            if training:
                gradient_teacher = decision_gradient_target(
                    task_objective, soft_mask, region_valid
                )
                dgd = dgd_loss(selection["logits"], gradient_teacher)
                loss = primary + args.lambda_dgd * dgd
            else:
                dgd = predicted.sum() * 0.0
                loss = primary

        if training:
            # Step 5: only selector and metric head are updated; Qwen stays frozen.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.selector.parameters())
                + list(model.metric_head.parameters()),
                5.0,
            )
            optimizer.step()

        # Step 6: evaluate the exact discrete optimization decision on CPU.
        solver_regret, violation = solver_metrics(
            predicted.detach().cpu().numpy(),
            batch["candidate_metrics"].cpu().numpy(),
            batch["candidate_valid"].cpu().numpy(),
            batch["source_id"].cpu().numpy(),
            tasks,
        )
        logs["loss"].append(float(loss.detach()))
        logs["task_regret"].append(float(task_regret.detach()))
        logs["optimal_ce"].append(float(optimal_ce.detach()))
        logs["metric"].append(float(metric_loss.detach()))
        logs["dgd"].append(float(dgd.detach()))
        logs["solver_regret"].append(solver_regret)
        logs["solver_violation"].append(violation)

    return {key: float(np.mean(values)) for key, values in logs.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--nuplan-task", required=True)
    parser.add_argument("--pointnav-task", required=True)
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
    parser.add_argument("--lambda-ce", type=float, default=0.25)
    parser.add_argument("--lambda-metric", type=float, default=1.0)
    parser.add_argument("--lambda-dgd", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", default="checkpoints/v83_optimizer.pt")
    parser.add_argument("--metrics", default="results/v83_metrics.json")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    # The language agent is executed before training and its JSON is cached.
    tasks = [
        OptimizationTask.load(args.nuplan_task),
        OptimizationTask.load(args.pointnav_task),
    ]

    image_side = args.region_grid * 32
    train_set = MappedOptimizationDataset(args.data, image_side)
    val_set = MappedOptimizationDataset(args.val, image_side)
    train_loader = DataLoader(
        train_set, args.batch, shuffle=True, num_workers=args.workers
    )
    val_loader = DataLoader(
        val_set, args.batch, shuffle=False, num_workers=args.workers
    )

    model = build_model(args, device)
    parameters = list(model.selector.parameters()) + list(
        model.metric_head.parameters()
    )
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)

    output_path = Path(args.output)
    metrics_path = Path(args.metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    best_regret = float("inf")
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(
            model, train_loader, optimizer, args, device, tasks, True
        )
        validation = run_epoch(
            model, val_loader, None, args, device, tasks, False
        )
        record = {"epoch": epoch, "train": train, "validation": validation}
        history.append(record)
        metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(record, indent=2))

        if validation["solver_regret"] < best_regret:
            best_regret = validation["solver_regret"]
            torch.save(
                {
                    "selector": model.selector.state_dict(),
                    "metric_head": model.metric_head.state_dict(),
                    "args": vars(args),
                    "tasks": [asdict(task) for task in tasks],
                },
                output_path,
            )


if __name__ == "__main__":
    main()
