"""Run one V8.4 sample and export the selected regions and final trajectory."""

import argparse
from pathlib import Path

import numpy as np
import torch

from flow_nd_planner_v84 import build_v84_model
from optimization_spec_v83 import OptimizationTask
from planning_dataset_v84 import PlanningDatasetV84
from train_decision_flow_nd_v84 import (
    move_batch,
    soft_candidate_warm_start,
    task_parameters,
    task_texts,
)


def add_batch_dimension(sample):
    return {
        key: value[None] if torch.is_tensor(value) else [value]
        for key, value in sample.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--nuplan-task", required=True)
    parser.add_argument("--pointnav-task", required=True)
    parser.add_argument("--nuplan-text", default="")
    parser.add_argument("--pointnav-text", default="")
    parser.add_argument("--vlm-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="results/v84_prediction.npz")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--region-grid", type=int, default=9)
    parser.add_argument("--prune-layer", type=int, default=4)
    parser.add_argument("--max-text-length", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--budget", type=float, default=0.15)
    parser.add_argument("--decision-temperature", type=float, default=0.25)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--nd-steps", type=int, default=6)
    parser.add_argument("--nd-step-size", type=float, default=0.15)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dataset = PlanningDatasetV84(args.data, args.region_grid * 32)
    raw = add_batch_dimension(dataset[args.index])
    batch = move_batch(raw, device)
    tasks = [
        OptimizationTask.load(args.nuplan_task),
        OptimizationTask.load(args.pointnav_task),
    ]
    weights, limits, limit_mask, penalty, task_vector = task_parameters(
        batch["source_id"], tasks, device
    )

    model = build_v84_model(args, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.selector.load_state_dict(checkpoint["selector"])
    model.metric_head.load_state_dict(checkpoint["metric_head"])
    model.flow.load_state_dict(checkpoint["flow"])
    model.selector.eval()
    model.metric_head.eval()
    model.flow.eval()

    with torch.no_grad():
        budget = torch.full((1,), args.budget, device=device)
        encoded, selection, valid = model.encode_and_select(
            batch["image"], task_texts(raw, batch["source_id"], args), budget
        )
        region_count = selection["logits"].shape[-1]
        k = max(1, round(args.budget * region_count))
        fused, selected = model.fuse_topk(
            encoded, selection["logits"], valid, k
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
        candidate_index = predicted_cost.argmin(-1)
        base = batch["candidate_trajectories"][
            torch.arange(1, device=device), candidate_index, :, :2
        ]
        flow_trajectory = model.flow.integrate(
            base,
            fused,
            batch["goal_state"],
            batch["ego_state"],
            task_vector,
            args.flow_steps,
        )
        final_trajectory = model.refiner(
            flow_trajectory,
            batch["goal_state"],
            batch["sdf"],
            batch["map_bounds"],
            weights,
            differentiable=False,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sample_id=np.str_(raw["sample_id"][0]),
        task_text=np.str_(task_texts(raw, batch["source_id"], args)[0]),
        selected_region_indices=selected[0].cpu().numpy(),
        token_logits=selection["logits"][0].cpu().numpy(),
        predicted_candidate_metrics=predicted_metrics[0].cpu().numpy(),
        predicted_candidate_costs=predicted_cost[0].cpu().numpy(),
        candidate_valid=batch["candidate_valid"][0].cpu().numpy(),
        candidate_count=batch["candidate_count"][0].cpu().numpy(),
        selected_candidate_index=candidate_index[0].cpu().numpy(),
        candidate_trajectory=base[0].cpu().numpy(),
        expert_trajectory=batch["expert_trajectory"][0].cpu().numpy(),
        expert_is_fallback=batch["expert_is_fallback"][0].cpu().numpy(),
        flow_trajectory=flow_trajectory[0].cpu().numpy(),
        final_trajectory=final_trajectory[0].cpu().numpy(),
    )
    print(f"saved {output}")


if __name__ == "__main__":
    main()
