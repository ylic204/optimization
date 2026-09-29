"""One complete inference: text agent -> pruned Qwen -> MILP -> trajectory."""

import argparse
import json
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from candidate_milp_solver_v83 import solve_candidate_milp
from mapped_optimization_dataset_v83 import MappedOptimizationDataset
from mapped_vlm_optimizer_v83 import build_model
from optimization_task_v83 import QwenOptimizationAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vlm-model", required=True)
    parser.add_argument("--task-text", required=True)
    parser.add_argument("--domain", choices=("nuplan", "pointnav"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget", type=float, default=0.15)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--output", default="solver_result.json")
    parser.add_argument("--trajectory-output", default="selected_trajectory.npy")
    parser.add_argument("--region-grid", type=int, default=9)
    parser.add_argument("--prune-layer", type=int, default=4)
    parser.add_argument("--max-text-length", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    # Step 1: load one exported planning instant.
    dataset = MappedOptimizationDataset(
        args.sample_dir, image_side=args.region_grid * 32
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

    # Step 2: load Qwen, token selector and coefficient-prediction head.
    model = build_model(args, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.selector.load_state_dict(checkpoint["selector"])
    model.metric_head.load_state_dict(checkpoint["metric_head"])
    model.selector.eval()
    model.metric_head.eval()

    # Step 3: the same Qwen model recognizes the natural-language objective.
    task = QwenOptimizationAgent(
        model.backbone.model, model.backbone.processor
    ).parse(args.task_text, args.domain)

    with torch.no_grad():
        # Step 4: score 81 visual regions from RGB and the task instruction.
        budget = torch.tensor([args.budget], device=device)
        encoded, selection, region_valid = model.encode_and_select(
            batch["image"], [args.task_text], budget
        )

        # Step 5: keep only K regions in later vision layers and the Qwen LLM.
        region_count = selection["logits"].shape[-1]
        k = max(1, round(args.budget * region_count))
        predicted_metrics, selected_regions = model.predict_with_topk(
            encoded, selection["logits"], region_valid, k, batch
        )

    # Step 6: construct and solve the language-conditioned mathematical model.
    predicted = predicted_metrics[0].float().cpu().numpy()
    valid = batch["candidate_valid"][0].cpu().numpy()
    solver_result = solve_candidate_milp(predicted, valid, task)

    # Step 7: return the selected continuous local trajectory to the controller.
    trajectory = batch["candidate_trajectories"][
        0, solver_result.selected_index
    ].cpu().numpy()
    np.save(args.trajectory_output, trajectory)

    report = {
        "task": asdict(task),
        "selected_candidate": solver_result.selected_index,
        "solver_objective": solver_result.objective,
        "predicted_metrics": solver_result.metric_values,
        "constraint_slacks": solver_result.constraint_slacks,
        "selected_qwen_regions": selected_regions[0].cpu().tolist(),
        "trajectory_file": args.trajectory_output,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
