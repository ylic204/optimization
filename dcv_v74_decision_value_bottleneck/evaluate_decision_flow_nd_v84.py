"""Standalone full/random/learned token evaluation for a V8.4 checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from flow_nd_planner_v84 import build_v84_model
from optimization_spec_v83 import OptimizationTask
from planning_dataset_v84 import PlanningDatasetV84
from train_decision_flow_nd_v84 import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--nuplan-task", required=True)
    parser.add_argument("--pointnav-task", required=True)
    parser.add_argument("--nuplan-text", default="")
    parser.add_argument("--pointnav-text", default="")
    parser.add_argument("--vlm-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="results/v84_test_metrics.json")
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
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dataset = PlanningDatasetV84(args.data, args.region_grid * 32)
    loader = DataLoader(
        dataset, args.batch, shuffle=False, num_workers=args.workers
    )
    tasks = [
        OptimizationTask.load(args.nuplan_task),
        OptimizationTask.load(args.pointnav_task),
    ]
    model = build_v84_model(args, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.selector.load_state_dict(checkpoint["selector"])
    model.metric_head.load_state_dict(checkpoint["metric_head"])
    model.flow.load_state_dict(checkpoint["flow"])

    result = validate(model, loader, args, device, tasks)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
